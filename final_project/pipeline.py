"""Collect NYC street-maintenance data, load SQLite, and train the two scores.

The dashboard reads the database only. Running this file rebuilds it.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DB_PATH = DATA / "nyc_infrastructure.sqlite"
SCHEMA_PATH = ROOT / "schema.sql"
POPULATION_PDF = ROOT / "reference" / "nyc-cd-decennial-2020-2010.pdf"
PAGE_SIZE = 50_000
OPEN_311 = "2022-01-01T00:00:00"
FIRST_WEEK = "2022-01-03"

BOROUGH_CODE = {
    "MANHATTAN": 1,
    "BRONX": 2,
    "BROOKLYN": 3,
    "QUEENS": 4,
    "STATEN ISLAND": 5,
}
BOROUGH_NAME = {code: name.title() if name != "STATEN ISLAND" else "Staten Island"
                for name, code in BOROUGH_CODE.items()}
BOROUGH_NAME[2] = "Bronx"

COMPLAINT_TYPES = [
    "Street Condition",
    "Sidewalk Condition",
    "Highway Condition",
    "Street Light Condition",
    "Traffic Signal Condition",
    "Curb Condition",
    "Bridge Condition",
    "Street Sign - Damaged",
    "Street Sign - Dangling",
    "Street Sign - Missing",
]
TYPE_GROUPS = {
    "Street Condition": "Street Condition",
    "Sidewalk Condition": "Sidewalk Condition",
    "Highway Condition": "Highway Condition",
    "Street Light Condition": "Street Light Condition",
    "Traffic Signal Condition": "Traffic Signal Condition",
    "Curb Condition": "Curb Condition",
    "Bridge Condition": "Bridge Condition",
    "Street Sign - Damaged": "Street Sign",
    "Street Sign - Dangling": "Street Sign",
    "Street Sign - Missing": "Street Sign",
}
GROUP_ORDER = [
    "Street Condition",
    "Sidewalk Condition",
    "Highway Condition",
    "Curb Condition",
    "Bridge Condition",
    "Street Light Condition",
    "Traffic Signal Condition",
    "Street Sign",
]

POISSON_FEATURES = [
    "weekofyear",
    "borough_code",
    "freeze_thaw_days",
    "precip",
    "snow",
    "share_fair_or_poor",
    "population",
    "lag52_filled",
]
HGB_FEATURES = POISSON_FEATURES + [
    "lag1",
    "lag2",
    "lag4",
    "roll4",
    "boro_cd",
    "type_code",
    "month",
]
UNFIXED_FEATURES = [
    "type_code",
    "is_pothole",
    "borough_code",
    "share_fair_or_poor",
    "month",
    "freeze_thaw_days",
    "precip",
    "backlog",
]

UNFIXED_SQL = """
UPDATE complaint
SET left_unfixed = CASE
  WHEN closed_at IS NOT NULL AND closed_at < created_at THEN NULL
  WHEN closed_at IS NOT NULL AND due_at IS NOT NULL AND closed_at > due_at THEN 1
  WHEN closed_at IS NOT NULL AND due_at IS NOT NULL THEN 0
  WHEN closed_at IS NOT NULL AND closed_at > datetime(created_at, '+14 days') THEN 1
  WHEN closed_at IS NOT NULL THEN 0
  WHEN due_at IS NOT NULL AND due_at < ? THEN 1
  WHEN due_at IS NULL AND datetime(created_at, '+14 days') < ? THEN 1
  ELSE NULL
END
"""

HOURS_SQL = """
UPDATE complaint
SET hours_to_close = ROUND((julianday(closed_at) - julianday(created_at)) * 24, 2)
WHERE closed_at IS NOT NULL
  AND closed_at >= created_at
"""

WEEKLY_SQL = """
INSERT INTO weekly_demand (
  boro_cd, complaint_type, week_start, complaint_count, rate_per_10k, unfixed_count, came_back_count
)
SELECT
  d.boro_cd,
  g.complaint_type,
  c.week_start,
  COALESCE(a.complaint_count, 0),
  CASE
    WHEN d.population > 0 THEN COALESCE(a.complaint_count, 0) * 10000.0 / d.population
  END,
  COALESCE(a.unfixed_count, 0),
  COALESCE(a.came_back_count, 0)
FROM community_district AS d
CROSS JOIN type_group_list AS g
CROSS JOIN week_calendar AS c
LEFT JOIN (
  SELECT
    boro_cd,
    type_group AS complaint_type,
    week_start,
    COUNT(*) AS complaint_count,
    COALESCE(SUM(left_unfixed), 0) AS unfixed_count,
    COALESCE(SUM(came_back), 0) AS came_back_count
  FROM complaint
  GROUP BY boro_cd, type_group, week_start
) AS a
  ON a.boro_cd = d.boro_cd
 AND a.complaint_type = g.complaint_type
 AND a.week_start = c.week_start
"""

DURATION_SQL = """
INSERT INTO duration_by_borough (borough_code, complaint_type, n, median_hours, p90_hours)
WITH ranked AS (
  SELECT
    borough_code,
    type_group AS complaint_type,
    hours_to_close,
    ROW_NUMBER() OVER (
      PARTITION BY borough_code, type_group
      ORDER BY hours_to_close
    ) AS rn,
    COUNT(*) OVER (PARTITION BY borough_code, type_group) AS n
  FROM complaint
  WHERE hours_to_close IS NOT NULL
)
SELECT
  borough_code,
  complaint_type,
  MAX(n) AS n,
  MAX(CASE WHEN rn = ((n + 1) / 2) THEN hours_to_close END) AS median_hours,
  MAX(CASE
    WHEN rn = CASE
      WHEN CAST(ROUND(n * 0.9) AS INT) < 1 THEN 1
      ELSE CAST(ROUND(n * 0.9) AS INT)
    END THEN hours_to_close
  END) AS p90_hours
FROM ranked
GROUP BY borough_code, complaint_type
"""

sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.float64, float)


def log(message: str) -> None:
    print(message, flush=True)


def parse_community_board(value) -> int | None:
    """Map '12 MANHATTAN' to boro_cd 112. Unspecified boards return None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value).strip().upper())
    match = re.match(r"^(\d{1,2})\s+([A-Z ]+)$", text)
    if not match:
        return None
    number = int(match.group(1))
    borough = match.group(2).strip()
    if borough not in BOROUGH_CODE or not 1 <= number <= 18:
        return None
    return BOROUGH_CODE[borough] * 100 + number


def type_group_for(complaint_type: str) -> str | None:
    return TYPE_GROUPS.get(complaint_type)


def rating_band(rating: float) -> str | None:
    """DOT groups on this file: Good 8-10, Fair 4 up to 8, Poor below 4."""
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return None
    if rating >= 8:
        return "good"
    if rating >= 4:
        return "fair"
    if rating > 0:
        return "poor"
    return None


def week_start_monday(timestamps: pd.Series) -> pd.Series:
    """Monday that starts the week containing each timestamp."""
    return timestamps.dt.to_period("W-SUN").dt.start_time.dt.normalize()


def history_count(counts: np.ndarray, weeks_back: int):
    index = len(counts) - weeks_back
    if index < 0:
        return None
    return float(counts[index])


def seasonal_count(counts: np.ndarray, weeks_ahead: int):
    """Count from 52 weeks before a future week. counts ends at the latest complete week."""
    index = len(counts) - 1 + weeks_ahead - 52
    if index < 0 or index >= len(counts):
        return None
    return float(counts[index])


def mark_came_back(frame: pd.DataFrame) -> np.ndarray:
    """Flag a request when the same cross streets had a close in the prior 90 days."""
    flag = np.zeros(len(frame), dtype=np.int64)
    work = frame.loc[
        :,
        ["borough_code", "type_group", "cross_street_1", "cross_street_2", "created_at", "closed_at"],
    ].copy()
    work["_i"] = np.arange(len(frame))
    work = work.dropna(subset=["cross_street_1", "cross_street_2", "created_at"])
    if work.empty:
        return flag
    ninety = np.timedelta64(90, "D")
    grouped = work.groupby(
        ["borough_code", "type_group", "cross_street_1", "cross_street_2"],
        sort=False,
    )
    for _, group in grouped:
        if len(group) < 2:
            continue
        group = group.sort_values("created_at")
        created = group["created_at"].to_numpy()
        closed = group["closed_at"].to_numpy()
        closed_ok = np.sort(closed[~pd.isna(closed)])
        if closed_ok.size == 0:
            continue
        low = np.searchsorted(closed_ok, created - ninety, side="right")
        high = np.searchsorted(closed_ok, created, side="left")
        hit = high > low
        if hit.any():
            flag[group["_i"].to_numpy()[hit]] = 1
    return flag


def parse_population_text(text: str) -> pd.DataFrame:
    """Read 2020 total population from the DCP community-district profile text."""
    cleaned = text.replace("\f", "\n")
    chunks = re.split(
        r"(?=(?:Bronx|Brooklyn|Manhattan|Queens|Staten Island) Community District \d+)",
        cleaned,
    )
    rows = []
    for chunk in chunks:
        match = re.match(
            r"\s*(Bronx|Brooklyn|Manhattan|Queens|Staten Island) Community District (\d+)",
            chunk,
        )
        if not match:
            continue
        borough = match.group(1)
        number = int(match.group(2))
        age = chunk.split("MUTUALLY")[0]
        line = next((ln for ln in age.splitlines() if ln.strip().startswith("Total population")), None)
        if line is None:
            continue
        numbers = re.findall(r"\d{1,3}(?:,\d{3})+", line)
        if len(numbers) < 2:
            continue
        population = int(numbers[1].replace(",", ""))
        code = BOROUGH_CODE[borough.upper()]
        rows.append(
            {
                "boro_cd": code * 100 + number,
                "borough_code": code,
                "district_number": number,
                "population": population,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("boro_cd")


def _download(url: str, dest: Path, timeout: int = 240) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    last_error = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "cs210-nyc-infrastructure"})
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            temporary.replace(dest)
            return
        except Exception as error:  # network and HTTP errors are retried
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Download failed for {url[:120]}: {last_error}")


def _csv_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _download_pages(kind: str, url_for_offset, header_token: str) -> list[Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    done = RAW / f"{kind}.done"
    if done.exists():
        pages = sorted(RAW.glob(f"{kind}_*.csv"))
        log(f"{kind}: using {len(pages)} cached pages")
        return pages
    pages = []
    for page in range(40):
        path = RAW / f"{kind}_{page:03d}.csv"
        offset = page * PAGE_SIZE
        if path.exists():
            first = path.open().readline()
            rows = _csv_rows(path) if header_token in first else 0
            if rows >= PAGE_SIZE:
                pages.append(path)
                continue
        log(f"{kind}: downloading page {page + 1}")
        _download(url_for_offset(offset), path)
        first = path.open().readline()
        if header_token not in first:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"{kind} page {page} was not a CSV")
        rows = _csv_rows(path)
        log(f"{kind}: page {page + 1} has {rows:,} rows")
        pages.append(path)
        if rows < PAGE_SIZE:
            done.write_text(str(page + 1))
            return pages
    raise RuntimeError(f"{kind} did not finish within 40 pages")


def download_311() -> list[Path]:
    quoted = ", ".join("'" + name.replace("'", "''") + "'" for name in COMPLAINT_TYPES)
    where = f"created_date >= '{OPEN_311}' AND complaint_type in ({quoted})"

    def url_for_offset(offset: int) -> str:
        query = urllib.parse.urlencode(
            {
                "$select": ",".join(
                    [
                        "unique_key",
                        "created_date",
                        "closed_date",
                        "due_date",
                        "agency",
                        "agency_name",
                        "complaint_type",
                        "descriptor",
                        "status",
                        "community_board",
                        "borough",
                        "latitude",
                        "longitude",
                        "incident_zip",
                        "cross_street_1",
                        "cross_street_2",
                    ]
                ),
                "$where": where,
                "$order": "unique_key",
                "$limit": PAGE_SIZE,
                "$offset": offset,
            }
        )
        return "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?" + query

    return _download_pages("requests311", url_for_offset, "unique_key")


def download_pavement() -> list[Path]:
    where = "systemrating > 0 AND ismultipass='0'"

    def url_for_offset(offset: int) -> str:
        query = urllib.parse.urlencode(
            {
                "$select": "boroughname,systemrating,locationgeometry_stlength,inspection",
                "$where": where,
                "$limit": PAGE_SIZE,
                "$offset": offset,
            }
        )
        return "https://data.cityofnewyork.us/resource/6yyb-pb25.csv?" + query

    return _download_pages("pavement", url_for_offset, "boroughname")


def download_districts() -> dict:
    dest = RAW / "districts.geojson"
    if not dest.exists():
        log("districts: downloading community district boundaries")
        _download("https://data.cityofnewyork.us/resource/5crt-au7u.geojson?$limit=100", dest, timeout=120)
    payload = json.loads(dest.read_text())
    return payload


def download_weather() -> pd.DataFrame:
    dest = RAW / "weather.json"
    if not dest.exists():
        end = (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=1)).date()
        last_error = None
        for _ in range(5):
            query = urllib.parse.urlencode(
                {
                    "latitude": "40.7829",
                    "longitude": "-73.9654",
                    "start_date": "2021-01-01",
                    "end_date": end.isoformat(),
                    "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum,snowfall_sum",
                    "temperature_unit": "fahrenheit",
                    "timezone": "America/New_York",
                }
            )
            url = "https://archive-api.open-meteo.com/v1/archive?" + query
            try:
                log(f"weather: downloading Central Park daily series through {end.isoformat()}")
                _download(url, dest, timeout=120)
                break
            except Exception as error:
                last_error = error
                dest.unlink(missing_ok=True)
                end = end - pd.Timedelta(days=2).to_pytimedelta()
        if not dest.exists():
            raise RuntimeError(f"Weather download failed: {last_error}")
    payload = json.loads(dest.read_text())
    daily = payload["daily"]
    frame = pd.DataFrame(
        {
            "weather_date": daily["time"],
            "tmin": daily["temperature_2m_min"],
            "tmax": daily["temperature_2m_max"],
            "precip": daily["precipitation_sum"],
            "snow": daily["snowfall_sum"],
        }
    )
    frame["freeze_thaw"] = ((frame["tmin"] < 32) & (frame["tmax"] > 32)).astype(int)
    return frame


def load_population() -> pd.DataFrame:
    if not POPULATION_PDF.exists():
        raise FileNotFoundError(f"Missing census profile PDF at {POPULATION_PDF}")
    text_path = RAW / "population.txt"
    RAW.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftotext", "-layout", str(POPULATION_PDF), str(text_path)],
        check=True,
        capture_output=True,
    )
    frame = parse_population_text(text_path.read_text(errors="replace"))
    if len(frame) != 59:
        raise RuntimeError(f"Expected 59 community districts in the census profile, found {len(frame)}")
    bronx_1 = int(frame.loc[frame["boro_cd"] == 201, "population"].iloc[0])
    if bronx_1 != 100929:
        raise RuntimeError(f"Bronx Community District 1 parsed as {bronx_1}, expected 100929")
    return frame


def _round_coordinates(value):
    if isinstance(value, list):
        if value and isinstance(value[0], (int, float)) and not isinstance(value[0], bool):
            return [round(float(item), 5) for item in value]
        return [_round_coordinates(item) for item in value]
    return value


def _short_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text[:80]


def _norm_street(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value).upper().strip())
    if text in {"", "N/A", "NA", "NONE", "UNKNOWN", ".", "-", "NULL"} or len(text) < 2:
        return None
    return text[:80]


def _as_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def prepare_complaints(pages: list[Path]) -> tuple[pd.DataFrame, dict]:
    log(f"requests: reading {len(pages)} pages")
    frames = [pd.read_csv(path, dtype=str) for path in pages]
    raw = pd.concat(frames, ignore_index=True)
    audit = {"downloaded": int(len(raw))}
    raw = raw.drop_duplicates("unique_key")
    audit["duplicates_removed"] = audit["downloaded"] - int(len(raw))
    raw["created_at"] = _as_timestamp(raw["created_date"])
    raw["closed_at"] = _as_timestamp(raw["closed_date"])
    raw["due_at"] = _as_timestamp(raw["due_date"])
    bad_date = int(raw["created_at"].isna().sum())
    raw = raw.loc[raw["created_at"].notna()].copy()
    audit["bad_created_date"] = bad_date
    raw["type_group"] = raw["complaint_type"].map(TYPE_GROUPS)
    unknown_type = int(raw["type_group"].isna().sum())
    raw = raw.loc[raw["type_group"].notna()].copy()
    audit["unknown_type"] = unknown_type
    raw["boro_cd"] = raw["community_board"].map(parse_community_board)
    audit["unspecified_district"] = int(raw["boro_cd"].isna().sum())
    raw = raw.loc[raw["boro_cd"].notna()].copy()
    raw["boro_cd"] = raw["boro_cd"].astype(int)
    raw["borough_code"] = raw["boro_cd"] // 100
    raw["week_dt"] = week_start_monday(raw["created_at"])
    raw["week_start"] = raw["week_dt"].dt.strftime("%Y-%m-%d")
    raw["is_pothole"] = raw["descriptor"].fillna("").str.upper().str.contains("POTHOLE").astype(int)
    raw["cross_street_1"] = raw["cross_street_1"].map(_norm_street)
    raw["cross_street_2"] = raw["cross_street_2"].map(_norm_street)
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    raw["descriptor"] = raw["descriptor"].fillna("").str.slice(0, 120)
    raw["descriptor"] = raw["descriptor"].where(raw["descriptor"].str.len() > 0, None)
    for column in ("agency", "agency_name", "status", "incident_zip"):
        raw[column] = raw[column].map(_short_text)
    log("requests: flagging came-back complaints")
    raw["came_back"] = mark_came_back(raw)
    audit["came_back"] = int(raw["came_back"].sum())
    audit["kept"] = int(len(raw))
    return raw, audit


def pavement_by_borough(pages: list[Path]) -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(path) for path in pages], ignore_index=True)
    frame["rating"] = pd.to_numeric(frame["systemrating"], errors="coerce")
    frame["length_ft"] = pd.to_numeric(frame["locationgeometry_stlength"], errors="coerce")
    frame = frame.loc[(frame["rating"] > 0) & (frame["length_ft"] > 0)].copy()
    frame["band"] = frame["rating"].map(rating_band)
    frame["borough_code"] = frame["boroughname"].str.upper().map(BOROUGH_CODE)
    frame = frame.loc[frame["borough_code"].notna() & frame["band"].notna()].copy()
    frame["inspection"] = pd.to_datetime(frame["inspection"], errors="coerce")
    rows = []
    for code, group in frame.groupby("borough_code"):
        total = float(group["length_ft"].sum())
        def share(band: str) -> float:
            return float(group.loc[group["band"] == band, "length_ft"].sum()) / total
        rows.append(
            {
                "borough_code": int(code),
                "miles_rated": total / 5280.0,
                "mean_rating": float(np.average(group["rating"], weights=group["length_ft"])),
                "share_good": share("good"),
                "share_fair": share("fair"),
                "share_poor": share("poor"),
                "share_fair_or_poor": share("fair") + share("poor"),
                "newest_inspection": group["inspection"].max().strftime("%Y-%m-%d"),
                "segment_count": int(len(group)),
                "grain": "borough",
            }
        )
    return pd.DataFrame(rows)


def weekly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["weather_date"] = pd.to_datetime(frame["weather_date"])
    frame["week_start"] = week_start_monday(frame["weather_date"]).dt.strftime("%Y-%m-%d")
    grouped = frame.groupby("week_start", as_index=False).agg(
        freeze_thaw_days=("freeze_thaw", "sum"),
        precip=("precip", "sum"),
        snow=("snow", "sum"),
        tmin_avg=("tmin", "mean"),
        days_observed=("weather_date", "count"),
    )
    grouped["weather_source"] = "central_park_observed"
    return grouped


def last_complete_monday(max_created: pd.Timestamp) -> pd.Timestamp:
    start = week_start_monday(pd.Series([max_created])).iloc[0]
    week_end = start + pd.Timedelta(days=6)
    if max_created.normalize() < week_end:
        return start - pd.Timedelta(days=7)
    return start


def calendar_weeks(last_monday: pd.Timestamp) -> list[str]:
    weeks = pd.date_range(FIRST_WEEK, last_monday, freq="W-MON")
    return [week.strftime("%Y-%m-%d") for week in weeks]


def add_seasonal_weather(weekly: pd.DataFrame, weeks: list[str]) -> pd.DataFrame:
    """Copy last year's Central Park week onto weeks the archive does not cover yet."""
    frame = weekly.set_index("week_start")
    for week in weeks:
        if week in frame.index and int(frame.at[week, "days_observed"]) >= 7:
            continue
        prior = (pd.Timestamp(week) - pd.Timedelta(days=364)).strftime("%Y-%m-%d")
        if prior not in frame.index:
            continue
        copied = frame.loc[prior].copy()
        copied["weather_source"] = "seasonal_prior_year"
        frame.loc[week] = copied
    return frame.reset_index()


def _executemany(conn: sqlite3.Connection, sql: str, frame: pd.DataFrame, columns: list[str]) -> None:
    if frame.empty:
        return
    cursor = conn.cursor()
    for start in range(0, len(frame), 20000):
        part = frame.iloc[start:start + 20000][columns]
        values = part.to_numpy(dtype=object)
        values = np.where(pd.isna(values), None, values)
        cursor.executemany(sql, values.tolist())
    conn.commit()


def _audit(conn: sqlite3.Connection, step: str, row_count: int, note: str = "") -> None:
    conn.execute(
        "INSERT INTO load_audit (step, row_count, note) VALUES (?, ?, ?)",
        (step, int(row_count), note),
    )


def build_database(complaints: pd.DataFrame, audit: dict, districts_geo: dict,
                   population: pd.DataFrame, pavement: pd.DataFrame, daily_weather: pd.DataFrame) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    temporary = DB_PATH.with_suffix(".building.sqlite")
    if temporary.exists():
        temporary.unlink()
    conn = sqlite3.connect(temporary)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())

    conn.executemany(
        "INSERT INTO borough (borough_code, borough_name) VALUES (?, ?)",
        list(BOROUGH_NAME.items()),
    )
    population = population.set_index("boro_cd")
    district_rows = []
    for feature in districts_geo["features"]:
        props = feature["properties"]
        boro_cd = int(props.get("boro_cd") or props.get("BoroCD"))
        borough_code = boro_cd // 100
        number = boro_cd % 100
        # Codes above 18 are joint-interest areas (parks, airports), not community districts.
        if borough_code not in BOROUGH_NAME or not 1 <= number <= 18 or boro_cd not in population.index:
            continue
        shape_area = float(props["shape_area"]) if props.get("shape_area") not in (None, "") else None
        pop = int(population.at[boro_cd, "population"]) if boro_cd in population.index else None
        density = None
        if pop and shape_area:
            density = pop / (shape_area / 27_878_400.0)
        district_rows.append(
            (
                boro_cd,
                borough_code,
                number,
                f"{BOROUGH_NAME[borough_code]} CD {number}",
                shape_area,
                pop,
                "2020 Census, NYC DCP Population Division, September 2021",
                density,
                json.dumps(_round_coordinates(feature["geometry"])),
            )
        )
    conn.executemany(
        """
        INSERT INTO community_district (
          boro_cd, borough_code, district_number, district_name, shape_area,
          population, pop_vintage, density_per_sq_mi, geometry
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        district_rows,
    )
    known_districts = {row[0] for row in district_rows}
    before = len(complaints)
    complaints = complaints.loc[complaints["boro_cd"].isin(known_districts)].copy()
    audit["district_not_in_map"] = before - len(complaints)

    agencies = (
        complaints.loc[complaints["agency"].notna(), ["agency", "agency_name"]]
        .drop_duplicates("agency")
    )
    conn.executemany(
        "INSERT OR IGNORE INTO agency (agency_code, agency_name) VALUES (?, ?)",
        list(agencies.itertuples(index=False, name=None)),
    )
    conn.executemany(
        "INSERT INTO complaint_type (complaint_type, type_group) VALUES (?, ?)",
        list(TYPE_GROUPS.items()),
    )
    conn.executemany(
        "INSERT INTO type_group_list (complaint_type) VALUES (?)",
        [(name,) for name in GROUP_ORDER],
    )

    last_monday = last_complete_monday(complaints["created_at"].max())
    weeks = calendar_weeks(last_monday)
    if len(weeks) <= 16:
        raise RuntimeError("Not enough complete weeks to hold out 12 weeks and train")
    conn.executemany("INSERT INTO week_calendar (week_start) VALUES (?)", [(week,) for week in weeks])
    future_weeks = [
        (pd.Timestamp(weeks[-1]) + pd.Timedelta(days=7 * step)).strftime("%Y-%m-%d")
        for step in range(1, 5)
    ]

    _executemany(
        conn,
        """
        INSERT INTO weather_day (weather_date, tmin, tmax, precip, snow, freeze_thaw)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        daily_weather,
        ["weather_date", "tmin", "tmax", "precip", "snow", "freeze_thaw"],
    )
    weekly = add_seasonal_weather(weekly_weather(daily_weather), future_weeks)
    _executemany(
        conn,
        """
        INSERT INTO weather_week (
          week_start, freeze_thaw_days, precip, snow, tmin_avg, days_observed, weather_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        weekly,
        ["week_start", "freeze_thaw_days", "precip", "snow", "tmin_avg", "days_observed", "weather_source"],
    )
    _executemany(
        conn,
        """
        INSERT INTO district_pavement (
          borough_code, miles_rated, mean_rating, share_good, share_fair, share_poor,
          share_fair_or_poor, newest_inspection, segment_count, grain
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pavement,
        [
            "borough_code",
            "miles_rated",
            "mean_rating",
            "share_good",
            "share_fair",
            "share_poor",
            "share_fair_or_poor",
            "newest_inspection",
            "segment_count",
            "grain",
        ],
    )

    complaints["created_at_text"] = complaints["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    complaints["closed_at_text"] = complaints["closed_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    complaints["due_at_text"] = complaints["due_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    complaints.loc[complaints["closed_at"].isna(), "closed_at_text"] = None
    complaints.loc[complaints["due_at"].isna(), "due_at_text"] = None
    log(f"requests: inserting {len(complaints):,} rows")
    _executemany(
        conn,
        """
        INSERT INTO complaint (
          unique_key, created_at, closed_at, due_at, agency_code, complaint_type, type_group,
          descriptor, status, boro_cd, borough_code, incident_zip, cross_street_1, cross_street_2,
          latitude, longitude, week_start, is_pothole, came_back
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        complaints.rename(columns={"agency": "agency_code"}),
        [
            "unique_key",
            "created_at_text",
            "closed_at_text",
            "due_at_text",
            "agency_code",
            "complaint_type",
            "type_group",
            "descriptor",
            "status",
            "boro_cd",
            "borough_code",
            "incident_zip",
            "cross_street_1",
            "cross_street_2",
            "latitude",
            "longitude",
            "week_start",
            "is_pothole",
            "came_back",
        ],
    )

    as_of = (last_monday + pd.Timedelta(days=6)).strftime("%Y-%m-%d") + " 23:59:59"
    log("sql: labeling unfixed requests and hours to close")
    conn.execute(UNFIXED_SQL, (as_of, as_of))
    conn.execute(HOURS_SQL)
    conn.commit()
    labeled = conn.execute("SELECT COUNT(*) FROM complaint WHERE left_unfixed IS NOT NULL").fetchone()[0]
    unfixed = conn.execute("SELECT COUNT(*) FROM complaint WHERE left_unfixed = 1").fetchone()[0]
    censored = conn.execute("SELECT COUNT(*) FROM complaint WHERE left_unfixed IS NULL").fetchone()[0]
    audit["labeled"] = labeled
    audit["unfixed"] = unfixed
    audit["censored"] = censored

    log("sql: building the district-week table, including weeks with zero requests")
    conn.execute(WEEKLY_SQL)
    conn.commit()
    weekly_total = conn.execute("SELECT COALESCE(SUM(complaint_count), 0) FROM weekly_demand").fetchone()[0]
    complaint_total = conn.execute(
        """
        SELECT COUNT(*) FROM complaint
        WHERE week_start IN (SELECT week_start FROM week_calendar)
        """
    ).fetchone()[0]
    if int(weekly_total) != int(complaint_total):
        raise RuntimeError(f"Weekly counts {weekly_total} do not match complaints {complaint_total}")

    log("sql: median and 90th percentile hours to close")
    conn.execute(DURATION_SQL)
    conn.commit()

    for step, count in audit.items():
        _audit(conn, step, count)
    meta = [
        (
            "311 street requests",
            "NYC Open Data erm2-nwe9",
            f"created from 2022-01-01 through {as_of[:10]}",
            audit["kept"],
            "Street, sidewalk, highway, curb, bridge, street light, traffic signal, and street sign complaints.",
        ),
        (
            "pavement",
            "NYC Open Data 6yyb-pb25, Street Pavement Ratings",
            str(pavement["newest_inspection"].max()),
            int(pavement["segment_count"].sum()),
            "Length-weighted borough shares. Rating 0 and repeat passes are excluded. Good is 8 to 10, fair is 4 up to 8, poor is below 4.",
        ),
        (
            "weather",
            "Open-Meteo archive, Central Park grid point",
            f"{daily_weather['weather_date'].min()} to {daily_weather['weather_date'].max()}",
            int(len(daily_weather)),
            "One station for every district. Freeze-thaw is a day with a low under 32 F and a high over 32 F.",
        ),
        (
            "population",
            "U.S. Census Bureau 2020 Census via NYC DCP, September 2021",
            "2020",
            int(len(population)),
            "Total population for each community district. Joint-interest areas are not in this file.",
        ),
        (
            "districts",
            "NYC Open Data 5crt-au7u",
            "community district boundaries",
            len(district_rows),
            "Borough code is the hundreds digit of boro_cd.",
        ),
        (
            "data_through",
            "last complete Monday-start week in the 311 extract",
            weeks[-1],
            len(weeks),
            as_of,
        ),
    ]
    conn.executemany(
        "INSERT INTO dataset_meta (name, source, vintage, row_count, note) VALUES (?, ?, ?, ?, ?)",
        meta,
    )
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    if DB_PATH.exists():
        DB_PATH.unlink()
    temporary.replace(DB_PATH)
    log(f"database: {DB_PATH}")
    return {"weeks": weeks, "future_weeks": future_weeks, "as_of": as_of, "last_week": weeks[-1]}


def _model_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    weekly = pd.read_sql_query(
        """
        SELECT w.boro_cd, w.complaint_type, w.week_start, w.complaint_count,
               d.borough_code, d.population, p.share_fair_or_poor,
               wx.freeze_thaw_days, wx.precip, wx.snow
        FROM weekly_demand w
        JOIN community_district d ON d.boro_cd = w.boro_cd
        LEFT JOIN district_pavement p ON p.borough_code = d.borough_code
        LEFT JOIN weather_week wx ON wx.week_start = w.week_start
        """,
        conn,
    )
    weekly = weekly.sort_values(["boro_cd", "complaint_type", "week_start"])
    grouped = weekly.groupby(["boro_cd", "complaint_type"], sort=False)["complaint_count"]
    weekly["lag1"] = grouped.shift(1)
    weekly["lag2"] = grouped.shift(2)
    weekly["lag4"] = grouped.shift(4)
    weekly["lag52"] = grouped.shift(52)
    weekly["roll4"] = grouped.transform(lambda series: series.shift(1).rolling(4, min_periods=1).mean())
    weekly["lag52_filled"] = weekly["lag52"].fillna(weekly["roll4"]).fillna(0)
    stamp = pd.to_datetime(weekly["week_start"])
    weekly["weekofyear"] = stamp.dt.isocalendar().week.astype(int)
    weekly["month"] = stamp.dt.month.astype(int)
    codes = {name: index for index, name in enumerate(GROUP_ORDER)}
    weekly["type_code"] = weekly["complaint_type"].map(codes).astype(int)
    for column in ("freeze_thaw_days", "precip", "snow", "share_fair_or_poor", "lag1", "lag2", "lag4", "roll4"):
        weekly[column] = weekly[column].fillna(0)
    weekly["population"] = weekly["population"].fillna(weekly["population"].median())
    return weekly


def _insert_metrics(conn, rows: list[tuple]) -> None:
    conn.executemany(
        """
        INSERT INTO model_metric (model_name, metric_name, metric_value, baseline_value, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_factors(conn, model_name: str, names: list[str], values: np.ndarray, detail: str) -> None:
    conn.executemany(
        """
        INSERT INTO factor_importance (model_name, feature_name, importance, detail)
        VALUES (?, ?, ?, ?)
        """,
        [(model_name, name, float(value), detail) for name, value in zip(names, values)],
    )


def train_load_forecast(conn: sqlite3.Connection, weeks: list[str], future_weeks: list[str]) -> str:
    log("model: training the weekly request forecast")
    panel = _model_frame(conn)
    holdout = set(weeks[-12:])
    train_weeks = set(weeks[:-12])
    early_cut = set(weeks[:4])
    train = panel.loc[panel["week_start"].isin(train_weeks - early_cut)].copy()
    test = panel.loc[panel["week_start"].isin(holdout)].copy()
    y_train = train["complaint_count"].to_numpy(dtype=float)
    y_test = test["complaint_count"].to_numpy(dtype=float)
    baseline_test = test["lag52_filled"].to_numpy(dtype=float)
    mae_base = float(mean_absolute_error(y_test, baseline_test))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[POISSON_FEATURES])
    x_test = scaler.transform(test[POISSON_FEATURES])
    poisson = PoissonRegressor(alpha=1.0, max_iter=400)
    poisson.fit(x_train, y_train)
    poisson_test = np.clip(poisson.predict(x_test), 0, None)
    mae_poisson = float(mean_absolute_error(y_test, poisson_test))

    boosting = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        early_stopping=True,
        random_state=210,
    )
    boosting.fit(train[HGB_FEATURES].to_numpy(dtype=float), y_train)
    boosting_test = np.clip(boosting.predict(test[HGB_FEATURES].to_numpy(dtype=float)), 0, None)
    mae_boost = float(mean_absolute_error(y_test, boosting_test))

    selected = "seasonal_baseline"
    best = mae_base
    chosen = baseline_test
    if mae_poisson < best:
        selected, best, chosen = "poisson_regression", mae_poisson, poisson_test
    if mae_boost < best:
        selected, best, chosen = "hist_gradient_boosting", mae_boost, boosting_test
    log(
        "model: holdout MAE "
        f"baseline {mae_base:.3f}, poisson {mae_poisson:.3f}, boosting {mae_boost:.3f}; using {selected}"
    )

    forecast = test[["boro_cd", "complaint_type", "week_start", "population"]].copy()
    forecast["predicted_count"] = chosen
    forecast["model_name"] = selected
    forecast["is_holdout"] = 1

    counts = {
        key: group["complaint_count"].to_numpy(dtype=float)
        for key, group in panel.groupby(["boro_cd", "complaint_type"], sort=False)
    }
    static = (
        panel.sort_values("week_start")
        .groupby(["boro_cd", "complaint_type"], as_index=False)
        .tail(1)
        .set_index(["boro_cd", "complaint_type"])
    )
    weather = pd.read_sql_query("SELECT * FROM weather_week", conn).set_index("week_start")
    future_rows = []
    feature_rows = []
    feature_index = []
    for (boro_cd, complaint_type), series in counts.items():
        info = static.loc[(boro_cd, complaint_type)]
        for step, week in enumerate(future_weeks, start=1):
            predicted = seasonal_count(series, step)
            if predicted is None:
                predicted = float(series[-4:].mean())
            model_name = "seasonal_baseline"
            if step == 1 and selected != "seasonal_baseline":
                model_name = selected
                wx = weather.loc[week] if week in weather.index else None
                feature_rows.append(
                    {
                        "weekofyear": int(pd.Timestamp(week).isocalendar().week),
                        "borough_code": int(info["borough_code"]),
                        "freeze_thaw_days": float(wx["freeze_thaw_days"]) if wx is not None else 0.0,
                        "precip": float(wx["precip"]) if wx is not None else 0.0,
                        "snow": float(wx["snow"]) if wx is not None else 0.0,
                        "share_fair_or_poor": float(info["share_fair_or_poor"]),
                        "population": float(info["population"]),
                        "lag52_filled": float(seasonal_count(series, 1) or series[-4:].mean()),
                        "lag1": history_count(series, 1) or 0.0,
                        "lag2": history_count(series, 2) or 0.0,
                        "lag4": history_count(series, 4) or 0.0,
                        "roll4": float(series[-4:].mean()),
                        "boro_cd": int(boro_cd),
                        "type_code": int(info["type_code"]),
                        "month": int(pd.Timestamp(week).month),
                    }
                )
                feature_index.append((int(boro_cd), complaint_type, week, float(info["population"]), model_name))
                continue
            population = float(info["population"])
            rate = float(predicted) * 10000.0 / population if population and population == population and population > 0 else None
            future_rows.append((int(boro_cd), complaint_type, week, float(predicted), rate, model_name, 0))

    if feature_rows and selected == "poisson_regression":
        future_frame = pd.DataFrame(feature_rows)
        predicted = np.clip(poisson.predict(scaler.transform(future_frame[POISSON_FEATURES])), 0, None)
    elif feature_rows and selected == "hist_gradient_boosting":
        future_frame = pd.DataFrame(feature_rows)
        predicted = np.clip(boosting.predict(future_frame[HGB_FEATURES].to_numpy(dtype=float)), 0, None)
    else:
        predicted = []
    for (boro_cd, complaint_type, week, population, model_name), value in zip(feature_index, predicted):
        future_rows.append(
            (
                boro_cd,
                complaint_type,
                week,
                float(value),
                float(value) * 10000.0 / population if population else None,
                model_name,
                0,
            )
        )

    forecast["predicted_rate_per_10k"] = np.where(
        forecast["population"] > 0,
        forecast["predicted_count"] * 10000.0 / forecast["population"],
        np.nan,
    )
    insert_rows = list(
        forecast[
            ["boro_cd", "complaint_type", "week_start", "predicted_count", "predicted_rate_per_10k", "model_name", "is_holdout"]
        ].itertuples(index=False, name=None)
    )
    insert_rows.extend(future_rows)

    def finite(value):
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    insert_rows = [tuple(finite(value) for value in row) for row in insert_rows]
    conn.executemany(
        """
        INSERT INTO load_forecast (
          boro_cd, complaint_type, week_start, predicted_count, predicted_rate_per_10k, model_name, is_holdout
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        insert_rows,
    )
    _insert_metrics(
        conn,
        [
            ("seasonal_baseline", "mae", mae_base, mae_base, "Last 12 complete weeks. Same district, 52 weeks earlier."),
            ("poisson_regression", "mae", mae_poisson, mae_base, "Poisson regression on scaled week, weather, pavement, population, and last year."),
            ("hist_gradient_boosting", "mae", mae_boost, mae_base, "Poisson loss, with short lags and district identity. One-step holdout."),
            ("load_forecast", "selected_model", best, mae_base, selected),
        ],
    )
    _insert_factors(
        conn,
        "poisson_regression",
        POISSON_FEATURES,
        poisson.coef_,
        "Change in the log of expected requests for a one-standard-deviation change.",
    )
    sample = test.sample(n=min(4000, len(test)), random_state=210)
    importance = permutation_importance(
        boosting,
        sample[HGB_FEATURES].to_numpy(dtype=float),
        sample["complaint_count"].to_numpy(dtype=float),
        n_repeats=3,
        random_state=210,
        scoring="neg_mean_absolute_error",
    )
    _insert_factors(
        conn,
        "hist_gradient_boosting",
        HGB_FEATURES,
        importance.importances_mean,
        "Holdout permutation importance. Higher means the feature moves the absolute error more.",
    )
    conn.commit()
    return selected


def _backlog(conn: sqlite3.Connection) -> pd.DataFrame:
    opened = pd.read_sql_query(
        """
        SELECT boro_cd, substr(created_at, 1, 10) AS day, COUNT(*) AS opened
        FROM complaint
        GROUP BY 1, 2
        """,
        conn,
    )
    closed = pd.read_sql_query(
        """
        SELECT boro_cd, substr(closed_at, 1, 10) AS day, COUNT(*) AS closed
        FROM complaint
        WHERE closed_at IS NOT NULL
        GROUP BY 1, 2
        """,
        conn,
    )
    frames = []
    for boro_cd, group in opened.groupby("boro_cd"):
        start = pd.Timestamp(group["day"].min())
        end = pd.Timestamp(group["day"].max())
        index = pd.date_range(start, end, freq="D")
        opened_daily = group.assign(day=pd.to_datetime(group["day"])).set_index("day")["opened"]
        closed_group = closed.loc[closed["boro_cd"] == boro_cd]
        closed_daily = (
            closed_group.assign(day=pd.to_datetime(closed_group["day"])).set_index("day")["closed"]
            if len(closed_group)
            else pd.Series(dtype=float)
        )
        stock = (
            opened_daily.reindex(index, fill_value=0).cumsum()
            - closed_daily.reindex(index, fill_value=0).cumsum()
        ).shift(1).fillna(0).clip(lower=0)
        frames.append(pd.DataFrame({"boro_cd": int(boro_cd), "day": index.strftime("%Y-%m-%d"), "backlog": stock.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def _stratified_sample(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    if len(frame) <= size:
        return frame
    pieces = []
    for _, group in frame.groupby("left_unfixed"):
        take = max(1, int(round(len(group) * size / len(frame))))
        pieces.append(group.sample(n=min(take, len(group)), random_state=210))
    sample = pd.concat(pieces)
    if len(sample) > size:
        sample = sample.sample(n=size, random_state=210)
    return sample


def train_unfixed(conn: sqlite3.Connection, holdout_start: str) -> str:
    log("model: training the unfixed-request score")
    complaints = pd.read_sql_query(
        """
        SELECT c.unique_key, c.created_at, c.boro_cd, c.borough_code, c.type_group,
               c.is_pothole, c.left_unfixed, c.week_start,
               p.share_fair_or_poor, w.freeze_thaw_days, w.precip
        FROM complaint c
        LEFT JOIN district_pavement p ON p.borough_code = c.borough_code
        LEFT JOIN weather_week w ON w.week_start = c.week_start
        WHERE c.created_at >= ?
           OR c.left_unfixed IS NOT NULL
        """,
        conn,
        params=[holdout_start],
    )
    # The WHERE keeps every labeled row and every holdout-week row. Pre-holdout unlabeled rows are omitted.
    complaints["day"] = complaints["created_at"].str.slice(0, 10)
    complaints["month"] = complaints["created_at"].str.slice(5, 7).astype(int)
    codes = {name: index for index, name in enumerate(GROUP_ORDER)}
    complaints["type_code"] = complaints["type_group"].map(codes).astype(int)
    complaints = complaints.merge(_backlog(conn), on=["boro_cd", "day"], how="left")
    complaints["backlog"] = complaints["backlog"].fillna(0)
    for column in ("share_fair_or_poor", "freeze_thaw_days", "precip"):
        complaints[column] = complaints[column].fillna(0)

    labeled = complaints.loc[complaints["left_unfixed"].notna()].copy()
    labeled["left_unfixed"] = labeled["left_unfixed"].astype(int)
    train = labeled.loc[labeled["created_at"] < holdout_start]
    test = labeled.loc[labeled["created_at"] >= holdout_start]
    fit = _stratified_sample(train, 250_000)
    log(f"model: unfixed fit rows {len(fit):,}, holdout labeled rows {len(test):,}")
    y_fit = fit["left_unfixed"].to_numpy(dtype=int)
    y_test = test["left_unfixed"].to_numpy(dtype=int)
    if len(test) == 0 or len(np.unique(y_test)) < 2:
        raise RuntimeError("The holdout does not contain both unfixed and fixed requests")
    type_rate = fit.groupby("type_group")["left_unfixed"].mean()
    overall = float(fit["left_unfixed"].mean())
    baseline = test["type_group"].map(type_rate).fillna(overall).to_numpy(dtype=float)
    ap_base = float(average_precision_score(y_test, baseline))

    scaler = StandardScaler()
    logistic = LogisticRegression(max_iter=400, solver="liblinear")
    logistic.fit(scaler.fit_transform(fit[UNFIXED_FEATURES]), y_fit)
    logistic_prob = logistic.predict_proba(scaler.transform(test[UNFIXED_FEATURES]))[:, 1]
    ap_logistic = float(average_precision_score(y_test, logistic_prob))
    auc_logistic = float(roc_auc_score(y_test, logistic_prob))

    boosting = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        early_stopping=True,
        random_state=210,
    )
    boosting.fit(fit[UNFIXED_FEATURES].to_numpy(dtype=float), y_fit)
    boost_prob = boosting.predict_proba(test[UNFIXED_FEATURES].to_numpy(dtype=float))[:, 1]
    ap_boost = float(average_precision_score(y_test, boost_prob))
    auc_boost = float(roc_auc_score(y_test, boost_prob))

    selected = "type_rate_baseline"
    best = ap_base
    if ap_logistic > best:
        selected, best = "logistic_regression", ap_logistic
    if ap_boost > best:
        selected, best = "hist_gradient_boosting_classifier", ap_boost
    log(
        "model: holdout average precision "
        f"baseline {ap_base:.3f}, logistic {ap_logistic:.3f}, boosting {ap_boost:.3f}; using {selected}"
    )

    score_frame = complaints.loc[complaints["created_at"] >= holdout_start].copy()
    if selected == "logistic_regression":
        probability = logistic.predict_proba(scaler.transform(score_frame[UNFIXED_FEATURES]))[:, 1]
    elif selected == "hist_gradient_boosting_classifier":
        probability = boosting.predict_proba(score_frame[UNFIXED_FEATURES].to_numpy(dtype=float))[:, 1]
    else:
        probability = score_frame["type_group"].map(type_rate).fillna(overall).to_numpy(dtype=float)
    score_frame = score_frame[["unique_key"]].copy()
    score_frame["probability"] = probability
    score_frame["model_name"] = selected
    conn.executemany(
        "INSERT INTO unfixed_score (unique_key, probability, model_name) VALUES (?, ?, ?)",
        list(score_frame.itertuples(index=False, name=None)),
    )
    note = f"Fit on {len(fit):,} labeled requests from before {holdout_start}. Scored on the later weeks."
    _insert_metrics(
        conn,
        [
            ("type_rate_baseline", "average_precision", ap_base, ap_base, "Each complaint group's rate before the holdout."),
            ("logistic_regression", "average_precision", ap_logistic, ap_base, note),
            ("logistic_regression", "roc_auc", auc_logistic, None, note),
            ("hist_gradient_boosting_classifier", "average_precision", ap_boost, ap_base, note),
            ("hist_gradient_boosting_classifier", "roc_auc", auc_boost, None, note),
            ("unfixed_score", "selected_model", best, ap_base, selected),
        ],
    )
    _insert_factors(
        conn,
        "logistic_regression",
        UNFIXED_FEATURES,
        logistic.coef_.ravel(),
        "Change in log-odds for a one-standard-deviation change.",
    )
    sample = test.sample(n=min(4000, len(test)), random_state=210)
    importance = permutation_importance(
        boosting,
        sample[UNFIXED_FEATURES].to_numpy(dtype=float),
        sample["left_unfixed"].to_numpy(dtype=int),
        n_repeats=3,
        random_state=210,
        scoring="average_precision",
    )
    _insert_factors(
        conn,
        "hist_gradient_boosting_classifier",
        UNFIXED_FEATURES,
        importance.importances_mean,
        "Holdout permutation importance for average precision.",
    )
    conn.commit()
    return selected


def train_models(bounds: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        train_load_forecast(conn, bounds["weeks"], bounds["future_weeks"])
        train_unfixed(conn, bounds["weeks"][-12])
    finally:
        conn.close()


def run_self_checks() -> None:
    assert parse_community_board("05 QUEENS") == 405
    assert parse_community_board("12 MANHATTAN") == 112
    assert parse_community_board("Unspecified BROOKLYN") is None
    assert type_group_for("Street Sign - Missing") == "Street Sign"
    assert rating_band(8) == "good"
    assert rating_band(4) == "fair"
    assert rating_band(3.9) == "poor"
    assert rating_band(0) is None
    stamps = pd.to_datetime(pd.Series(["2022-01-05", "2022-01-03", "2022-01-09"]))
    starts = week_start_monday(stamps).dt.strftime("%Y-%m-%d").tolist()
    assert starts == ["2022-01-03", "2022-01-03", "2022-01-03"], starts
    series = np.array([10.0] * 52 + [5.0])
    assert seasonal_count(series, 1) == 10.0
    assert history_count(series, 1) == 5.0

    came = pd.DataFrame(
        {
            "borough_code": [1, 1, 1],
            "type_group": ["Street Condition"] * 3,
            "cross_street_1": ["1 AVENUE", "1 AVENUE", "2 AVENUE"],
            "cross_street_2": ["2 AVENUE", "2 AVENUE", "3 AVENUE"],
            "created_at": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-02-01"]),
            "closed_at": pd.to_datetime(["2022-01-10", pd.NaT, pd.NaT]),
        }
    )
    flagged = mark_came_back(came)
    assert flagged.tolist() == [0, 1, 0], flagged

    connection = sqlite3.connect(":memory:")
    connection.executescript(SCHEMA_PATH.read_text())
    connection.execute("INSERT INTO borough VALUES (4, 'Queens')")
    connection.execute(
        """
        INSERT INTO community_district (
          boro_cd, borough_code, district_number, district_name, population
        ) VALUES (405, 4, 5, 'Queens CD 5', 100000)
        """
    )
    connection.execute("INSERT INTO type_group_list VALUES ('Street Condition')")
    connection.executemany("INSERT INTO week_calendar VALUES (?)", [("2022-01-03",), ("2022-01-10",)])
    connection.execute("INSERT INTO agency VALUES ('DOT', 'Department of Transportation')")
    rows = [
        ("a", "2022-01-04 00:00:00", "2022-01-20 00:00:00", "2022-01-10 00:00:00", 1),
        ("b", "2022-01-04 00:00:00", "2022-01-05 00:00:00", "2022-01-10 00:00:00", 0),
        ("c", "2022-01-04 00:00:00", "2022-01-20 00:00:00", None, 1),
        ("d", "2022-01-04 00:00:00", "2022-01-10 00:00:00", None, 0),
        ("e", "2022-01-04 00:00:00", None, "2022-02-01 00:00:00", 1),
        ("f", "2022-01-04 00:00:00", None, None, 1),
        ("g", "2024-01-01 00:00:00", None, None, None),
    ]
    connection.executemany(
        """
        INSERT INTO complaint (
          unique_key, created_at, closed_at, due_at, agency_code, complaint_type, type_group,
          boro_cd, borough_code, week_start, is_pothole, left_unfixed
        ) VALUES (?, ?, ?, ?, 'DOT', 'Street Condition', 'Street Condition', 405, 4, '2022-01-03', 1, NULL)
        """,
        [(key, created, closed, due) for key, created, closed, due, _expected in rows],
    )
    connection.execute(UNFIXED_SQL, ("2022-06-01 00:00:00", "2022-06-01 00:00:00"))
    found = dict(connection.execute("SELECT unique_key, left_unfixed FROM complaint"))
    for key, _created, _closed, _due, expected in rows:
        assert found[key] == expected, (key, found[key], expected)
    connection.execute(WEEKLY_SQL)
    counts = dict(connection.execute("SELECT week_start, complaint_count FROM weekly_demand ORDER BY 1"))
    assert counts["2022-01-03"] == 7, counts
    assert counts["2022-01-10"] == 0, counts
    connection.close()

    if POPULATION_PDF.exists():
        text = subprocess.run(
            ["pdftotext", "-layout", str(POPULATION_PDF), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        parsed = parse_population_text(text)
        assert len(parsed) == 59
        assert int(parsed.loc[parsed["boro_cd"] == 201, "population"].iloc[0]) == 100929
    log("checks: passed")


def main() -> None:
    run_self_checks()
    RAW.mkdir(parents=True, exist_ok=True)
    weather = download_weather()
    districts = download_districts()
    population = load_population()
    pavement_pages = download_pavement()
    request_pages = download_311()
    complaints, audit = prepare_complaints(request_pages)
    pavement = pavement_by_borough(pavement_pages)
    bounds = build_database(complaints, audit, districts, population, pavement, weather)
    train_models(bounds)
    log("pipeline: finished")


if __name__ == "__main__":
    main()
