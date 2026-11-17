"""Maintenance-desk view of NYC street requests, forecasts, and past-due work."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "nyc_infrastructure.sqlite"

GROUPS = [
    "All types",
    "Street Condition",
    "Sidewalk Condition",
    "Highway Condition",
    "Curb Condition",
    "Bridge Condition",
    "Street Light Condition",
    "Traffic Signal Condition",
    "Street Sign",
]

FEATURE_LABELS = {
    "weekofyear": "Week of year",
    "borough_code": "Borough",
    "freeze_thaw_days": "Freeze-thaw days",
    "precip": "Rain",
    "snow": "Snow",
    "share_fair_or_poor": "Fair or poor pavement",
    "population": "Population",
    "lag52_filled": "Same week last year",
    "lag1": "Last week",
    "lag2": "Two weeks ago",
    "lag4": "Four weeks ago",
    "roll4": "Trailing four-week average",
    "boro_cd": "Community district",
    "type_code": "Complaint group",
    "month": "Month",
    "is_pothole": "Pothole descriptor",
    "backlog": "Open requests that morning",
}

WEEK_SQL = """
SELECT
  d.boro_cd,
  d.district_name,
  d.borough_code,
  b.borough_name,
  d.population,
  d.geometry,
  f.complaint_type,
  f.predicted_count,
  f.predicted_rate_per_10k,
  f.model_name,
  f.is_holdout,
  COALESCE(w.complaint_count, 0) AS complaint_count,
  w.rate_per_10k,
  COALESCE(w.unfixed_count, 0) AS unfixed_count,
  COALESCE(w.came_back_count, 0) AS came_back_count,
  p.share_good,
  p.share_fair,
  p.share_poor,
  p.mean_rating,
  p.newest_inspection,
  p.miles_rated,
  p.grain,
  wx.freeze_thaw_days,
  wx.precip,
  wx.snow,
  wx.weather_source
FROM load_forecast AS f
JOIN community_district AS d ON d.boro_cd = f.boro_cd
JOIN borough AS b ON b.borough_code = d.borough_code
LEFT JOIN weekly_demand AS w
  ON w.boro_cd = f.boro_cd
 AND w.complaint_type = f.complaint_type
 AND w.week_start = f.week_start
LEFT JOIN district_pavement AS p ON p.borough_code = d.borough_code
LEFT JOIN weather_week AS wx ON wx.week_start = f.week_start
WHERE f.week_start = ?
"""


def db_mtime() -> float:
    return DB_PATH.stat().st_mtime


@st.cache_data(show_spinner=False)
def read_sql(sql: str, params: tuple, mtime: float) -> pd.DataFrame:
    del mtime
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def number(value, digits=0):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if digits == 0:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def week_label(week: str) -> str:
    stamp = pd.Timestamp(week)
    return f"Week of {stamp.strftime('%b')} {stamp.day}, {stamp.year}"


def collapse(frame: pd.DataFrame, complaint: str) -> pd.DataFrame:
    view = frame if complaint == "All types" else frame.loc[frame["complaint_type"] == complaint]
    if view.empty:
        return view
    grouped = view.groupby(
        ["boro_cd", "district_name", "borough_code", "borough_name", "population", "geometry"],
        as_index=False,
        dropna=False,
    ).agg(
        predicted_count=("predicted_count", "sum"),
        complaint_count=("complaint_count", "sum"),
        unfixed_count=("unfixed_count", "sum"),
        came_back_count=("came_back_count", "sum"),
        model_name=("model_name", "first"),
        is_holdout=("is_holdout", "max"),
        share_good=("share_good", "first"),
        share_fair=("share_fair", "first"),
        share_poor=("share_poor", "first"),
        mean_rating=("mean_rating", "first"),
        newest_inspection=("newest_inspection", "first"),
        miles_rated=("miles_rated", "first"),
        grain=("grain", "first"),
        freeze_thaw_days=("freeze_thaw_days", "first"),
        precip=("precip", "first"),
        snow=("snow", "first"),
        weather_source=("weather_source", "first"),
    )
    grouped["predicted_rate_per_10k"] = grouped["predicted_count"] * 10000.0 / grouped["population"]
    grouped["rate_per_10k"] = grouped["complaint_count"] * 10000.0 / grouped["population"]
    return grouped


def geojson_for(frame: pd.DataFrame) -> dict:
    features = []
    for row in frame.itertuples(index=False):
        if not row.geometry:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {"boro_cd": str(row.boro_cd)},
                "geometry": json.loads(row.geometry),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def watch_flags(current: pd.DataFrame, latest_unfixed: pd.DataFrame) -> pd.DataFrame:
    marked = current.copy()
    k = 5 if marked["borough_name"].nunique() == 1 else 10
    k = min(k, len(marked))
    high_forecast = set(marked.nlargest(k, "predicted_count")["boro_cd"])
    backlog = latest_unfixed if not latest_unfixed.empty else marked
    high_backlog = set(backlog.nlargest(min(k, len(backlog)), "unfixed_count")["boro_cd"])
    marked["desk_note"] = ""
    mask = marked["boro_cd"].isin(high_forecast - high_backlog)
    marked.loc[mask, "desk_note"] = "High forecast, not on the latest past-due list"
    return marked


def main() -> None:
    st.set_page_config(page_title="NYC Infrastructure Intelligence", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background: #E4EEF3;
            border-radius: 8px;
            padding: 0.4rem 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("NYC Infrastructure Intelligence")
    st.caption(
        "A desk view for DOT borough crews: where street-maintenance requests are likely to land, "
        "and which districts already have work past due."
    )

    if not DB_PATH.exists():
        st.error("The database has not been built yet.")
        st.write("From this folder, run the pipeline, then reload this page.")
        st.code("python pipeline.py", language="bash")
        st.stop()

    mtime = db_mtime()
    try:
        meta = read_sql("SELECT name, source, vintage, row_count, note FROM dataset_meta", (), mtime)
        weeks = read_sql(
            """
            SELECT week_start, MAX(is_holdout) AS is_holdout
            FROM load_forecast
            GROUP BY week_start
            ORDER BY week_start
            """,
            (),
            mtime,
        )
    except sqlite3.Error as error:
        st.error("The database could not be read.")
        st.write(str(error))
        st.stop()

    if weeks.empty:
        st.info("The request history is loaded. The model scores are not in the database yet. Reload after training finishes.")
        st.stop()

    meta_map = meta.set_index("name")
    data_through = meta_map.at["data_through", "vintage"] if "data_through" in meta_map.index else weeks["week_start"].max()
    holdout_weeks = weeks.loc[weeks["is_holdout"] == 1, "week_start"]
    default_week = holdout_weeks.max() if len(holdout_weeks) else weeks["week_start"].max()

    st.sidebar.header("Filters")
    boroughs = ["All boroughs"] + read_sql(
        "SELECT borough_name FROM borough ORDER BY borough_code", (), mtime
    )["borough_name"].tolist()
    borough = st.sidebar.selectbox("Borough", boroughs)
    complaint = st.sidebar.selectbox("Complaint group", GROUPS, index=1)
    week_options = weeks["week_start"].tolist()
    week = st.sidebar.selectbox(
        "Week",
        week_options,
        index=week_options.index(default_week),
        format_func=week_label,
    )
    district_rows = read_sql(
        """
        SELECT d.boro_cd, d.district_name, b.borough_name
        FROM community_district d
        JOIN borough b ON b.borough_code = d.borough_code
        ORDER BY d.boro_cd
        """,
        (),
        mtime,
    )
    if borough != "All boroughs":
        district_rows = district_rows.loc[district_rows["borough_name"] == borough]
    district_labels = ["All districts"] + district_rows["district_name"].tolist()
    district_name = st.sidebar.selectbox("District", district_labels)

    try:
        raw_week = read_sql(WEEK_SQL, (week,), mtime)
    except sqlite3.Error as error:
        st.error("That week could not be queried.")
        st.write(str(error))
        st.stop()

    view = collapse(raw_week, complaint)
    if borough != "All boroughs":
        view = view.loc[view["borough_name"] == borough]
    if view.empty:
        st.warning("No districts match these filters.")
        st.stop()

    selected = view if district_name == "All districts" else view.loc[view["district_name"] == district_name]
    if selected.empty:
        st.warning("That district has no row for this complaint group.")
        st.stop()

    latest = collapse(
        read_sql(WEEK_SQL, (data_through,), mtime),
        complaint,
    )
    if borough != "All boroughs":
        latest = latest.loc[latest["borough_name"] == borough]
    view = watch_flags(view, latest)

    future = week > data_through
    weather_source = view["weather_source"].dropna().iloc[0] if view["weather_source"].notna().any() else ""
    model_name = selected["model_name"].mode().iloc[0] if len(selected) else ""
    if future or weather_source == "seasonal_prior_year":
        st.info(
            "Weather for this week is Central Park's weather from the same week last year. "
            "It is a stand-in, not a forecast from the National Weather Service."
        )
    elif int(selected["is_holdout"].max()) == 1:
        st.caption(
            "Counts are requests that were actually opened. The prediction for each district was scored one week ahead, "
            "without using that week's count."
        )

    population = float(selected["population"].sum()) if district_name == "All districts" else float(selected["population"].iloc[0])
    actual = float(selected["complaint_count"].sum())
    predicted = float(selected["predicted_count"].sum())
    unfixed = float(selected["unfixed_count"].sum())
    actual_rate = actual * 10000.0 / population if population else None
    predicted_rate = predicted * 10000.0 / population if population else None

    metric_cols = st.columns(4)
    metric_cols[0].metric("Requests opened", "—" if future else number(actual))
    metric_cols[1].metric("Predicted requests", number(predicted))
    metric_cols[2].metric(
        "Per 10,000 residents",
        number(actual_rate, 1) if not future else number(predicted_rate, 1),
        help="Actual rate for a completed week. Predicted rate for a week that has not finished.",
    )
    latest_slice = latest if district_name == "All districts" else latest.loc[latest["district_name"] == district_name]
    past_due_value = unfixed if not future else float(latest_slice["unfixed_count"].sum()) if len(latest_slice) else 0
    metric_cols[3].metric("Past due" if not future else "Past due in the latest week", number(past_due_value))
    if not future and unfixed == 0 and week == data_through:
        st.caption(
            "Requests opened this week are still inside the 14-day window, so none are past due yet. "
            "Choose an earlier week to see requests that missed the deadline."
        )
    through = pd.Timestamp(data_through)
    st.caption(
        f"Model stored for this slice: {model_name.replace('_', ' ')}. "
        f"Data runs through the week of {through.strftime('%b')} {through.day}, {through.year}."
    )

    map_choice = st.radio(
        "Color the map by",
        ["Predicted requests", "Requests opened", "Past due", "Predicted rate per 10k"],
        horizontal=True,
    )
    color_column = {
        "Predicted requests": "predicted_count",
        "Requests opened": "complaint_count",
        "Past due": "unfixed_count",
        "Predicted rate per 10k": "predicted_rate_per_10k",
    }[map_choice]
    map_frame = view.copy()
    map_frame["boro_cd"] = map_frame["boro_cd"].astype(str)
    chart_col, table_col = st.columns([1.15, 0.85])
    with chart_col:
        figure = px.choropleth(
            map_frame,
            geojson=geojson_for(view),
            locations="boro_cd",
            featureidkey="properties.boro_cd",
            color=color_column,
            hover_name="district_name",
            hover_data={
                "boro_cd": False,
                "predicted_count": ":.1f",
                "complaint_count": True,
                "unfixed_count": True,
                "predicted_rate_per_10k": ":.1f",
                "desk_note": True,
            },
            color_continuous_scale="YlOrRd",
        )
        figure.update_geos(fitbounds="locations", visible=False)
        figure.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(title=""),
        )
        st.plotly_chart(figure, width="stretch")
        st.caption("Zero is the light end of the scale. The map is community districts, not zip codes.")
    with table_col:
        table = view.sort_values("predicted_count", ascending=False)[
            [
                "district_name",
                "complaint_count",
                "predicted_count",
                "rate_per_10k",
                "predicted_rate_per_10k",
                "unfixed_count",
                "came_back_count",
                "desk_note",
            ]
        ].rename(
            columns={
                "district_name": "District",
                "complaint_count": "Opened",
                "predicted_count": "Predicted",
                "rate_per_10k": "Opened per 10k",
                "predicted_rate_per_10k": "Predicted per 10k",
                "unfixed_count": "Past due",
                "came_back_count": "Came back",
                "desk_note": "Desk note",
            }
        )
        st.dataframe(
            table,
            hide_index=True,
            height=520,
            width="stretch",
            column_config={
                "Predicted": st.column_config.NumberColumn(format="%.1f"),
                "Opened per 10k": st.column_config.NumberColumn(format="%.1f"),
                "Predicted per 10k": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    st.subheader("Actual and predicted, over the year")
    borough_code = 0 if borough == "All boroughs" else int(view["borough_code"].iloc[0])
    district_code = 0
    if district_name != "All districts":
        district_code = int(district_rows.loc[district_rows["district_name"] == district_name, "boro_cd"].iloc[0])
    history = read_sql(
        """
        SELECT w.week_start,
               SUM(w.complaint_count) AS complaint_count,
               SUM(w.unfixed_count) AS unfixed_count
        FROM weekly_demand AS w
        JOIN community_district AS d ON d.boro_cd = w.boro_cd
        WHERE (? = 'All types' OR w.complaint_type = ?)
          AND (? = 0 OR d.borough_code = ?)
          AND (? = 0 OR w.boro_cd = ?)
        GROUP BY w.week_start
        ORDER BY w.week_start
        """,
        (complaint, complaint, borough_code, borough_code, district_code, district_code),
        mtime,
    )
    forecast_history = read_sql(
        """
        SELECT f.week_start, SUM(f.predicted_count) AS predicted_count
        FROM load_forecast AS f
        JOIN community_district AS d ON d.boro_cd = f.boro_cd
        WHERE (? = 'All types' OR f.complaint_type = ?)
          AND (? = 0 OR d.borough_code = ?)
          AND (? = 0 OR f.boro_cd = ?)
        GROUP BY f.week_start
        ORDER BY f.week_start
        """,
        (complaint, complaint, borough_code, borough_code, district_code, district_code),
        mtime,
    )
    if not history.empty:
        history = history.tail(60)
        trend = history.merge(forecast_history, on="week_start", how="left")
        trend["week"] = pd.to_datetime(trend["week_start"])
        long = trend.melt(
            id_vars="week",
            value_vars=["complaint_count", "predicted_count"],
            var_name="series",
            value_name="requests",
        )
        long["series"] = long["series"].map(
            {"complaint_count": "Opened", "predicted_count": "Predicted"}
        )
        trend_fig = px.line(long, x="week", y="requests", color="series", markers=False)
        trend_fig.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            height=320,
            legend_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="Requests",
        )
        st.plotly_chart(trend_fig, width="stretch")
        st.caption(
            "The predicted line covers the last 12 complete weeks and the next four. "
            "The first future week uses the stored model with last year's weather. "
            "The three weeks after that are the same district and complaint group, 52 weeks earlier."
        )

    factor_col, duration_col = st.columns(2)
    with factor_col:
        st.subheader("What the models used")
        factors = read_sql(
            """
            SELECT model_name, feature_name, importance, detail
            FROM factor_importance
            WHERE model_name IN ('poisson_regression', 'hist_gradient_boosting')
            """,
            (),
            mtime,
        )
        if factors.empty:
            st.write("Factor rows are not in the database yet.")
        else:
            show = factors.copy()
            show["Factor"] = show["feature_name"].map(FEATURE_LABELS).fillna(show["feature_name"])
            show["Model"] = show["model_name"].map(
                {
                    "poisson_regression": "Poisson regression",
                    "hist_gradient_boosting": "Gradient boosting",
                }
            )
            show["Value"] = show["importance"]
            st.dataframe(
                show[["Model", "Factor", "Value"]].sort_values(["Model", "Value"], ascending=[True, False]),
                hide_index=True,
                width="stretch",
                column_config={"Value": st.column_config.NumberColumn(format="%+.3f")},
            )
            st.caption(
                "Poisson values are coefficients on scaled inputs. "
                "Boosting values are how much the holdout error moves when the feature is shuffled."
            )
        if view["mean_rating"].notna().any():
            pavement = view.drop_duplicates("borough_code")
            st.write(
                "Pavement is the latest borough survey, shared by every district in that borough. "
                "Good is 8 to 10, fair is 4 up to 8, and poor is below 4. Rating 0 is excluded."
            )
            pavement_table = pavement[["borough_name", "miles_rated", "mean_rating", "share_good", "share_fair", "share_poor", "newest_inspection"]].copy()
            for column in ("share_good", "share_fair", "share_poor"):
                pavement_table[column] = pavement_table[column].map(lambda value: "—" if pd.isna(value) else f"{value:.0%}")
            st.dataframe(
                pavement_table.rename(columns={
                    "borough_name": "Borough",
                    "miles_rated": "Rated miles",
                    "mean_rating": "Mean rating",
                    "share_good": "Good",
                    "share_fair": "Fair",
                    "share_poor": "Poor",
                    "newest_inspection": "Newest inspection",
                }),
                hide_index=True,
                width="stretch",
                column_config={
                    "Rated miles": st.column_config.NumberColumn(format="%.0f"),
                    "Mean rating": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        if view["freeze_thaw_days"].notna().any():
            st.write(
                f"This week's Central Park weather: {number(view['freeze_thaw_days'].iloc[0])} freeze-thaw days, "
                f"{number(view['precip'].iloc[0], 1)} mm of rain, {number(view['snow'].iloc[0], 1)} cm of snow."
            )
    with duration_col:
        st.subheader("Hours from open to close")
        duration = read_sql(
            """
            SELECT b.borough_name, d.complaint_type, d.n, d.median_hours, d.p90_hours
            FROM duration_by_borough AS d
            JOIN borough AS b ON b.borough_code = d.borough_code
            WHERE (? = 'All types' OR d.complaint_type = ?)
              AND (? = 'All boroughs' OR b.borough_name = ?)
            ORDER BY d.median_hours DESC
            """,
            (complaint, complaint, borough, borough),
            mtime,
        )
        if duration.empty:
            st.write("No closed requests match this filter.")
        else:
            duration_fig = px.bar(
                duration,
                x="median_hours",
                y="borough_name" if borough == "All boroughs" else "complaint_type",
                color="complaint_type" if borough == "All boroughs" else None,
                orientation="h",
                labels={"median_hours": "Median hours", "borough_name": "", "complaint_type": ""},
            )
            duration_fig.update_layout(
                margin=dict(l=0, r=0, t=8, b=0),
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                legend_title="",
            )
            st.plotly_chart(duration_fig, width="stretch")
            st.caption("The median is the lower middle from a SQL window ranking. The 90th percentile is in the table below.")
            st.dataframe(
                duration.rename(
                    columns={
                        "borough_name": "Borough",
                        "complaint_type": "Group",
                        "n": "Closed requests",
                        "median_hours": "Median hours",
                        "p90_hours": "90th percentile hours",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Median hours": st.column_config.NumberColumn(format="%.1f"),
                    "90th percentile hours": st.column_config.NumberColumn(format="%.1f"),
                },
            )

    st.subheader("Requests most likely to stay past due")
    if future:
        st.write("This week has not started in the 311 file, so there is no request list yet. The past-due counts above are from the latest complete week.")
    else:
        risky = read_sql(
            """
            SELECT c.created_at, d.district_name, c.complaint_type, c.descriptor,
                   c.status, c.cross_street_1, c.cross_street_2,
                   c.left_unfixed, c.came_back, s.probability, c.hours_to_close
            FROM complaint AS c
            JOIN community_district AS d ON d.boro_cd = c.boro_cd
            LEFT JOIN unfixed_score AS s ON s.unique_key = c.unique_key
            WHERE c.week_start = ?
              AND (? = 'All types' OR c.type_group = ?)
              AND (? = 'All boroughs' OR d.borough_code = (
                    SELECT borough_code FROM borough WHERE borough_name = ?
              ))
              AND (? = 'All districts' OR d.district_name = ?)
            ORDER BY s.probability DESC
            LIMIT 25
            """,
            (week, complaint, complaint, borough, borough, district_name, district_name),
            mtime,
        )
        if risky.empty:
            st.write("No requests were opened for this filter.")
        else:
            risky["probability"] = risky["probability"].map(lambda value: None if pd.isna(value) else f"{value:.0%}")
            risky["left_unfixed"] = risky["left_unfixed"].map({1: "Past due", 0: "Closed in time"}).fillna("Deadline not reached")
            st.dataframe(
                risky.rename(
                    columns={
                        "created_at": "Opened",
                        "district_name": "District",
                        "complaint_type": "Type",
                        "descriptor": "Descriptor",
                        "status": "Status",
                        "cross_street_1": "Cross street 1",
                        "cross_street_2": "Cross street 2",
                        "left_unfixed": "Outcome",
                        "came_back": "Came back",
                        "probability": "Score",
                        "hours_to_close": "Hours to close",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Past due means the request was still open after its due date, or after 14 days when no due date was recorded. "
                "A request that is still inside that window is not labeled yet. Came back means the same complaint group "
                "returned to the same cross streets within 90 days of an earlier close."
            )

    with st.expander("Model scores on the last 12 weeks"):
        metrics = read_sql(
            """
            SELECT model_name, metric_name, metric_value, baseline_value, notes
            FROM model_metric
            ORDER BY model_name, metric_name
            """,
            (),
            mtime,
        )
        st.dataframe(metrics, hide_index=True, width="stretch")
        st.write(
            "The load forecast is mean absolute error against the seasonal baseline. "
            "The unfixed score is average precision and ROC-AUC against each complaint group's historical rate. "
            "A model is stored only when it beats that baseline. The split is the last 12 complete weeks, not a random sample."
        )

    with st.expander("SQL the desk can rerun"):
        query_name = st.selectbox(
            "Query",
            [
                "Freeze-thaw weeks and potholes",
                "Past-due pothole share by year",
                "Came-back requests by borough",
                "Rows dropped while loading",
                "Week-over-week change",
            ],
        )
        queries = {
            "Freeze-thaw weeks and potholes": "SELECT season, potholes, avg_freeze_thaw_days, avg_precip_mm FROM v_pothole_season",
            "Past-due pothole share by year": """
                SELECT substr(created_at, 1, 4) AS year,
                       COUNT(*) AS potholes,
                       SUM(CASE WHEN left_unfixed = 1 THEN 1 ELSE 0 END) AS past_due,
                       ROUND(AVG(CASE WHEN left_unfixed IS NOT NULL THEN left_unfixed END), 3) AS past_due_share
                FROM complaint
                WHERE is_pothole = 1
                GROUP BY 1
                ORDER BY 1
            """,
            "Came-back requests by borough": """
                SELECT b.borough_name, c.type_group,
                       SUM(c.came_back) AS came_back_requests,
                       COUNT(*) AS requests
                FROM complaint c
                JOIN borough b ON b.borough_code = c.borough_code
                GROUP BY b.borough_name, c.type_group
                ORDER BY came_back_requests DESC
            """,
            "Rows dropped while loading": "SELECT step, row_count, note FROM load_audit ORDER BY step",
        }
        try:
            if query_name == "Week-over-week change":
                if district_code == 0:
                    st.write("Choose one community district to run the window query.")
                else:
                    group = "Street Condition" if complaint == "All types" else complaint
                    result = read_sql(
                        """
                        SELECT week_start,
                               complaint_count,
                               complaint_count - LAG(complaint_count) OVER (ORDER BY week_start) AS week_over_week,
                               AVG(complaint_count) OVER (
                                 ORDER BY week_start
                                 ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
                               ) AS trailing_4_week_avg
                        FROM weekly_demand
                        WHERE boro_cd = ?
                          AND complaint_type = ?
                        ORDER BY week_start
                        """,
                        (district_code, group),
                        mtime,
                    )
                    st.dataframe(result.tail(26), hide_index=True, width="stretch")
            else:
                st.dataframe(read_sql(queries[query_name], (), mtime), hide_index=True, width="stretch")
        except sqlite3.Error as error:
            st.error("The query failed.")
            st.write(str(error))

    with st.expander("How to read this, and what it does not say"):
        st.markdown(
            """
            311 counts the people who reported a problem, not every defect on the street.
            One Central Park weather series is applied to every district. A future week uses last year's weather for that week.
            Pavement is the latest DOT survey for the borough, not a rating from the morning of the call. Every district in a borough shares that figure.
            Population is the 2020 Census count published by City Planning in September 2021.
            A high score is not a reason to skip a new call.
            Repair blitzes can close potholes outside the 311 clock, and complaint volume is a weak stand-in for which streets the city has underinvested in.
            """
        )
        st.dataframe(
            meta.rename(columns={"name": "Dataset", "source": "Source", "vintage": "Vintage", "row_count": "Rows", "note": "Note"}),
            hide_index=True,
            width="stretch",
        )


if __name__ == "__main__":
    main()
