-- NYC Infrastructure Intelligence
-- SQLite is the system of record. The Streamlit app only reads these tables.

PRAGMA foreign_keys = ON;

CREATE TABLE borough (
  borough_code INTEGER PRIMARY KEY,
  borough_name TEXT NOT NULL
);

CREATE TABLE community_district (
  boro_cd INTEGER PRIMARY KEY,
  borough_code INTEGER NOT NULL REFERENCES borough(borough_code),
  district_number INTEGER NOT NULL,
  district_name TEXT NOT NULL,
  shape_area REAL,
  population INTEGER,
  pop_vintage TEXT,
  density_per_sq_mi REAL,
  geometry TEXT
);

CREATE TABLE agency (
  agency_code TEXT PRIMARY KEY,
  agency_name TEXT
);

CREATE TABLE complaint_type (
  complaint_type TEXT PRIMARY KEY,
  type_group TEXT NOT NULL
);

CREATE TABLE type_group_list (
  complaint_type TEXT PRIMARY KEY
);

CREATE TABLE week_calendar (
  week_start TEXT PRIMARY KEY
);

CREATE TABLE weather_day (
  weather_date TEXT PRIMARY KEY,
  tmin REAL,
  tmax REAL,
  precip REAL,
  snow REAL,
  freeze_thaw INTEGER NOT NULL
);

CREATE TABLE weather_week (
  week_start TEXT PRIMARY KEY,
  freeze_thaw_days INTEGER,
  precip REAL,
  snow REAL,
  tmin_avg REAL,
  days_observed INTEGER,
  weather_source TEXT NOT NULL
);

CREATE TABLE district_pavement (
  borough_code INTEGER PRIMARY KEY REFERENCES borough(borough_code),
  miles_rated REAL,
  mean_rating REAL,
  share_good REAL,
  share_fair REAL,
  share_poor REAL,
  share_fair_or_poor REAL,
  newest_inspection TEXT,
  segment_count INTEGER,
  grain TEXT NOT NULL
);

CREATE TABLE complaint (
  unique_key TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  closed_at TEXT,
  due_at TEXT,
  agency_code TEXT REFERENCES agency(agency_code),
  complaint_type TEXT NOT NULL,
  type_group TEXT NOT NULL,
  descriptor TEXT,
  status TEXT,
  boro_cd INTEGER NOT NULL REFERENCES community_district(boro_cd),
  borough_code INTEGER NOT NULL REFERENCES borough(borough_code),
  incident_zip TEXT,
  cross_street_1 TEXT,
  cross_street_2 TEXT,
  latitude REAL,
  longitude REAL,
  week_start TEXT NOT NULL,
  hours_to_close REAL,
  is_pothole INTEGER NOT NULL,
  left_unfixed INTEGER,
  came_back INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE weekly_demand (
  boro_cd INTEGER NOT NULL REFERENCES community_district(boro_cd),
  complaint_type TEXT NOT NULL,
  week_start TEXT NOT NULL,
  complaint_count INTEGER NOT NULL,
  rate_per_10k REAL,
  unfixed_count INTEGER NOT NULL,
  came_back_count INTEGER NOT NULL,
  PRIMARY KEY (boro_cd, complaint_type, week_start)
);

CREATE TABLE load_forecast (
  boro_cd INTEGER NOT NULL REFERENCES community_district(boro_cd),
  complaint_type TEXT NOT NULL,
  week_start TEXT NOT NULL,
  predicted_count REAL NOT NULL,
  predicted_rate_per_10k REAL,
  model_name TEXT NOT NULL,
  is_holdout INTEGER NOT NULL,
  PRIMARY KEY (boro_cd, complaint_type, week_start)
);

CREATE TABLE unfixed_score (
  unique_key TEXT PRIMARY KEY REFERENCES complaint(unique_key),
  probability REAL NOT NULL,
  model_name TEXT NOT NULL
);

CREATE TABLE factor_importance (
  model_name TEXT NOT NULL,
  feature_name TEXT NOT NULL,
  importance REAL NOT NULL,
  detail TEXT,
  PRIMARY KEY (model_name, feature_name)
);

CREATE TABLE model_metric (
  model_name TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  metric_value REAL,
  baseline_value REAL,
  notes TEXT,
  PRIMARY KEY (model_name, metric_name)
);

CREATE TABLE duration_by_borough (
  borough_code INTEGER NOT NULL,
  complaint_type TEXT NOT NULL,
  n INTEGER NOT NULL,
  median_hours REAL,
  p90_hours REAL,
  PRIMARY KEY (borough_code, complaint_type)
);

CREATE TABLE dataset_meta (
  name TEXT PRIMARY KEY,
  source TEXT,
  vintage TEXT,
  row_count INTEGER,
  note TEXT
);

CREATE TABLE load_audit (
  step TEXT PRIMARY KEY,
  row_count INTEGER,
  note TEXT
);

CREATE INDEX idx_complaint_week ON complaint (boro_cd, type_group, week_start);
CREATE INDEX idx_complaint_created ON complaint (created_at);
CREATE INDEX idx_complaint_duration ON complaint (borough_code, type_group, hours_to_close);
CREATE INDEX idx_weekly_lookup ON weekly_demand (week_start, complaint_type);
CREATE INDEX idx_forecast_lookup ON load_forecast (week_start, complaint_type);

CREATE VIEW v_open_past_due AS
SELECT
  c.unique_key,
  c.created_at,
  c.due_at,
  c.boro_cd,
  c.borough_code,
  c.type_group,
  c.descriptor,
  c.status,
  c.cross_street_1,
  c.cross_street_2
FROM complaint c
WHERE c.left_unfixed = 1
  AND c.closed_at IS NULL;

CREATE VIEW v_pothole_season AS
SELECT
  CASE
    WHEN w.freeze_thaw_days >= 1 THEN 'Week had a freeze-thaw day'
    ELSE 'Week had no freeze-thaw day'
  END AS season,
  COUNT(*) AS potholes,
  ROUND(AVG(w.freeze_thaw_days), 2) AS avg_freeze_thaw_days,
  ROUND(AVG(w.precip), 2) AS avg_precip_mm
FROM complaint c
JOIN weather_week w ON w.week_start = c.week_start
WHERE c.is_pothole = 1
GROUP BY 1;
