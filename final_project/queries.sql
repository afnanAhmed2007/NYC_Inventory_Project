-- Management queries used by the dashboard.
-- Bind the ? placeholders from the borough, district, complaint group, and week filters.

-- Requests opened in weeks that had a freeze-thaw day, versus weeks that did not.
-- Limited to complaints whose descriptor says pothole.
SELECT season, potholes, avg_freeze_thaw_days, avg_precip_mm
FROM v_pothole_season;

-- Lower median and 90th percentile hours from open to close, by borough and group.
SELECT b.borough_name, d.complaint_type, d.n, d.median_hours, d.p90_hours
FROM duration_by_borough d
JOIN borough b ON b.borough_code = d.borough_code
ORDER BY b.borough_name, d.complaint_type;

-- Potholes that were still open past the deadline, or that closed after it, by year.
SELECT substr(created_at, 1, 4) AS year,
       COUNT(*) AS potholes,
       SUM(CASE WHEN left_unfixed = 1 THEN 1 ELSE 0 END) AS past_due,
       ROUND(AVG(CASE WHEN left_unfixed IS NOT NULL THEN left_unfixed END), 3) AS past_due_share
FROM complaint
WHERE is_pothole = 1
GROUP BY 1
ORDER BY 1;

-- Came-back requests: same borough, complaint group, and cross streets within 90 days
-- of an earlier close. The flag is stored on the request during the load.
SELECT b.borough_name,
       c.type_group,
       SUM(c.came_back) AS came_back_requests,
       COUNT(*) AS requests
FROM complaint c
JOIN borough b ON b.borough_code = c.borough_code
GROUP BY b.borough_name, c.type_group
ORDER BY came_back_requests DESC;

-- Week-over-week change and a 4-week trailing average. Window functions, one district.
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
ORDER BY week_start;

-- Districts whose predicted count is high for the chosen week, with the latest
-- past-due count beside it. The app flags a district when it is in the high-forecast
-- set and not in the latest past-due set.
SELECT d.district_name,
       f.predicted_count,
       w.unfixed_count AS unfixed_in_forecast_week,
       latest.unfixed_count AS unfixed_in_latest_week
FROM load_forecast f
JOIN community_district d ON d.boro_cd = f.boro_cd
LEFT JOIN weekly_demand w
  ON w.boro_cd = f.boro_cd
 AND w.complaint_type = f.complaint_type
 AND w.week_start = f.week_start
LEFT JOIN weekly_demand latest
  ON latest.boro_cd = f.boro_cd
 AND latest.complaint_type = f.complaint_type
 AND latest.week_start = ?
WHERE f.week_start = ?
  AND f.complaint_type = ?
ORDER BY f.predicted_count DESC;
