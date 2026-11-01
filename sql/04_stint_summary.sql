-- Stint length and average pace per driver, compound and stint.
SELECT year, race, driver, stint, compound,
       COUNT(*) AS laps, MAX(tyre_life) AS end_age,
       ROUND(AVG(lap_time_s), 3) AS avg_clean_lap_s
FROM clean_laps
JOIN (SELECT DISTINCT year y, race r, driver d, lap_number n, stint FROM laps)
  ON y = year AND r = race AND d = driver AND n = lap_number
GROUP BY year, race, driver, stint, compound
ORDER BY year, race, driver, stint;
