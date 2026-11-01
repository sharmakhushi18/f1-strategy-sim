-- Raw degradation curve: average clean lap time by compound and tyre age.
SELECT compound, tyre_life, COUNT(*) AS laps, ROUND(AVG(lap_time_s), 3) AS avg_lap_s
FROM clean_laps
GROUP BY compound, tyre_life
HAVING COUNT(*) >= 3
ORDER BY compound, tyre_life;
