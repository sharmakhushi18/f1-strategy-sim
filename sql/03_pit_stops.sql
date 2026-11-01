-- Pit stops with in-lap and out-lap time (LEAD over ALL laps, then filter).
WITH ordered AS (
    SELECT year, race, driver, lap_number, lap_time_s, is_pit_in,
           LEAD(lap_time_s) OVER (PARTITION BY year, race, driver
                                  ORDER BY lap_number) AS out_lap_s
    FROM laps
)
SELECT year, race, driver, lap_number AS in_lap, lap_time_s AS in_lap_s, out_lap_s
FROM ordered
WHERE is_pit_in = 1;
