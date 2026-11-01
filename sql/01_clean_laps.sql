-- Representative racing laps: no lap 1, in/out laps, or SC/VSC/yellow laps.
CREATE VIEW IF NOT EXISTS clean_laps AS
SELECT year, race, driver, lap_number, lap_time_s, compound, tyre_life
FROM laps
WHERE is_pit_in = 0 AND is_pit_out = 0 AND track_status = '1'
  AND lap_number > 1 AND lap_time_s IS NOT NULL;
