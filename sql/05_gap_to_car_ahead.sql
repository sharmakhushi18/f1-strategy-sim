-- Gap to the car directly ahead at the end of each lap (self-join on position).
SELECT a.year, a.race, a.lap_number, a.driver AS car, b.driver AS car_ahead,
       ROUND(a.lap_end_s - b.lap_end_s, 3) AS gap_s
FROM laps a
JOIN laps b ON a.year = b.year AND a.race = b.race
           AND a.lap_number = b.lap_number AND b.position = a.position - 1;
