DROP TABLE IF EXISTS batter1;


CREATE TABLE IF NOT EXISTS batter1(
SELECT tbc.game_id, team_id, atBat , Hit,
UNIX_TIMESTAMP(g.local_date) as DateOfGame
FROM team_batting_counts tbc
join game g
on g.game_id = tbc.game_id
having game_id = 12560);


DROP TABLE IF EXISTS batter2;

CREATE TABLE IF NOT EXISTS batter2(
SELECT *, SUM(Hit)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Hits,
	SUM(atBat)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_atBat
FROM batter1
ORDER BY team_id, DateOfGame);


DROP TABLE IF EXISTS batter3;


# Calculating the rolling stats

CREATE TABLE IF NOT EXISTS batter3(
SELECT *, IF(RollingSum_atBat > 0, RollingSum_Hits/RollingSum_atBat, 0) as RollingAverage
FROM batter2
ORDER BY team_id, DateOfGame);