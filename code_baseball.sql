# ------------------------------- pitcher stats - starting pitcher ---------------------------------

drop table if exists pitcher1;

create table if not exists pitcher1(
select g.game_id, home_team_id, away_team_id, home_pitcher,
away_pitcher, b.winner_home_or_away, UNIX_TIMESTAMP(g.local_date) as DateOfGame
from game g
join boxscore b
on g.game_id = b.game_id
order by game_id);



# adding home pitcher and away pitcher columns

drop table if exists pitcher2;

create table if not exists pitcher2(
select p1.*,
p.Strikeout as Home_Pitcher_Strikeout,
p.Walk as Home_Pitcher_Walk,
p.Hit_By_Pitch as Home_Pitcher_Hit_By_Pitch,
p.Home_Run as Home_Pitcher_Home_run,
p.Hit as Home_Pitcher_Hit
from
(select *
from pitcher_counts pc
where pc.startingPitcher = 1) as p
join pitcher1 p1
on p1.game_id = p.game_id and p1.home_pitcher = p.pitcher);


drop table if exists pitcher3;

create table if not exists pitcher3(
select p2.*,
p.Strikeout as Away_Pitcher_Strikeout,
p.Walk as Away_Pitcher_Walk,
p.Hit_By_Pitch as Away_Pitcher_Hit_By_Pitch,
p.Home_Run as Away_Pitcher_Home_run,
p.Hit as Away_Pitcher_Hit
from
(select *
from pitcher_counts pc
where pc.startingPitcher = 1) as p
join pitcher2 p2
on p2.game_id = p.game_id and p2.away_pitcher = p.pitcher);


# Calculating the rolling stats for last 5 games and the current game

drop table if exists pitcher4;

create table if not exists pitcher4(
select p.*,
SUM(Home_Pitcher_Strikeout)
	OVER(PARTITION BY home_pitcher
	ORDER BY DateOfGame
	ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Home_Pitcher_Strikeout,
SUM(Home_Pitcher_Walk)
	OVER(PARTITION BY home_pitcher
	ORDER BY DateOfGame
	ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Home_Pitcher_walk,
SUM(Home_Pitcher_Hit_By_Pitch)
	OVER(PARTITION BY home_pitcher
	ORDER BY DateOfGame
	ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Home_Pitcher_Hit_By_Pitch,
SUM(Home_Pitcher_Home_run)
	OVER(PARTITION BY home_pitcher
	ORDER BY DateOfGame
	ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Home_Pitcher_Home_run,
SUM(Home_Pitcher_Hit)
	OVER(PARTITION BY home_pitcher
	ORDER BY DateOfGame
	ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Home_Pitcher_hit,
SUM(Away_Pitcher_Strikeout)
	OVER(PARTITION BY away_pitcher
	ORDER BY DateOfGame
	ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
	) AS RollingSum_away_Pitcher_Strikeout,
SUM(Away_Pitcher_Walk)
	OVER(PARTITION BY away_pitcher
	ORDER BY DateOfGame
	ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
	) AS RollingSum_away_Pitcher_walk,
SUM(Away_Pitcher_Hit_By_Pitch)
	OVER(PARTITION BY away_pitcher
	ORDER BY DateOfGame
	ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
	) AS RollingSum_away_Pitcher_Hit_By_Pitch,
SUM(Away_Pitcher_Home_run)
	OVER(PARTITION BY away_pitcher
	ORDER BY DateOfGame
	ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
	) AS RollingSum_away_Pitcher_home_run,
SUM(Away_Pitcher_Hit)
	OVER(PARTITION BY away_pitcher
	ORDER BY DateOfGame
	ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
	) AS RollingSum_away_Pitcher_hit
from pitcher3 p);




#----------------------------------- batter sats - Combined team --------------------------------------

DROP TABLE if exists batter1;


CREATE TABLE IF NOT EXISTS batter1(
SELECT tbc.game_id, team_id, atBat , Hit, Home_Run,
Strikeout, Walk, Sac_Fly,
UNIX_TIMESTAMP(g.local_date) as DateOfGame
FROM team_batting_counts tbc
join game g
on g.game_id = tbc.game_id);


DROP TABLE if exists batter2;

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
	) AS RollingSum_atBat,
	SUM(Home_Run)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Home_run,
	SUM(Strikeout)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Strikeout,
	SUM(Walk)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Walk,
	SUM(Sac_Fly)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Sac_Fly
FROM batter1
ORDER BY team_id, DateOfGame);


DROP TABLE if exists batter3;


# Calculating the rolling stats

CREATE TABLE IF NOT EXISTS batter3(
SELECT *, IF(RollingSum_atBat > 0, RollingSum_Hits/RollingSum_atBat, 0) as RollingAverage,
IF(RollingSum_Home_run > 0, RollingSum_atBat/RollingSum_Home_run, 0)as atBatsPerHomerun,
IF(RollingSum_Strikeout > 0 ,RollingSum_Walk/RollingSum_Strikeout, 0) as WalkStrikeoutRatio,
(RollingSum_Hits - RollingSum_Home_run)/(RollingSum_atBat - RollingSum_Strikeout - RollingSum_Home_run + RollingSum_Sac_Fly) as BABIP,
IF(RollingSum_Hits > 0, RollingSum_Home_run/RollingSum_Hits, 0) as HomerunPerHit
FROM batter2
ORDER BY team_id, DateOfGame);



# -------------------------------- Pitcher stats - Combined Team ------------------------------------

DROP TABLE if exists inning_combined;

CREATE TABLE inning_combined(
SELECT game_id, pitcher, MAX(outs) as outs
FROM inning
GROUP BY game_id, pitcher);


DROP TABLE if exists PitcherStats_Combined;

CREATE TABLE IF NOT EXISTS PitcherStats_Combined(
SELECT p.*, UNIX_TIMESTAMP(g.local_date) as DateOfGame
FROM
	(SELECT pc.game_id, pc.team_id, pc.homeTeam,
	SUM(Home_Run) as Home_Run ,
	SUM(Walk) as Walk ,
	SUM(Hit_By_Pitch) as Hit_By_Pitch ,
	SUM(Strikeout) as Strikeout,
	SUM(i.outs) as outs,
	SUM(Hit) as Hit
	FROM pitcher_counts pc
	JOIN inning_combined i
	ON i.game_id = pc.game_id AND i.pitcher = pc.pitcher
	GROUP BY game_id, team_id, homeTeam) AS p
JOIN game g
ON p.game_id = g.game_id
ORDER BY game_id, team_id
);



DROP TABLE if exists Pitcher_IP_Combined;

CREATE TABLE IF NOT EXISTS Pitcher_IP_Combined(
	SELECT *, outs/3 as Innings_Pitched ,
	3 + ((13 * Home_Run + 3 * (Walk + Hit_By_Pitch) - 2 * Strikeout) / (outs/3)) as DICE,
	9 * (Walk/outs)*3 as Bases_On_Ball,
	9 * (Hit/outs)*3 as Hit_p9i
	FROM PitcherStats_Combined );



DROP TABLE if exists batter_counts_combined;

CREATE TABLE IF NOT EXISTS batter_counts_combined(
SELECT game_id, team_id, COUNT(DISTINCT(batter)) as batter_count
from batter_counts bc
group by game_id, team_id
ORDER BY game_id, team_id);


DROP TABLE if exists Pitcher_stats_combined2;

CREATE TABLE IF NOT EXISTS Pitcher_stats_combined2
(SELECT pic.*, bc.batter_count
FROM batter_counts_combined bc
JOIN Pitcher_IP_Combined pic
ON pic.game_id = bc.game_id and pic.team_id != bc.team_id
ORDER BY game_id, team_id)
;


DROP TABLE if exists Pitcher_stats_combined3;

CREATE TABLE IF NOT EXISTS Pitcher_stats_combined3(
SELECT *, SUM(Innings_Pitched)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Innings_Pitched,
	SUM(DICE)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_DICE,
	SUM(Bases_On_Ball)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Bases_On_Ball,
	SUM(Hit_p9i)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_Hit_p9i,
	SUM(batter_count)
	OVER(PARTITION BY team_id
	ORDER BY DateOfGame
	RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
	) AS RollingSum_batter_count
FROM Pitcher_stats_combined2
ORDER BY team_id, DateOfGame);

select * from Pitcher_stats_combined3;


# ------------------------ Combining pitcher and batting stats -------------------------------------

DROP TABLE if exists combined1;

CREATE TABLE IF NOT EXISTS combined1
(SELECT
	b.team_id,
	b.game_id,
	b.RollingAverage,
	b.atBatsPerHomerun,
	b.WalkStrikeoutRatio,
	b.BABIP,
	b.HomerunPerHit,
	psc.RollingSum_Innings_Pitched,
	psc.RollingSum_DICE,
	psc.RollingSum_Bases_On_Ball,
	psc.RollingSum_Hit_p9i,
	psc.RollingSum_batter_count
FROM batter3 b
JOIN Pitcher_stats_combined3 psc
ON b.team_id = psc.team_id AND b.game_id = psc.game_id
) ORDER BY game_id ;


select * from combined1;



DROP TABLE if exists combined2;

CREATE TABLE IF NOT EXISTS combined2(
select p.*,
c.RollingAverage as home_RollingAverage,
c.atBatsPerHomerun as home_atBatsPerHomerun,
c.WalkStrikeoutRatio as home_WalkStrikeoutRatio,
c.BABIP as home_BABIP,
c.HomerunPerHit as home_HomerunPerHit,
c.RollingSum_Innings_Pitched as home_RollingSum_Innings_Pitched,
c.RollingSum_DICE as home_RollingSum_DICE,
c.RollingSum_Bases_On_Ball as home_RollingSum_Bases_On_Ball,
c.RollingSum_batter_count as home_RollingSum_batter_count,
c.RollingSum_Hit_p9i as home_RollingSum_Hit_p9i
from pitcher4 p
join combined1 c
on p.game_id = c.game_id and p.home_team_id =c.team_id );


DROP TABLE if exists combined3;

CREATE TABLE IF NOT EXISTS combined3(
select p.*,
c.RollingAverage as away_RollingAverage,
c.atBatsPerHomerun as away_atBatsPerHomerun,
c.WalkStrikeoutRatio as away_WalkStrikeoutRatio,
c.BABIP as away_BABIP,
c.HomerunPerHit as away_HomerunPerHit,
c.RollingSum_Innings_Pitched as away_RollingSum_Innings_Pitched,
c.RollingSum_DICE as away_RollingSum_DICE,
c.RollingSum_Bases_On_Ball as away_RollingSum_Bases_On_Ball,
c.RollingSum_batter_count as away_RollingSum_batter_count,
c.RollingSum_Hit_p9i as away_RollingSum_Hit_p9i
from combined2 p
join combined1 c
on p.game_id = c.game_id and p.away_team_id = c.team_id );




DROP TABLE if exists combined3_1;

CREATE TABLE IF NOT EXISTS combined3_1(
select * from combined3
where winner_home_or_away != '' );




UPDATE combined3_1
set winner_home_or_away = 1
where winner_home_or_away = "H";


UPDATE combined3_1
set winner_home_or_away = 0
where winner_home_or_away = "A";




DROP TABLE if exists combined4_1;

CREATE TABLE IF NOT EXISTS combined4_1(
select * from combined3
where winner_home_or_away != '' and
RollingSum_Home_Pitcher_Strikeout is not null and
RollingSum_Home_Pitcher_walk is not null and
RollingSum_Home_Pitcher_Hit_By_Pitch is not null and
RollingSum_Home_Pitcher_Home_run is not null and
RollingSum_Home_Pitcher_hit is not null and
RollingSum_away_Pitcher_Strikeout is not null and
RollingSum_away_Pitcher_walk is not null and
RollingSum_away_Pitcher_Hit_By_Pitch is not null and
RollingSum_away_Pitcher_home_run is not null and
RollingSum_away_Pitcher_hit is not null and
home_RollingAverage is not null and
home_atBatsPerHomerun is not null and
home_WalkStrikeoutRatio is not null and
home_BABIP is not null and
home_HomerunPerHit is not null and
home_RollingSum_Innings_Pitched is not null and
home_RollingSum_DICE is not null and
home_RollingSum_Bases_On_Ball is not null and
home_RollingSum_batter_count is not null and
home_RollingSum_Hit_p9i is not null and
away_RollingAverage is not null and
away_atBatsPerHomerun is not null and
away_WalkStrikeoutRatio is not null and
away_BABIP is not null and
away_HomerunPerHit is not null and
away_RollingSum_Innings_Pitched is not null and
away_RollingSum_DICE is not null and
away_RollingSum_Bases_On_Ball is not null and
away_RollingSum_batter_count is not null and
away_RollingSum_Hit_p9i is not null);









drop table if exists game_scores;

create table if not exists game_scores
(select game_id , home_team_id, away_team_id , home_w , away_w , home_l , away_l
from game g
order by home_team_id, game_id);



drop table if exists game_scores_1;

create table if not exists game_scores_1
(select gs.* , b.winner_home_or_away
from game_scores gs
join boxscore b
on gs.game_id = b.game_id
having b.winner_home_or_away  != ''
);

drop table if exists game_scores_2;

create table if not exists game_scores_2
(select gs.*, g.winner_home_or_away as away_win_or_lose
from game_scores_1 gs
join game_scores_1 g
on gs.game_id = g.game_id);


UPDATE game_scores_2
set winner_home_or_away = 1
where winner_home_or_away = "H";


UPDATE game_scores_2
set winner_home_or_away = 0
where winner_home_or_away = "A";

UPDATE game_scores_2
set away_win_or_lose = 0
where away_win_or_lose = "H";


UPDATE game_scores_2
set away_win_or_lose = 1
where away_win_or_lose = "A";



drop table if exists win_scores;

create table if not exists win_scores
(SELECT *, sum(winner_home_or_away) over(PARTITION by home_team_id
										order by game_id
										rows between UNBOUNDED PRECEDING and 1 PRECEDING) as rolling_sum_wins,
		COUNT(home_team_id)  over(PARTITION by home_team_id
										order by game_id
										rows between UNBOUNDED PRECEDING and 1 PRECEDING) as rolling_count_matches,
		sum(away_win_or_lose) over(PARTITION by away_team_id
										order by game_id
										rows between UNBOUNDED PRECEDING and 1 PRECEDING) as rolling_sum_wins_away,
		COUNT(away_team_id)  over(PARTITION by away_team_id
										order by game_id
										rows between UNBOUNDED PRECEDING and 1 PRECEDING) as rolling_count_matches_away
from game_scores_2 gs
order by home_team_id , game_id);



drop table if exists win_per;

create table if not exists win_per
(select *, IF(rolling_count_matches > 0, rolling_sum_wins/rolling_count_matches, 0) as win_per_home,
		IF(rolling_count_matches_away > 0, rolling_sum_wins_away/rolling_count_matches_away, 0) as win_per_away
from win_scores );



drop table if exists combined5;

create table if not exists combined5(
select c.*, wp.rolling_sum_wins , wp.rolling_count_matches , wp.rolling_sum_wins_away , wp.rolling_count_matches_away ,
		wp.win_per_home , wp.win_per_away
from combined4_1 c
join win_per wp
on c.game_id = wp.game_id );



drop table if exists combined5_1;

create table if not exists combined5_1(
	select *,
	(win_per_home - (win_per_home * win_per_away))/(win_per_home + win_per_away - (2 * win_per_home * win_per_away)) as log5_home,
	(win_per_away - (win_per_home * win_per_away))/(win_per_home + win_per_away - (2 * win_per_home * win_per_away)) as log5_away
	from combined5);


--UPDATE combined5_1
--set winner_home_or_away = 1
--where winner_home_or_away = "H";
--
--
--UPDATE combined5_1
--set winner_home_or_away = 0
--where winner_home_or_away = "A";