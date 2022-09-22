# ---------------------------- Assignment ------------------------------------------



# Batting average historic

SELECT batter, ROUND((SUM(Hit)/SUM(atBat)),3) as battingaverage
FROM baseball.batter_counts as bc
GROUP BY batter
HAVING sum(atBat) > 0
ORDER by batter;

# Used sum(atbat)> 0 because of error division by 0


# Creating a Table and storing Data

#DROP TABLE HistoricAverage ;

CREATE TABLE IF NOT EXISTS HistoricAverage(
player int,
BattingAverage DECIMAL(5,3)
);





INSERT INTO HistoricAverage(player, BattingAverage)
SELECT * 
FROM
	(SELECT batter, ROUND((SUM(Hit)/SUM(atBat)),3) as battingAverage
	FROM baseball.batter_counts as bc
	GROUP BY batter
	HAVING sum(atBat) > 0 
	ORDER BY batter) AS stats;





# ----------------------------------------------------------------------------------------

#batting avergae annual for single player

# Used sum(atbat)> 0 because of error division by 0


SELECT player, YearOfGame, ROUND((SUM(hit)/SUM(bat)),3) as battingaverage
FROM
(SELECT bc.batter as player, bc.atBat as bat, bc.Hit as hit, EXTRACT(YEAR from g.local_date) as YearOfGame
FROM batter_counts bc 
JOIN game g 
ON bc.game_id = g.game_id) as batting
GROUP BY player, YearOfGame 
HAVING player = 407832 AND SUM(bat)>0
ORDER BY YearOfGame ;



#batting avergae annual for each player

SELECT player, YearOfGame, ROUND((SUM(hit)/SUM(bat)),3) as battingaverage
FROM
(SELECT bc.batter as player, bc.atBat as bat, bc.Hit as hit, EXTRACT(YEAR from g.local_date) as YearOfGame
FROM batter_counts bc 
JOIN game g 
ON bc.game_id = g.game_id) as batting
GROUP BY player, YearOfGame 
HAVING SUM(bat)>0
ORDER BY player, YearOfGame;



# Creating a Table And Storing Data

#DROP TABLE AnnualAverage;

CREATE TABLE IF NOT EXISTS AnnualAverage(
player int,
YearOfGame int,
BattingAverage DECIMAL(5,3)
);


INSERT INTO AnnualAverage
SELECT * FROM
(SELECT player as p, YearOfGame as yog, ROUND((SUM(hit)/SUM(bat)),3) as battingaverage
FROM
(SELECT bc.batter as player, bc.atBat as bat, bc.Hit as hit, EXTRACT(YEAR from g.local_date) as YearOfGame
FROM batter_counts bc 
JOIN game g 
ON bc.game_id = g.game_id) as batting
GROUP BY p, yog
HAVING SUM(bat)>0 
ORDER BY yog, p) as aa;




# ----------------------------------------------------------------------------------------


# Rolling average over last 100 days

# Creating a table player stats and inserting data into it

#DROP TABLE PlayerStat;

CREATE TABLE IF NOT EXISTS PlayerStat(
player int,
atbat int,
hit int, 
DateOfGame DATETIME
);


INSERT INTO PlayerStat(player, atbat, hit, DateOfGame)
SELECT * FROM 
(SELECT bc.batter as player, bc.atBat as bat, bc.Hit as hit, g.local_date as DateOfGame
FROM batter_counts bc 
JOIN game g 
ON bc.game_id = g.game_id) as p
ORDER BY player, DateOfGame;




# Method 1- Here I have created a rolling dates table and then calculated the rolling average from it
# creating a rolling stats table and inserting DATA 

#DROP TABLE RollingStats;

CREATE TABLE IF NOT EXISTS RollingStats(
player int,
atbat int,
hit int, 
DateOfGame DATETIME,
RollingDate DATETIME,
DaysDifference int
);




INSERT INTO RollingStats 
SELECT *
FROM
	(SELECT ps1.player, ps2.atbat as bat, ps2.hit as bathit, ps1.DateOfGame as dog, ps2.DateOfGame AS RollingDates, DATEDIFF(ps1.DateOfGame, ps2.DateOfGame)
	FROM PlayerStat ps1
	LEFT JOIN PlayerStat ps2
	ON ps1.player = ps2.player AND DATEDIFF(ps1.DateOfGame, ps2.DateOfGame) BETWEEN 1 AND 100) as stats
	ORDER BY player, dog, RollingDates;

# Date difference is between 1 and 100 because we need to calculate for last 100 days prior to the current day (exclude current day)
	

# For single batter
	
SELECT player, DateOfGame, ROUND((SUM(hit)/SUM(atbat)),3) as RollingAverage, SUM(hit) as SumOfHits, SUM(atbat) as SumOfAtBat
FROM RollingStats rs 
WHERE player = 110029
GROUP BY player, DateOfGame
ORDER BY player, DateOfGame;



# For each batter
SELECT player, DateOfGame, ROUND((SUM(hit)/SUM(atbat)),3) as RollingAverage, SUM(hit) as SumOfHits, SUM(atbat) as SumOfAtBat
FROM RollingStats rs 
GROUP BY player, DateOfGame
ORDER BY player, DateOfGame;



# Creating a Rolling Average Table and inserting the data 

#DROP TABLE RollingAverage;

CREATE TABLE IF NOT EXISTS RollingAverage(
player int,
DateOfGame DATETIME,
RollingAverage DECIMAL(5,3),
SumOfHits int,
SumOfAtBat int
);



INSERT INTO RollingAverage 
SELECT * FROM 
	(SELECT player, DateOfGame, ROUND((SUM(hit)/SUM(atbat)),3) as RollingAverage, SUM(hit) as SumOfHits, SUM(atbat) as SumOfAtBat
	FROM RollingStats rs 
	GROUP BY player, DateOfGame
	HAVING SUM(atbat)>0) as stats
ORDER BY player, DateOfGame;





# Method 2 to calculate the rolling average over last 100 days prior to this game

SELECT  player, dog as DateOfGame, ROUND((sum(bathit)/SUM(bat)),3) as RollingAverage, SUM(bat), sum(bathit)
FROM
(SELECT ps1.player, ps1.DateOfGame as dog, ps2.DateOfGame AS RollingDates, ps2.atbat as bat, ps2.hit as bathit
FROM PlayerStat ps1
LEFT JOIN PlayerStat ps2
ON ps1.player = ps2.player AND DATEDIFF(ps1.DateOfGame, ps2.DateOfGame) BETWEEN 1 AND 100
ORDER BY ps1.DateOfGame) as p
GROUP by player, dog
ORDER BY player, dog;