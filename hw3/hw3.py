# This code if without the transformer

import sys

# from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def main():
    appName = "PySpark MariaDB"
    master = "local"

    # Create Spark session
    spark = (
        SparkSession.builder.appName(appName)
        .config("spark.jars", "mariadb-java-client-3.0.8.jar")
        .master(master)
        .getOrCreate()
    )

    sql1 = """select
                batter,
                atBat,
                Hit,
                bc.game_id,
                DATE(g.local_date) as DateOfGame
            from baseball.batter_counts bc
            join baseball.game g
            on bc.game_id = g.game_id
            order by batter, DateOfGame
           """

    database = "baseball"
    user = "root"  # use your username
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    # Create a data frame by reading data from mariaDB via JDBC
    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql1)
        .option("user", user)
        .option("password", "Sameer!12")  # add your password in place of pwd
        .option("driver", jdbc_driver)
        .load()
    )

    print(df.show())

    df = df.withColumn("DateOfGame", df.DateOfGame.cast("timestamp"))

    # create window by casting timestamp to long (number of seconds)
    w = (
        Window.partitionBy("batter")
        .orderBy(F.col("DateOfGame").cast("long"))
        .rangeBetween(-100 * 86400, -1)
    )

    df_test = (
        df.withColumn("rolling_average", F.sum("Hit").over(w) / F.sum("atBat").over(w))
        .withColumn("Rolling_SumOfHits", F.sum("Hit").over(w))
        .withColumn("Rolling_SumOfatBat", F.sum("atBat").over(w))
    )
    print("Rolling Average \n")
    print(df_test.show(20))


if __name__ == "__main__":
    sys.exit(main())
