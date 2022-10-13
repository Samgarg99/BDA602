import sys

from pyspark import StorageLevel, keyword_only
from pyspark.ml import Pipeline, Transformer

# from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml.feature import CountVectorizer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# rom pyspark.sql.functions import col, concat, lit, split, when
from pyspark.sql.window import Window

# imports for transformer


# Creating a custom transformer


class RollingAverageTransformer(
    Transformer,
    HasInputCols,
    HasOutputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None):
        super(RollingAverageTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCols()
        print(input_cols)

        dataset = dataset.withColumn("DateOfGame", dataset.DateOfGame.cast("timestamp"))

        w = (
            Window.partitionBy("batter")
            .orderBy(F.col("DateOfGame").cast("long"))
            .rangeBetween(-100 * 86400, -1)
        )

        dataset = (
            dataset.withColumn(output_col[0], F.sum("Hit").over(w))
            .withColumn(output_col[1], F.sum("atBat").over(w))
            .withColumn(output_col[2], F.sum("Hit").over(w) / F.sum("atBat").over(w))
        )

        return dataset


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
                game_id,
                batter,
                atBat,
                Hit
            from baseball.batter_counts bc
           """

    sql2 = """select
                game_id,
                g.local_date
            from baseball.game g
            """

    database = "baseball"
    user = "root"
    pwd = "abc"
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    # Create a data frame by reading data from mariaDB via JDBC
    df1 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql1)
        .option("user", user)
        .option("password", pwd)
        .option("driver", jdbc_driver)
        .load()
    )

    df2 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql2)
        .option("user", user)
        .option("password", pwd)
        .option("driver", jdbc_driver)
        .load()
    )

    df1.createOrReplaceTempView("batter_count")
    df1.persist(StorageLevel.MEMORY_ONLY)

    df2.createOrReplaceTempView("game")
    df2.persist(StorageLevel.MEMORY_ONLY)

    # Method 1 for joining both tables
    '''df = spark.sql("""select
                batter,
                atBat,
                Hit,
                bc.game_id,
                DATE(g.local_date) as DateOfGame
            from batter_count bc
            join game g
            on bc.game_id = g.game_id
            order by batter, DateOfGame
    """)'''

    # Method 2 for joining both the tables
    df = (
        df1.join(df2, on="game_id", how="inner")
        .select(
            df1.batter, df1.atBat, df1.Hit, df2.game_id, df2.local_date.cast("DATE")
        )
        .withColumnRenamed("local_date", "DateOfGame")
    )

    df1.unpersist()
    df2.unpersist()

    df.createOrReplaceTempView("player_stats")
    df.persist(StorageLevel.MEMORY_ONLY)

    Rolling_Average_Transformer = RollingAverageTransformer(
        inputCols=["batter", "atBat", "Hit", "game_id", "DateOfGame"],
        outputCols=["Rolling_SumOfHits", "Rolling_SumOfatBat", "Rolling_Average"],
    )

    pipeline = Pipeline(stages=[Rolling_Average_Transformer])
    model = pipeline.fit(df)
    df = model.transform(df)

    print(df.show(200))


if __name__ == "__main__":
    sys.exit(main())
