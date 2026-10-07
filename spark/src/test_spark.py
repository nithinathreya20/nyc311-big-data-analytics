from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("NYC311-Spark-Test")
    .master("local[*]")
    .getOrCreate()
)

print("===================================")
print("Spark is working!")
print("Spark version:", spark.version)
print("===================================")

data = [
    ("FAST", 10),
    ("MODERATE", 30),
    ("SLOW", 100)
]

df = spark.createDataFrame(data, ["category", "hours"])

df.show()

spark.stop()