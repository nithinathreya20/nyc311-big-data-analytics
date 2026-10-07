from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType
)


# --------------------------------------------------
# 1. Create Spark session
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("NYC311-Data-Loading-Test")
    .master("local[*]")
    .getOrCreate()
)


# --------------------------------------------------
# 2. Define the schema
# --------------------------------------------------

schema = StructType([
    StructField("unique_key", StringType(), True),
    StructField("created_date", StringType(), True),
    StructField("closed_date", StringType(), True),
    StructField("agency", StringType(), True),
    StructField("agency_name", StringType(), True),
    StructField("complaint_type", StringType(), True),
    StructField("descriptor", StringType(), True),
    StructField("descriptor_2", StringType(), True),
    StructField("location_type", StringType(), True),
    StructField("incident_zip", StringType(), True),
    StructField("incident_address", StringType(), True),
    StructField("street_name", StringType(), True),
    StructField("cross_street_1", StringType(), True),
    StructField("cross_street_2", StringType(), True),
    StructField("intersection_street_1", StringType(), True),
    StructField("intersection_street_2", StringType(), True),
    StructField("address_type", StringType(), True),
    StructField("city", StringType(), True),
    StructField("landmark", StringType(), True),
    StructField("facility_type", StringType(), True),
    StructField("status", StringType(), True),
    StructField("community_board", StringType(), True),
    StructField("council_district", StringType(), True),
    StructField("police_precinct", StringType(), True),
    StructField("borough", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("resolution_hours", DoubleType(), True),
    StructField("resolution_category", StringType(), True)
])


# --------------------------------------------------
# 3. Read ONE year first
# --------------------------------------------------

input_path = "data/nyc311_transfer/2020.tsv"

df = (
    spark.read
    .option("sep", "\t")
    .option("header", "false")
    .schema(schema)
    .csv(input_path)
)


# --------------------------------------------------
# 4. Basic validation
# --------------------------------------------------

print("\n======================================")
print("NYC 311 Spark Data Loading Test")
print("======================================")

print("\nSchema:")
df.printSchema()

print("\nFirst 5 records:")
df.show(5, truncate=False)

print("\nNumber of columns:")
print(len(df.columns))

print("\nNumber of rows:")
print(df.count())

print("\nResolution category distribution:")
df.groupBy("resolution_category").count().show()

print("\n======================================")
print("Data loading test completed")
print("======================================")


spark.stop()