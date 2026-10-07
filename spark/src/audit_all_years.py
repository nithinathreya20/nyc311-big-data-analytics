from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

spark = (
    SparkSession.builder.appName("NYC311-All-Years-Audit")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

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
    StructField("resolution_category", StringType(), True),
])

# Read all 7 TSV partitions at once
input_path = "data/nyc311_transfer/*.tsv"
df = (
    spark.read.option("sep", "\t")
    .option("header", "false")
    .schema(schema)
    .csv(input_path)
)

# Extract year from filename
df = df.withColumn(
    "request_year",
    F.regexp_extract(F.input_file_name(), r"(\d{4})\.tsv$", 1),
)

print("\n" + "=" * 65)
print("NYC 311 MULTI-YEAR COMPREHENSIVE DATA AUDIT (2020 - 2026)")
print("=" * 65)

# Year-by-year volume and anomaly profiling
metrics = (
    df.groupBy("request_year")
    .agg(
        F.count("*").alias("total_records"),
        F.sum(F.when(F.col("resolution_hours") < 0, 1).otherwise(0)).alias(
            "neg_durations"
        ),
        F.sum(
            F.when(F.col("resolution_hours") < (2.0 / 60.0), 1).otherwise(0)
        ).alias("instant_closures"),
        F.sum(
            F.when(F.col("resolution_hours") > (180 * 24.0), 1).otherwise(0)
        ).alias("zombies_gt_180d"),
        F.sum(
            F.when(
                (F.col("latitude") < 40.47)
                | (F.col("latitude") > 40.92)
                | (F.col("longitude") < -74.26)
                | (F.col("longitude") > -73.70)
                | F.col("latitude").isNull(),
                1,
            ).otherwise(0)
        ).alias("out_of_bounds_geo"),
        F.sum(
            F.when(
                (F.col("borough") == "Unspecified") | F.col("borough").isNull(),
                1,
            ).otherwise(0)
        ).alias("unspecified_borough"),
    )
    .orderBy("request_year")
)

metrics.show(10, truncate=False)

print("\nTarget Label Distribution Across All Years:")
df.groupBy("resolution_category").count().show()

total_rows = df.count()
print(f"Total Combined Records: {total_rows:,}")
print("=" * 65)

spark.stop()