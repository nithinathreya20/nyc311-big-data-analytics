from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

spark = (
    SparkSession.builder.appName("NYC311-Outlier-Audit")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
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

df = (
    spark.read.option("sep", "\t")
    .option("header", "false")
    .schema(schema)
    .csv("data/nyc311_transfer/2020.tsv")
)

total = df.count()

instant = df.filter(F.col("resolution_hours") < (2.0 / 60.0)).count()
negative = df.filter(F.col("resolution_hours") < 0).count()
zombie = df.filter(F.col("resolution_hours") > (180 * 24.0)).count()
missing_geo = df.filter(
    F.col("latitude").isNull() | F.col("longitude").isNull()
).count()
out_geo = df.filter(
    (F.col("latitude") < 40.47)
    | (F.col("latitude") > 40.92)
    | (F.col("longitude") < -74.26)
    | (F.col("longitude") > -73.70)
).count()
bad_borough = df.filter(
    (F.col("borough") == "Unspecified") | F.col("borough").isNull()
).count()

# Multi-criteria clean dataframe to see what survives
df_clean = df.filter(
    (F.col("resolution_hours") >= (2.0 / 60.0))
    & (F.col("resolution_hours") <= (180 * 24.0))
    & (F.col("latitude").between(40.47, 40.92))
    & (F.col("longitude").between(-74.26, -73.70))
    & (F.col("borough") != "Unspecified")
    & (F.col("borough").isNotNull())
)
clean_count = df_clean.count()

print("\n" + "=" * 60)
print(f"TOTAL AUDITED RECORDS (2020.tsv): {total:,}")
print("=" * 60)
print(f"1. Instant closures (< 2 mins) : {instant:,} ({instant/total*100:.2f}%)")
print(
    f"2. Negative durations (< 0 hrs): {negative:,} ({negative/total*100:.2f}%)"
)
print(f"3. Zombie cases (> 180 days)   : {zombie:,} ({zombie/total*100:.2f}%)")
print(
    f"4. Missing Lat/Lon (NULL)      : {missing_geo:,} ({missing_geo/total*100:.2f}%)"
)
print(
    f"5. Outside NYC Bounding Box    : {out_geo:,} ({out_geo/total*100:.2f}%)"
)
print(
    f"6. Unspecified / NULL Borough  : {bad_borough:,} ({bad_borough/total*100:.2f}%)"
)
print("-" * 60)
print(
    f"SURVIVING CLEAN RECORDS        : {clean_count:,} ({clean_count/total*100:.2f}%)"
)
print(f"TOTAL DROPPED ANOMALIES        : {total - clean_count:,} ({(total - clean_count)/total*100:.2f}%)")
print("=" * 60)

spark.stop()
