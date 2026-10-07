import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

spark = (
    SparkSession.builder.appName("NYC311-Dashboard-Metrics-Export")
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

df = (
    spark.read.option("sep", "\t")
    .option("header", "false")
    .schema(schema)
    .csv("data/nyc311_transfer/*.tsv")
)
df = df.filter(
    (F.col("borough").isNotNull()) & (F.col("borough") != "Unspecified")
)

# 1. Macro KPIs
total_requests = df.count()
tier_counts = {
    r["resolution_category"]: r["count"]
    for r in df.groupBy("resolution_category").count().collect()
}
median_res = df.approxQuantile("resolution_hours", [0.5], 0.01)[0]
p75_res = df.approxQuantile("resolution_hours", [0.75], 0.01)[0]

# 2. Equity: Borough x Agency Breach Rates
equity = (
    df.groupBy("borough", "agency")
    .agg(
        F.count("*").alias("volume"),
        F.round(
            F.avg(
                F.when(F.col("resolution_category") == "SLOW", 1.0).otherwise(
                    0.0
                )
            )
            * 100,
            2,
        ).alias("breach_pct"),
        F.round(F.avg("resolution_hours"), 2).alias("avg_hours"),
    )
    .filter(F.col("volume") > 1000)
    .orderBy(F.desc("breach_pct"))
    .toPandas()
    .to_dict(orient="records")
)

# 3. Bottleneck Analysis: Top 25 Complaints
bottlenecks = (
    df.groupBy("complaint_type")
    .agg(
        F.count("*").alias("volume"),
        F.round(F.avg("resolution_hours"), 2).alias("avg_hours"),
        F.round(
            F.avg(
                F.when(F.col("resolution_category") == "SLOW", 1.0).otherwise(
                    0.0
                )
            )
            * 100,
            2,
        ).alias("breach_rate"),
    )
    .orderBy(F.desc("volume"))
    .limit(25)
    .toPandas()
    .to_dict(orient="records")
)

metrics_payload = {
    "kpis": {
        "total_requests": total_requests,
        "median_hours": round(median_res, 2),
        "p75_hours": round(p75_res, 2),
        "fast_pct": round(
            (tier_counts.get("FAST", 0) / total_requests) * 100, 2
        ),
        "moderate_pct": round(
            (tier_counts.get("MODERATE", 0) / total_requests) * 100, 2
        ),
        "slow_pct": round(
            (tier_counts.get("SLOW", 0) / total_requests) * 100, 2
        ),
    },
    "model_eval": {
        "person_a_roc_auc": 0.9161,
        "person_b_weighted_f1": 0.7766,
    },
    "equity": equity,
    "bottlenecks": bottlenecks,
}

with open("models/dashboard_metrics.json", "w") as f:
    json.dump(metrics_payload, f, indent=2)

print(">>> Dashboard analytical metrics written to models/dashboard_metrics.json")
spark.stop()