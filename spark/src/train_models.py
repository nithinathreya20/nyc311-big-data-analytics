from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

# 1. Start SparkSession with executor memory allocation
spark = (
    SparkSession.builder.appName("NYC311-MLlib-Dual-Pipelines")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# 2. Schema definition matching the 29 validated columns
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

# 3. Read dataset
input_path = "data/nyc311_transfer/*.tsv"
df_raw = (
    spark.read.option("sep", "\t")
    .option("header", "false")
    .schema(schema)
    .csv(input_path)
)

# 4. Feature Engineering & Conditioning
timestamp_col = F.to_timestamp("created_date", "yyyy-MM-dd'T'HH:mm:ss.SSS")

df_filtered = df_raw.filter(
    (F.col("borough").isNotNull())
    & (F.col("borough") != "Unspecified")
    & (F.col("resolution_category").isin("FAST", "MODERATE", "SLOW"))
)

# Top 30 complaint types binning
top_complaints = [
    r["complaint_type"]
    for r in df_filtered.groupBy("complaint_type")
    .count()
    .orderBy(F.desc("count"))
    .limit(30)
    .collect()
]

df_features = (
    df_filtered.withColumn("hour", F.hour(timestamp_col))
    .withColumn("day_of_week", F.dayofweek(timestamp_col))
    .withColumn("month", F.month(timestamp_col))
    .withColumn(
        "is_weekend",
        F.when(F.dayofweek(timestamp_col).isin(1, 7), 1.0).otherwise(0.0),
    )
    .withColumn(
        "complaint_type_clean",
        F.when(
            F.col("complaint_type").isin(top_complaints),
            F.col("complaint_type"),
        ).otherwise("OTHER"),
    )
    # Person A Target: SLA Breach (1 if SLOW, 0 if FAST/MODERATE)
    .withColumn(
        "label_binary",
        F.when(F.col("resolution_category") == "SLOW", 1.0).otherwise(0.0),
    )
)

# 5. Pipeline Stages
index_agency = StringIndexer(
    inputCol="agency", outputCol="agency_idx", handleInvalid="keep"
)
index_complaint = StringIndexer(
    inputCol="complaint_type_clean",
    outputCol="complaint_idx",
    handleInvalid="keep",
)
index_borough = StringIndexer(
    inputCol="borough", outputCol="borough_idx", handleInvalid="keep"
)
index_target_multi = StringIndexer(
    inputCol="resolution_category",
    outputCol="label_multiclass",
    stringOrderType="frequencyDesc",
)

ohe = OneHotEncoder(
    inputCols=["agency_idx", "complaint_idx", "borough_idx"],
    outputCols=["agency_vec", "complaint_vec", "borough_vec"],
)

assembler = VectorAssembler(
    inputCols=[
        "agency_vec",
        "complaint_vec",
        "borough_vec",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "latitude",
        "longitude",
    ],
    outputCol="features",
    handleInvalid="skip",
)

# Sample 5% for local training (~1,000,000 records)
train_df, test_df = (
    df_features.sample(fraction=0.05, seed=42)
    .randomSplit([0.8, 0.2], seed=42)
)
train_df.cache()

# -------------------------------------------------------------
# PERSON A: Binary Classification (Logistic Regression SLA Breach)
# -------------------------------------------------------------
print("\n>>> Fitting Person A Pipeline (Binary SLA Breach Risk)...")
lr = LogisticRegression(
    featuresCol="features", labelCol="label_binary", maxIter=20
)
pipeline_a = Pipeline(
    stages=[index_agency, index_complaint, index_borough, ohe, assembler, lr]
)

model_a = pipeline_a.fit(train_df)
preds_a = model_a.transform(test_df)

eval_a = BinaryClassificationEvaluator(
    labelCol="label_binary", metricName="areaUnderROC"
)
auc = eval_a.evaluate(preds_a)
print(f">>> Person A Model ROC-AUC: {auc:.4f}")

# -------------------------------------------------------------
# PERSON B: Multi-Class Classification (Random Forest Turnaround Tiers)
# -------------------------------------------------------------
print("\n>>> Fitting Person B Pipeline (Multi-Class Velocity Tiers)...")
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label_multiclass",
    numTrees=20,
    maxDepth=8,
    seed=42,
)
pipeline_b = Pipeline(
    stages=[
        index_agency,
        index_complaint,
        index_borough,
        index_target_multi,
        ohe,
        assembler,
        rf,
    ]
)

model_b = pipeline_b.fit(train_df)
preds_b = model_b.transform(test_df)

eval_b = MulticlassClassificationEvaluator(
    labelCol="label_multiclass", metricName="f1"
)
f1 = eval_b.evaluate(preds_b)
print(f">>> Person B Model Weighted F1-Score: {f1:.4f}")

# -------------------------------------------------------------
# Save Pipeline Artifacts for Streamlit Dashboard
# -------------------------------------------------------------
model_a.write().overwrite().save("models/spark_model_binary")
model_b.write().overwrite().save("models/spark_model_multiclass")
print(
    "\n>>> Successfully saved models to models/spark_model_binary and models/spark_model_multiclass."
)

spark.stop()