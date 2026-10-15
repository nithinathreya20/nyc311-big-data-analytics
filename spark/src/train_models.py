"""
NYC 311 - Spark MLlib training and evaluation (corrected version).

Replaces spark/src/train_models.py. What changed and why
--------------------------------------------------------
1. TIMESTAMP BUG FIXED. The old script parsed created_date with the ISO pattern only.
   Records from 2022 onward use 'MM/dd/yyyy hh:mm:ss a', so hour / day_of_week / month
   were NULL for them and VectorAssembler(handleInvalid="skip") silently dropped those
   rows. This version parses both formats (same approach as the Hive queries), and
   prints how many rows have an unparseable timestamp instead of hiding them.
2. EVERY METRIC IS COMPUTED from the test-set predictions: confusion matrices,
   per-class precision/recall/F1, macro/weighted F1, PR-AUC, accuracy. Nothing is a
   typed-in constant. Counts (dataset, sample, train, test) are measured.
3. BASELINES are evaluated on the same test rows: majority class, and a rule that uses
   complaint type only (fitted on the training rows only).
4. The top-30 complaint vocabulary is computed from TRAINING rows only (the old script
   used all rows before splitting).
5. test_predictions_sample is built from real test rows (real unique_key) scored by the
   saved pipelines. The old file contained hand-written rows.
6. Optional temporal split: --split temporal trains on requests created before the
   latest year and tests on the latest year.

Defaults reproduce the original design: 5% Bernoulli sample, seed 42, 80/20 random split,
LogisticRegression(maxIter=20), RandomForest(numTrees=20, maxDepth=8, seed=42).

Usage (from the repository root):
    python spark/src/train_models.py
    python spark/src/train_models.py --split temporal
    python spark/src/train_models.py --input "data/nyc311_transfer/*.tsv" --fraction 0.05
"""
import argparse
import json
import os
import tempfile
from datetime import datetime, timezone

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.functions import vector_to_array

CLASS_ORDER = ["FAST", "MODERATE", "SLOW"]
CLASS_TITLES = {"FAST": "FAST (<24h)", "MODERATE": "MODERATE (24\u201372h)", "SLOW": "SLOW (>72h)"}
ISO_FMT = "yyyy-MM-dd'T'HH:mm:ss.SSS"
LEGACY_FMT = "MM/dd/yyyy hh:mm:ss a"

# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--input", default="/home/nithin-athreya/nyc311_transfer/*.tsv")
ap.add_argument("--fraction", type=float, default=0.05, help="Bernoulli sample fraction")
ap.add_argument("--split", choices=["random", "temporal"], default="random")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--driver-memory", default="6g")
ap.add_argument("--model-dir", default="models")
ap.add_argument("--metrics-path", default="models/dashboard_metrics.json")
args = ap.parse_args()

spark = (
    SparkSession.builder.appName("NYC311-MLlib-Dual-Pipelines")
    .master("local[*]")
    .config("spark.driver.memory", args.driver_memory)
    # Spark 4 enables ANSI mode by default, which makes to_timestamp() RAISE on a string that
    # does not match the pattern. We try two formats with coalesce(), so we need NULL instead.
    .config("spark.sql.ansi.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# --------------------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------------------
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
df_raw = spark.read.option("sep", "\t").option("header", "false").schema(schema).csv(args.input)

# --------------------------------------------------------------------------------------
# 2. Filter and submission-time features
# --------------------------------------------------------------------------------------
df_filtered = df_raw.filter(
    F.col("borough").isNotNull()
    & (F.col("borough") != "Unspecified")
    & F.col("resolution_category").isin(*CLASS_ORDER)
)

# Both timestamp formats (2020-2021 ISO, 2022+ 'MM/dd/yyyy hh:mm:ss a').
created_clean = F.regexp_replace("created_date", '"', "")
created_ts = F.coalesce(
    F.to_timestamp(created_clean, ISO_FMT),
    F.to_timestamp(created_clean, LEGACY_FMT),
)

df_base = (
    df_filtered.withColumn("created_ts", created_ts)
    .withColumn("hour", F.hour("created_ts"))
    .withColumn("day_of_week", F.dayofweek("created_ts"))
    .withColumn("month", F.month("created_ts"))
    .withColumn("year", F.year("created_ts"))
    .withColumn("is_weekend", F.when(F.col("day_of_week").isin(1, 7), 1.0).otherwise(0.0))
    .withColumn("label_binary", F.when(F.col("resolution_category") == "SLOW", 1.0).otherwise(0.0))
)

# Rows with an unparseable timestamp cannot be used; report them rather than hiding them.
n_total_filtered = df_base.count()
n_bad_ts = df_base.filter(F.col("created_ts").isNull()).count()
n_null_geo = df_base.filter(F.col("latitude").isNull() | F.col("longitude").isNull()).count()
print(f"\nRows after filters: {n_total_filtered:,}")
print(f"  unparseable created_date: {n_bad_ts:,}")
print(f"  null latitude/longitude : {n_null_geo:,}")
df_base = df_base.filter(F.col("created_ts").isNotNull())
df_base = df_base.dropna(subset=["latitude", "longitude", "agency", "complaint_type"])
n_usable = df_base.count()
print(f"Usable rows (all features present): {n_usable:,}")

year_counts = {int(r["year"]): r["count"] for r in df_base.groupBy("year").count().collect()}
print("Usable rows by creation year:", dict(sorted(year_counts.items())))

# --------------------------------------------------------------------------------------
# 3. Sample and split
# --------------------------------------------------------------------------------------
if args.split == "random":
    train_df, test_df = df_base.sample(fraction=args.fraction, seed=args.seed).randomSplit(
        [0.8, 0.2], seed=args.seed
    )
    split_desc = {"type": "random", "train": 0.80, "test": 0.20, "seed": args.seed}
else:
    last_year = max(year_counts)
    train_df = df_base.filter(F.col("year") < last_year).sample(fraction=args.fraction, seed=args.seed)
    test_df = df_base.filter(F.col("year") == last_year).sample(fraction=args.fraction, seed=args.seed)
    split_desc = {
        "type": "temporal",
        "train": f"created before {last_year}",
        "test": f"created in {last_year}",
        "seed": args.seed,
    }

# Complaint vocabulary from TRAINING rows only.
top_complaints = [
    r["complaint_type"]
    for r in train_df.groupBy("complaint_type").count().orderBy(F.desc("count")).limit(30).collect()
]


def add_complaint_clean(d):
    return d.withColumn(
        "complaint_type_clean",
        F.when(F.col("complaint_type").isin(top_complaints), F.col("complaint_type")).otherwise("OTHER"),
    )


train_df = add_complaint_clean(train_df).cache()
test_df = add_complaint_clean(test_df).cache()
n_train, n_test = train_df.count(), test_df.count()
n_sample = n_train + n_test
print(f"Measured sample: {n_sample:,}  train: {n_train:,}  test: {n_test:,}")
if train_df.select("resolution_category").distinct().count() < 3:
    raise SystemExit("Training rows do not contain all three resolution categories; "
                     "check the input path, filters and sampling fraction.")

# --------------------------------------------------------------------------------------
# 4. Pipelines
# --------------------------------------------------------------------------------------
index_agency = StringIndexer(inputCol="agency", outputCol="agency_idx", handleInvalid="keep")
index_complaint = StringIndexer(inputCol="complaint_type_clean", outputCol="complaint_idx", handleInvalid="keep")
index_borough = StringIndexer(inputCol="borough", outputCol="borough_idx", handleInvalid="keep")
index_target_multi = StringIndexer(
    inputCol="resolution_category", outputCol="label_multiclass", stringOrderType="frequencyDesc"
)
ohe = OneHotEncoder(
    inputCols=["agency_idx", "complaint_idx", "borough_idx"],
    outputCols=["agency_vec", "complaint_vec", "borough_vec"],
)
assembler = VectorAssembler(
    inputCols=[
        "agency_vec", "complaint_vec", "borough_vec",
        "hour", "day_of_week", "month", "is_weekend", "latitude", "longitude",
    ],
    outputCol="features",
    handleInvalid="error",  # all features were checked above; fail loudly instead of dropping rows
)

lr = LogisticRegression(featuresCol="features", labelCol="label_binary", maxIter=20)
pipeline_a = Pipeline(stages=[index_agency, index_complaint, index_borough, ohe, assembler, lr])

rf = RandomForestClassifier(
    featuresCol="features", labelCol="label_multiclass", numTrees=20, maxDepth=8, seed=args.seed
)
pipeline_b = Pipeline(
    stages=[index_agency, index_complaint, index_borough, index_target_multi, ohe, assembler, rf]
)

print("\n>>> Fitting binary pipeline (logistic regression, SLOW vs not SLOW)...")
model_a = pipeline_a.fit(train_df)
preds_a = model_a.transform(test_df).cache()
print(">>> Fitting multiclass pipeline (random forest)...")
model_b = pipeline_b.fit(train_df)
preds_b = model_b.transform(test_df).cache()

# --------------------------------------------------------------------------------------
# 5. Metric helpers (all computed from predictions)
# --------------------------------------------------------------------------------------


def confusion_counts(pred_df, label_col, pred_col="prediction"):
    rows = pred_df.groupBy(label_col, pred_col).count().collect()
    return {(int(r[label_col]), int(r[pred_col])): r["count"] for r in rows}


def per_class(cm, k):
    tp = cm.get((k, k), 0)
    fp = sum(v for (t, p), v in cm.items() if p == k and t != k)
    fn = sum(v for (t, p), v in cm.items() if t == k and p != k)
    support = tp + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, support


def summarize(cm, classes):
    n = sum(cm.values())
    stats = {k: per_class(cm, k) for k in classes}
    acc = sum(cm.get((k, k), 0) for k in classes) / n
    macro_f1 = sum(s[2] for s in stats.values()) / len(classes)
    w_f1 = sum(s[2] * s[3] for s in stats.values()) / n
    w_prec = sum(s[0] * s[3] for s in stats.values()) / n
    w_rec = sum(s[1] * s[3] for s in stats.values()) / n
    return n, stats, acc, macro_f1, w_f1, w_prec, w_rec


def mc_eval(pred_df, label_col, metric):
    return MulticlassClassificationEvaluator(labelCol=label_col, metricName=metric).evaluate(pred_df)


def bin_eval(pred_df, label_col, metric, raw_col="rawPrediction"):
    return BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_col, metricName=metric).evaluate(pred_df)


# ---- Binary model
cm_a = confusion_counts(preds_a, "label_binary")
tn, fp, fn, tp = (cm_a.get((0, 0), 0), cm_a.get((0, 1), 0), cm_a.get((1, 0), 0), cm_a.get((1, 1), 0))
n_a, stats_a, acc_a, macro_a, wf1_a, wprec_a, wrec_a = summarize(cm_a, [0, 1])
auc_a = bin_eval(preds_a, "label_binary", "areaUnderROC")
pr_auc_a = bin_eval(preds_a, "label_binary", "areaUnderPR")
prec1, rec1, f1_1, _ = stats_a[1]
assert abs(acc_a - mc_eval(preds_a, "label_binary", "accuracy")) < 1e-6

# ---- Multiclass model
labels_b = model_b.stages[3].labels  # StringIndexerModel for resolution_category
cm_b = confusion_counts(preds_b, "label_multiclass")
n_b, stats_b, acc_b, macro_b, wf1_b, wprec_b, wrec_b = summarize(cm_b, list(range(len(labels_b))))
wf1_eval = mc_eval(preds_b, "label_multiclass", "f1")
if abs(wf1_eval - wf1_b) > 1e-3:
    print(f"WARNING: weighted F1 from evaluator ({wf1_eval:.4f}) != from confusion matrix ({wf1_b:.4f})")

# ---- Baselines, fitted on training rows only
train_counts = (
    train_df.groupBy("complaint_type_clean", "resolution_category").count()
    .groupBy("complaint_type_clean")
    .agg(
        F.sum("count").alias("n"),
        F.sum(F.when(F.col("resolution_category") == "SLOW", F.col("count")).otherwise(0)).alias("n_slow"),
        F.max(F.struct("count", "resolution_category")).alias("best"),
    )
    .select(
        "complaint_type_clean",
        (F.col("n_slow") / F.col("n")).alias("slow_rate"),
        F.col("best.resolution_category").alias("majority_cat"),
    )
)
global_slow_rate = train_df.agg(F.avg("label_binary")).first()[0]
global_majority = (
    train_df.groupBy("resolution_category").count().orderBy(F.desc("count")).first()["resolution_category"]
)
base_df = (
    test_df.join(F.broadcast(train_counts), "complaint_type_clean", "left")
    .withColumn("slow_rate", F.coalesce("slow_rate", F.lit(global_slow_rate)))
    .withColumn("majority_cat", F.coalesce("majority_cat", F.lit(global_majority)))
    .cache()
)
base_auc = bin_eval(base_df, "label_binary", "areaUnderROC", raw_col="slow_rate")
base_pred_cat = F.col("majority_cat")
cm_base = {}
for r in base_df.groupBy("resolution_category", "majority_cat").count().collect():
    cm_base[(CLASS_ORDER.index(r["resolution_category"]), CLASS_ORDER.index(r["majority_cat"]))] = r["count"]
_, _, base_acc, base_macro, base_wf1, _, _ = summarize(cm_base, [0, 1, 2])
test_class_counts = {r["resolution_category"]: r["count"] for r in test_df.groupBy("resolution_category").count().collect()}
maj_fast_acc = test_class_counts.get("FAST", 0) / n_test
maj_notslow_acc = 1 - test_class_counts.get("SLOW", 0) / n_test

print("\n================ RESULTS (test split) ================")
print(f"Binary LR   : ROC-AUC {auc_a:.4f} | PR-AUC {pr_auc_a:.4f} | acc {acc_a:.4f} | "
      f"SLOW precision {prec1:.4f} recall {rec1:.4f} F1 {f1_1:.4f}")
print(f"              confusion TN {tn:,} FP {fp:,} FN {fn:,} TP {tp:,}")
print(f"Multiclass RF: weighted F1 {wf1_b:.4f} | macro F1 {macro_b:.4f} | acc {acc_b:.4f}")
print(f"Baseline (complaint type only): ROC-AUC {base_auc:.4f} | weighted F1 {base_wf1:.4f} | acc {base_acc:.4f}")
print(f"Baseline (always FAST)   acc {maj_fast_acc:.4f} | (always not-SLOW) acc {maj_notslow_acc:.4f}")

# --------------------------------------------------------------------------------------
# 6. Save models
# --------------------------------------------------------------------------------------
model_a.write().overwrite().save(os.path.join(args.model_dir, "spark_model_binary"))
model_b.write().overwrite().save(os.path.join(args.model_dir, "spark_model_multiclass"))

# --------------------------------------------------------------------------------------
# 7. Real test-prediction sample (real unique_key, scored by the saved pipelines)
# --------------------------------------------------------------------------------------
sample_rows = test_df.orderBy(F.rand(args.seed)).limit(25)
sa = model_a.transform(sample_rows).select(
    "unique_key", "agency", "complaint_type", "borough", "hour", "month", "is_weekend",
    "resolution_category", "label_binary",
    vector_to_array("probability")[1].alias("p_slow"),
)
sb = model_b.transform(sample_rows).select(
    "unique_key", vector_to_array("probability").alias("probs"), "prediction"
)
joined = sa.join(sb, "unique_key").collect()
test_predictions_sample = []
for r in joined:
    probs = {labels_b[i]: round(float(p), 4) for i, p in enumerate(r["probs"])}
    pred_tier = labels_b[int(r["prediction"])]
    test_predictions_sample.append({
        "id": r["unique_key"],
        "agency": r["agency"],
        "complaint_type": r["complaint_type"],
        "borough": r["borough"],
        "hour": int(r["hour"]),
        "month": int(r["month"]),
        "is_weekend": int(r["is_weekend"]),
        "actual_tier": r["resolution_category"],
        "actual_breach": int(r["label_binary"]),
        "pred_breach_prob": round(float(r["p_slow"]), 4),
        "pred_binary_decision": "SLA BREACH" if r["p_slow"] >= 0.5 else "SLA MET",
        "pred_tier": pred_tier,
        "tier_probabilities": {k: probs.get(k, 0.0) for k in CLASS_ORDER},
        "is_correct": pred_tier == r["resolution_category"],
    })

# --------------------------------------------------------------------------------------
# 8. Write metrics JSON (keeps kpis / equity / bottlenecks written by export_dashboard_metrics.py)
# --------------------------------------------------------------------------------------
existing = {}
try:
    with open(args.metrics_path, "r") as f:
        existing = json.load(f)
except Exception:
    existing = {}

now = datetime.now(timezone.utc)
class_metrics = []
for idx, name in enumerate(labels_b):
    p, r_, f1, sup = stats_b[idx]
    class_metrics.append({
        "class_name": CLASS_TITLES[name],
        "class_index": idx,
        "precision": round(p, 4),
        "recall": round(r_, 4),
        "f1_score": round(f1, 4),
        "support": sup,
        "support_pct": round(100.0 * sup / n_b, 2),
    })
class_metrics.sort(key=lambda c: CLASS_ORDER.index(c["class_name"].split(" ")[0]))

provenance = {
    "evaluation_run_id": "SPARK-EVAL-" + now.strftime("%Y%m%dT%H%M%SZ"),
    "evaluation_timestamp": now.isoformat(),
    "dataset_version": "NYC 311 cleaned (Pig preprocessed, 29 fields)",
    "dataset_source": args.input,
    "dataset_total_records": n_total_filtered,
    "usable_records": n_usable,
    "records_unparseable_created_date": n_bad_ts,
    "usable_records_by_year": {str(k): v for k, v in sorted(year_counts.items())},
    "sample_fraction": args.fraction,
    "sampling_method": "simple random (Bernoulli) sample; not stratified",
    "sample_records": n_sample,
    "split": split_desc,
    "split_ratio": {"train": 0.80, "test": 0.20, "seed": args.seed},
    "train_records": n_train,
    "test_records": n_test,
    "random_seed": args.seed,
    "features_assembled": [
        "agency_vec (OneHotEncoded StringIndexer, handleInvalid=keep)",
        "complaint_vec (top 30 of TRAINING rows + OTHER, OneHotEncoded)",
        "borough_vec (OneHotEncoded StringIndexer)",
        "hour (0-23, from created_date)",
        "day_of_week (1-7, from created_date)",
        "month (1-12, from created_date)",
        "is_weekend (1.0/0.0)",
        "latitude", "longitude",
    ],
    "data_leakage_safeguards": [
        "closed_date (excluded from feature pipeline)",
        "resolution_hours (excluded from feature pipeline)",
        "resolution_category (used only as target label)",
        "complaint vocabulary computed from training rows only",
    ],
    "saved_artifacts": {"binary": "models/spark_model_binary", "multiclass": "models/spark_model_multiclass"},
}

model_eval = {
    "person_a_roc_auc": round(auc_a, 4),
    "person_b_weighted_f1": round(wf1_b, 4),
    "person_a": {
        "model_name": "Binary SLA Breach Classifier",
        "algorithm": "LogisticRegression (Spark MLlib)",
        "target": "label_binary (1 if SLOW >72h else 0)",
        "hyperparameters": {"maxIter": 20, "regParam": 0.0, "elasticNetParam": 0.0, "threshold": 0.50},
        "metrics": {
            "roc_auc": round(auc_a, 4),
            "pr_auc": round(pr_auc_a, 4),
            "accuracy": round(acc_a, 4),
            "precision": round(prec1, 4),   # SLOW (positive) class
            "recall": round(rec1, 4),       # SLOW (positive) class
            "f1_score": round(f1_1, 4),     # SLOW (positive) class
            "weighted_precision": round(wprec_a, 4),
            "weighted_recall": round(wrec_a, 4),
            "weighted_f1": round(wf1_a, 4),
        },
        "confusion_matrix": {
            "true_negatives": tn, "false_positives": fp,
            "false_negatives": fn, "true_positives": tp,
            "total_test_samples": n_a,
            "tn_rate_pct": round(100 * tn / n_a, 2), "fp_rate_pct": round(100 * fp / n_a, 2),
            "fn_rate_pct": round(100 * fn / n_a, 2), "tp_rate_pct": round(100 * tp / n_a, 2),
        },
    },
    "person_b": {
        "model_name": "Multi-Class Resolution Tier Classifier",
        "algorithm": "RandomForestClassifier (Spark MLlib)",
        "target": "label_multiclass (" + ", ".join(f"{i}: {n}" for i, n in enumerate(labels_b)) + ")",
        "hyperparameters": {"numTrees": 20, "maxDepth": 8, "seed": args.seed},
        "metrics": {
            "weighted_f1": round(wf1_b, 4),
            "macro_f1": round(macro_b, 4),
            "accuracy": round(acc_b, 4),
            "weighted_precision": round(wprec_b, 4),
            "weighted_recall": round(wrec_b, 4),
        },
        "class_metrics": class_metrics,
        "confusion_matrix": {
            "labels": CLASS_ORDER,
            "matrix": [
                [cm_b.get((labels_b.index(t), labels_b.index(p)), 0) for p in CLASS_ORDER]
                for t in CLASS_ORDER
            ],
            "note": "rows = actual, columns = predicted",
        },
    },
    "baselines": {
        "complaint_type_only": {
            "description": "fitted on training rows; score = SLOW rate of complaint type, prediction = majority category",
            "roc_auc": round(base_auc, 4),
            "weighted_f1": round(base_wf1, 4),
            "macro_f1": round(base_macro, 4),
            "accuracy": round(base_acc, 4),
        },
        "always_fast_accuracy": round(maj_fast_acc, 4),
        "always_not_slow_accuracy": round(maj_notslow_acc, 4),
    },
}

existing["provenance"] = provenance
existing["model_eval"] = model_eval
existing["test_predictions_sample"] = test_predictions_sample

# atomic write
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(args.metrics_path)) or ".", suffix=".json")
with os.fdopen(fd, "w") as f:
    json.dump(existing, f, indent=2)
os.replace(tmp, args.metrics_path)
print(f"\n>>> Metrics written to {args.metrics_path}")
spark.stop()