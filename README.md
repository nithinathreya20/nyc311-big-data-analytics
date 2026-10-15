# NYC 311 Big Data Analytics

A batch Big Data pipeline for NYC 311 service requests. It cleans and labels the data with Apache Pig, stores and summarises it with Hive, trains Spark MLlib classifiers to predict the **resolution-time category** of a request from information available at submission, and presents everything in a Streamlit dashboard.

**Research question:** can machine learning predict the eventual resolution-time category of an NYC 311 request using only information available when the request is submitted?

| Category | Definition |
|---|---|
| `FAST` | resolved in less than 24 hours |
| `MODERATE` | resolved in 24 to 72 hours, inclusive |
| `SLOW` | resolved in more than 72 hours |

The models classify **historical, closed requests**. They do not estimate the remaining time of an open request, and the project does not implement real-time or backlog prediction.

---

## 1. Technology stack

| Component | Role |
|---|---|
| Hadoop HDFS | Storage for raw, cleaned and aggregate data |
| Apache Pig | Cleaning, deduplication, duration calculation, labelling |
| Apache Hive | Year-partitioned external table, SQL analytics, dashboard aggregates |
| Apache Spark MLlib | Feature engineering, model training and evaluation (local mode, `local[*]`) |
| Python, Pandas, Plotly | Pipeline utilities and dashboard charts |
| Streamlit | Interactive dashboard |

Nothing in this repository implements streaming, workflow schedulers, cloud services or a prediction API.

## 2. Architecture

```text
NYC Open Data (yearly CSV, 44 columns)
        |
        v
HDFS  /nyc311/raw/full/<year>/
        |
        v
Apache Pig  (pig/clean_311_final.pig, one run per year)
        |
        v
HDFS  /nyc311/cleaned_final/<year>/    29 tab-separated fields
        |
        +--------------------------------+
        |                                |
        v                                v
Apache Hive                      Local TSV copy (manual transfer)
external table, PARTITIONED      data/nyc311_transfer/*.tsv
BY (year)                                |
        |                                v
        v                          Spark MLlib (train_models.py)
10 aggregate CSVs                        |
/nyc311/dashboard/ ->                    v
dashboard/data/                  models/spark_model_binary
        |                        models/spark_model_multiclass
        |                        models/dashboard_metrics.json
        +----------------+---------------+
                         |
                         v
                  Streamlit dashboard
```

Filtering and labelling happen once, in Pig. Hive and Spark both read the same cleaned records. The HDFS-to-local copies (cleaned TSVs for Spark, aggregate CSVs for the dashboard) are done by hand; no script in this repository performs them.

## 3. Repository structure

```text
.
├── app.py                           Streamlit entry point
├── dashboard/
│   ├── components/                  charts.py, kpis.py
│   ├── utils/                       data_loader.py, validation.py
│   ├── views/                       overview, request_analytics, resolution_analysis,
│   │                                ml_insights, data_info
│   └── data/                        10 Hive aggregate exports (CSV part files)
├── data/sample/                     small raw-format sample files
├── docs/                            pipeline.md, schema.md
├── hive/
│   ├── create_tables.hql            database nyc311, external table service_requests
│   ├── add_partitions.hql           year partitions 2020-2026
│   ├── analytics.hql                10 exploratory queries
│   └── create_dashboard_data.hql    10 INSERT OVERWRITE DIRECTORY exports
├── models/
│   ├── spark_model_binary/          saved logistic-regression PipelineModel
│   ├── spark_model_multiclass/      saved random-forest PipelineModel
│   └── dashboard_metrics.json       metrics, provenance, KPIs, equity/bottleneck tables
├── pig/clean_311_final.pig          cleaning pipeline
├── spark/src/
│   ├── train_models.py              training and evaluation
│   ├── export_dashboard_metrics.py  KPI, equity and bottleneck tables
│   ├── audit_data_quality.py        anomaly profile for one year
│   ├── audit_all_years.py           anomaly profile for all years
│   ├── load_data.py                 single-year load check
│   └── test_spark.py                Spark smoke test
├── tests/test_dashboard_data.py     13 unit tests for the dashboard data layer
├── hadoop.env                       HDFS settings (NameNode, replication factor 1)
└── requirements.txt                 streamlit, pandas, plotly
```

## 4. Prerequisites

- Java compatible with your Hadoop, Hive and Spark versions
- Hadoop/HDFS, Apache Pig (with Piggybank for `CSVExcelStorage`), Apache Hive with HiveServer2
- Apache Spark / PySpark (not listed in `requirements.txt`; install the version matching your Spark and Java setup, and make sure the `pyspark` package and the Spark runtime versions agree)
- Python 3.10+

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pyspark          # for the Spark scripts
```

## 5. Data preparation

1. Download the yearly CSV files for 2020 to the present from the [NYC 311 Service Requests](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9) dataset (ID `erm2-nwe9`). The Pig script expects the 44-column export.
2. Upload them to HDFS, one directory per year:

```bash
hdfs dfs -mkdir -p /nyc311/raw/full/2020
hdfs dfs -put 311_2020.csv /nyc311/raw/full/2020/
# repeat for each year
```

Record the data snapshot date. The dataset is a rolling export, so counts depend on when you download it.

## 6. Running the pipeline

### Step 1: Clean with Pig (once per year)

```bash
pig -x mapreduce -param YEAR=2020 pig/clean_311_final.pig
# repeat for 2021 ... 2026
hdfs dfs -ls /nyc311/cleaned_final/2020
```

Cleaning rules (in order): remove header rows; require unique_key, created_date, closed_date, agency and complaint_type (so only **closed** requests are kept); deduplicate by `unique_key`; parse both timestamp formats (`yyyy-MM-dd'T'HH:mm:ss.SSS` for 2020-2021 and `MM/dd/yyyy hh:mm:ss a` from 2022); require closed >= created; compute `resolution_hours`; keep 2 minutes to 180 days; keep coordinates inside the NYC bounding box (lat 40.47-40.92, lon -74.26 to -73.70); assign the category; write 29 tab-separated fields.

### Step 2: Create the Hive table and partitions

```bash
beeline -u 'jdbc:hive2://localhost:10000/default' -f hive/create_tables.hql
beeline -u 'jdbc:hive2://localhost:10000/default' -f hive/add_partitions.hql
```

### Step 3: Run the analytics and export dashboard aggregates

```bash
beeline -u 'jdbc:hive2://localhost:10000/default' -f hive/analytics.hql
beeline -u 'jdbc:hive2://localhost:10000/default' -f hive/create_dashboard_data.hql
```

Copy each `/nyc311/dashboard/<name>/` directory into `dashboard/data/<name>/`, for example:

```bash
hdfs dfs -get /nyc311/dashboard/yearly_requests dashboard/data/
```

### Step 4: Copy the cleaned data for Spark

```bash
mkdir -p data/nyc311_transfer
for y in 2020 2021 2022 2023 2024 2025 2026; do
  hdfs dfs -getmerge /nyc311/cleaned_final/$y data/nyc311_transfer/$y.tsv
done
```

### Step 5: Train and evaluate the models

```bash
python spark/src/train_models.py
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--input` | `data/nyc311_transfer/*.tsv` | cleaned TSV files |
| `--fraction` | `0.05` | Bernoulli sample fraction |
| `--split` | `random` | `random` (80/20) or `temporal` (train before the latest year, test on the latest year) |
| `--seed` | `42` | sampling, split and forest seed |
| `--driver-memory` | `6g` | Spark driver memory |

The script trains both pipelines, computes every metric from the test predictions, evaluates a complaint-type baseline, saves the models, and writes `models/dashboard_metrics.json` (keeping any KPI/equity/bottleneck blocks already there).

### Step 6: Export dashboard KPIs

```bash
python spark/src/export_dashboard_metrics.py
```

Run this after training; it preserves the model-evaluation block.

### Step 7: Launch the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501. The dashboard reads the aggregate CSVs and the JSON file; it does not load PySpark or retrain anything.

Pages: Overview, Request Analytics, Resolution Analysis, ML Insights, Data Info.

## 7. Machine learning

Both pipelines use only attributes known at submission time.

| Feature | Transformation |
|---|---|
| `agency` | StringIndexer (`handleInvalid=keep`) then OneHotEncoder |
| `complaint_type` | top 30 types **in the training rows** kept, others = `OTHER`; index, one-hot |
| `borough` | StringIndexer then OneHotEncoder |
| `created_date` | `hour`, `day_of_week`, `month`, `is_weekend` (both timestamp formats parsed) |
| `latitude`, `longitude` | numeric |

**Excluded to prevent leakage:** `closed_date`, `resolution_hours` and `resolution_category` (used only as the label).

| | Binary model | Multiclass model |
|---|---|---|
| Task | `SLOW` vs not `SLOW` | `FAST` / `MODERATE` / `SLOW` |
| Algorithm | LogisticRegression, `maxIter=20` | RandomForest, 20 trees, depth 8, seed 42 |
| Main metrics | ROC-AUC, PR-AUC, confusion matrix | weighted F1, macro F1, per-class metrics, confusion matrix |

Sampling is a simple random (Bernoulli) sample, **not stratified**, followed by a random 80/20 split.

### Results (run `SPARK-EVAL-20261015T100753Z`)

Input: 20,627,368 usable requests (2020 to September 2026); 5% sample of 1,031,402; train 824,840; test 206,562.

| Measure | Model | Complaint-type-only baseline |
|---|---|---|
| Binary ROC-AUC | 0.9285 | 0.8992 |
| Binary accuracy | 0.8605 | 0.7314 (always not-SLOW) |
| Multiclass weighted F1 | 0.7762 | 0.7719 |
| Multiclass macro F1 | 0.6729 | 0.6810 |
| MODERATE-class F1 | 0.4057 | n/a |

Reading these honestly: complaint type (and through it, agency) carries most of the signal. The binary model clearly improves on the baseline; the multiclass model improves weighted F1 only marginally and recognises the intermediate `MODERATE` class poorly.

## 8. Testing

```bash
python3 -m unittest tests.test_dashboard_data
```

The 13 tests check the dashboard loaders, aggregate totals, formatting helpers, chart construction and the internal consistency of `dashboard_metrics.json` (confusion matrices sum to the test size, supports agree, accuracy equals weighted recall, macro and weighted F1 recompute from per-class values). They do **not** test the Pig, Hive or Spark code.

## 9. Limitations

- **Random split, no temporal validation.** Use `--split temporal` to test on the latest year.
- **Closed requests only.** Requests still open are absent, and recent months are biased toward fast closure.
- **Single run, fixed hyperparameters,** no cross-validation or confidence intervals, no calibration check.
- **Class imbalance is not addressed;** `MODERATE` is the smallest class (about 14%).
- **Case-variant complaint types** (for example `APPLIANCE` and `Appliance`) are not merged.
- **Spark runs in local mode;** no run-time or scalability measurements are stored, so no scalability claim is made.
- **The what-if panel on the ML Insights page is a rule-based illustration,** not the trained models, unless it has been connected to the saved pipelines.
- 311 data reflect who reports problems, not all underlying problems. Borough or agency differences are differences in recorded requests, not verdicts on service equity.

## 10. Reproducibility notes

- Record the download date, year range and row counts for each run. `models/dashboard_metrics.json` stores the run identifier, timestamp, measured counts and split settings.
- Do not treat existing generated files as proof that the pipeline just ran.
- 2026 is a partial year (January to September in the current aggregates).
- The raw record count and per-rule drop counts are not stored; run `spark/src/audit_data_quality.py` and `audit_all_years.py` against your data and keep the output if you need them.

## 11. Data source

[NYC Open Data, 311 Service Requests from 2020 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9) (`erm2-nwe9`).