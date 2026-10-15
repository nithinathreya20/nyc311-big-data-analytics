# NYC 311 Big Data Analytics Pipeline

This document explains the project's end-to-end architecture, processing stages, data movement, model-training workflow, and dashboard generation. It is intended to help a new contributor understand and reproduce the pipeline.

For field definitions and exact column order, see [`schema.md`](schema.md).

## 1. Project Scope

The project analyzes historical NYC 311 service requests to understand resolution patterns across time, boroughs, agencies, and complaint categories. It also trains machine-learning models to classify the observed resolution-duration category from selected request attributes.

**Dataset:** [NYC 311 Service Requests from 2020 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9)  
**Dataset ID:** `erm2-nwe9`  
**Configured year range:** 2020–2026

### Resolution categories

| Category | Definition |
|---|---|
| `FAST` | Resolution in less than 24 hours |
| `MODERATE` | Resolution from 24 to 72 hours, inclusive |
| `SLOW` | Resolution in more than 72 hours |

The categories are derived from `created_date` and `closed_date`. The analytical dataset contains service requests with valid closure information and describes their observed historical resolution outcomes.

## 2. Architecture Overview

```text
NYC Open Data
    |
    v
Raw CSV files
    |
    v
Hadoop HDFS
/nyc311/raw/full/$YEAR/
    |
    v
Apache Pig
Cleaning, validation, deduplication,
duration calculation and categorization
    |
    v
Cleaned TSV data in HDFS
/nyc311/cleaned_final/$YEAR/
    |
    +-----------------------------------+
    |                                   |
    v                                   v
Apache Hive                         Local TSV transfer
Partitioned data and analytics      data/nyc311_transfer/*.tsv
    |                                   |
    v                                   v
Dashboard aggregates              Apache Spark MLlib
/nyc311/dashboard/                 Feature engineering,
    |                              model training and evaluation
    |                                   |
    v                                   v
Local dashboard data               Saved model pipelines
and visualizations                 and evaluation metrics
    |                                   |
    +----------------+------------------+
                     |
                     v
             Streamlit Dashboard
```

Hive and Spark process the cleaned dataset for complementary purposes. Hive produces descriptive analytics and dashboard aggregates, while Spark MLlib trains classification models. The Streamlit application brings the generated analytics, model metrics, and prediction functionality together.

## 3. Main Data Locations

| Stage | Location | Description |
|---|---|---|
| Raw data | `/nyc311/raw/full/$YEAR/*.csv` | Source CSV files in HDFS |
| Cleaned data | `/nyc311/cleaned_final/$YEAR` | Headerless, tab-separated records |
| Hive table | `nyc311.service_requests` | External table partitioned by year |
| Hive aggregates | `/nyc311/dashboard/` | Aggregated datasets for dashboard visualizations |
| Spark input | `data/nyc311_transfer/*.tsv` | Local copies of cleaned TSV data |
| Binary model | `models/spark_model_binary/` | Saved binary Logistic Regression pipeline |
| Multiclass model | `models/spark_model_multiclass/` | Saved multiclass Random Forest pipeline |
| Model metrics | `models/dashboard_metrics.json` | Evaluation metrics and dashboard KPI metadata |
| Dashboard | `app.py`, `dashboard/` | Streamlit application and supporting files |

The cleaned TSV schema contains 29 fields. The Hive partition key `year` is stored as partition metadata and is not an additional field in each TSV record. See [`schema.md`](schema.md) for the complete schema.

## 4. Stage 1: Acquire and Ingest Data

Download the required year-wise CSV files from NYC Open Data and upload them to HDFS using the project's directory convention.

Expected layout:

```text
/nyc311/raw/full/2020/
/nyc311/raw/full/2021/
/nyc311/raw/full/2022/
/nyc311/raw/full/2023/
/nyc311/raw/full/2024/
/nyc311/raw/full/2025/
/nyc311/raw/full/2026/
```

Verify that the source files are available:

```bash
hdfs dfs -ls /nyc311/raw/full
hdfs dfs -ls /nyc311/raw/full/2020
```

Use the intended source snapshot and year range consistently across the processing stages.

## 5. Stage 2: Clean Data with Apache Pig

**Script:** `pig/clean_311_final.pig`

The Pig script processes the raw files for a specified year and writes cleaned records to the corresponding HDFS output directory.

```text
Input:  /nyc311/raw/full/$YEAR/*.csv
Output: /nyc311/cleaned_final/$YEAR
```

### Processing steps

1. Remove CSV header rows.
2. Validate required identifiers, timestamps, agency, and complaint category.
3. Deduplicate requests using `unique_key`.
4. Parse supported timestamp formats.
5. Validate that closure time is not earlier than creation time.
6. Calculate `resolution_hours`.
7. Retain durations from 2 minutes through 180 days, inclusive.
8. Apply the configured NYC geographic bounds to latitude and longitude.
9. Assign the `FAST`, `MODERATE`, or `SLOW` resolution category.
10. Write a headerless, tab-separated output with 29 fields.

Run the script for each year with available raw input:

```bash
pig -x mapreduce -param YEAR=2020 pig/clean_311_final.pig
```

Repeat the command for each required year by changing the `YEAR` parameter. Confirm that the cleaned output was generated:

```bash
hdfs dfs -ls /nyc311/cleaned_final/2020
hdfs dfs -du -h /nyc311/cleaned_final/2020
```

## 6. Stage 3: Create the Hive Table and Partitions

Hive provides SQL access to the cleaned data stored in HDFS.

**Scripts:**
- `hive/create_tables.hql`
- `hive/add_partitions.hql`

The table contract is:

```text
Database:     nyc311
Table:        service_requests
Location:     /nyc311/cleaned_final
Storage:      Tab-delimited text
Partition key: year INT
```

Run the table and partition scripts through Beeline:

```bash
beeline -u 'jdbc:hive2://localhost:10000/default' \
  -f hive/create_tables.hql

beeline -u 'jdbc:hive2://localhost:10000/default' \
  -f hive/add_partitions.hql
```

The partition script registers the configured year partitions. Confirm the table and partition coverage:

```sql
USE nyc311;
SHOW PARTITIONS service_requests;

SELECT year, COUNT(*)
FROM service_requests
GROUP BY year
ORDER BY year;
```

## 7. Stage 4: Run Hive Analytics

**Scripts:**
- `hive/analytics.hql`
- `hive/create_dashboard_data.hql`

The analytics queries summarize request volume and resolution performance across several dimensions:

- Yearly request volume
- Resolution-category distribution
- Average resolution duration by year
- Borough-level performance
- Agency-level performance
- Most frequent complaint categories
- Complaint category by resolution category
- Monthly and hourly request patterns
- Overall resolution-time statistics

Run the analytical queries:

```bash
beeline -u 'jdbc:hive2://localhost:10000/default' \
  -f hive/analytics.hql
```

### Generate dashboard aggregates

The dashboard-export script writes aggregate datasets under `/nyc311/dashboard/` using Hive `INSERT OVERWRITE DIRECTORY`.

| Output directory | Contents |
|---|---|
| `yearly_requests` | Request count by year |
| `resolution_distribution` | Count and percentage by resolution category |
| `yearly_resolution` | Average resolution hours by year |
| `borough_performance` | Request volume, average duration, and resolution-category percentages by borough |
| `top_complaints` | Top 15 complaint categories by volume |
| `complaint_resolution` | Request count by complaint category and resolution category |
| `agency_performance` | Request volume, average duration, and resolution-category percentages by agency |
| `monthly_requests` | Request count by year and month |
| `hourly_requests` | Request count by hour of day |
| `resolution_stats` | Average, median, 90th-percentile, and 95th-percentile resolution hours |

Generate the aggregate outputs:

```bash
beeline -u 'jdbc:hive2://localhost:10000/default' \
  -f hive/create_dashboard_data.hql
```

These aggregates provide compact datasets for the dashboard without requiring every visualization to process the full row-level dataset.

## 8. Stage 5: Prepare Spark Input Data

The Spark training workflow reads local files matching:

```text
data/nyc311_transfer/*.tsv
```

Transfer the cleaned TSV files from HDFS into this directory. Preserve the 29-field order and tab-delimited format.

Before training, ensure that:

1. Each input file has no header row.
2. Fields follow the cleaned schema defined in [`schema.md`](schema.md).
3. `latitude`, `longitude`, and `resolution_hours` are represented as numeric values.
4. The resolution category uses the labels `FAST`, `MODERATE`, and `SLOW`.
5. The local data covers the intended years for the experiment.

The helper `spark/src/load_data.py` provides a single-year data-loading check. The multi-year training input is supplied through the configured local TSV transfer directory.

## 9. Stage 6: Train and Evaluate Spark MLlib Models

**Script:** `spark/src/train_models.py`

The training script loads the local TSV dataset, applies its input filters, builds the feature vector, trains both classifiers, evaluates their performance, and saves the resulting pipelines and evaluation metadata.

### Feature engineering

| Feature | Transformation |
|---|---|
| Agency | String indexing and one-hot encoding |
| Complaint category | Keep the top 30 categories, map the remaining categories to `OTHER`, then index and one-hot encode |
| Borough | String indexing and one-hot encoding |
| Hour | Derived from `created_date` |
| Day of week | Derived from `created_date` |
| Month | Derived from `created_date` |
| Weekend | Binary feature derived from day of week |
| Latitude | Numeric feature |
| Longitude | Numeric feature |

### Classification tasks

| Task | Model | Target | Main metric |
|---|---|---|---|
| Binary classification | Logistic Regression | `SLOW` versus `FAST`/`MODERATE` | ROC-AUC |
| Multiclass classification | Random Forest with 20 trees and maximum depth 8 | `FAST`, `MODERATE`, or `SLOW` | Weighted F1 |

The current training configuration samples 5% of the available input using seed `42` and splits the sample into 80% training and 20% testing using seed `42`.

The evaluation workflow also calculates supporting classification metrics, including accuracy, precision, and recall where applicable.

Run training from the repository root:

```bash
python spark/src/train_models.py
```

The workflow saves the trained pipelines to:

```text
models/spark_model_binary/
models/spark_model_multiclass/
```

Evaluation and provenance information is written to:

```text
models/dashboard_metrics.json
```

The models classify resolution-duration categories derived from historical closure outcomes. Their evaluation metrics describe performance on the configured test split.

## 10. Stage 7: Generate Dashboard KPIs and Model Metadata

**Script:** `spark/src/export_dashboard_metrics.py`

This script computes dashboard-oriented KPIs from the local cleaned TSV inputs and writes them to `models/dashboard_metrics.json`.

The metrics include:
- Total processed records
- Resolution-category percentages
- Median and 75th-percentile resolution duration
- Borough and agency slow-resolution rates
- High-volume complaint summaries
- Model evaluation and provenance metadata used by the dashboard

Run the exporter from the repository root:

```bash
python spark/src/export_dashboard_metrics.py
```

Run model training before exporting the final dashboard metrics so the metadata can include the latest evaluation results.

## 11. Stage 8: Publish Dashboard Data and Launch Streamlit

The dashboard entry point is `app.py`, with supporting pages, components, and utilities in `dashboard/`.

Before launch:
1. Export or copy the Hive aggregate outputs into the local dashboard data paths.
2. Ensure `models/dashboard_metrics.json` contains the KPI and evaluation information for the current pipeline run.
3. Ensure the saved model directories are available for any prediction functionality.
4. Verify that the dashboard data files use the expected columns and formats.

Launch the application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://localhost:8501
```

The dashboard presents the generated analytics, resolution trends, borough and agency comparisons, complaint-category insights, and model evaluation information through an interactive interface.

## 12. Validation and Reproduction Checklist

Use the following checklist to verify a complete run.

### HDFS and Pig
- [ ] Raw files are available for all intended years.
- [ ] Pig completes successfully for each year.
- [ ] Cleaned output exists for each processed year.
- [ ] Cleaned rows follow the documented 29-field TSV schema.
- [ ] Resolution durations and categories follow the documented rules.

### Hive
- [ ] The `nyc311.service_requests` table exists.
- [ ] Partitions match the processed year range.
- [ ] Counts by year are available.
- [ ] Analytics queries execute successfully.
- [ ] All expected dashboard aggregate directories are generated.

### Spark
- [ ] Local TSV files follow the cleaned schema.
- [ ] Both classification pipelines train successfully.
- [ ] Model artifacts are saved.
- [ ] Evaluation metrics and metadata are generated.

### Dashboard
- [ ] Hive aggregates have been exported to local dashboard paths.
- [ ] KPI metadata is available.
- [ ] Dashboard pages load successfully.
- [ ] Visualizations and model-related views display the generated outputs.

Run the automated tests from the repository root:

```bash
pytest -q
```

## 13. End-to-End Execution Order

For a reproducible run, execute the stages in this order:

1. Download the required NYC 311 CSV files and upload them to HDFS.
2. Run `pig/clean_311_final.pig` for each input year.
3. Create the Hive table and register the processed year partitions.
4. Run `hive/analytics.hql`.
5. Generate aggregate outputs with `hive/create_dashboard_data.hql`.
6. Transfer cleaned TSV files from HDFS to `data/nyc311_transfer/`.
7. Train and evaluate the models with `spark/src/train_models.py`.
8. Generate dashboard KPIs and metadata with `spark/src/export_dashboard_metrics.py`.
9. Copy or export the Hive aggregates to the local dashboard data paths.
10. Run `pytest -q`.
11. Launch the application with `streamlit run app.py`.

## 14. Implementation Reference

| File | Responsibility |
|---|---|
| `pig/clean_311_final.pig` | Cleans raw records and writes the 29-field TSV dataset |
| `hive/create_tables.hql` | Creates the external Hive table |
| `hive/add_partitions.hql` | Registers year partitions |
| `hive/analytics.hql` | Defines descriptive Hive analytics |
| `hive/create_dashboard_data.hql` | Produces dashboard aggregate datasets |
| `spark/src/load_data.py` | Provides a single-year Spark input-loading check |
| `spark/src/load_all_years.py` | Multi-year loading utility location |
| `spark/src/train_models.py` | Feature engineering, model training, evaluation, and model saving |
| `spark/src/export_dashboard_metrics.py` | Computes dashboard KPIs and exports metrics metadata |
| `app.py` | Streamlit application entry point |
| `dashboard/` | Dashboard pages, components, utilities, and local data |
| `docs/schema.md` | Detailed schema and data contracts |