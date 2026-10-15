# NYC 311 Big Data Analytics

A big data analytics project for analyzing NYC 311 service requests, understanding resolution patterns, comparing agency and borough performance, and predicting service-request resolution categories using distributed data processing and machine learning.

## 1. Project Overview

The project combines distributed data processing, SQL analytics, machine learning, and interactive visualization to analyze NYC 311 service requests at scale.

**Key objectives**
- Clean and validate large volumes of service-request data.
- Analyze request trends and resolution performance across years, boroughs, agencies, and complaint categories.
- Train machine learning models to classify resolution outcomes.
- Present analytical findings, model metrics, and predictions through an interactive dashboard.

## 2. Technology Stack

- **Hadoop HDFS:** Distributed data storage
- **Apache Pig:** Data cleaning and preprocessing
- **Apache Hive:** SQL-based analytics and aggregations
- **Apache Spark MLlib:** Model training and evaluation
- **Streamlit:** Interactive dashboard
- **Python:** Pipeline utilities, model workflows, validation, and visualization

## 3. Architecture

```text
NYC 311 Raw Data
       |
       v
Hadoop HDFS
       |
       v
Apache Pig
(Data Cleaning and Validation)
       |
       v
Cleaned Data in HDFS
       |
       v
Apache Hive
(Analytics and Aggregations)
       |
       +----------------------+
       |                      |
       v                      v
Dashboard Aggregates    Spark Input Data
                              |
                              v
                        Spark MLlib
                    (Training and Evaluation)
                              |
                              v
                     Models and Metrics
       |                      |
       +-----------+----------+
                   |
                   v
           Streamlit Dashboard
```

The dashboard consumes generated aggregate datasets and model artifacts. Follow the actual scripts and configuration in the repository for the precise data-transfer and execution steps.

## 4. Repository Structure

```text
.
├── app.py
├── dashboard/
│   ├── views/
│   ├── components/
│   ├── utils/
│   └── data/
├── data/
├── hive/
│   ├── create_tables.hql
│   ├── add_partitions.hql
│   ├── analytics.hql
│   └── create_dashboard_data.hql
├── pig/
│   └── clean_311_final.pig
├── spark/
│   └── src/
│       ├── load_data.py
│       ├── load_all_years.py
│       ├── train_models.py
│       ├── export_dashboard_metrics.py
│       └── audit_data_quality.py
├── models/
├── tests/
├── requirements.txt
└── docker-compose.yml
```

- `pig/`: Data-cleaning scripts.
- `hive/`: Table definitions, partition management, and analytics queries.
- `spark/src/`: Data-loading utilities, model training, evaluation, and metric export.
- `models/`: Saved model artifacts and generated metrics.
- `dashboard/`: Streamlit pages, components, utilities, and aggregate data.
- `tests/`: Automated checks for dashboard data and related functionality.
- `docker-compose.yml`: Container configuration for the services defined in the Compose file.

## 5. Prerequisites

Install and configure the software required by the pipeline:

- Java version compatible with your Hadoop and Hive installations
- Hadoop and HDFS
- Apache Pig
- Apache Hive and HiveServer2
- Apache Spark with PySpark MLlib
- Python and pip
- Docker and Docker Compose, if using the provided container configuration

Check the versions and environment variables used by your installation before running the pipeline. The required services must be running and accessible.

Install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If Spark dependencies are not included in `requirements.txt`, install or configure PySpark separately according to the project's Spark environment.

## 6. Data Preparation

The project uses NYC 311 Service Requests data.

1. Obtain the required source dataset from the NYC Open Data portal.
2. Organize the raw files according to the directory and naming conventions expected by the ingestion scripts.
3. Place or upload the raw files into the configured HDFS input location.
4. Verify that the files are accessible before running Pig.

The Pig cleaning script expects raw data under the configured year-wise HDFS paths. Confirm the paths in the script and configuration before execution.

**Important:** The dashboard and model results depend on the data actually processed. Record the source data range and use consistent inputs when reproducing the reported results.

## 7. Reproducing the Pipeline

Run the stages in dependency order. Use the repository's orchestration script if one is provided and verified; otherwise, execute the corresponding scripts individually.

### Step 1: Start the required services

Start Hadoop, HDFS, HiveServer2, and any other services required by your environment.

Verify HDFS availability:

```bash
hdfs dfs -ls /
```

### Step 2: Ingest and clean the data

Upload the raw data to the configured HDFS location and execute the Pig cleaning pipeline:

```bash
pig -x mapreduce pig/clean_311_final.pig
```

Confirm that the cleaned output exists at the destination configured in the script:

```bash
hdfs dfs -ls /nyc311/cleaned_final
```

Adjust the command and paths if the script is configured differently.

### Step 3: Initialize Hive and partitions

Execute the Hive table and partition scripts in the required order:

```bash
beeline -u 'jdbc:hive2://localhost:10000/default' \
  -f hive/create_tables.hql

beeline -u 'jdbc:hive2://localhost:10000/default' \
  -f hive/add_partitions.hql
```

Use the correct HiveServer2 URL and database for your environment. Verify that the table and expected year partitions exist before proceeding.

### Step 4: Run Hive analytics

Execute the analytics and dashboard-export queries:

```bash
beeline -u 'jdbc:hive2://localhost:10000/default' \
  -f hive/analytics.hql

beeline -u 'jdbc:hive2://localhost:10000/default' \
  -f hive/create_dashboard_data.hql
```

Verify the output directories and ensure the generated aggregates are available to the dashboard.

### Step 5: Prepare Spark inputs and train the models

Check the input paths configured in the Spark scripts. The model-training workflow expects processed TSV data in the configured local transfer directory.

Prepare or transfer the cleaned data as required by the existing pipeline, then run:

```bash
python spark/src/train_models.py
```

The training workflow builds binary and multiclass classification pipelines, evaluates model performance, and saves model artifacts. Run the metric-export script only after the required evaluation outputs have been generated:

```bash
python spark/src/export_dashboard_metrics.py
```

Confirm that the saved models and generated metrics correspond to the latest successful training and evaluation run.

## 8. Launch the Dashboard

After the required aggregate datasets and model artifacts have been generated and validated, start Streamlit:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

The dashboard provides:
- Request volume and temporal trends
- Resolution-time analysis
- Borough and agency comparisons
- Complaint-category analysis
- Model evaluation metrics and prediction insights
- Data coverage and pipeline information

The dashboard is intended to consume generated outputs. Starting it should not require retraining models or rerunning the full data pipeline.

## 9. Machine Learning

The project includes two Spark MLlib classification tasks:

**Binary classification:** Classifies requests according to the existing slow-resolution target definition.

**Multiclass classification:** Predicts the project's resolution categories.

The workflow uses feature transformation pipelines, model training, and evaluation metrics appropriate to each task. The dashboard should display metrics generated from actual evaluation predictions and predictions produced by the saved model artifacts.

These are historical resolution-outcome classification tasks. Their performance should not be interpreted as proof of prospective resolution-time prediction without appropriate temporal evaluation.

## 10. Validation and Testing

Run the available automated tests:

```bash
pytest -q
```

Before treating a run as reproducible, verify:

- HDFS contains the expected raw and cleaned data.
- Hive tables and partitions match the processed dataset.
- Hive queries execute successfully and produce the expected outputs.
- Exported aggregate totals agree with the corresponding query results.
- Spark model artifacts and evaluation metrics are generated successfully.
- The dashboard loads the correct outputs and handles missing or stale artifacts appropriately.

For a full reproduction, test with a manageable sample dataset first, then repeat with the intended full dataset.

## 11. Reproducibility Notes

- Follow the actual paths and configuration values in the scripts.
- Use consistent source data, filters, and partitions when comparing results.
- Distinguish complete years from partial-year coverage.
- Do not treat existing generated files as proof that the pipeline has just executed successfully.
- Record the input dataset, execution date, model version, and evaluation results for each run.
- If a required service or stage fails, resolve the issue before treating downstream outputs as current.

## 12. Data Source

**NYC Open Data – 311 Service Requests**

Dataset: [NYC 311 Service Requests](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9)

The available time range depends on the source data downloaded and processed for a particular run.

## 13. Project Outcome

This project demonstrates an end-to-end big data analytics workflow combining distributed storage, large-scale preprocessing, SQL analytics, machine learning, and interactive visualization to investigate NYC municipal service-request resolution patterns.