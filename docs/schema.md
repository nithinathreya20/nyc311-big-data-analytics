# NYC 311 Data Schema and Data Contracts

This document defines the data structures used by the NYC 311 Big Data Analytics project: the raw source, Pig-cleaned records, Hive table and partitions, Spark MLlib inputs and targets, and Hive-generated dashboard aggregates.

## 1. Scope and Source

- **Source:** NYC Open Data, *311 Service Requests from 2020 to Present*.
- **Dataset ID:** `erm2-nwe9`
- **API:** <https://data.cityofnewyork.us/resource/erm2-nwe9>
- **Processing years in project scope:** 2020–2026.
- **Primary storage:** HDFS.
- **Cleaned-data format:** tab-separated values (TSV), without a header row.
- **Record grain:** one deduplicated service request (`unique_key`) that passes the cleaning rules.

The cleaned dataset is not a complete representation of every request submitted. The current Pig script requires both `created_date` and `closed_date`, so open requests are excluded. The resulting analysis describes retained, closed requests and their observed resolution durations, not the current open backlog.

## 2. End-to-End Data Flow

```text
NYC Open Data (raw CSV)
        |
        v
HDFS: /nyc311/raw/full/$YEAR/*.csv
        |
        v
Apache Pig: pig/clean_311_final.pig
  - removes header rows
  - validates required fields and timestamps
  - deduplicates by unique_key
  - calculates resolution_hours
  - filters temporal and spatial outliers
  - derives resolution_category
        |
        v
HDFS: /nyc311/cleaned_final/$YEAR
        |
        +-------------------------------+
        |                               |
        v                               v
Apache Hive                      Local Spark input
nyc311.service_requests          data/nyc311_transfer/*.tsv
        |                               |
        v                               v
Hive analytics and exports       Spark MLlib pipelines
        |                               |
        v                               v
HDFS dashboard aggregates        Saved models and metrics
        |                               |
        +---------------+---------------+
                        v
              Local dashboard data
                        |
                        v
                Streamlit dashboard
```

Hive dashboard aggregates are written under `/nyc311/dashboard/`. The Streamlit application reads generated local dashboard files and model artifacts; it does not query Hive live on each page render. The Spark training script reads the local TSV transfer directory, so cleaned HDFS data must be transferred or exported to that location before training.

## 3. Raw Source Schema

The NYC Open Data CSV contains more columns than the cleaned project dataset. The Pig script reads the source fields in this order:

| # | Raw field | Meaning |
|---:|---|---|
| 1 | `unique_key` | Service-request identifier |
| 2 | `created_date` | Request creation timestamp |
| 3 | `closed_date` | Request closure timestamp |
| 4 | `agency` | Agency code |
| 5 | `agency_name` | Agency name |
| 6 | `complaint_type` | Complaint category |
| 7 | `descriptor` | Complaint descriptor |
| 8 | `descriptor_2` | Secondary descriptor |
| 9 | `location_type` | Location type |
| 10 | `incident_zip` | Incident ZIP code |
| 11 | `incident_address` | Incident address |
| 12 | `street_name` | Street name |
| 13 | `cross_street_1` | First cross street |
| 14 | `cross_street_2` | Second cross street |
| 15 | `intersection_street_1` | First intersection street |
| 16 | `intersection_street_2` | Second intersection street |
| 17 | `address_type` | Address type |
| 18 | `city` | City |
| 19 | `landmark` | Landmark |
| 20 | `facility_type` | Facility type |
| 21 | `status` | Request status in the source record |
| 22 | `due_date` | Source due date |
| 23 | `resolution_description` | Source resolution description |
| 24 | `resolution_action_updated_date` | Resolution-action update timestamp |
| 25 | `community_board` | Community board |
| 26 | `council_district` | Council district |
| 27 | `police_precinct` | Police precinct |
| 28 | `bbl` | Borough-block-lot identifier |
| 29 | `borough` | Borough |
| 30 | `x_coordinate_state_plane` | State-plane X coordinate |
| 31 | `y_coordinate_state_plane` | State-plane Y coordinate |
| 32 | `open_data_channel_type` | Channel used to submit the request |
| 33 | `park_facility_name` | Park or facility name |
| 34 | `park_borough` | Borough associated with a park |
| 35 | `vehicle_type` | Vehicle type, where applicable |
| 36 | `taxi_company_borough` | Taxi-company borough |
| 37 | `taxi_pick_up_location` | Taxi pickup location |
| 38 | `bridge_highway_name` | Bridge or highway name |
| 39 | `bridge_highway_direction` | Bridge or highway direction |
| 40 | `road_ramp` | Road-ramp information |
| 41 | `bridge_highway_segment` | Bridge or highway segment |
| 42 | `latitude` | Geographic latitude |
| 43 | `longitude` | Geographic longitude |
| 44 | `location` | Source location representation |

The raw schema above reflects the field order declared in `pig/clean_311_final.pig`. Confirm the source CSV version if NYC Open Data changes its schema.

## 4. Cleaned Record Schema

The Pig output contains **29 fields**, in the exact order below. It is stored as TSV without a header row.

| # | Field | Logical type | Description |
|---:|---|---|---|
| 1 | `unique_key` | String | Request identifier; used for deduplication |
| 2 | `created_date` | String | Original request creation timestamp |
| 3 | `closed_date` | String | Original request closure timestamp |
| 4 | `agency` | String | Responsible agency code |
| 5 | `agency_name` | String | Responsible agency name |
| 6 | `complaint_type` | String | Complaint category |
| 7 | `descriptor` | String | Complaint descriptor |
| 8 | `descriptor_2` | String | Secondary complaint descriptor |
| 9 | `location_type` | String | Location type |
| 10 | `incident_zip` | String | Incident ZIP code |
| 11 | `incident_address` | String | Incident address |
| 12 | `street_name` | String | Street name |
| 13 | `cross_street_1` | String | First cross street |
| 14 | `cross_street_2` | String | Second cross street |
| 15 | `intersection_street_1` | String | First intersection street |
| 16 | `intersection_street_2` | String | Second intersection street |
| 17 | `address_type` | String | Address type |
| 18 | `city` | String | City |
| 19 | `landmark` | String | Landmark |
| 20 | `facility_type` | String | Facility type |
| 21 | `status` | String | Status copied from the source record |
| 22 | `community_board` | String | Community board |
| 23 | `council_district` | String | Council district |
| 24 | `police_precinct` | String | Police precinct |
| 25 | `borough` | String | Borough |
| 26 | `latitude` | Double | Geographic latitude |
| 27 | `longitude` | Double | Geographic longitude |
| 28 | `resolution_hours` | Double | Calculated elapsed time between creation and closure, in hours |
| 29 | `resolution_category` | String | Derived resolution-duration class: `FAST`, `MODERATE`, or `SLOW` |

### Fields intentionally omitted from the cleaned output

The raw source fields `due_date`, `resolution_description`, `resolution_action_updated_date`, `bbl`, `x_coordinate_state_plane`, `y_coordinate_state_plane`, `open_data_channel_type`, `park_facility_name`, `park_borough`, `vehicle_type`, `taxi_company_borough`, `taxi_pick_up_location`, `bridge_highway_name`, `bridge_highway_direction`, `road_ramp`, `bridge_highway_segment`, and `location` are not included in the 29-field output. Some raw fields are also used only for parsing or filtering.

## 5. Hive Table Contract

The cleaned data is registered as an external Hive table:

```text
Database: nyc311
Table:    service_requests
Location: /nyc311/cleaned_final
Format:   TextFile, tab-delimited
```

The Hive table contains the same 29 data columns and adds a **partition column**:

| Field | Hive type | Role |
|---|---|---|
| `year` | `INT` | Partition key representing the data year |

The `year` partition is **not** one of the 29 fields in each TSV row. Hive partition metadata supplies it based on the registered partition location.

Expected year partition locations:

```text
/nyc311/cleaned_final/2020
/nyc311/cleaned_final/2021
/nyc311/cleaned_final/2022
/nyc311/cleaned_final/2023
/nyc311/cleaned_final/2024
/nyc311/cleaned_final/2025
/nyc311/cleaned_final/2026
```

The actual set of populated partitions depends on which years have been processed and registered.

## 6. Cleaning and Validation Rules

The current Pig script applies these rules before writing the cleaned dataset:

1. **Header removal:** removes rows whose `unique_key` equals the header label.
2. **Required fields:** requires non-empty `unique_key`, `created_date`, `closed_date`, and `agency`, and a non-null `complaint_type`.
3. **Deduplication:** groups by `unique_key` and retains one record from each group.
4. **Timestamp parsing:** supports the timestamp patterns configured in the script, including ISO-like timestamps and legacy `MM/dd/yyyy hh:mm:ss a` values.
5. **Chronology:** requires parseable timestamps and `closed_date >= created_date`.
6. **Duration boundary:** retains durations from 2 minutes through 180 days, inclusive.
7. **Coordinates:** requires coordinates inside the project's configured NYC bounding box:
   - Latitude: `40.47` to `40.92`
   - Longitude: `-74.26` to `-73.70`
8. **TSV safety:** replaces tab characters in selected text fields before writing.
9. **Target derivation:** assigns a resolution category using the duration thresholds below.

These are project-specific operational filters, not a guarantee that every retained record is error-free or that the bounding box represents a definitive administrative boundary.

## 7. Derived Resolution Fields

### `resolution_hours`

Calculated from the difference between closure and creation timestamps:

```text
resolution_hours = (closed_date - created_date) in hours
```

Only records with valid chronological timestamps and a duration inside the configured boundary are retained.

### `resolution_category`

| Value | Rule |
|---|---|
| `FAST` | `resolution_hours < 24` |
| `MODERATE` | `24 <= resolution_hours <= 72` |
| `SLOW` | `resolution_hours > 72` |

The category is derived from the **observed historical closure duration**. It is not a source-provided field.

## 8. Spark MLlib Input and Feature Contract

### Input

The training script reads:

```text
data/nyc311_transfer/*.tsv
```

It declares the same 29-column schema as the cleaned TSV, with `latitude`, `longitude`, and `resolution_hours` read as doubles and the remaining columns read as strings.

Before feature engineering, the current training script filters out rows with a null `borough`, `borough = 'Unspecified'`, or a `resolution_category` outside the three defined classes.

### Features used by the current training script

| Feature | Construction |
|---|---|
| `agency_vec` | `agency` indexed with `StringIndexer`, then one-hot encoded |
| `complaint_vec` | Top 30 complaint types retained as categories; all others mapped to `OTHER`, then indexed and one-hot encoded |
| `borough_vec` | `borough` indexed and one-hot encoded |
| `hour` | Hour of day derived from `created_date` |
| `day_of_week` | Day of week derived from `created_date` |
| `month` | Month derived from `created_date` |
| `is_weekend` | Binary indicator derived from day of week |
| `latitude` | Numeric latitude |
| `longitude` | Numeric longitude |

The assembled Spark feature vector contains these encoded categorical and numeric features. The current code parses `created_date` using the configured ISO timestamp pattern; timestamp parsing should be checked when input data contains a different supported format.

### Targets and models

| Task | Target | Model | Main evaluation metric |
|---|---|---|---|
| Binary classification | `1` if category is `SLOW`, otherwise `0` | Spark MLlib Logistic Regression | ROC-AUC |
| Multiclass classification | `resolution_category` indexed to numeric labels | Spark MLlib Random Forest | Weighted F1 |

The current training script samples 5% of the available input with seed `42`, then uses an 80/20 random train/test split with seed `42`. This is a random split, not a chronological or stratified split.

### Leakage and interpretation

The feature vector does **not** include `closed_date`, `resolution_hours`, or `resolution_category`. The category is used as a target, not as an input feature. This avoids direct target leakage from those fields into the feature vector.

However, the labels are computed from actual closure times, and the training data consists of closed historical requests. The model should therefore be described as classifying the eventual resolution-duration category from selected request attributes. Evaluation on a random holdout does not by itself establish performance on future periods or prove prospective performance on open requests. A time-based holdout is needed to evaluate that claim.

## 9. Hive Dashboard Aggregate Contracts

`hive/create_dashboard_data.hql` writes comma-delimited aggregate files under `/nyc311/dashboard/`. These are **derived aggregate datasets**, not row-level request records.

| Output directory | Grain | Columns written, in order |
|---|---|---|
| `yearly_requests` | One row per year | `request_year`, `total_requests` |
| `resolution_distribution` | One row per resolution category | `resolution_category`, `total_requests`, `percentage` |
| `yearly_resolution` | One row per year | `request_year`, `avg_resolution_hours` |
| `borough_performance` | One row per borough | `borough`, `total_requests`, `avg_resolution_hours`, `fast_percentage`, `moderate_percentage`, `slow_percentage` |
| `top_complaints` | One row per selected complaint type, limited to 15 | `complaint_type`, `total_requests` |
| `complaint_resolution` | One row per complaint type and resolution category | `complaint_type`, `resolution_category`, `total_requests` |
| `agency_performance` | One row per agency | `agency`, `total_requests`, `avg_resolution_hours`, `fast_percentage`, `moderate_percentage`, `slow_percentage` |
| `monthly_requests` | One row per year and month | `request_year`, `request_month`, `total_requests` |
| `hourly_requests` | One row per hour of day | `request_hour`, `total_requests` |
| `resolution_stats` | One overall summary row | `average_hours`, `median_hours`, `p90_hours`, `p95_hours` |

These files are written using `INSERT OVERWRITE DIRECTORY`; each run replaces the corresponding output directory's contents. The local dashboard data files must be exported or copied from these HDFS outputs using the project's configured process before the dashboard can consume them.

## 10. Model Artifacts and Metrics

The current training and metric-export workflow uses these paths:

```text
models/spark_model_binary/
models/spark_model_multiclass/
models/dashboard_metrics.json
```

- `spark_model_binary/`: saved binary Logistic Regression pipeline.
- `spark_model_multiclass/`: saved multiclass Random Forest pipeline.
- `dashboard_metrics.json`: model-evaluation metadata, KPI values, feature/provenance information, and evaluation metrics consumed by the dashboard.

Metrics should represent the output of the latest successful model evaluation. Existing JSON values or model directories alone do not prove that the full pipeline has just run or that all dashboard aggregates were generated from the same dataset version.

## 11. Schema Invariants

The following checks should pass before downstream processing:

- Every cleaned TSV row has exactly 29 fields in the documented order.
- The TSV has no header row.
- `unique_key` is non-empty and unique after deduplication.
- `created_date` and `closed_date` parse under the formats supported by the pipeline.
- `closed_date` is not earlier than `created_date`.
- `resolution_hours` is within the configured inclusive range of 2 minutes to 180 days.
- `resolution_category` agrees with the documented duration thresholds.
- `latitude` and `longitude` fall within the configured geographic bounds.
- Hive column order matches the TSV order, and `year` is configured as a partition column rather than a row field.
- Spark reads the same 29-column order and numeric coordinate types.
- Dashboard aggregate counts reconcile with the relevant Hive query results for the same data version.

## 12. Source of Truth

Implementation details are defined by these repository files:

- `pig/clean_311_final.pig` — raw field order, cleaning rules, derived fields, and cleaned output order.
- `hive/create_tables.hql` — Hive table columns, types, storage format, location, and partition key.
- `hive/add_partitions.hql` — year partition registration.
- `hive/analytics.hql` — analytical query definitions.
- `hive/create_dashboard_data.hql` — dashboard aggregate output schemas.
- `spark/src/train_models.py` — Spark input schema, feature engineering, targets, train/test split, models, and evaluation.
- `spark/src/export_dashboard_metrics.py` — model-metric/provenance export.

If this document and an implementation file disagree, inspect the implementation and update the schema documentation with the same change.