# COEN542_GROUP2
# Scalable Big Data Pipeline for Network Intrusion Detection Using the CICIDS2017 Dataset

## Project Overview

This project implements an end-to-end Big Data pipeline for network intrusion detection using the CICIDS2017 dataset.

### Implemented Architecture

CICIDS2017 → Python Producer → Apache Kafka → Spark Structured Streaming → Parquet / HDFS → Apache Spark → Spark MLlib Random Forest → Evaluation & Scalability → Streamlit Dashboard

## Technologies

| Technology                 | Role |
| Python                     | Data ingestion and pipeline scripts |
| Apache Kafka               | Streaming ingestion and decoupling |
| Apache HDFS                | Scalable storage |
| Apache Spark               | Distributed processing |
| Spark Structured Streaming | Kafka stream processing |
| Spark MLlib                | Random Forest classification |
| Parquet                    | Columnar data storage |
| Streamlit                  | Interactive dashboard |
| Docker Compose             | Kafka deployment |

## Dataset and ML Task

The pipeline processes **2,830,743 network-flow records** with **78 ML features**.

Binary classification uses:

- `BENIGN` → `0.0`
- All attack labels → `1.0`

## Repository Structure

```text
COEN542_GitHub/
├── dashboard/
│   └── app.py
├── ingestion/
│   ├── kafka_consumer_test.py
│   └── kafka_producer.py
├── processing/
│   ├── 01_clean_unify.py
│   ├── 02_prepare_ml.py
│   ├── 03_train_model.py
│   ├── 04_scalability_experiment.py
│   └── spark_streaming_consumer.py
├── results/
│   ├── confusion_matrix.csv
│   ├── feature_importance.csv
│   ├── metrics.csv
│   └── scalability_results.csv
├── models/
│   └── random_forest_binary/
├── docker-compose.yml
└── requirements.txt
```

## Pipeline Stages

### 1. Cleaning and Unification

`processing/01_clean_unify.py` prepares the CICIDS2017 flow data into a unified dataset.

### 2. Kafka Ingestion

`ingestion/kafka_producer.py` replays network-flow records to the Kafka topic:

```text
network-flows
```

The configured topic uses 3 partitions.

### 3. Spark Structured Streaming

`processing/spark_streaming_consumer.py` consumes Kafka data with Spark Structured Streaming and writes network-flow data to HDFS storage.

### 4. ML Preparation

`processing/02_prepare_ml.py` prepares the data for binary classification, including label conversion and feature preparation.

### 5. Random Forest

`processing/03_train_model.py` trains a Spark MLlib Random Forest classifier.

Configuration:

- 50 trees
- Maximum depth: 12
- 80/20 train/test split
- Random seed: 42
- 78 input features

### 6. Evaluation

Recorded results:

| Metric             | Value  |
| Accuracy           | 0.9979 |
| Weighted Precision | 0.9979 |
| Weighted Recall    | 0.9979 |
| Weighted F1        | 0.9979 |

Additional outputs:

- Confusion matrix: `results/confusion_matrix.csv`
- Feature importance: `results/feature_importance.csv`
- Metrics: `results/metrics.csv`

### 7. Scalability Experiment

The Random Forest training workload was evaluated at 25%, 50%, and 100% of the prepared dataset.

| Dataset size | Rows      | Training time |
| 25%          | 708,804   | 267.47 s |
| 50%          | 1,416,895 | 359.25 s |
| 100%         | 2,830,743 | 774.67 s |

Results are stored in `results/scalability_results.csv`.

The larger workload caused greater memory pressure and disk spilling, so runtime increased non-linearly as the dataset size increased.

## HDFS

The repository includes the implemented HDFS configuration in:

```text
config/hdfs/
├── core-site.xml
├── hdfs-site.xml
└── README.md
```

The configuration uses Hadoop 3.4.1, `hdfs://localhost:9000` as the default
filesystem, and replication factor 1 for the single-node development setup.

HDFS project directories:

```text
/coen542/raw
/coen542/processed
/coen542/models
```

Processed ML dataset:

```text
/coen542/processed/ml_binary
```

The dataset contains 2,830,743 rows and 79 columns (78 features plus
`binary_label`).

## Why HDFS Data Is Not Committed

The actual NameNode/DataNode storage directories are not included in GitHub.
They are HDFS-managed storage and can be large and machine-specific.

GitHub contains the reproducible configuration and setup instructions instead.

See `config/hdfs/README.md` for complete setup and verification commands.


## Dashboard

The interactive dashboard is implemented in:

```text
dashboard/app.py
```

It presents:

- Dataset summary
- Label distribution
- Benign vs attack distribution
- Attack-type distribution
- Random Forest configuration
- Model metrics
- Confusion matrix
- Feature importance
- Scalability results

Launch with:

```bash
streamlit run dashboard/app.py
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project also requires Apache Kafka, Apache Spark, Apache Hadoop/HDFS, Java, and Docker/Docker Compose for the Kafka setup.

## Running the Pipeline

### Prepare data

```bash
python3 processing/01_clean_unify.py
python3 processing/02_prepare_ml.py
```

### Start Kafka

```bash
docker compose up -d
```

Kafka is configured for:

```text
localhost:9092
```

### Start Spark Structured Streaming

```bash
python3 processing/spark_streaming_consumer.py --sink parquet
```

### Replay network flows

In another terminal:

```bash
python3 ingestion/kafka_producer.py --rate 400
```

The producer supports `--rate`, `--limit`, `--skip`, and `--topic`.

### Train the model

```bash
python3 processing/03_train_model.py
```

### Run scalability experiment

```bash
python3 processing/04_scalability_experiment.py
```

The repository contains the recorded scalability results from the completed experiment.

### Launch dashboard

```bash
streamlit run dashboard/app.py
```

## Reproducibility

The repository contains the main source code, dependency information, Kafka configuration, model artifacts, and recorded analytical results.

The large raw CICIDS2017 dataset and generated streaming/storage directories are not included in the GitHub repository. They should be obtained separately and placed in the expected project data locations before reproducing the full pipeline.

## Important Results

The completed implementation processed:

- 2,830,743 network flows
- 78 ML features
- Binary benign-vs-attack classification
- Random Forest with 50 trees
- Maximum tree depth of 12
- 80/20 train/test split

Recorded metrics:

- Accuracy: 0.9979
- Weighted Precision: 0.9979
- Weighted Recall: 0.9979
- Weighted F1: 0.9979

## Limitations

This repository documents the implemented project pipeline.

The reported ML experiment is a binary Random Forest classification experiment. Multiclass classification and anomaly-detection results are not included in the reported results.

The HDFS environment is a single-node deployment. It demonstrates HDFS storage and Spark integration rather than a multi-node production cluster.

## Project Deliverables

The repository supports the project's main implementation deliverables:

- Kafka-based ingestion
- HDFS storage
- Spark distributed processing
- Spark MLlib classification
- Scalability experiment
- Interactive Streamlit dashboard
- Source code and dependency information
- Recorded model and scalability results
