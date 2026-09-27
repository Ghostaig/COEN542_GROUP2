import csv

import pandas as pd
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    sum as spark_sum,
    when
)


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="COEN542 Cybersecurity Big Data Dashboard",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# PAGE TITLE
# =========================================================

st.title("🛡️ COEN542 Cybersecurity Big Data Dashboard")

st.write(
    "CICIDS2017 Intrusion Detection Big Data Pipeline"
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = "/mnt/bigdata/coen542-project"

DATA_PATH = (
    f"{PROJECT_ROOT}/"
    "data/streaming_store/network_flows"
)

MODEL_RESULTS_PATH = (
    f"{PROJECT_ROOT}/"
    "data/processed/model_results"
)

SCALABILITY_RESULTS_PATH = (
    f"{PROJECT_ROOT}/"
    "data/processed/scalability_results_clean.csv"
)


# =========================================================
# SPARK SESSION
# =========================================================

@st.cache_resource
def create_spark_session():

    spark = (
        SparkSession.builder
        .appName("COEN542-Dashboard")
        .master("local[4]")
        .config("spark.driver.memory", "2g")
        .config(
            "spark.local.dir",
            f"{PROJECT_ROOT}/spark-tmp"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


spark = create_spark_session()


# =========================================================
# LOAD PARQUET DATA
# =========================================================

@st.cache_resource
def load_data():

    return spark.read.parquet(DATA_PATH)


df = load_data()


# =========================================================
# SPARK DATASET SUMMARY
# =========================================================

@st.cache_data
def calculate_dataset_summary():

    summary = (
        df
        .agg(
            count("*").alias("total_flows"),

            spark_sum(
                when(
                    col("Label") == "BENIGN",
                    1
                ).otherwise(0)
            ).alias("benign_count"),

            spark_sum(
                when(
                    col("Label") != "BENIGN",
                    1
                ).otherwise(0)
            ).alias("attack_count")
        )
        .collect()[0]
    )

    attack_counts = (
        df
        .filter(col("Label") != "BENIGN")
        .groupBy("Label")
        .count()
        .orderBy(col("count").desc())
        .collect()
    )

    return summary, attack_counts


summary, attack_counts = calculate_dataset_summary()


# =========================================================
# EXTRACT DATASET VALUES
# =========================================================

total_flows = int(summary["total_flows"])

benign_count = int(summary["benign_count"])

attack_count = int(summary["attack_count"])


# Streaming output contains:
#
# 78 network features
# Label
# source_file
# _processing_ts
#
# Total = 81 columns
# Features = 81 - 3 = 78

feature_count = len(df.columns) - 3


benign_percentage = (
    benign_count / total_flows
) * 100


attack_percentage = (
    attack_count / total_flows
) * 100


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.header("📊 Dataset Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Flows",
        f"{total_flows:,}"
    )


with col2:

    st.metric(
        "Features",
        feature_count
    )


with col3:

    st.metric(
        "Benign Traffic",
        f"{benign_percentage:.2f}%"
    )


with col4:

    st.metric(
        "Attack Traffic",
        f"{attack_percentage:.2f}%"
    )


# =========================================================
# BENIGN VS ATTACK
# =========================================================

st.header("🚨 Benign vs Attack Traffic")


binary_df = pd.DataFrame(
    {
        "Traffic Type": [
            "Benign",
            "Attack"
        ],
        "Count": [
            benign_count,
            attack_count
        ]
    }
)


st.bar_chart(
    binary_df,
    x="Traffic Type",
    y="Count"
)


# =========================================================
# ATTACK TYPE DISTRIBUTION
# =========================================================

st.header("🔍 Attack Type Distribution")


attack_table = []


for row in attack_counts:

    attack_label = row["Label"]

    attack_count_value = int(row["count"])

    attack_percentage_value = (
        attack_count_value / attack_count
    ) * 100

    attack_table.append(
        {
            "Attack Type": attack_label,
            "Count": attack_count_value,
            "Percentage": round(
                attack_percentage_value,
                2
            )
        }
    )


attack_df = pd.DataFrame(
    attack_table
)


st.dataframe(
    attack_df,
    use_container_width=True,
    hide_index=True
)


st.bar_chart(
    attack_df,
    x="Attack Type",
    y="Count"
)


# =========================================================
# RANDOM FOREST RESULTS
# =========================================================

st.header(
    "🤖 Random Forest Intrusion Detection Results"
)


# =========================================================
# LOAD MODEL METRICS
# =========================================================

@st.cache_data
def load_model_metrics():

    metrics_file = (
        f"{MODEL_RESULTS_PATH}/metrics.csv"
    )

    metrics = {}

    with open(
        metrics_file,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            metrics[row["metric"]] = float(
                row["value"]
            )

    return metrics


metrics = load_model_metrics()


# =========================================================
# MODEL METRIC CARDS
# =========================================================

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )


with metric2:

    st.metric(
        "Weighted Precision",
        f"{metrics['weighted_precision'] * 100:.2f}%"
    )


with metric3:

    st.metric(
        "Weighted Recall",
        f"{metrics['weighted_recall'] * 100:.2f}%"
    )


with metric4:

    st.metric(
        "Weighted F1",
        f"{metrics['weighted_f1'] * 100:.2f}%"
    )


st.caption(
    "Precision, recall, and F1 are weighted metrics "
    "reported by the Spark MLlib evaluation."
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

st.subheader("⚙️ Random Forest Configuration")


config_col1, config_col2, config_col3, config_col4 = (
    st.columns(4)
)


with config_col1:

    st.metric(
        "Trees",
        "50"
    )


with config_col2:

    st.metric(
        "Maximum Depth",
        "12"
    )


with config_col3:

    st.metric(
        "Features",
        "78"
    )


with config_col4:

    st.metric(
        "Train/Test Split",
        "80 / 20"
    )


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("📊 Confusion Matrix")


@st.cache_data
def load_confusion_matrix():

    return pd.read_csv(
        f"{MODEL_RESULTS_PATH}/confusion_matrix.csv"
    )


confusion_df = load_confusion_matrix()


confusion_matrix = (
    confusion_df
    .pivot(
        index="actual",
        columns="predicted",
        values="count"
    )
    .fillna(0)
)


confusion_matrix.index = [
    "BENIGN",
    "ATTACK"
]


confusion_matrix.columns = [
    "BENIGN",
    "ATTACK"
]


st.dataframe(
    confusion_matrix,
    use_container_width=True
)


st.caption(
    "Rows = actual class. "
    "Columns = predicted class. "
    "BENIGN = 0, ATTACK = 1."
)


# =========================================================
# CONFUSION MATRIX INTERPRETATION
# =========================================================

tn = 453591
fp = 467
fn = 726
tp = 110376


cm_col1, cm_col2, cm_col3, cm_col4 = (
    st.columns(4)
)


with cm_col1:

    st.metric(
        "True Negatives",
        f"{tn:,}"
    )


with cm_col2:

    st.metric(
        "False Positives",
        f"{fp:,}"
    )


with cm_col3:

    st.metric(
        "False Negatives",
        f"{fn:,}"
    )


with cm_col4:

    st.metric(
        "True Positives",
        f"{tp:,}"
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader(
    "🔎 Top Random Forest Feature Importances"
)


@st.cache_data
def load_feature_importance():

    return pd.read_csv(
        f"{MODEL_RESULTS_PATH}/feature_importance.csv"
    )


feature_df = load_feature_importance()


feature_df = feature_df.sort_values(
    "importance",
    ascending=True
)


st.bar_chart(
    feature_df,
    x="feature",
    y="importance",
    horizontal=True
)


st.caption(
    "The chart shows the 15 features with the highest "
    "Random Forest feature-importance values."
)


# =========================================================
# SCALABILITY EXPERIMENT
# =========================================================

st.header("📈 Spark Scalability Experiment")


st.write(
    "The same Random Forest workload was evaluated using "
    "25%, 50%, and 100% of the prepared dataset to examine "
    "how training time changes as the amount of data increases."
)


# =========================================================
# LOAD COMPLETED SCALABILITY RESULTS
# =========================================================

@st.cache_data
def load_scalability_results():

    results = pd.read_csv(
        SCALABILITY_RESULTS_PATH
    )

    results["Training Time (minutes)"] = (
        results["training_time_seconds"] / 60
    )

    return results


scalability_df = load_scalability_results()


# =========================================================
# FORMAT SCALABILITY TABLE
# =========================================================

display_scalability_df = (
    scalability_df[
        [
            "dataset_size",
            "rows",
            "training_time_seconds",
            "Training Time (minutes)"
        ]
    ]
    .copy()
)


display_scalability_df.columns = [
    "Dataset Size",
    "Rows",
    "Training Time (seconds)",
    "Training Time (minutes)"
]


display_scalability_df[
    "Training Time (seconds)"
] = display_scalability_df[
    "Training Time (seconds)"
].round(2)


display_scalability_df[
    "Training Time (minutes)"
] = display_scalability_df[
    "Training Time (minutes)"
].round(2)


# =========================================================
# EXPERIMENT RESULTS TABLE
# =========================================================

st.subheader("Experiment Results")


st.dataframe(
    display_scalability_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# TRAINING TIME CHART
# =========================================================

st.subheader("Training Time vs Dataset Size")


chart_df = scalability_df[
    [
        "dataset_size",
        "training_time_seconds"
    ]
].copy()


chart_df.columns = [
    "Dataset Size",
    "Training Time (seconds)"
]


chart_df = chart_df.set_index(
    "Dataset Size"
)


st.line_chart(
    chart_df,
    y="Training Time (seconds)",
    use_container_width=True
)


# =========================================================
# TRAINING TIME METRICS
# =========================================================

st.subheader("Training Time Summary")


time_col1, time_col2, time_col3 = st.columns(3)


for index, column in enumerate(
    [time_col1, time_col2, time_col3]
):

    if index < len(scalability_df):

        row = scalability_df.iloc[index]

        with column:

            st.metric(
                f"{row['dataset_size']} Dataset",
                f"{row['training_time_seconds']:.2f} sec",
                f"{row['training_time_seconds'] / 60:.2f} min"
            )


# =========================================================
# SCALABILITY INTERPRETATION
# =========================================================

st.subheader("Scalability Interpretation")


time_25 = float(
    scalability_df.iloc[0]["training_time_seconds"]
)

time_50 = float(
    scalability_df.iloc[1]["training_time_seconds"]
)

time_100 = float(
    scalability_df.iloc[2]["training_time_seconds"]
)


increase_25_to_50 = time_50 - time_25

increase_50_to_100 = time_100 - time_50


st.markdown(
    f"""
The experiment shows that increasing the dataset size does not
produce a perfectly linear increase in Random Forest training time.

- **25%:** 708,804 rows required approximately **{time_25:.2f} seconds**
  ({time_25 / 60:.2f} minutes).
- **50%:** 1,416,895 rows required approximately **{time_50:.2f} seconds**
  ({time_50 / 60:.2f} minutes).
- **100%:** 2,830,743 rows required approximately **{time_100:.2f} seconds**
  ({time_100 / 60:.2f} minutes).

The increase from **25% to 50%** was approximately
**{increase_25_to_50:.2f} seconds**.

The increase from **50% to 100%** was approximately
**{increase_50_to_100:.2f} seconds**.

The larger increase at 100% is consistent with the memory pressure
and disk-spilling warnings observed during the larger Spark workloads.
When Spark has to spill intermediate data to disk, additional disk I/O
is introduced, which can increase execution time.

Therefore, the experiment demonstrates that Spark can process the
increasing workload while also showing that available memory and disk
I/O can become important performance factors as the dataset grows.
"""
)


# =========================================================
# SCALABILITY EXPERIMENT CONFIGURATION
# =========================================================

st.subheader("⚙️ Scalability Experiment Configuration")


scale_config1, scale_config2, scale_config3, scale_config4 = (
    st.columns(4)
)


with scale_config1:

    st.metric(
        "Random Forest Trees",
        "50"
    )


with scale_config2:

    st.metric(
        "Maximum Depth",
        "12"
    )


with scale_config3:

    st.metric(
        "Features",
        "78"
    )


with scale_config4:

    st.metric(
        "Train/Test Split",
        "80 / 20"
    )


st.info(
    "The scalability experiment used the same Random Forest "
    "configuration at 25%, 50%, and 100% of the prepared dataset. "
    "The displayed results are from the completed experiment and "
    "are not recalculated when the dashboard loads."
)


# =========================================================
# BIG DATA PIPELINE
# =========================================================

st.header("🏗️ Big Data Pipeline")


st.markdown(
    """
```text
CICIDS2017
     ↓
   Kafka
     ↓
Spark Structured Streaming
     ↓
   Hdfs
     ↓
Spark MLlib
     ↓
Random Forest
     ↓
Streamlit Dashboard
"""
)


# =========================================================
# TECHNOLOGY ROLES
# =========================================================

st.subheader("🔧 Big Data Technology Roles")


technology_col1, technology_col2, technology_col3 = (
    st.columns(3)
)


with technology_col1:

    st.markdown(
        """
### Apache Kafka

Used as the ingestion layer to
replay CICIDS2017 network-flow
records as a stream of messages.

It decouples data production
from downstream processing.
"""
    )


with technology_col2:

    st.markdown(
        """
### Apache Spark

Used for distributed data
processing, feature preparation,
Spark MLlib Random Forest
classification, and scalability
experiments.
"""
    )


with technology_col3:

    st.markdown(
        """
### HDFS

Used as the scalable distributed
storage layer for the processed
Parquet dataset used by Spark.
"""
    )


# =========================================================
# PROJECT SUMMARY
# =========================================================

st.header("📌 Project Summary")


st.markdown(
    f"""
This dashboard presents results from the COEN542 Big Data
network intrusion detection pipeline using the CICIDS2017
dataset.

**Dataset**

- Total flows: **{total_flows:,}**
- Network features: **{feature_count}**
- Benign traffic: **{benign_count:,} flows**
- Attack traffic: **{attack_count:,} flows**

**Analytics**

- Binary intrusion classification
- Spark MLlib Random Forest
- 50 trees
- Maximum depth: 12
- 78 input features
- 80/20 train/test split

**Big Data Technologies**

- Apache Kafka for ingestion
- Apache Spark for distributed processing and machine learning
- HDFS for scalable storage
- Streamlit for interactive visualization

**Scalability**

The completed experiment evaluated
**25%, 50%, and 100%** of the prepared dataset and measured
Random Forest training time for each workload.
"""
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "COEN542 — Scalable Big Data Pipeline for Network "
    "Intrusion Detection Using the CICIDS2017 Dataset"
)
