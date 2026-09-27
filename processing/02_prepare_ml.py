from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


# ---------------------------------------------------------
# COEN542 Big Data Project
# Step 2: Prepare CICIDS2017 data for binary classification
#
# BENIGN -> 0
# All attack types -> 1
# ---------------------------------------------------------


INPUT_PATH = (
    "/mnt/bigdata/coen542-project/"
    "data/streaming_store/network_flows"
)

OUTPUT_PATH = (
    "/mnt/bigdata/coen542-project/"
    "data/processed/ml_binary"
)

SPARK_TMP = "/mnt/bigdata/coen542-project/spark-tmp"


# ---------------------------------------------------------
# Start Spark
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("COEN542-Prepare-Binary-ML")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .config("spark.local.dir", SPARK_TMP)
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# Read streaming output
# ---------------------------------------------------------

print("\n[INFO] Reading streaming Parquet data...")
print(f"[INFO] Input: {INPUT_PATH}")

df = spark.read.parquet(INPUT_PATH)

print(f"[INFO] Input columns: {len(df.columns)}")


# ---------------------------------------------------------
# Create binary target
#
# BENIGN = 0
# Any other label = 1
# ---------------------------------------------------------

print("\n[INFO] Creating binary target...")

df = df.withColumn(
    "binary_label",
    when(col("Label") == "BENIGN", 0.0).otherwise(1.0)
)


# ---------------------------------------------------------
# Remove metadata columns
#
# Label is replaced by binary_label.
# source_file and _processing_ts are metadata and should
# not be used as ML features.
# ---------------------------------------------------------

feature_columns = [
    c for c in df.columns
    if c not in [
        "Label",
        "source_file",
        "_processing_ts",
        "binary_label"
    ]
]

df_ml = df.select(
    *feature_columns,
    "binary_label"
)


# ---------------------------------------------------------
# Display schema
# ---------------------------------------------------------

print("\n[INFO] ML dataset schema:")
df_ml.printSchema()


# ---------------------------------------------------------
# Verify binary-label distribution
# ---------------------------------------------------------

print("\n[INFO] Binary label distribution:")

(
    df_ml
    .groupBy("binary_label")
    .count()
    .orderBy("binary_label")
    .show()
)


# ---------------------------------------------------------
# Verify row count
# ---------------------------------------------------------

row_count = df_ml.count()

print(f"[INFO] ML rows: {row_count}")

if row_count != 2830743:
    raise RuntimeError(
        f"Unexpected row count: {row_count}. "
        "Expected 2,830,743."
    )


# ---------------------------------------------------------
# Write ML-ready Parquet
# ---------------------------------------------------------

print(f"\n[INFO] Writing ML dataset to:")
print(f"[INFO] {OUTPUT_PATH}")

(
    df_ml
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


print("\n[INFO] ML dataset successfully created.")
print(f"[INFO] Rows: {row_count}")
print(f"[INFO] Features: {len(feature_columns)}")
print(f"[INFO] Output: {OUTPUT_PATH}")


spark.stop()
