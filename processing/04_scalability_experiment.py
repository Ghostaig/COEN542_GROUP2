import time

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

INPUT_PATH = "/mnt/bigdata/coen542-project/data/processed/ml_binary"
RESULT_PATH = "/mnt/bigdata/coen542-project/data/processed/scalability_results"

SPARK_TMP = "/mnt/bigdata/coen542-project/spark-tmp"


spark = (
    SparkSession.builder
    .appName("COEN542-Scalability-Experiment")
    .master("local[4]")
    .config("spark.driver.memory", "2g")
    .config("spark.local.dir", SPARK_TMP)
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


print("[INFO] Reading ML dataset...")
df = spark.read.parquet(INPUT_PATH)

total_rows = df.count()

print(f"[INFO] Total dataset rows: {total_rows}")


# ---------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------

feature_columns = [
    c for c in df.columns
    if c != "binary_label"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

df_vector = assembler.transform(df).select(
    "features",
    "binary_label"
)


# ---------------------------------------------------------
# Experiment sizes
# ---------------------------------------------------------

experiments = [
    ("25%", 0.25),
    ("50%", 0.50),
    ("100%", 1.00),
]


results = []


for name, fraction in experiments:

    print()
    print("=" * 60)
    print(f"STARTING {name} DATASET EXPERIMENT")
    print("=" * 60)

    # -----------------------------------------------------
    # Select dataset size
    # -----------------------------------------------------

    if fraction == 1.0:
        experiment_df = df_vector
    else:
        experiment_df = df_vector.sample(
            withReplacement=False,
            fraction=fraction,
            seed=42
        )

    experiment_rows = experiment_df.count()

    print(f"[INFO] Dataset size: {name}")
    print(f"[INFO] Rows: {experiment_rows}")


    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    train_df, test_df = experiment_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    print("[INFO] Train/test split created.")


    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

    rf = RandomForestClassifier(
        labelCol="binary_label",
        featuresCol="features",
        numTrees=50,
        maxDepth=12,
        seed=42
    )


    # -----------------------------------------------------
    # Measure training time
    # -----------------------------------------------------

    start_time = time.perf_counter()

    print("[INFO] Training Random Forest...")

    model = rf.fit(train_df)

    end_time = time.perf_counter()

    training_time = end_time - start_time

    print(
        f"[INFO] Training completed in "
        f"{training_time:.2f} seconds "
        f"({training_time / 60:.2f} minutes)"
    )


    results.append(
        (
            name,
            experiment_rows,
            training_time
        )
    )


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print()
print("=" * 60)
print("SCALABILITY EXPERIMENT RESULTS")
print("=" * 60)

for name, rows, seconds in results:

    print(
        f"{name:>5} | "
        f"{rows:>10,} rows | "
        f"{seconds:>10.2f} seconds | "
        f"{seconds / 60:>8.2f} minutes"
    )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_df = spark.createDataFrame(
    results,
    ["dataset_size", "rows", "training_time_seconds"]
)

results_df.write.mode("overwrite").option(
    "header", "true"
).csv(RESULT_PATH)

print()
print(f"[INFO] Results saved to:")
print(RESULT_PATH)


spark.stop()
