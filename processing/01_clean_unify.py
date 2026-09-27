"""
COEN542 - Stage: Processing (Cleaning + Unification) - v4
Adds spark.local.dir so Spark's shuffle/scratch space writes to the
external drive instead of the root partition's /tmp.
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

RAW_DIR = "/mnt/bigdata/coen542-project/data/raw/MachineLearningCVE"
OUTPUT_DIR = "/mnt/bigdata/coen542-project/data/processed/unified_flows.parquet"

FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

def dedupe_columns(cols):
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 1
            new_cols.append(c)
    return new_cols

def main():
    start = time.time()

    spark = (
        SparkSession.builder
        .appName("COEN542-CleanUnify-v4")
        .master("local[*]")
        .config("spark.driver.memory", "3g")
        .config("spark.local.dir", "/mnt/bigdata/coen542-project/spark-tmp")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"[INFO] Loading {len(FILES)} CSV files from {RAW_DIR}")

    all_dfs = []
    for fname in FILES:
        path = f"{RAW_DIR}/{fname}"
        df = spark.read.csv(path, header=True, inferSchema=True)

        stripped = [c.strip() for c in df.columns]
        deduped = dedupe_columns(stripped)
        df = df.toDF(*deduped)

        df = df.withColumn("source_file", F.lit(fname))
        all_dfs.append(df)
        print(f"[INFO] Loaded {fname}: {df.count()} rows, {len(df.columns)} columns")

    unified = all_dfs[0]
    for df in all_dfs[1:]:
        unified = unified.unionByName(df)

    total_rows = unified.count()
    print(f"[INFO] Unified dataset: {total_rows} total rows, {len(unified.columns)} columns")

    cols = unified.columns
    dupes = [c for c in set(cols) if cols.count(c) > 1]
    print(f"[INFO] Duplicate columns after dedupe: {dupes}")

    unified = unified.withColumn(
        "Label",
        F.regexp_replace(F.col("Label"), r"[^\x00-\x7F]+", " - ")
    )
    unified = unified.withColumn(
        "Label",
        F.regexp_replace(F.col("Label"), r"\s+", " ")
    )
    unified = unified.withColumn("Label", F.trim(F.col("Label")))

    problem_cols = ["Flow Bytes/s", "Flow Packets/s"]
    for col_name in problem_cols:
        unified = unified.withColumn(
            col_name,
            F.when(F.col(col_name).isin(float("inf"), float("-inf")), None)
             .otherwise(F.col(col_name).cast(DoubleType()))
        )

    unified = unified.dropna(subset=["Label"])
    unified = unified.fillna(0, subset=problem_cols)

    cleaned_rows = unified.count()
    print(f"[INFO] Rows after cleaning: {cleaned_rows} (dropped {total_rows - cleaned_rows})")

    print("[INFO] Label distribution (normalized):")
    unified.groupBy("Label").count().orderBy(F.desc("count")).show(30, truncate=False)

    print(f"[INFO] Writing unified Parquet to {OUTPUT_DIR}")
    unified.write.mode("overwrite").parquet(OUTPUT_DIR)

    elapsed = time.time() - start
    print(f"[INFO] Done in {elapsed:.2f} seconds")

    spark.stop()

if __name__ == "__main__":
    main()
