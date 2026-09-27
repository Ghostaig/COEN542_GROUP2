"""
COEN542 - Stage: Ingestion -> Storage bridge (Spark Structured Streaming) - v2
Adds spark.local.dir so Spark's scratch space writes to the external
drive instead of the root partition's /tmp.
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import StructType

REFERENCE_PARQUET = "/mnt/bigdata/coen542-project/data/processed/unified_flows.parquet"
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "network-flows"
STORAGE_OUTPUT = "/mnt/bigdata/coen542-project/data/streaming_store/network_flows"
CHECKPOINT_DIR = "/mnt/bigdata/coen542-project/data/checkpoints/network_flows"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sink", choices=["console", "parquet"], default="console")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("COEN542-StreamingConsumer")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.local.dir", "/mnt/bigdata/coen542-project/spark-tmp")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"[INFO] Reading reference schema from {REFERENCE_PARQUET}")
    reference_df = spark.read.parquet(REFERENCE_PARQUET)
    json_schema: StructType = reference_df.schema
    print(f"[INFO] Schema has {len(json_schema.fields)} fields")

    print(f"[INFO] Subscribing to Kafka topic '{TOPIC}' at {KAFKA_BOOTSTRAP}")
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    parsed = (
        raw_stream
        .selectExpr("CAST(value AS STRING) AS json_str")
        .select(from_json(col("json_str"), json_schema).alias("data"))
        .select("data.*")
        .withColumn("_processing_ts", current_timestamp())
    )

    if args.sink == "console":
        query = (
            parsed.writeStream
            .format("console")
            .outputMode("append")
            .option("truncate", "true")
            .option("numRows", 5)
            .trigger(processingTime="5 seconds")
            .start()
        )
        print(f"[INFO] Console sink started. Running for {args.timeout}s...")
        if args.timeout > 0:
            query.awaitTermination(args.timeout)
            query.stop()
        else:
            query.awaitTermination()

    else:
        query = (
            parsed.writeStream
            .format("parquet")
            .option("path", STORAGE_OUTPUT)
            .option("checkpointLocation", CHECKPOINT_DIR)
            .outputMode("append")
            .trigger(processingTime="10 seconds")
            .start()
        )
        print(f"[INFO] Parquet sink started, writing to {STORAGE_OUTPUT}")
        print(f"[INFO] Checkpoint at {CHECKPOINT_DIR}")
        print(f"[INFO] Running indefinitely - Ctrl+C to stop")
        query.awaitTermination()

    spark.stop()


if __name__ == "__main__":
    main()
