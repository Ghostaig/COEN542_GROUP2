"""
COEN542 - Stage: Ingestion (Kafka Producer) - v3
Adds --skip so chunked runs are continuous through the dataset instead of
each restarting from row 0.

Usage:
    python3 kafka_producer.py --rate 400 --skip 0 --limit 943581        # chunk 1
    python3 kafka_producer.py --rate 400 --skip 943581 --limit 943581   # chunk 2
    python3 kafka_producer.py --rate 400 --skip 1887162                 # chunk 3, rest of dataset
"""

import argparse
import json
import time

import pyarrow.dataset as ds
from kafka import KafkaProducer

PARQUET_PATH = "/mnt/bigdata/coen542-project/data/processed/unified_flows.parquet"
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "network-flows"
BATCH_SIZE = 5000


def main():
    parser = argparse.ArgumentParser(description="Kafka producer for CICIDS2017 flow replay")
    parser.add_argument("--rate", type=float, default=500,
                         help="Target messages per second (0 = as fast as possible)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Max number of rows to send THIS RUN (default: all remaining)")
    parser.add_argument("--skip", type=int, default=0,
                         help="Number of rows to skip from the start of the dataset "
                              "(use this to continue where a previous chunk left off)")
    parser.add_argument("--topic", type=str, default=TOPIC)
    args = parser.parse_args()

    print(f"[INFO] Connecting to Kafka at {KAFKA_BOOTSTRAP} ...")
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=5,
        acks=1,
    )
    print(f"[INFO] Connected. Streaming to topic '{args.topic}'")
    if args.skip > 0:
        print(f"[INFO] Skipping first {args.skip} rows (continuing from a previous chunk)")

    delay_per_msg = (1.0 / args.rate) if args.rate > 0 else 0.0

    dataset = ds.dataset(PARQUET_PATH, format="parquet")
    rows_seen = 0
    total_sent = 0
    start_time = time.time()
    last_report = start_time

    try:
        for batch in dataset.to_batches(batch_size=BATCH_SIZE):
            df_batch = batch.to_pydict()
            n_rows = len(df_batch["Label"])
            col_names = df_batch.keys()

            for i in range(n_rows):
                rows_seen += 1

                if rows_seen <= args.skip:
                    continue

                if args.limit is not None and total_sent >= args.limit:
                    raise StopIteration

                record = {k: df_batch[k][i] for k in col_names}
                record["_ingest_ts"] = time.time()
                record["_dataset_row_index"] = rows_seen - 1

                producer.send(args.topic, value=record)
                total_sent += 1

                if delay_per_msg > 0:
                    time.sleep(delay_per_msg)

                now = time.time()
                if now - last_report >= 5.0:
                    elapsed = now - start_time
                    rate_actual = total_sent / elapsed if elapsed > 0 else 0
                    print(f"[INFO] Sent {total_sent} messages this run "
                          f"(dataset row {rows_seen - 1}) | "
                          f"{rate_actual:.1f} msg/s actual | {elapsed:.1f}s elapsed")
                    last_report = now

    except StopIteration:
        pass
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        producer.flush()
        producer.close()
        elapsed = time.time() - start_time
        rate = total_sent / elapsed if elapsed > 0 else 0
        last_row_sent = args.skip + total_sent - 1 if total_sent > 0 else args.skip - 1
        print(f"[INFO] Done. Sent {total_sent} messages in {elapsed:.2f}s ({rate:.1f} msg/s average)")
        print(f"[INFO] Dataset rows covered this run: {args.skip} to {last_row_sent}")
        print(f"[INFO] To continue next chunk, use: --skip {last_row_sent + 1}")


if __name__ == "__main__":
    main()
