"""
COEN542 - Quick consumer test: prints incoming messages so you can verify
the producer is actually delivering data before wiring up Spark
Structured Streaming as the real consumer.

Usage:
    python3 kafka_consumer_test.py
    (Ctrl+C to stop)
"""

import json
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "network-flows"

def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=15000,  # stop automatically if no messages for 15s
    )
    print(f"[INFO] Listening on topic '{TOPIC}' (will stop after 15s of silence)...")

    count = 0
    for msg in consumer:
        count += 1
        if count <= 3:
            print(f"[MSG {count}] partition={msg.partition} "
                  f"Label={msg.value.get('Label')} "
                  f"DestPort={msg.value.get('Destination Port')} "
                  f"ingest_ts={msg.value.get('_ingest_ts')}")
        elif count % 100 == 0:
            print(f"[INFO] ... {count} messages received so far")

    print(f"[INFO] Total messages received: {count}")

if __name__ == "__main__":
    main()
