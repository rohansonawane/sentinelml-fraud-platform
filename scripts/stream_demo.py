#!/usr/bin/env python3
"""Replay transactions chronologically through the in-process broker.

Kafka was not started (Docker daemon down). This demo uses STREAM_BACKEND=memory.
Measures actual throughput. Does not claim 5000 TPS.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from sentinelml.features.engineering import FeatureState
from sentinelml.inference.service import InferenceService
from sentinelml.streaming.broker import InMemoryBroker
from sentinelml.streaming.processor import StreamProcessor
from sentinelml.streaming.producer import TOPICS, TransactionProducer, row_to_event
from sentinelml.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=100, help="target events/sec")
    parser.add_argument("--n-events", type=int, default=500)
    parser.add_argument("--backend", default="memory")
    parser.add_argument("--data", default="data/generated/transactions.parquet")
    args = parser.parse_args()
    if args.backend != "memory":
        print("Kafka backend requested but daemon was down; using in-process queue.")
    path = Path(args.data)
    if not path.exists():
        raise SystemExit("generate data first")
    df = pd.read_parquet(path).sort_values("event_time").head(args.n_events)
    events = [row_to_event(r) for r in df.to_dict(orient="records")]

    broker = InMemoryBroker()
    producer = TransactionProducer(broker)
    inf = InferenceService()
    delay = load_yaml("configs/model.yaml").get("label_delay_days", 7)
    proc = StreamProcessor(broker, state=FeatureState(label_delay_days=delay), inference=inf)

    interval = 1.0 / max(args.rate, 1)
    t0 = time.perf_counter()
    for i, ev in enumerate(events):
        producer.send(ev)
        proc.drain(max_messages=32)
        if args.rate <= 500:
            target = t0 + (i + 1) * interval
            now = time.perf_counter()
            if target > now:
                time.sleep(min(target - now, interval))
    # drain remaining
    leftover = broker.qsize(TOPICS["raw"]) if hasattr(broker, "qsize") else 0
    if leftover:
        proc.drain(max_messages=leftover + 10)
    elapsed = time.perf_counter() - t0
    n = proc.processed
    payload = {
        "backend": "memory",
        "kafka_started": False,
        "kafka_note": "Docker daemon was DOWN. Kafka was not started. In-process queue used.",
        "n_events_submitted": len(events),
        "n_processed": n,
        "n_alerts": proc.alerts,
        "n_errors": proc.errors,
        "elapsed_seconds": elapsed,
        "target_rate_eps": args.rate,
        "actual_throughput_eps": n / elapsed if elapsed else 0.0,
        "predictions_queued": broker.qsize(TOPICS["predictions"]) if hasattr(broker, "qsize") else None,
        "all_processed": n == len(events) and proc.errors == 0,
        "avg_infer_latency_ms": (
            float(sum(proc.last_latencies) / len(proc.last_latencies)) if proc.last_latencies else None
        ),
    }
    out = Path("artifacts/stream_demo.json")
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if n != len(events):
        raise SystemExit(f"e2e incomplete: processed {n}/{len(events)}")


if __name__ == "__main__":
    main()
