#!/usr/bin/env python3
"""Measured load test against TestClient or a live API. No invented numbers."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from sentinelml.api.app import create_app
from sentinelml.streaming.producer import row_to_event


async def _hammer_async(base: str, events: list[dict], concurrency: int) -> list[float]:
    latencies: list[float] = []
    errors = 0

    async def one(client: httpx.AsyncClient, ev: dict) -> None:
        nonlocal errors
        t0 = time.perf_counter()
        try:
            r = await client.post("/api/v1/score", json=ev, timeout=10.0)
            if r.status_code >= 400:
                errors += 1
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - t0) * 1000.0)

    sem = asyncio.Semaphore(concurrency)

    async def bounded(client: httpx.AsyncClient, ev: dict) -> None:
        async with sem:
            await one(client, ev)

    async with httpx.AsyncClient(base_url=base) as client:
        await asyncio.gather(*[bounded(client, ev) for ev in events])
    return latencies, errors  # type: ignore[return-value]


def _hammer_testclient(n: int, events: list[dict]) -> tuple[list[float], int]:
    app = create_app()
    latencies: list[float] = []
    errors = 0
    with TestClient(app) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise SystemExit("API failed health check")
        for ev in events[:n]:
            t0 = time.perf_counter()
            r = client.post("/api/v1/score", json=ev)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if r.status_code >= 400:
                errors += 1
    return latencies, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--base-url", default="", help="If empty, use in-process TestClient")
    args = parser.parse_args()
    path = Path("data/generated/transactions.parquet")
    if not path.exists():
        raise SystemExit("generate data first")
    df = pd.read_parquet(path).sort_values("event_time").head(args.requests)
    events = [row_to_event(r) for r in df.to_dict(orient="records")]

    t0 = time.perf_counter()
    if args.base_url:
        latencies, errors = asyncio.run(_hammer_async(args.base_url, events, args.concurrency))
    else:
        latencies, errors = _hammer_testclient(args.requests, events)
    elapsed = time.perf_counter() - t0
    arr = np.array(latencies) if latencies else np.array([0.0])
    payload = {
        "n_requests": len(latencies),
        "concurrency": args.concurrency if args.base_url else 1,
        "backend": args.base_url or "TestClient",
        "elapsed_seconds": elapsed,
        "throughput_rps": len(latencies) / elapsed if elapsed else 0.0,
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_p99": float(np.percentile(arr, 99)),
        "error_rate": errors / len(latencies) if latencies else 1.0,
        "errors": errors,
        "note": "Measured on this host. Do not claim 5000 TPS.",
    }
    out = Path("artifacts/benchmark.json")
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
