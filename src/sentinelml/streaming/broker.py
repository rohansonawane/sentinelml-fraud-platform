"""Pluggable broker: in-process queue (smoke) + Kafka client (when broker is up).

Docker daemon was DOWN at discovery. Kafka is not started for smoke.
`make stream-demo` uses the memory backend.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterator
from threading import Lock
from typing import Any

from sentinelml.utils.logging import get_logger

log = get_logger("broker")


class Broker(ABC):
    @abstractmethod
    def publish(self, topic: str, value: dict[str, Any], key: str | None = None) -> None: ...

    @abstractmethod
    def consume(self, topic: str, max_messages: int = 100) -> list[dict[str, Any]]: ...

    @abstractmethod
    def topics(self) -> list[str]: ...


class InMemoryBroker(Broker):
    def __init__(self) -> None:
        self._topics: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._seen: set[str] = set()
        self._lock = Lock()

    def publish(self, topic: str, value: dict[str, Any], key: str | None = None) -> None:
        idem = value.get("transaction_id") or value.get("event_id")
        with self._lock:
            if idem and f"{topic}:{idem}" in self._seen:
                return
            if idem:
                self._seen.add(f"{topic}:{idem}")
            payload = dict(value)
            if key:
                payload["_key"] = key
            self._topics[topic].append(payload)

    def consume(self, topic: str, max_messages: int = 100) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._lock:
            q = self._topics[topic]
            for _ in range(min(max_messages, len(q))):
                out.append(q.popleft())
        return out

    def peek_all(self, topic: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._topics[topic])

    def topics(self) -> list[str]:
        with self._lock:
            return list(self._topics.keys())

    def qsize(self, topic: str) -> int:
        with self._lock:
            return len(self._topics[topic])

    def iter_topic(self, topic: str) -> Iterator[dict[str, Any]]:
        yield from self.peek_all(topic)


class KafkaBroker(Broker):
    def __init__(self, bootstrap: str = "localhost:9092") -> None:
        from kafka import KafkaConsumer, KafkaProducer

        self.bootstrap = bootstrap
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            request_timeout_ms=2000,
            api_version_auto_timeout_ms=2000,
        )
        self._consumers: dict[str, KafkaConsumer] = {}

    def _consumer(self, topic: str) -> Any:
        from kafka import KafkaConsumer

        if topic not in self._consumers:
            self._consumers[topic] = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap,
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                auto_offset_reset="earliest",
                consumer_timeout_ms=500,
                enable_auto_commit=True,
                group_id=f"sentinelml-{topic}",
            )
        return self._consumers[topic]

    def publish(self, topic: str, value: dict[str, Any], key: str | None = None) -> None:
        self._producer.send(topic, value=value, key=key)
        self._producer.flush(timeout=2)

    def consume(self, topic: str, max_messages: int = 100) -> list[dict[str, Any]]:
        cons = self._consumer(topic)
        out: list[dict[str, Any]] = []
        for i, msg in enumerate(cons):
            out.append(msg.value)
            if i + 1 >= max_messages:
                break
        return out

    def topics(self) -> list[str]:
        return list(self._consumers.keys())


def build_broker(backend: str = "memory", kafka_bootstrap: str = "localhost:9092") -> Broker:
    if backend == "kafka":
        try:
            broker = KafkaBroker(kafka_bootstrap)
            log.info("kafka_broker_connected", bootstrap=kafka_bootstrap)
            return broker
        except Exception as exc:
            log.warning("kafka_unavailable_falling_back_to_memory", error=str(exc))
    return InMemoryBroker()
