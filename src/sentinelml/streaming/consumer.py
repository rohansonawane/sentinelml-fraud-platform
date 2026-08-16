from __future__ import annotations

from sentinelml.streaming.broker import Broker
from sentinelml.streaming.producer import TOPICS


class TopicConsumer:
    def __init__(self, broker: Broker, topic: str = TOPICS["predictions"]) -> None:
        self.broker = broker
        self.topic = topic
        self.received: list[dict] = []

    def poll(self, max_messages: int = 500) -> list[dict]:
        batch = self.broker.consume(self.topic, max_messages=max_messages)
        self.received.extend(batch)
        return batch
