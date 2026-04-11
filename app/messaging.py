from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

try:
    import pika
except ImportError:
    pika = None


class RunEventPublisher:
    def __init__(self) -> None:
        self._url = (os.getenv("ATLAS_RABBITMQ_URL") or "").strip()
        self._exchange = (os.getenv("ATLAS_RABBITMQ_EXCHANGE") or "atlas.runs").strip()
        self._routing_prefix = (os.getenv("ATLAS_RABBITMQ_ROUTING_PREFIX") or "atlas").strip(".")

    def publish(self, event_type: str, payload: dict[str, Any]) -> bool:
        if not self._url or pika is None:
            return False

        envelope = {
            "event_type": event_type,
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        routing_key = f"{self._routing_prefix}.{event_type}".replace(" ", "-")

        connection = pika.BlockingConnection(pika.URLParameters(self._url))
        try:
            channel = connection.channel()
            channel.exchange_declare(exchange=self._exchange, exchange_type="topic", durable=True)
            channel.basic_publish(
                exchange=self._exchange,
                routing_key=routing_key,
                body=json.dumps(envelope, default=str),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,
                ),
            )
            return True
        finally:
            connection.close()
