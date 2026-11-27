"""Publishers for the replay: Pub/Sub (real or emulator) and a JSON-lines writer for dry runs.

A publisher returns a future-like object per message (``done()``, ``exception()``, ``result(timeout)``), exactly like
``google.cloud.pubsub_v1`` does, so the replay loop is identical for every target and tests can use a fake.
Credentials: Application Default Credentials only (or ``PUBSUB_EMULATOR_HOST``, which needs none). Nothing is embedded.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, Protocol

EMULATOR_ENV = "PUBSUB_EMULATOR_HOST"


class PublisherSetupError(RuntimeError):
    """Credentials, topic or configuration problem detected before publishing."""


class FutureLike(Protocol):
    def done(self) -> bool: ...
    def exception(self, timeout: float | None = None) -> BaseException | None: ...
    def result(self, timeout: float | None = None) -> Any: ...


class Publisher(Protocol):
    def publish(self, data: bytes, ordering_key: str, attributes: Mapping[str, str]) -> FutureLike: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class _DoneFuture:
    """An already-completed future (dry runs publish synchronously)."""

    def __init__(self, error: BaseException | None = None, value: Any = None) -> None:
        self._error, self._value = error, value

    def done(self) -> bool:
        return True

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._error

    def result(self, timeout: float | None = None) -> Any:
        if self._error:
            raise self._error
        return self._value


class JsonlPublisher:
    """Dry-run target: appends each message body as one JSON line to a file (``-`` = stdout)."""

    def __init__(self, target: str | Path, *, append: bool = False) -> None:
        self.target = str(target)
        self.count = 0
        self._handle: IO[bytes]
        if self.target == "-":
            import sys

            self._handle = sys.stdout.buffer
            self._owns = False
        else:
            path = Path(self.target)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("ab" if append else "wb")
            self._owns = True

    def publish(self, data: bytes, ordering_key: str, attributes: Mapping[str, str]) -> FutureLike:
        self._handle.write(data + b"\n")
        self.count += 1
        return _DoneFuture(value=str(self.count))

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self.flush()
        if self._owns:
            self._handle.close()


@dataclasses.dataclass(frozen=True)
class BatchConfig:
    max_messages: int = 100
    max_bytes: int = 1_000_000
    max_latency_s: float = 0.05
    flow_control_messages: int = 10_000
    flow_control_bytes: int = 50_000_000

    def validate(self) -> None:
        for name in ("max_messages", "max_bytes", "flow_control_messages", "flow_control_bytes"):
            if getattr(self, name) <= 0:
                raise PublisherSetupError(f"batching.{name} must be positive")
        if self.max_messages > 1000:
            raise PublisherSetupError(
                "batching.max_messages must be <= 1000 (Pub/Sub limit per publish request)"
            )
        if self.max_bytes > 10_000_000:
            raise PublisherSetupError(
                "batching.max_bytes must be <= 10,000,000 (Pub/Sub limit per publish request)"
            )
        if self.max_latency_s < 0:
            raise PublisherSetupError("batching.max_latency_s must be >= 0")


class PubSubPublisher:
    """``google.cloud.pubsub_v1`` publisher with message ordering, batching and blocking flow control.

    Ordering is enabled, so messages sharing an ordering key (the client id) are delivered in publish order.
    Flow control blocks ``publish`` when too many messages are outstanding, which is the producer's backpressure.
    """

    def __init__(self, project: str, topic: str, batch: BatchConfig, *, client: Any = None) -> None:
        batch.validate()
        try:
            from google.cloud import pubsub_v1
            from google.cloud.pubsub_v1.types import (
                BatchSettings,
                LimitExceededBehavior,
                PublisherOptions,
                PublishFlowControl,
            )
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise PublisherSetupError(
                "google-cloud-pubsub is not installed (pip install -r streaming/producer/requirements.txt)"
            ) from exc
        if client is None:
            try:
                client = pubsub_v1.PublisherClient(
                    batch_settings=BatchSettings(
                        max_messages=batch.max_messages,
                        max_bytes=batch.max_bytes,
                        max_latency=batch.max_latency_s,
                    ),
                    publisher_options=PublisherOptions(
                        enable_message_ordering=True,
                        flow_control=PublishFlowControl(
                            message_limit=batch.flow_control_messages,
                            byte_limit=batch.flow_control_bytes,
                            limit_exceeded_behavior=LimitExceededBehavior.BLOCK,
                        ),
                    ),
                )
            except Exception as exc:
                if exc.__class__.__name__ == "DefaultCredentialsError":
                    raise PublisherSetupError(
                        "no Google credentials found. Use Application Default Credentials "
                        "(`gcloud auth application-default login` or GOOGLE_APPLICATION_CREDENTIALS pointing at a key file "
                        f"outside the repository), or set {EMULATOR_ENV} to use the Pub/Sub emulator"
                    ) from exc
                raise
        self.client = client
        self.topic_path = client.topic_path(project, topic)

    def check_topic(self) -> None:
        """Fail early (and clearly) if the topic cannot be found; never creates it."""
        try:
            self.client.get_topic(request={"topic": self.topic_path})
        except Exception as exc:
            name = exc.__class__.__name__
            if name == "NotFound":
                raise PublisherSetupError(
                    f"topic {self.topic_path} does not exist. The cloud topic is provisioned by the infrastructure owner; "
                    "against the emulator use --create-topic-in-emulator"
                ) from exc
            if name in {"PermissionDenied", "Unauthenticated", "Forbidden"}:
                raise PublisherSetupError(f"not allowed to read topic {self.topic_path}: {exc}") from exc
            raise PublisherSetupError(f"cannot reach topic {self.topic_path}: {exc}") from exc

    def create_topic(self) -> bool:
        """Create the topic. Only allowed against the emulator; returns True if it was created."""
        if not os.environ.get(EMULATOR_ENV):
            raise PublisherSetupError(
                f"refusing to create a topic outside the emulator ({EMULATOR_ENV} is not set)"
            )
        try:
            self.client.create_topic(request={"name": self.topic_path})
            return True
        except Exception as exc:
            if exc.__class__.__name__ == "AlreadyExists":
                return False
            raise

    def publish(self, data: bytes, ordering_key: str, attributes: Mapping[str, str]) -> FutureLike:
        return self.client.publish(self.topic_path, data, ordering_key=ordering_key, **dict(attributes))

    def flush(self) -> None:
        # pubsub_v1 flushes pending batches in stop(); batches also flush by size/latency while running
        pass

    def close(self) -> None:
        try:
            self.client.stop()
        finally:
            transport = getattr(self.client, "transport", None)
            close = getattr(transport, "close", None)
            if close:
                try:
                    close()
                except Exception:  # pragma: no cover - best effort
                    pass
