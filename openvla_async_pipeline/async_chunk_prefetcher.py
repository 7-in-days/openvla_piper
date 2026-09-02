from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from openvla_pipeline.model_io import ContractError, validate_action_chunk


@dataclass(frozen=True)
class PreparedChunk:
    """Background inference result together with its observation age."""

    sequence: int
    request_id: str
    actions: np.ndarray
    server_inference_ms: float
    requested_at: float
    ready_at: float

    @property
    def inference_wall_ms(self) -> float:
        return (self.ready_at - self.requested_at) * 1000.0


class AsyncChunkPrefetcher:
    """One in-flight inference plus one latest-only pending trigger.

    The worker captures the robot observation inside ``infer``. Therefore a pending
    trigger does not retain camera frames in RAM, and the observation is read only
    when the policy server is ready to start the next inference.
    """

    def __init__(
        self,
        infer: Callable[[int], dict[str, Any]],
        expected_shape: tuple[int, int],
    ) -> None:
        self._infer = infer
        self._expected_shape = expected_shape
        self._condition = threading.Condition()
        self._pending_sequence: int | None = None
        self._result: PreparedChunk | None = None
        self._error: BaseException | None = None
        self._closed = False
        self._next_sequence = 0
        self._replaced_requests = 0
        self._worker = threading.Thread(
            target=self._run,
            name="piper-openvla-inference",
            daemon=True,
        )
        self._worker.start()

    def request(self) -> int:
        """Schedule inference; replace only a not-yet-started trigger."""
        with self._condition:
            self._raise_if_unavailable()
            sequence = self._next_sequence
            self._next_sequence += 1
            if self._pending_sequence is not None:
                self._replaced_requests += 1
            self._pending_sequence = sequence
            self._condition.notify_all()
            return sequence

    def get(self, timeout_s: float, max_age_s: float) -> PreparedChunk:
        if timeout_s <= 0 or max_age_s <= 0:
            raise ValueError("timeout_s and max_age_s must be positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._result is None and self._error is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"OpenVLA action chunk was not ready within {timeout_s:.3f}s"
                    )
                self._condition.wait(remaining)
            self._raise_if_unavailable()
            if self._result is None:
                raise RuntimeError("async prefetcher closed without an action chunk")
            result = self._result
            self._result = None
            self._condition.notify_all()

        age_s = time.monotonic() - result.requested_at
        if age_s > max_age_s:
            raise ContractError(
                "OpenVLA action chunk is stale: "
                f"sequence={result.sequence} age={age_s:.3f}s limit={max_age_s:.3f}s"
            )
        return result

    @property
    def replaced_requests(self) -> int:
        with self._condition:
            return self._replaced_requests

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending_sequence = None
            self._condition.notify_all()
        self._worker.join(timeout=1.0)

    def _raise_if_unavailable(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"OpenVLA async inference failed: {type(self._error).__name__}: {self._error}"
            ) from self._error
        if self._closed:
            raise RuntimeError("OpenVLA async prefetcher is closed")

    def _run(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._closed
                    and (self._pending_sequence is None or self._result is not None)
                ):
                    self._condition.wait()
                if self._closed:
                    return
                sequence = self._pending_sequence
                self._pending_sequence = None

            assert sequence is not None
            requested_at = time.monotonic()
            try:
                response = self._infer(sequence)
                actions = validate_action_chunk(
                    response.get("actions"),
                    self._expected_shape,
                ).copy()
                request_id = str(response.get("request_id", ""))
                if not request_id:
                    raise ContractError("OpenVLA response request_id is missing")
                server_inference_ms = float(response.get("inference_ms", float("nan")))
                if not np.isfinite(server_inference_ms) or server_inference_ms < 0:
                    raise ContractError(
                        f"invalid server inference_ms: {server_inference_ms}"
                    )
                result = PreparedChunk(
                    sequence=sequence,
                    request_id=request_id,
                    actions=actions,
                    server_inference_ms=server_inference_ms,
                    requested_at=requested_at,
                    ready_at=time.monotonic(),
                )
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
                return

            with self._condition:
                if self._closed:
                    return
                self._result = result
                self._condition.notify_all()
