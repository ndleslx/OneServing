"""Dynamic request admission and mixed prefill/decode batch scheduling."""

from __future__ import annotations

import contextvars
import queue
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .model import CONTEXT, ROWS


@dataclass(eq=False)
class RequestState:
    prompt_ids: list[int]
    max_tokens: int
    stop_token_ids: frozenset[int]
    stop_strings: tuple[str, ...] = ()
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: queue.SimpleQueue[tuple[str, Any]] = field(default_factory=queue.SimpleQueue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    submitted_ns: int = field(default_factory=time.monotonic_ns)
    prefill_offset: int = 0
    generated: list[int] = field(default_factory=list)
    token_times_ns: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    stop_match: str | None = None
    runtime_state: Any = None

    def metrics(self) -> dict[str, float | int | None]:
        ttft = ((self.token_times_ns[0] - self.submitted_ns) / 1e6
                if self.token_times_ns else None)
        tpot = ((self.token_times_ns[-1] - self.token_times_ns[0]) /
                (len(self.token_times_ns) - 1) / 1e6
                if len(self.token_times_ns) > 1 else None)
        return {"ttft_ms": ttft, "tpot_ms": tpot,
                "output_tokens": len(self.generated), "prompt_tokens": len(self.prompt_ids)}


@dataclass(frozen=True)
class BatchSlot:
    request: RequestState
    token_ids: tuple[int, ...]
    position: int
    phase: str
    completes_prefill: bool


class RequestCapacityError(RuntimeError):
    def __init__(self, request: RequestState, message: str):
        super().__init__(message)
        self.request = request


class BatchRunner(Protocol):
    def start(self) -> None: ...
    def close(self) -> None: ...
    def admit(self, request: RequestState) -> int: ...
    def run_batch(self, slots: list[BatchSlot]) -> list[int | None]: ...
    def release(self, request: RequestState) -> None: ...


class Scheduler:
    def __init__(self, runner: BatchRunner, stop_token_ids: frozenset[int],
                 max_active: int = 8, max_queued: int = 64,
                 on_trace: Callable[[str, dict], None] | None = None,
                 decode_tokens: Callable[[list[int]], str] | None = None):
        if not (1 <= max_active <= 8):
            raise ValueError("The compiled batch capacity is at most eight requests")
        self.runner = runner
        self.stop_token_ids = stop_token_ids
        self.max_active = max_active
        self.incoming: queue.Queue[RequestState | None] = queue.Queue(maxsize=max_queued)
        self.pending: deque[RequestState] = deque()
        self.max_queued = max_queued
        self.active: list[RequestState] = []
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="pypto-batch-scheduler", daemon=False)
        self.ready = threading.Event()
        self.on_trace = on_trace
        self.decode_tokens = decode_tokens
        self.batch_number = 0
        self.lock = threading.Lock()
        self.error: BaseException | None = None

    def _trace(self, name: str, **fields: object) -> None:
        if self.on_trace is not None:
            self.on_trace(name, {"time_ns": time.monotonic_ns(), **fields})

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(timeout=600):
            raise RuntimeError("Batch runner startup timed out")
        if self.error is not None:
            raise RuntimeError("Batch runner failed to start") from self.error

    def submit(self, prompt_ids: list[int], max_tokens: int,
               stop_strings: tuple[str, ...] = ()) -> RequestState:
        if not self.ready.is_set() or self.stopping.is_set() or self.error is not None:
            raise RuntimeError("Batch scheduler is unavailable")
        if not prompt_ids or max_tokens < 1 or len(prompt_ids) + max_tokens > CONTEXT:
            raise ValueError("Prompt and completion exceed the supported context")
        request = RequestState(prompt_ids, max_tokens, self.stop_token_ids,
                               stop_strings=stop_strings)
        with self.lock:
            if len(self.pending) + self.incoming.qsize() >= self.max_queued:
                raise RuntimeError("Request queue is full")
        try:
            self.incoming.put_nowait(request)
        except queue.Full as exc:
            raise RuntimeError("Request queue is full") from exc
        self._trace("queued", request_id=request.request_id)
        return request

    def stop(self) -> None:
        self.stopping.set()
        try:
            self.incoming.put_nowait(None)
        except queue.Full:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=120)
        if self.thread.is_alive():
            raise RuntimeError("Batch scheduler failed to stop")

    def counts(self) -> dict[str, int]:
        with self.lock:
            return {"active": len(self.active),
                    "queued": len(self.pending) + self.incoming.qsize(),
                    "batches": self.batch_number}

    def _stage_pending(self) -> int:
        staged = 0
        while True:
            try:
                request = self.incoming.get_nowait()
            except queue.Empty:
                return staged
            if request is None:
                self.stopping.set()
                return staged
            with self.lock:
                self.pending.append(request)
            staged += 1

    def _admit(self) -> None:
        self._stage_pending()
        while len(self.active) < self.max_active:
            with self.lock:
                request = self.pending.popleft() if self.pending else None
            if request is None:
                return
            if request.cancelled.is_set():
                request.events.put(("cancelled", request.metrics()))
                continue
            try:
                request.prefill_offset = self.runner.admit(request)
                if not 0 <= request.prefill_offset < len(request.prompt_ids):
                    raise ValueError("Invalid cached prefix position")
                self.active.append(request)
                self._trace("admitted", request_id=request.request_id,
                            prefix_tokens=request.prefill_offset)
            except BaseException as exc:
                request.events.put(("error", str(exc)))

    def _slot(self, request: RequestState) -> BatchSlot:
        if request.prefill_offset < len(request.prompt_ids):
            position = request.prefill_offset
            tokens = tuple(request.prompt_ids[position:position + ROWS])
            return BatchSlot(request, tokens, position, "prefill",
                             position + len(tokens) == len(request.prompt_ids))
        return BatchSlot(request, (request.generated[-1],),
                         len(request.prompt_ids) + len(request.generated) - 1,
                         "decode", False)

    def _finish(self, request: RequestState, kind: str, value: Any) -> None:
        request.events.put((kind, value))
        try:
            self.runner.release(request)
        finally:
            self._trace("finished", request_id=request.request_id, reason=kind,
                        metrics=request.metrics())

    def _step(self) -> None:
        self._admit()
        current = [request for request in self.active if not request.cancelled.is_set()]
        for request in self.active:
            if request.cancelled.is_set():
                self._finish(request, "cancelled", request.metrics())
        self.active = current
        if not self.active:
            time.sleep(0.005)
            return
        slots = [self._slot(request) for request in self.active[:self.max_active]]
        self.batch_number += 1
        self._trace("batch_submit", batch=self.batch_number,
                    slots=[{"request_id": slot.request.request_id,
                            "phase": slot.phase, "position": slot.position,
                            "tokens": len(slot.token_ids)} for slot in slots])
        try:
            context = contextvars.copy_context()
            submitted_ns = time.monotonic_ns()
            future = self.executor.submit(context.run, self.runner.run_batch, slots)
            self._trace("device_submit", batch=self.batch_number)
            while True:
                staged = self._stage_pending()
                if staged:
                    self._trace("host_staged", batch=self.batch_number,
                                requests=staged, after_submit_ns=time.monotonic_ns() - submitted_ns)
                try:
                    outputs = future.result(timeout=0.05)
                    break
                except FutureTimeout:
                    continue
            self._trace("device_complete", batch=self.batch_number)
            if len(outputs) != len(slots):
                raise RuntimeError("Batch runner returned the wrong slot count")
        except RequestCapacityError as exc:
            self._finish(exc.request, "error", str(exc))
            self.active = [request for request in self.active if request is not exc.request]
            return
        except BaseException as exc:
            for slot in slots:
                self._finish(slot.request, "error", str(exc))
            ids = {slot.request.request_id for slot in slots}
            self.active = [request for request in self.active if request.request_id not in ids]
            raise
        self._trace("batch_complete", batch=self.batch_number)
        finished = []
        for slot, token in zip(slots, outputs):
            request = slot.request
            if request.cancelled.is_set():
                self._finish(request, "cancelled", request.metrics())
                finished.append(request)
                continue
            if slot.phase == "prefill":
                request.prefill_offset += len(slot.token_ids)
            if slot.phase == "prefill" and not slot.completes_prefill:
                if token is not None:
                    raise RuntimeError("Intermediate prefill chunk emitted a token")
                continue
            if token is None:
                raise RuntimeError("Final prefill or decode slot omitted a token")
            request.generated.append(token)
            at_ns = time.monotonic_ns()
            request.token_times_ns.append(at_ns)
            reason = "stop" if token in request.stop_token_ids else None
            if reason is None and request.stop_strings and self.decode_tokens is not None:
                rendered = self.decode_tokens(request.generated)
                matches = [(rendered.index(value), value) for value in request.stop_strings
                           if value in rendered]
                if matches:
                    request.stop_match = min(matches)[1]
                    reason = "stop"
            if reason is None and len(request.generated) >= request.max_tokens:
                reason = "length"
            request.events.put(("token", {"token_id": token, "time_ns": at_ns,
                                          "finish_reason": reason,
                                          "generated": tuple(request.generated),
                                          "stop_match": request.stop_match}))
            if reason is not None:
                request.finish_reason = reason
                self._finish(request, "done", {"finish_reason": reason,
                                               "stop_match": request.stop_match,
                                               "metrics": request.metrics()})
                finished.append(request)
        if finished:
            ids = {request.request_id for request in finished}
            self.active = [request for request in self.active if request.request_id not in ids]

    def _run(self) -> None:
        try:
            self.runner.start()
            self.ready.set()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pypto-device") as self.executor:
                while not self.stopping.is_set():
                    self._step()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            try:
                for request in self.active:
                    try:
                        self._finish(request, "error", "Batch scheduler stopped")
                    except BaseException as exc:
                        self.error = self.error or exc
                self.active.clear()
                while True:
                    try:
                        request = self.incoming.get_nowait()
                    except queue.Empty:
                        break
                    if request is not None:
                        request.events.put(("error", "Batch scheduler stopped"))
                with self.lock:
                    while self.pending:
                        self.pending.popleft().events.put(("error", "Batch scheduler stopped"))
            finally:
                try:
                    self.runner.close()
                except BaseException as exc:
                    self.error = self.error or exc
