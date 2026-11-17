"""Dynamic request admission and mixed prefill/decode batch scheduling."""

from __future__ import annotations

import contextvars
import queue
import secrets
import torch
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .distributed import SamplingParams


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
    sampling: SamplingParams = field(default_factory=SamplingParams)
    generator: torch.Generator = field(default_factory=lambda: torch.Generator().manual_seed(secrets.randbits(63)))
    detokenizer: Any = None
    text: str = ""

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
    def run_batch(self, slots: list[BatchSlot]) -> list[list[int] | None]: ...
    def release(self, request: RequestState) -> None: ...


class Scheduler:
    def __init__(self, runner: BatchRunner, stop_token_ids: frozenset[int],
                 max_active: int = 32, max_queued: int = 128,
                 max_model_len: int = 133120, token_budget: int = 16384,
                 detokenizer_factory=None,
                 on_trace: Callable[[str, dict], None] | None = None,
                 decode_tokens: Callable[[list[int]], str] | None = None,
                 long_prefill_token_threshold: int = 512,
                 speculative_tokens: int = 3):
        if not (1 <= max_active <= 32):
            raise ValueError("The device slot capacity is 32 requests")
        if (type(long_prefill_token_threshold) is not int or
                not 0 <= long_prefill_token_threshold <= max_model_len):
            raise ValueError("long_prefill_token_threshold must be a nonnegative integer <= max_model_len")
        if type(speculative_tokens) is not int or speculative_tokens not in (0, 3):
            raise ValueError("speculative_tokens must be 0 or 3")
        if type(token_budget) is not int or token_budget < max_active * (1 + speculative_tokens):
            raise ValueError("Token budget must cover every active decode/verification request")
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.speculative_tokens = speculative_tokens
        self._prefill_order: deque[str] = deque()
        self.max_model_len = max_model_len
        self.token_budget = token_budget
        self.detokenizer_factory = detokenizer_factory
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
        if not self.ready.wait(timeout=1800):
            raise RuntimeError("Batch runner startup timed out")
        if self.error is not None:
            raise RuntimeError("Batch runner failed to start") from self.error

    def submit(self, prompt_ids: list[int], max_tokens: int,
               stop_strings: tuple[str, ...] = (),
               sampling: SamplingParams | None = None,
               ignore_eos: bool = False) -> RequestState:
        if not self.ready.is_set() or self.stopping.is_set() or self.error is not None:
            raise RuntimeError("Batch scheduler is unavailable")
        if not prompt_ids or max_tokens < 1 or len(prompt_ids) + max_tokens > self.max_model_len:
            raise ValueError("Prompt and completion exceed the supported context")
        request = RequestState(prompt_ids, max_tokens, frozenset() if ignore_eos else self.stop_token_ids,
                               stop_strings=stop_strings, sampling=sampling or SamplingParams())
        if request.sampling.seed is not None:
            request.generator.manual_seed(request.sampling.seed)
        if self.detokenizer_factory is not None:
            request.detokenizer = self.detokenizer_factory()
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
                if getattr(self.runner, "terminal_device_error", None) is not None:
                    raise  # Fatal devicefailure must reach owner-thread finallyclose.

    def _prefill_limit(self, request: RequestState) -> int:
        remaining = len(request.prompt_ids) - request.prefill_offset
        limit = min(self.long_prefill_token_threshold, remaining) if self.long_prefill_token_threshold else remaining
        # A miss must reach the committed prefix checkpoint before continuing.
        # This boundary is independent of the model's internal 16-row tiles.
        if request.prefill_offset < 32:
            limit = min(limit, 32 - request.prefill_offset)
        return limit

    def _decode_cost(self, request: RequestState) -> int:
        remaining = max(0, request.max_tokens - len(request.generated) - 1)
        return 1 + min(self.speculative_tokens, remaining)

    def _slot(self, request: RequestState, prefill_quota: int | None = None) -> BatchSlot:
        if request.prefill_offset < len(request.prompt_ids):
            position = request.prefill_offset
            quota = self._prefill_limit(request)
            if prefill_quota is not None:
                quota = min(quota, prefill_quota)
            if quota <= 0:
                raise RuntimeError("Scheduled prefill requires a positive token allocation")
            tokens = tuple(request.prompt_ids[position:position + quota])
            return BatchSlot(request, tokens, position, "prefill",
                             position + len(tokens) == len(request.prompt_ids))
        return BatchSlot(request, (request.generated[-1],),
                         len(request.prompt_ids) + len(request.generated) - 1,
                         "decode", False)

    def _batch_slots(self, requests: list[RequestState]) -> list[BatchSlot]:
        prefills = [r for r in requests if r.prefill_offset < len(r.prompt_ids)]
        reserved = sum(self._decode_cost(r) for r in requests if r not in prefills)
        available = self.token_budget - reserved
        if available < 0:
            raise RuntimeError("Decode/verification reservation exceeds the token budget")
        # Keep a stable round-robin queue across admission, completion and cancel.
        # A selected request receives its full legal chunk before the next one;
        # only the last grant may be limited by the remaining global budget.
        # This permits large chunks without reducing resident request capacity.
        by_id = {r.request_id: r for r in prefills}
        self._prefill_order = deque(key for key in self._prefill_order if key in by_id)
        known = set(self._prefill_order)
        self._prefill_order.extend(r.request_id for r in prefills if r.request_id not in known)
        quotas = {}
        allocation_trace = []
        for _ in range(len(self._prefill_order)):
            if available <= 0:
                break
            key = self._prefill_order.popleft()
            request = by_id[key]
            remaining = len(request.prompt_ids) - request.prefill_offset
            limit = self._prefill_limit(request)
            grant = min(limit, available)
            reasons = []
            if grant == remaining:
                reasons.append('remaining_prompt')
            if self.long_prefill_token_threshold and grant == self.long_prefill_token_threshold:
                reasons.append('positive_threshold')
            if request.prefill_offset < 32 and grant == 32 - request.prefill_offset:
                reasons.append('prefix_checkpoint')
            if grant == available:
                reasons.append('global_budget')
            allocation_trace.append({'request_id': key, 'position': request.prefill_offset,
                'remaining_prompt': remaining, 'legal_request_limit': limit,
                'budget_before': available, 'tokens': grant,
                'budget_after': available - grant, 'limiting_factors': reasons})
            quotas[key] = grant
            available -= grant
            self._prefill_order.append(key)
        allocation = list(quotas.values())
        slots = [self._slot(r, quotas.get(r.request_id)) for r in requests
                 if r.request_id not in by_id or r.request_id in quotas]
        target_tokens = reserved + sum(allocation)
        if target_tokens > self.token_budget:
            raise RuntimeError("Scheduler exceeded its target-model token budget")
        self._trace("token_budget", decode_verification_reserved=reserved,
                    prefill_tokens=sum(allocation), target_tokens=target_tokens,
                    budget_before_prefill=self.token_budget - reserved,
                    budget_after_prefill=available, allocation_order=allocation_trace,
                    unselected_request_ids=[r.request_id for r in prefills if r.request_id not in quotas],
                    prefill_chunk_mode='budget' if self.long_prefill_token_threshold == 0 else 'capped',
                    resident_requests=len(requests), selected_prefills=len(quotas),
                    deferred_prefills=len(prefills) - len(quotas),
                    max_num_batched_tokens=self.token_budget,
                    long_prefill_token_threshold=self.long_prefill_token_threshold)
        return slots

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
        slots = self._batch_slots(self.active[:self.max_active])
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
                    if future.done():
                        # Distinguish a completed native TimeoutError from a poll
                        # that raced successful completion. This reads the SAME
                        # Future result; it never resubmits device work.
                        outputs = future.result()
                        break
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
        for slot, tokens in zip(slots, outputs):
            request = slot.request
            if request.cancelled.is_set():
                self._finish(request, "cancelled", request.metrics())
                finished.append(request)
                continue
            if slot.phase == "prefill":
                request.prefill_offset += len(slot.token_ids)
            if slot.phase == "prefill" and not slot.completes_prefill:
                if tokens is not None:
                    raise RuntimeError("Intermediate prefill chunk emitted a token")
                continue
            if not isinstance(tokens, list) or not tokens:
                raise RuntimeError("Final prefill or decode omitted committed token IDs")
            for token in tokens:
                request.generated.append(token)
                at_ns = time.monotonic_ns()
                request.token_times_ns.append(at_ns)
                reason = "stop" if token in request.stop_token_ids else None
                if request.detokenizer is not None:
                    request.text += request.detokenizer.push(token)
                elif self.decode_tokens is not None:
                    request.text = self.decode_tokens(request.generated)
                if reason is None and request.stop_strings:
                    matches = [(request.text.index(value), value) for value in request.stop_strings
                               if value in request.text]
                    if matches:
                        request.stop_match = min(matches)[1]
                        reason = "stop"
                if reason is None and len(request.generated) >= request.max_tokens:
                    reason = "length"
                if reason is not None and request.detokenizer is not None:
                    request.text += request.detokenizer.finish()
                request.events.put(("token", {"token_id": token, "time_ns": at_ns,
                                              "finish_reason": reason,
                                              "generated": tuple(request.generated),
                                              "text": request.text,
                                              "stop_match": request.stop_match}))
                if reason is not None:
                    request.finish_reason = reason
                    self._finish(request, "done", {"finish_reason": reason,
                                                   "stop_match": request.stop_match,
                                                   "metrics": request.metrics()})
                    finished.append(request)
                    break
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
