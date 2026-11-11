"""Phase rotation without changing the public request/stop/HTTP contract."""
import contextvars
import time
from concurrent.futures import TimeoutError as FutureTimeout

from .schedule import Scheduler as BaseScheduler, BatchSlot, RequestState, RequestCapacityError
from .model import ROWS
from .scheduling_policy import PrefillDecodePolicy, select_short_mixed


class Scheduler(BaseScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefill_decode_policy = PrefillDecodePolicy()
        self.current_step_isolated = False
        self.short_selection = None

    def _select_step_requests(self):
        self.short_selection = None
        if getattr(self.runner, 'enable_mixed_prefill', False):
            self.short_selection = select_short_mixed(self.active, self.max_active)
        if self.short_selection is not None:
            self.current_step_isolated = True
            self.prefill_decode_policy.last_phase = (
                'decode' if self.short_selection[0] == 'mixed128' else 'prefill')
            return list(self.short_selection[1])
        self.current_step_isolated = any(
            len(q.prompt_ids) - q.prefill_offset > 128 for q in self.active)
        if self.current_step_isolated:
            return self.prefill_decode_policy.choose(self.active, 'de59', self.max_active)
        return list(self.active[:self.max_active])

    def _slot(self, request: RequestState) -> BatchSlot:
        remaining = len(request.prompt_ids) - request.prefill_offset
        if remaining and (self.short_selection is not None or self.current_step_isolated):
            position = request.prefill_offset
            rows = 128 if self.short_selection is not None else (512 if remaining > 64 else ROWS)
            tokens = tuple(request.prompt_ids[position:position + rows])
            return BatchSlot(request, tokens, position, 'prefill',
                             position + len(tokens) == len(request.prompt_ids))
        if request.prefill_offset < len(request.prompt_ids):
            position = request.prefill_offset
            window = 64 if len(request.prompt_ids) - position > ROWS else ROWS
            tokens = tuple(request.prompt_ids[position:position + window])
            return BatchSlot(request, tokens, position, "prefill",
                             position + len(tokens) == len(request.prompt_ids))
        return BatchSlot(request, (request.generated[-1],),
                         len(request.prompt_ids) + len(request.generated) - 1,
                         "decode", False)

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
        slots = [self._slot(request) for request in self._select_step_requests()]
        self.batch_number += 1
        self._trace("batch_submit", batch=self.batch_number,
                    slots=[{"request_id": slot.request.request_id,
                            "phase": slot.phase, "position": slot.position,
                            "tokens": len(slot.token_ids)} for slot in slots])
        future = None
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
            if future is not None and not future.done():
                try:
                    future.result()
                except BaseException:
                    pass
            self._finish(exc.request, "error", str(exc))
            self.active = [request for request in self.active if request is not exc.request]
            return
        except BaseException as exc:
            if future is not None and not future.done():
                try:
                    future.result()
                except BaseException:
                    pass
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
