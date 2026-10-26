"""Independent two-device data replicas and request routing."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .engine import InferenceEngine, percentile
from .schedule import Scheduler
from .runtime import TensorParallelRunner


class TensorParallelEngine(InferenceEngine):
    def __init__(self, model_root: Path, device_ids: tuple[int, int],
                 max_active: int = 2, prefix_cache: bool = True):
        if len(device_ids) != 2 or len(set(device_ids)) != 2:
            raise ValueError("TP=2 requires two distinct assigned devices")
        super().__init__(model_root, device_ids[0], max_active=max_active,
                         prefix_cache=prefix_cache)
        self.device_id = list(device_ids)
        self.runner = TensorParallelRunner(model_root, device_ids,
                                           on_cache_event=self._on_cache)
        for shard in self.runner.shards:
            shard.allow_prefix_hits = prefix_cache
        self.scheduler = Scheduler(self.runner, self.scheduler.stop_token_ids,
                                   max_active=max_active, on_trace=self._on_trace,
                                   decode_tokens=self.decode)

    def snapshot(self) -> dict:
        with self.lock:
            completed = list(self.completed)
            recent = list(self.events)[-32:]
            errors, cancelled = self.errors, self.cancelled
        timing = {}
        for name in ("ttft_ms", "tpot_ms"):
            values = [float(record[name]) for record in completed if record[name] is not None]
            timing[name] = {"p50": percentile(values, 0.50),
                            "p95": percentile(values, 0.95),
                            "p99": percentile(values, 0.99)}
        return {"devices": self.device_id, "scheduler": self.scheduler.counts(),
                "requests": {"completed": len(completed), "cancelled": cancelled,
                             "errors": errors}, "timing_ms": timing,
                "prefix_cache_enabled": self.runner.shards[0].allow_prefix_hits,
                "tp": self.runner.stats(), "recent_requests": completed[-32:],
                "recent_events": recent}


class ReplicaRouter:
    def __init__(self, model_root: Path, device_ids: tuple[int, int],
                 max_active: int = 2, prefix_cache: bool = True):
        if len(device_ids) != 2 or len(set(device_ids)) != 2:
            raise ValueError("DP=2 requires two distinct assigned devices")
        self.device_ids = device_ids
        self.device_id = list(device_ids)
        self.replicas = [InferenceEngine(model_root, device,
                                         max_active=max_active,
                                         prefix_cache=prefix_cache) for device in device_ids]
        self.lock = threading.Lock()
        self.next_replica = 0
        self.routes: list[dict] = []
        self.stopping = threading.Event()
        self.scheduler = self

    @property
    def error(self):
        return next((replica.scheduler.error for replica in self.replicas
                     if replica.scheduler.error is not None), None)

    def start(self) -> None:
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda replica: replica.start(), self.replicas))
            assert results == [None, None]
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        self.stopping.set()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda replica: replica.stop(), self.replicas))

    def counts(self) -> dict[str, int]:
        counts = [replica.scheduler.counts() for replica in self.replicas]
        return {key: sum(item[key] for item in counts)
                for key in ("active", "queued", "batches")}

    def decode(self, token_ids: list[int]) -> str:
        return self.replicas[0].decode(token_ids)

    def _dispatch(self, method: str, *args):
        with self.lock:
            index = self.next_replica
            self.next_replica = (index + 1) % len(self.replicas)
        request = getattr(self.replicas[index], method)(*args)
        with self.lock:
            self.routes.append({"request_id": request.request_id,
                                "replica": index, "device": self.device_ids[index]})
        return request

    def submit_tokens(self, prompt_ids: list[int], max_tokens: int,
                      stop_strings: tuple[str, ...] = ()):
        return self._dispatch("submit_tokens", prompt_ids, max_tokens, stop_strings)

    def submit_text(self, prompt: str, max_tokens: int,
                    stop_strings: tuple[str, ...] = ()):
        return self._dispatch("submit_text", prompt, max_tokens, stop_strings)

    def submit_chat(self, messages: list[dict[str, str]], max_tokens: int,
                    stop_strings: tuple[str, ...] = ()):
        return self._dispatch("submit_chat", messages, max_tokens, stop_strings)

    def snapshot(self) -> dict:
        replicas = [replica.snapshot() for replica in self.replicas]
        records = []
        for replica in self.replicas:
            with replica.lock:
                records.extend(list(replica.completed))
        timing = {}
        for name in ("ttft_ms", "tpot_ms"):
            values = [float(record[name]) for record in records if record[name] is not None]
            timing[name] = {"p50": percentile(values, 0.50),
                            "p95": percentile(values, 0.95),
                            "p99": percentile(values, 0.99)}
        with self.lock:
            routes = list(self.routes[-64:])
        return {"devices": list(self.device_ids), "scheduler": self.counts(),
                "prefix_cache_enabled": self.replicas[0].runner.allow_prefix_hits,
                "requests": {key: sum(item["requests"][key] for item in replicas)
                             for key in ("completed", "cancelled", "errors")},
                "timing_ms": timing, "recent_requests": records[-32:],
                "routes": routes, "replicas": replicas}
