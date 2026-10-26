"""Request lifecycle, tokenization, and service metrics."""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

from transformers import AutoTokenizer

from .runtime import ModelRunner
from .schedule import RequestState, Scheduler


MODEL_NAME = "Qwen3.8-27B-PyPTO"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class InferenceEngine:
    def __init__(self, model_root: Path, device_id: int,
                 max_active: int = 2, prefix_cache: bool = True):
        if not 1 <= max_active <= 2:
            raise ValueError("This runtime has validated at most two active device slots")
        self.model_root = model_root
        self.device_id = device_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_root, local_files_only=True, trust_remote_code=False)
        eos = self.tokenizer.eos_token_id
        eos_ids = frozenset({eos} if isinstance(eos, int) else (eos or []))
        self.lock = threading.Lock()
        self.events: deque[dict] = deque(maxlen=10000)
        self.completed: deque[dict] = deque(maxlen=10000)
        self.errors = 0
        self.cancelled = 0
        self.runner = ModelRunner(model_root, device_id,
                                  on_cache_event=self._on_cache)
        self.runner.allow_prefix_hits = prefix_cache
        self.scheduler = Scheduler(self.runner, eos_ids, max_active=max_active,
                                   on_trace=self._on_trace,
                                   decode_tokens=self.decode)

    def _on_cache(self, name: str, fields: dict) -> None:
        with self.lock:
            self.events.append({"event": "cache_" + name, **fields})

    def _on_trace(self, name: str, fields: dict) -> None:
        with self.lock:
            self.events.append({"event": name, **fields})
            if name == "finished":
                if fields["reason"] == "done":
                    self.completed.append({"request_id": fields["request_id"],
                                           **fields["metrics"]})
                elif fields["reason"] == "cancelled":
                    self.cancelled += 1
                else:
                    self.errors += 1

    def start(self) -> None:
        self.scheduler.start()

    def stop(self) -> None:
        self.scheduler.stop()

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def submit_tokens(self, prompt_ids: list[int], max_tokens: int,
                      stop_strings: tuple[str, ...] = ()) -> RequestState:
        if any(not isinstance(token, int) or not 0 <= token < 248320 for token in prompt_ids):
            raise ValueError("Prompt contains an invalid token ID")
        if any(not value for value in stop_strings):
            raise ValueError("Stop strings must be nonempty")
        return self.scheduler.submit(prompt_ids, max_tokens, stop_strings)

    def submit_text(self, prompt: str, max_tokens: int,
                    stop_strings: tuple[str, ...] = ()) -> RequestState:
        return self.submit_tokens(self.tokenizer.encode(prompt, add_special_tokens=False),
                                  max_tokens, stop_strings)

    def submit_chat(self, messages: list[dict[str, str]], max_tokens: int,
                    stop_strings: tuple[str, ...] = ()) -> RequestState:
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        prompt_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return self.submit_tokens(prompt_ids, max_tokens, stop_strings)

    def snapshot(self) -> dict:
        with self.lock:
            completed = list(self.completed)
            recent = list(self.events)[-32:]
            errors, cancelled = self.errors, self.cancelled
        values = {name: [float(item[name]) for item in completed if item[name] is not None]
                  for name in ("ttft_ms", "tpot_ms")}
        timing = {name: {"p50": percentile(items, 0.50),
                         "p95": percentile(items, 0.95),
                         "p99": percentile(items, 0.99)}
                  for name, items in values.items()}
        return {"device": self.device_id, "scheduler": self.scheduler.counts(),
                "prefix_cache_enabled": self.runner.allow_prefix_hits,
                "requests": {"completed": len(completed), "cancelled": cancelled,
                             "errors": errors}, "timing_ms": timing,
                "cache": self.runner.cache.stats(), "memory": self.runner.memory_stats,
                "provenance": self.runner.provenance,
                "recent_requests": completed[-32:],
                "recent_events": recent}
