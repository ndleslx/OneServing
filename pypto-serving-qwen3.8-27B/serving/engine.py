"""Request lifecycle, tokenization, and service metrics."""

from __future__ import annotations

import threading
import os
import json
from collections import deque
from pathlib import Path

from transformers import AutoTokenizer

from .distributed import ServingConfig, SamplingParams
from .schedule import RequestState, Scheduler


MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-27B")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class InferenceEngine:
    def __init__(self, config: ServingConfig):
        from .runtime import TensorParallelRunner
        self.config = config
        self.model_root = config.model_root
        self.device_id = list(config.devices)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_root, local_files_only=True, trust_remote_code=False)
        eos = self.tokenizer.eos_token_id
        generation = json.loads((config.model_root / "generation_config.json").read_text())
        eos = generation.get("eos_token_id", eos)
        eos_ids = frozenset({eos} if isinstance(eos, int) else (eos or []))
        self.lock = threading.Lock()
        self.events: deque[dict] = deque(maxlen=10000)
        self.completed: deque[dict] = deque(maxlen=10000)
        self.errors = 0
        self.cancelled = 0
        self.runner = TensorParallelRunner(config, on_cache_event=self._on_cache)
        self.runner.allow_prefix_hits = config.prefix_cache
        self.scheduler = Scheduler(self.runner, eos_ids, max_active=config.max_num_seqs,
                                   max_model_len=config.max_model_len,
                                   token_budget=config.max_num_batched_tokens,
                                   long_prefill_token_threshold=config.long_prefill_token_threshold,
                                   speculative_tokens=config.speculative_tokens,
                                   on_trace=self._on_trace, decode_tokens=self.decode,
                                   detokenizer_factory=lambda: IncrementalDetokenizer(self.tokenizer))

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
                      stop_strings: tuple[str, ...] = (),
                      sampling: SamplingParams | None = None,
                      ignore_eos: bool = False) -> RequestState:
        if any(not isinstance(token, int) or not 0 <= token < 248320 for token in prompt_ids):
            raise ValueError("Prompt contains an invalid token ID")
        if any(not value for value in stop_strings):
            raise ValueError("Stop strings must be nonempty")
        return self.scheduler.submit(prompt_ids, max_tokens, stop_strings, sampling, ignore_eos)

    def submit_text(self, prompt: str, max_tokens: int,
                    stop_strings: tuple[str, ...] = (),
                    sampling: SamplingParams | None = None,
                    ignore_eos: bool = False) -> RequestState:
        return self.submit_tokens(self.tokenizer.encode(prompt, add_special_tokens=False),
                                  max_tokens, stop_strings, sampling, ignore_eos)

    def submit_chat(self, messages: list[dict[str, str]], max_tokens: int,
                    stop_strings: tuple[str, ...] = (),
                    sampling: SamplingParams | None = None,
                    ignore_eos: bool = False) -> RequestState:
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        prompt_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return self.submit_tokens(prompt_ids, max_tokens, stop_strings, sampling, ignore_eos)

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
                "recent_events": recent, "model_execution": self.runner.stats()}


class IncrementalDetokenizer:
    """Decode new tokenizer pieces with a bounded committed context window.

    Incomplete UTF-8 suffixes stay pending until the next piece arrives. The
    retained prefix provides the decoder context without decoding all prior IDs.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pieces: list[str] = []
        self.prefix = 0
        self.read = 0
        self.closed = False

    def push(self, token_id: int) -> str:
        if self.closed:
            raise RuntimeError("Detokenizer is closed")
        token = self.tokenizer.convert_ids_to_tokens([token_id], skip_special_tokens=True)
        self.pieces.extend(token)
        before = self.tokenizer.convert_tokens_to_string(self.pieces[self.prefix:self.read])
        after = self.tokenizer.convert_tokens_to_string(self.pieces[self.prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            if not after.startswith(before):
                raise RuntimeError("Tokenizer changed an already committed prefix")
            delta = after[len(before):]
            previous_read = self.read
            self.read = len(self.pieces)
            self.prefix = previous_read
            if self.prefix > 64:
                self.pieces = self.pieces[self.prefix:]
                self.read -= self.prefix
                self.prefix = 0
            return delta
        return ""

    def finish(self) -> str:
        if self.closed:
            return ""
        self.closed = True
        before = self.tokenizer.convert_tokens_to_string(self.pieces[self.prefix:self.read])
        after = self.tokenizer.convert_tokens_to_string(self.pieces[self.prefix:])
        if not after.startswith(before):
            raise RuntimeError("Tokenizer changed final committed prefix")
        return after[len(before):]
