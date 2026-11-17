"""Deployment contract and request-local sampling for the TP2 engine."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ServingConfig:
    model_root: Path
    devices: tuple[int, int]
    max_model_len: int = 133120
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 32
    speculative_tokens: int = 3
    prefix_cache: bool = True
    served_model_name: str = "qwen3.8-27B"
    long_prefill_token_threshold: int = 0

    def __post_init__(self):
        if type(self.long_prefill_token_threshold) is not int or not 0 <= self.long_prefill_token_threshold <= self.max_model_len:
            raise ValueError("long_prefill_token_threshold must be a non-negative integer <= max_model_len (0: budget-driven)")
        if len(self.devices) != 2 or len(set(self.devices)) != 2:
            raise ValueError("TP2 requires two distinct assigned logical devices")
        if not self.model_root.is_dir():
            raise ValueError("Model directory does not exist")
        if not 1 <= self.max_num_seqs <= 32:
            raise ValueError("max_num_seqs must be in [1, 32]")
        if not 1 <= self.max_model_len <= 262144:
            raise ValueError("max_model_len exceeds the checkpoint position range")
        if self.max_num_batched_tokens < self.max_num_seqs * (self.speculative_tokens + 1):
            raise ValueError("Token budget must accommodate one verification step per slot")
        if self.speculative_tokens not in (0, 3):
            raise ValueError("This deployment supports MTP3, or disabled diagnostic mode")

    @classmethod
    def from_env(cls):
        if int(os.environ.get("TP", "2")) != 2 or int(os.environ.get("DP", "1")) != 1:
            raise ValueError("This model package targets TP2/DP1")
        if int(os.environ.get("EP", "1")) != 1:
            raise ValueError("This dense model requires EP1")
        if os.environ.get("SPECULATIVE_METHOD", "qwen3_5_mtp") != "qwen3_5_mtp":
            raise ValueError("Unsupported speculative method")
        if os.environ.get("PREFIX_CACHE", "1") not in ("0", "1"):
            raise ValueError("PREFIX_CACHE must be 0 or 1")
        assignment = os.environ.get("TASK_DEVICE")
        if not assignment:
            raise ValueError("Explicit assigned devices are required")
        try:
            threshold = int(os.environ.get("LONG_PREFILL_TOKEN_THRESHOLD", "0"))
        except ValueError as error:
            raise ValueError("LONG_PREFILL_TOKEN_THRESHOLD must be a non-negative integer <= max_model_len (0: budget-driven)") from error
        return cls(Path(os.environ["MODEL_ROOT"]), tuple(map(int, assignment.split(","))),
                   int(os.environ.get("MAX_MODEL_LEN", "133120")),
                   int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "16384")),
                   int(os.environ.get("MAX_NUM_SEQS", "32")),
                   int(os.environ.get("NUM_SPECULATIVE_TOKENS", "3")),
                   os.environ.get("PREFIX_CACHE", "1") == "1",
                   os.environ.get("SERVED_MODEL_NAME", "qwen3.8-27B"), threshold)


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("seed must be an integer in [0, 2**63)")


def probabilities(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Apply each request's filters to gathered CPU logits, then normalize."""
    if logits.device.type != "cpu" or logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("Sampling requires finite one-dimensional CPU logits")
    values = logits.float()
    if params.temperature == 0:
        result = torch.zeros_like(values)
        result[int(values.argmax())] = 1
        return result
    values = values / params.temperature
    if params.top_k is not None and params.top_k < values.numel():
        order = torch.argsort(values, descending=True, stable=True)[:params.top_k]
        filtered = torch.full_like(values, -torch.inf)
        filtered[order] = values[order]
        values = filtered
    result = torch.softmax(values, dim=0)
    if params.top_p < 1:
        ordered, indices = torch.sort(result, descending=True, stable=True)
        keep = (torch.cumsum(ordered, dim=0) - ordered) < params.top_p
        ordered = torch.where(keep, ordered, 0)
        result = torch.zeros_like(result).scatter_(0, indices, ordered)
        result /= result.sum()
    return result


def draw(probs: torch.Tensor, generator: torch.Generator) -> int:
    return int(torch.multinomial(probs, 1, generator=generator))


def verify_drafts(target: list[torch.Tensor], proposals: list[torch.Tensor],
                  draft_ids: list[int], params: SamplingParams,
                  generator: torch.Generator) -> tuple[list[int], int]:
    """Exact speculative rejection sampling; count excludes the input anchor.

    target[i] predicts draft_ids[i]; the final target predicts a bonus token.
    Proposal distributions already include the same request sampling filters.
    """
    if len(target) != len(draft_ids) + 1 or len(proposals) != len(draft_ids):
        raise ValueError("Speculative probability layout mismatch")
    accepted = []
    for index, token in enumerate(draft_ids):
        p, q = target[index], proposals[index]
        if params.temperature == 0:
            valid = token == int(p.argmax())
        else:
            q_token = float(q[token])
            if q_token <= 0:
                raise ValueError("A draft token has zero proposal probability")
            valid = float(torch.rand((), generator=generator)) < min(1., float(p[token]) / q_token)
        if not valid:
            if params.temperature == 0:
                replacement = int(p.argmax())
            else:
                residual = torch.clamp(p - q, min=0)
                mass = float(residual.sum())
                if mass <= 0:
                    raise RuntimeError("Rejected proposal has no residual probability mass")
                replacement = draw(residual / mass, generator)
            return accepted + [replacement], len(accepted)
        accepted.append(token)
    bonus = int(target[-1].argmax()) if params.temperature == 0 else draw(target[-1], generator)
    return accepted + [bonus], len(accepted)
