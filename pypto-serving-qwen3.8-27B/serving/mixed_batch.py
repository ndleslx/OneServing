"""Host metadata for the TP1 mixed graph (no device dependencies)."""

from itertools import accumulate
from typing import Literal, Sequence


def select_tp1_graph(
    phases: Sequence[str], token_counts: Sequence[int], mixed_enabled: bool,
    *, capacity: int, rows: int,
) -> Literal["prefill", "decode", "mixed"]:
    if not phases or len(phases) != len(token_counts):
        raise ValueError("Batch phases and token counts must be nonempty and aligned")
    if len(phases) > capacity:
        raise ValueError(f"Batch has {len(phases)} slots; maximum is {capacity}")
    if any(phase not in ("prefill", "decode") for phase in phases):
        raise ValueError("Invalid batch phase")
    if any(not 1 <= count <= rows for count in token_counts):
        raise ValueError(f"Batch slots must contain between 1 and {rows} tokens")
    if any(phase == "decode" and count != 1 for phase, count in zip(phases, token_counts)):
        raise ValueError("Decode slots must contain exactly one token")
    has_prefill = "prefill" in phases
    if mixed_enabled:
        return "mixed"
    return "prefill" if has_prefill else "decode"


def packed_token_offsets(token_counts: Sequence[int]) -> tuple[int, ...]:
    """Exclusive request boundaries in the dense token buffer, including its end."""
    return tuple(accumulate(token_counts, initial=0))
