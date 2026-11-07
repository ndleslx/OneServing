"""Compare fixed TP1 mixed steps after warmup; this is not an HTTP benchmark."""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

from test_mixed_npu import MODEL_ROOT, MixedNPUTests

if TYPE_CHECKING:
    from serving.runtime import ModelRunner
    from serving.schedule import BatchSlot, RequestState


def benchmark(runner: ModelRunner, batch_slot: type[BatchSlot],
              request_state: type[RequestState], mixed: bool,
              prefills: int, decodes: int, chunk: int, steps: int, warmup: int) -> None:
    runner.enable_mixed = mixed
    requests = []
    try:
        decode_requests = [request_state([5 + index], steps + 2, frozenset())
                           for index in range(decodes)]
        for request in decode_requests:
            runner.admit(request)
            requests.append(request)
        runner.run_batch([batch_slot(request, (5 + index,), 0, "prefill", True)
                          for index, request in enumerate(decode_requests)])
        prefill_requests = [request_state([2 + index] * (steps * chunk + 1), 2, frozenset())
                           for index in range(prefills)]
        for request in prefill_requests:
            runner.admit(request)
            requests.append(request)
        samples = []
        for step in range(steps):
            slots = [batch_slot(request, (2 + index,) * chunk, step * chunk, "prefill", False)
                     for index, request in enumerate(prefill_requests)]
            slots += [batch_slot(request, (6 + index,), step + 1, "decode", False)
                      for index, request in enumerate(decode_requests)]
            start = time.monotonic()
            runner.run_batch(slots)
            elapsed = time.monotonic() - start
            if step >= warmup:
                samples.append(elapsed)
        mean = sum(samples) / len(samples)
        print(f"{'mixed' if mixed else 'baseline'}: {prefills}P+{decodes}D, "
              f"mean_step_ms={mean * 1000:.3f}, "
              f"input_tokens_per_second={(prefills * chunk + decodes) / mean:.3f}, "
              f"samples={len(samples)}")
    finally:
        for request in requests:
            runner.release(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefills", type=int, default=4)
    parser.add_argument("--decodes", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if MODEL_ROOT is None:
        parser.error("Set PYPTO_QWEN_TEST_MODEL_ROOT and PYPTO_QWEN_TEST_DEVICE")
    from serving.model import BATCH, CONTEXT, ROWS
    if min(args.prefills, args.decodes) < 1 or args.prefills + args.decodes > BATCH:
        parser.error(f"Use at least one P and one D, with at most {BATCH} requests")
    if not 1 <= args.chunk <= ROWS or not 0 <= args.warmup < args.steps:
        parser.error("Invalid chunk size or warmup/step count")
    if args.steps * args.chunk + 1 > CONTEXT:
        parser.error("The prefill workload exceeds the supported context")
    MixedNPUTests.setUpClass()
    try:
        for use_mixed in (False, True):
            benchmark(MixedNPUTests.runner, MixedNPUTests.BatchSlot,
                      MixedNPUTests.RequestState, use_mixed, args.prefills, args.decodes,
                      args.chunk, args.steps, args.warmup)
    finally:
        MixedNPUTests.tearDownClass()
