"""Opt-in end-to-end comparison against the existing TP1 model graph."""

import os
from pathlib import Path
import unittest
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch


MODEL_ROOT = os.environ.get("PYPTO_QWEN_TEST_MODEL_ROOT")


@unittest.skipUnless(MODEL_ROOT, "set PYPTO_QWEN_TEST_MODEL_ROOT for the NPU test")
class MixedNPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODEL_ROOT is None:
            raise unittest.SkipTest("set PYPTO_QWEN_TEST_MODEL_ROOT for the NPU test")
        import torch
        from serving.model import BLOCK_SIZE, POOL_BLOCKS
        from serving.runtime import ModelRunner
        from serving.schedule import BatchSlot, RequestState

        cls.torch = torch
        cls.block_size = BLOCK_SIZE
        cls.pool_blocks = POOL_BLOCKS
        cls.BatchSlot = BatchSlot
        cls.RequestState = RequestState
        cls.previous_setting = os.environ.get("PYPTO_QWEN_MIXED")
        os.environ["PYPTO_QWEN_MIXED"] = "1"
        cls.runner = ModelRunner(Path(MODEL_ROOT), int(os.environ.get("PYPTO_QWEN_TEST_DEVICE", "0")))
        cls.runner.allow_prefix_hits = False
        try:
            cls.runner.start(include_reference_graphs=True)
        except BaseException:
            cls.runner.close()
            cls.restore_setting()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            cls.runner.close()
        finally:
            cls.restore_setting()

    @classmethod
    def restore_setting(cls):
        if cls.previous_setting is None:
            os.environ.pop("PYPTO_QWEN_MIXED", None)
        else:
            os.environ["PYPTO_QWEN_MIXED"] = cls.previous_setting

    def setUp(self):
        self.original_run_batch = self.runner.run_batch
        poison = [(handle, self.torch.full(tuple(handle.shape), float("nan"), dtype=handle.dtype))
                  for handle in (self.runner.traces, self.runner.logits)]

        def run_checked(slots):
            for handle, host in poison:
                self.runner.worker.copy_to(handle.data_ptr, host.data_ptr(),
                                           host.numel() * host.element_size())
            return self.original_run_batch(slots)

        self.runner.run_batch = run_checked

    def tearDown(self):
        self.runner.run_batch = self.original_run_batch

    def read_kv(self, request):
        resident = self.runner.requests[request.request_id]
        table = self.runner.cache.table(resident.pages)
        length = resident.pages.length
        blocks = (length + self.block_size - 1) // self.block_size
        block_elements = self.block_size * 256
        tensors = []
        for pool in self.runner.resident[2:4]:
            data = self.torch.empty((16, 4, blocks * self.block_size, 256),
                                    dtype=self.torch.bfloat16)
            for layer in range(16):
                for head in range(4):
                    for block in range(blocks):
                        physical = int(table[block])
                        offset = ((layer * 4 + head) * self.pool_blocks + physical) * block_elements * 2
                        self.runner.worker.copy_from(
                            data[layer, head, block * self.block_size].data_ptr(),
                            pool.data_ptr, block_elements * 2, src_offset=offset)
            tensors.append(data[:, :, :length, :].contiguous())
        return tensors

    def run_case(self, mixed, chunk, history, reverse):
        self.runner.enable_mixed = mixed
        prefill = self.RequestState([2] * (history + chunk + 1), 2, frozenset())
        decode_length = history if history else 1
        decode = self.RequestState([5] * decode_length, 2, frozenset())
        self.runner.admit(decode)
        try:
            if history == 0:
                self.runner.run_batch([self.BatchSlot(decode, (5,), 0, "prefill", False)])
                self.runner.admit(prefill)
            else:
                self.runner.admit(prefill)
                for offset in range(0, history, self.block_size):
                    count = min(self.block_size, history - offset)
                    self.runner.run_batch([
                        self.BatchSlot(prefill, (2,) * count, offset, "prefill", False),
                        self.BatchSlot(decode, (5,) * count, offset, "prefill", False),
                    ])
            slots = [
                self.BatchSlot(prefill, (2,) * chunk, history, "prefill", False),
                self.BatchSlot(decode, (6,), decode_length, "decode", False),
            ]
            if reverse:
                slots.reverse()
            tokens = self.runner.run_batch(slots)
            states = [self.runner._read_state(self.runner.requests[request.request_id].slot)
                      for request in (prefill, decode)]
            kv = [self.read_kv(request) for request in (prefill, decode)]
            logits = [self.runner._logit_row(self.runner.requests[request.request_id].slot, row)
                      for request, row in ((prefill, chunk - 1), (decode, 0))]
            return tokens, states, kv, logits
        finally:
            self.runner.release(prefill)
            self.runner.release(decode)

    def test_mixed_matches_baseline(self):
        for case in ((1, 0, False), (16, 0, True), (7, 17, False)):
            with self.subTest(case=case):
                baseline = self.run_case(False, *case)
                mixed = self.run_case(True, *case)
                self.assert_equivalent(baseline, mixed)

    def run_multi_case(self, mixed, counts, reverse=False):
        self.runner.enable_mixed = mixed
        requests = [self.RequestState([2 + index] * 64, 4, frozenset())
                    for index in range(len(counts))]
        try:
            for request in requests:
                self.runner.admit(request)
            # Seed distinct histories across a page boundary.
            for offset, count in ((0, 16), (16, 1)):
                self.runner.run_batch([
                    self.BatchSlot(request, (2 + index,) * count, offset, "prefill", False)
                    for index, request in enumerate(requests)
                ])
            slots = [self.BatchSlot(request, (2 + index,) * count, 17,
                                    "decode" if count == 1 else "prefill", True)
                     for index, (request, count) in enumerate(zip(requests, counts))]
            if reverse:
                slots.reverse()
            self.runner.run_batch(slots)
            # Remove a request and permute survivors to exercise state compaction
            # after the packed path. The final batch has a partial row tile.
            removed = requests.pop(1)
            self.runner.release(removed)
            slots = list(reversed([
                self.BatchSlot(slot.request, (9,), slot.position + len(slot.token_ids),
                               "prefill" if index == 0 else "decode", True)
                for index, slot in enumerate(slots) if slot.request is not removed
            ]))
            tokens = self.runner.run_batch(slots)
            states, kv, logits = [], [], []
            for slot in slots:
                index = self.runner.requests[slot.request.request_id].slot
                states.append(self.runner._read_state(index))
                kv.append(self.read_kv(slot.request))
                logits.append(self.runner._logit_row(index, 0))
            return tokens, states, kv, logits
        finally:
            for request in requests:
                self.runner.release(request)

    def test_multi_request_packing_and_compaction(self):
        for counts in ((16, 7, 16, 3, 1, 1), (7, 1, 1, 1, 1, 1, 1, 1),
                       (16, 16, 16, 16, 16, 16, 16, 1)):
            for reverse in (False, True):
                with self.subTest(counts=counts, reverse=reverse):
                    self.assert_equivalent(self.run_multi_case(False, counts, reverse),
                                           self.run_multi_case(True, counts, reverse))

    def run_slot_reuse_case(self, mixed):
        self.runner.enable_mixed = mixed
        requests = [self.RequestState([token] * 64, 4, frozenset())
                    for token in (2, 3, 4)]
        try:
            for request in requests:
                self.runner.admit(request)
            guard = (patch.object(self.runner, "_compact", side_effect=AssertionError(
                "Packed execution must preserve resident state slots")) if mixed else nullcontext())
            with guard:
                self.runner.run_batch([
                    self.BatchSlot(request, (token,) * 7, 0, "prefill", False)
                    for request, token in zip(requests, (2, 3, 4))])
                removed = requests.pop(1)
                self.runner.release(removed)
                replacement = self.RequestState([8] * 64, 4, frozenset())
                self.runner.admit(replacement)
                requests.append(replacement)
                expected_slots = {r.request_id: self.runner.requests[r.request_id].slot
                                  for r in requests}
                slots = [self.BatchSlot(requests[1], (4,), 7, "decode", False),
                         self.BatchSlot(replacement, (8,) * 3, 0, "prefill", True),
                         self.BatchSlot(requests[0], (2,), 7, "decode", False)]
                tokens = self.runner.run_batch(slots)
                slots = [self.BatchSlot(slot.request, (9,),
                                       slot.position + len(slot.token_ids), "decode", False)
                         for slot in reversed(slots)]
                tokens = self.runner.run_batch(slots)
            if mixed:
                self.assertEqual(expected_slots, {
                    r.request_id: self.runner.requests[r.request_id].slot for r in requests})
            states = [self.runner._read_state(self.runner.requests[s.request.request_id].slot)
                      for s in slots]
            kv = [self.read_kv(s.request) for s in slots]
            logits = [self.runner._logit_row(self.runner.requests[s.request.request_id].slot,
                                              len(s.token_ids) - 1) for s in slots]
            return tokens, states, kv, logits
        finally:
            for request in requests:
                self.runner.release(request)

    def test_slot_reuse_without_state_compaction(self):
        self.assert_equivalent(self.run_slot_reuse_case(False),
                               self.run_slot_reuse_case(True))

    def run_scheduler_case(self, mixed):
        from serving.mixed_batch import select_tp1_graph
        from serving.schedule import Scheduler

        self.runner.enable_mixed = mixed
        scheduler = Scheduler(self.runner, frozenset(), max_active=2)
        decode = self.RequestState([5], 3, frozenset())
        prefill = self.RequestState([2] * 17, 2, frozenset())
        selected = []

        def record_selection(phases, counts, enabled, **limits):
            graph = select_tp1_graph(phases, counts, enabled, **limits)
            selected.append((tuple(phases), graph))
            return graph

        try:
            with ThreadPoolExecutor(max_workers=1) as scheduler.executor:
                with patch("serving.runtime.select_tp1_graph", side_effect=record_selection):
                    scheduler.incoming.put(decode)
                    scheduler._step()
                    scheduler.incoming.put(prefill)
                    for _ in range(3):
                        scheduler._step()
            self.assertFalse(scheduler.active)
            self.assertEqual(decode.finish_reason, "length")
            self.assertEqual(prefill.finish_reason, "length")
            expected_graph = "mixed" if mixed else "prefill"
            self.assertIn((("decode", "prefill"), expected_graph), selected)
            return decode.generated, prefill.generated
        finally:
            for request in scheduler.active:
                self.runner.release(request)

    def test_scheduler_runs_mixed_batch(self):
        try:
            self.assertEqual(self.run_scheduler_case(False), self.run_scheduler_case(True))
        finally:
            self.runner.enable_mixed = True

    def assert_equivalent(self, baseline, mixed):
        self.assertEqual(baseline[0], mixed[0])
        for baseline_group, mixed_group in zip(baseline[1:3], mixed[1:3]):
            for baseline_slot, mixed_slot in zip(baseline_group, mixed_group):
                for expected, actual in zip(baseline_slot, mixed_slot):
                    self.torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        for expected, actual in zip(baseline[3], mixed[3]):
            self.torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)

    def run_prefix_case(self, mixed):
        self.runner.enable_mixed = mixed
        decode = self.RequestState([5], 2, frozenset())
        prefill = self.RequestState([2] * 65, 2, frozenset())
        self.runner.admit(decode)
        try:
            self.runner.run_batch([self.BatchSlot(decode, (5,), 0, "prefill", False)])
            self.assertEqual(self.runner.admit(prefill), 64)
            tokens = self.runner.run_batch([
                self.BatchSlot(prefill, (2,), 64, "prefill", True),
                self.BatchSlot(decode, (6,), 1, "decode", False),
            ])
            states = [self.runner._read_state(self.runner.requests[request.request_id].slot)
                      for request in (prefill, decode)]
            kv = [self.read_kv(request) for request in (prefill, decode)]
            logits = [self.runner._logit_row(self.runner.requests[request.request_id].slot, 0)
                      for request in (prefill, decode)]
            return tokens, states, kv, logits
        finally:
            self.runner.release(prefill)
            self.runner.release(decode)

    def test_prefix_reuse_matches_baseline(self):
        self.runner.allow_prefix_hits = True
        seed = self.RequestState([2] * 65, 2, frozenset())
        self.runner.admit(seed)
        try:
            for offset in range(0, 64, self.block_size):
                self.runner.run_batch([
                    self.BatchSlot(seed, (2,) * self.block_size, offset, "prefill", False),
                ])
        finally:
            self.runner.release(seed)
        try:
            self.assert_equivalent(self.run_prefix_case(False), self.run_prefix_case(True))
        finally:
            self.runner.allow_prefix_hits = False


if __name__ == "__main__":
    unittest.main()
