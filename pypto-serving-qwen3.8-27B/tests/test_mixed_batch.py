import unittest
from itertools import product

from serving.mixed_batch import packed_token_offsets, select_tp1_graph


class MixedBatchSelectionTests(unittest.TestCase):
    def select(self, phases, counts, enabled=True, capacity=8):
        return select_tp1_graph(phases, counts, enabled, capacity=capacity, rows=16)

    def test_every_phase_order_up_to_capacity(self):
        for size in range(1, 9):
            for phases in product(("prefill", "decode"), repeat=size):
                counts = [7 if phase == "prefill" else 1 for phase in phases]
                with self.subTest(phases=phases):
                    self.assertEqual(self.select(phases, counts), "mixed")

    def test_disabled_mixed_uses_baseline(self):
        self.assertEqual(self.select(["decode", "prefill"], [1, 16], False), "prefill")
        self.assertEqual(self.select(["decode"] * 8, [1] * 8, False), "decode")

    def test_one_token_prefill_keeps_its_phase(self):
        self.assertEqual(self.select(["prefill", "decode"], [1, 1]), "mixed")

    def test_capacity_is_supplied_by_caller(self):
        self.assertEqual(self.select(["prefill"] * 8 + ["decode"] * 8,
                                     [16] * 8 + [1] * 8, capacity=16), "mixed")

    def test_invalid_inputs_are_rejected(self):
        for phases, counts in (([], []), (["decode"], []), (["unknown"], [1]),
                               (["prefill"], [0]), (["prefill"], [17]),
                               (["decode"], [2]), (["prefill"] * 9, [1] * 9)):
            with self.subTest(phases=phases, counts=counts), self.assertRaises(ValueError):
                self.select(phases, counts)

    def test_offsets_preserve_request_boundaries_without_per_slot_padding(self):
        for counts in ((16, 1), (1, 7, 1, 16), (16, 16, 16, 16, 1, 1), (1,) * 8):
            with self.subTest(counts=counts):
                offsets = packed_token_offsets(counts)
                self.assertEqual(offsets[0], 0)
                self.assertEqual(offsets[-1], sum(counts))
                source = [(slot, row) for slot, count in enumerate(counts) for row in range(count)]
                for slot, count in enumerate(counts):
                    self.assertEqual(source[offsets[slot]:offsets[slot + 1]],
                                     [(slot, row) for row in range(count)])


if __name__ == "__main__":
    unittest.main()
