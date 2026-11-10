"""CPU-only contract checks; these do not certify new source on an NPU."""
import importlib
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import itertools
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / 'serving' / 'optimized'
package = types.ModuleType('export_test_serving')
package.__path__ = [str(KERNEL)]
sys.modules[package.__name__] = package
model = types.ModuleType(package.__name__ + '.model')
for name, value in dict(BATCH=8, ROWS=16, BLOCK_SIZE=16, CONTEXT=2048,
                        HIDDEN=5120, POOL_BLOCKS=1032, VOCAB=248320).items():
    setattr(model, name, value)
sys.modules[model.__name__] = model
policy = importlib.import_module(package.__name__ + '.scheduling_policy')
schedule = importlib.import_module(package.__name__ + '.schedule_optimized')
base_schedule = importlib.import_module(package.__name__ + '.schedule')


class Runner:
    enable_mixed_prefill = False

    def __init__(self, token=123, delay=0):
        self.token, self.delay = token, delay
        self.requests = {}
        self.batches, self.releases = [], []
        self.running = threading.Event()
        self.completed = threading.Event()

    def admit(self, q):
        assert not self.running.is_set()
        q.runtime_state = SimpleNamespace()
        self.requests[q.request_id] = q.runtime_state
        return 0

    def release(self, q):
        assert not self.running.is_set(), 'release raced accepted model work'
        self.releases.append(q.request_id)
        self.requests.pop(q.request_id, None)
        q.runtime_state = None

    def run_batch(self, slots):
        self.batches.append(tuple(slots))
        self.completed.clear(); self.running.set()
        try:
            if self.delay:
                time.sleep(self.delay)
            return [self.token if s.phase == 'decode' or s.completes_prefill else None for s in slots]
        finally:
            self.running.clear(); self.completed.set()


def request(n, budget=3, stop=()):
    return base_schedule.RequestState(list(range(n)), budget, frozenset(stop))


class SchedulingTests(unittest.TestCase):
    def scheduler(self, runner):
        return schedule.Scheduler(runner, frozenset(), max_active=8)

    def run_steps(self, s, count):
        with ThreadPoolExecutor(max_workers=1) as s.executor:
            for _ in range(count):
                s._step()

    def test_large_prefill_decode_rotation(self):
        runner = Runner(); s = self.scheduler(runner)
        p, d = request(1024), request(16)
        d.prefill_offset = 16; d.generated = [5]
        s.active = [p, d]
        with ThreadPoolExecutor(max_workers=1) as s.executor:
            for _ in range(4):
                s._step()
        self.assertEqual([[(x.phase, len(x.token_ids)) for x in row] for row in runner.batches],
                         [[('prefill', 512)], [('decode', 1)], [('prefill', 512)],
                          [('decode', 1), ('decode', 1)]])
        self.assertEqual(p.prefill_offset, 1024)

    def test_default_small_prefills_keep_original64_windows(self):
        s = self.scheduler(Runner()); q = request(128); s.active = [q]
        s._select_step_requests()
        self.assertEqual(len(s._slot(q).token_ids), 64)
        self.assertIsNone(s.short_selection)

    def test_mixed_is_explicit_opt_in(self):
        runner = Runner(); s = self.scheduler(runner)
        p, d = request(1024), request(16)
        d.prefill_offset = 16; d.generated = [5]; s.active = [p, d]
        self.assertEqual(s._select_step_requests(), [p])
        runner.enable_mixed_prefill = True
        self.assertEqual(s._select_step_requests(), [p, d])
        self.assertEqual(len(s._slot(p).token_ids), 128)
        self.assertEqual(s._slot(d).token_ids, (5,))

    def test_budget_one_has_one_model_call_and_release(self):
        runner = Runner(); s = self.scheduler(runner); q = request(8, 1)
        s.incoming.put(q); self.run_steps(s, 2)
        self.assertEqual(len(runner.batches), 1)
        self.assertEqual(q.generated, [123]); self.assertEqual(q.finish_reason, 'length')
        self.assertEqual(runner.releases, [q.request_id]); self.assertEqual(s.active, [])

    def test_eos_finishes_before_budget(self):
        runner = Runner(7); s = self.scheduler(runner); q = request(8, 99, (7,))
        s.incoming.put(q); self.run_steps(s, 2)
        self.assertEqual(q.generated, [7]); self.assertEqual(q.finish_reason, 'stop')
        self.assertEqual(len(runner.batches), 1)

    def test_cancel_before_admit_and_during_model(self):
        runner = Runner(delay=.03); s = self.scheduler(runner)
        cancelled = request(8); cancelled.cancelled.set(); s.incoming.put(cancelled)
        q = request(8); s.incoming.put(q)
        def cancel():
            self.assertTrue(runner.running.wait(.5)); q.cancelled.set()
        t = threading.Thread(target=cancel); t.start(); self.run_steps(s, 1); t.join()
        self.assertEqual(q.generated, [])
        self.assertEqual(runner.releases, [q.request_id])
        self.assertEqual(len(runner.batches), 1)

    def test_staging_error_joins_forward_before_release(self):
        runner = Runner(delay=.04); s = self.scheduler(runner); q = request(8)
        s.incoming.put(q); original = s._stage_pending; calls = []
        def stage():
            calls.append(1)
            if runner.batches:
                raise RuntimeError('injected staging observer error')
            return original()
        s._stage_pending = stage
        with self.assertRaisesRegex(RuntimeError, 'injected staging'):
            self.run_steps(s, 1)
        self.assertTrue(runner.completed.is_set())
        self.assertEqual(runner.releases, [q.request_id])

    def test_incoming_is_staged_without_concurrent_admit(self):
        runner = Runner(delay=.06); s = self.scheduler(runner); q = request(8)
        next_q = request(8); s.incoming.put(q)
        def incoming():
            self.assertTrue(runner.running.wait(.5)); s.incoming.put(next_q)
        t = threading.Thread(target=incoming); t.start(); self.run_steps(s, 1); t.join()
        self.assertIn(next_q, s.pending)
        self.assertNotIn(next_q.request_id, runner.requests)


class ResidentTests(unittest.TestCase):
    def test_every_live_owner_retained_for_partial_selections(self):
        checks = 0
        for occupied in (('a', 'b', 'c', 'd'), ('a', None, 'b', None, 'c')):
            slots = list(occupied) + [None] * (8 - len(occupied))
            residents = {key: SimpleNamespace(slot=i) for i, key in enumerate(slots) if key is not None}
            owners = tuple(residents)
            for n in range(1, len(owners) + 1):
                for chosen in itertools.permutations(owners, n):
                    order, moves = policy.resident_permutation(slots, residents, chosen)
                    self.assertEqual(set(x for x in order if x is not None), set(owners))
                    self.assertEqual(tuple(order[:n]), chosen)
                    self.assertEqual(len(order), 8)
                    checks += 1
        self.assertEqual(checks, 79)

    def test_bad_owner_table_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'index differs'):
            policy.resident_permutation(['a'] + [None] * 7, {'a': SimpleNamespace(slot=1)}, ['a'])

    def test_actual_runtime_compactor_preserves_unscheduled_payload(self):
        import torch
        # Stub only the external execution/loader modules; run the exported
        # runtime method and the actual exported copy-plan algorithm unchanged.
        runtime_base = types.ModuleType(package.__name__ + '.runtime')
        runtime_base.ModelRunner = type('Base', (), {})
        sys.modules[runtime_base.__name__] = runtime_base
        weight = types.ModuleType(package.__name__ + '.weight'); weight.TextWeights = object
        sys.modules[weight.__name__] = weight
        pypto = types.ModuleType('pypto'); runtime_api = types.ModuleType('pypto.runtime')
        runtime_api.DeviceTensor = object; runtime_api.DistributedWorker = object
        sys.modules['pypto'] = pypto; sys.modules['pypto.runtime'] = runtime_api
        module = importlib.import_module(package.__name__ + '.runtime_optimized')
        class Worker:
            def run(self, graph, conv, recurrent, plan, count, **kwargs):
                for values in (conv, recurrent):
                    saved = None
                    for mode, source, dest in plan[:count].tolist():
                        if mode == 0: values[dest] = values[source].clone()
                        elif mode == 1: saved = values[source].clone()
                        else: values[dest] = saved
        checks = 0
        for selection in itertools.permutations(('a', 'b', 'c', 'd'), 2):
            r = module.ModelRunner.__new__(module.ModelRunner)
            r.slot_ids = ['a', None, 'b', 'c', None, 'd', None, None]
            r.requests = {key: SimpleNamespace(slot=i) for i, key in enumerate(r.slot_ids) if key}
            r.conv_words = torch.arange(24).reshape(8, 3).clone()
            r.recurrent_words = (r.conv_words + 100).clone()
            old = {key: (r.conv_words[v.slot].clone(), r.recurrent_words[v.slot].clone()) for key, v in r.requests.items()}
            r.local_plan_tensor = torch.full((12, 3), -1, dtype=torch.int32)
            r.local_compact_graph = object(); r.config = None; r.worker = Worker(); r.async281 = None
            r.preserved_local_compactions = 0
            slots = [SimpleNamespace(request=SimpleNamespace(request_id=key)) for key in selection]
            r._compact(slots)
            self.assertEqual(r.slot_ids[:2], list(selection))
            for key, resident in r.requests.items():
                self.assertTrue(torch.equal(r.conv_words[resident.slot], old[key][0]))
                self.assertTrue(torch.equal(r.recurrent_words[resident.slot], old[key][1]))
            checks += 1
        self.assertEqual(checks, 12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
