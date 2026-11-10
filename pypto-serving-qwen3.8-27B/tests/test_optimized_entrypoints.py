"""CPU checks for backend selection and the pinned NZ binding boundary."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BackendTests(unittest.TestCase):
    def route(self, backend='auto', tp='1', dp='1', **extra):
        package = types.ModuleType('api_dispatch_probe')
        package.__path__ = [str(ROOT / 'serving')]
        optimized = types.ModuleType(package.__name__ + '.optimized')
        optimized.__path__ = [str(ROOT / 'serving/optimized')]
        modules = {package.__name__: package, optimized.__name__: optimized}
        for suffix in ('optimized.api', 'api_legacy'):
            module = types.ModuleType(package.__name__ + '.' + suffix)
            module.app = suffix
            modules[module.__name__] = module
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(
                os.environ, dict(PYPTO_BACKEND=backend, TP=tp, DP=dp, **extra), clear=True):
            return load(package.__name__ + '.api', ROOT / 'serving/api.py').app

    def test_auto_single_and_legacy_topologies(self):
        for backend, tp, dp, expected in (
            ('auto', '1', '1', 'optimized.api'),
            ('optimized', '1', '1', 'optimized.api'),
            ('legacy', '1', '1', 'api_legacy'),
            ('auto', '2', '1', 'api_legacy'),
            ('auto', '1', '2', 'api_legacy')):
            with self.subTest(backend=backend, tp=tp, dp=dp):
                self.assertEqual(self.route(backend, tp, dp), expected)

    def test_unsupported_and_inert_options_rejected(self):
        for options in (dict(backend='unknown'), dict(backend='optimized', tp='2'),
                        dict(backend='legacy', PYPTO_MIXED_PREFILL='1'),
                        dict(backend='auto', dp='2', PYPTO_ASYNC_PREPARE='1')):
            with self.subTest(options=options), self.assertRaises(RuntimeError):
                self.route(**options)


class NZBindingTests(unittest.TestCase):
    def setUp(self):
        self.module = load('nz_binding_probe', ROOT / 'serving/optimized/nz_binding.py')
        self.calls = []
        def original(shape, info):
            self.calls.append((shape, info))
            if info.layout == 'NZ' and info.shape and info.shape[0] < 0 and len(shape) != 3:
                raise TypeError('dynamic NZ rank')
            return ('original', tuple(shape))
        self.adapted = self.module._with_logical_nz_shapes(original)

    def info(self, shape, layout='NZ'):
        return types.SimpleNamespace(shape=shape, layout=layout)

    def test_exact_logical_nz_only(self):
        self.assertEqual(self.adapted((64, 5120, 17408), self.info((64, 5120, 17408))),
                         [64, 5120, 17408])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.adapted((8, 16, 32), self.info((-1, 16, 32))), [8, 16, 32])

    def test_nd_mismatch_blocked_and_unknown_delegate(self):
        for shape, info in (((8, 16), self.info((8, 16), 'ND')),
                            ((64, 5120, 17408), self.info((64, 1088, 320, 16, 16))),
                            ((64, 5120, 17409), self.info((64, 5120, 17408))),
                            ((8, 16), self.info(None))):
            self.assertEqual(self.adapted(shape, info), ('original', shape))
        self.assertEqual(len(self.calls), 4)

    def test_dynamic_rank_guard_is_preserved(self):
        with self.assertRaisesRegex(TypeError, 'dynamic NZ rank'):
            self.adapted((8, 1, 16, 32), self.info((-1, 1, 16, 32)))

    def test_unpinned_runtime_rejected_without_modifying_files(self):
        path = ROOT / 'serving/optimized/nz_binding.py'
        before = path.read_bytes()
        with mock.patch.object(self.module.importlib, 'import_module',
                               return_value=types.SimpleNamespace(__file__=str(path))):
            with self.assertRaisesRegex(RuntimeError, 'pinned PyPTO'):
                self.module.install_nz_bindings()
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
