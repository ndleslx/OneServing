"""Opt-in compiler regression checks, without an NPU or model weights."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("PYPTO_QWEN_TEST_COMPILE") == "1",
                     "set PYPTO_QWEN_TEST_COMPILE=1 with the pinned PyPTO installed")
class MixedCompileTests(unittest.TestCase):
    def test_baseline_and_packed_graph_codegen(self):
        from pypto.ir import DistributedConfig
        from pypto.runtime import RunConfig
        from serving.model import mixed_fwd, prefill_fwd

        for kernel in (prefill_fwd, mixed_fwd):
            with self.subTest(kernel=kernel.__name__), TemporaryDirectory() as output:
                config = RunConfig(
                    platform="a2a3", save_kernels_dir=output,
                    distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0))
                # Exercise the real parser, pass pipeline and codegen. PTOAS and
                # device binary assembly are outside this compiler-only check.
                with patch("pypto.jit.decorator._ptoas_available", return_value=False):
                    kernel.compile(config=config)
                self.assertTrue(list(Path(output).rglob("*.pto")))


if __name__ == "__main__":
    unittest.main()
