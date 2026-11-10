"""Select the source-compiled optimized single-card or existing multi-card API."""
import os

backend = os.environ.get('PYPTO_BACKEND', 'auto')
if backend not in ('auto', 'optimized', 'legacy'):
    raise RuntimeError('PYPTO_BACKEND must be auto, optimized, or legacy')
single_device = os.environ.get('TP', '1') == '1' and os.environ.get('DP', '1') == '1'
if backend == 'optimized' and not single_device:
    raise RuntimeError('The optimized backend supports TP1/DP1')
if backend == 'optimized' or (backend == 'auto' and single_device):
    from .optimized.api import app
else:
    if any(os.environ.get(flag, '0') != '0' for flag in ('PYPTO_MIXED_PREFILL', 'PYPTO_ASYNC_PREPARE')):
        raise RuntimeError('Mixed prefill and async prepare require the optimized backend')
    from .api_legacy import app
