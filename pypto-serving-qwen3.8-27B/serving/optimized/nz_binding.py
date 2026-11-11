"""Process-local NZ shape compatibility for the pinned PyPTO release.

The model passes already packed NZ storage with logical bank dimensions. The
validated compiler can retain those dimensions in its entry metadata. Keep the
same exact-shape rule used by the tested runtime without editing site-packages.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import importlib
from pathlib import Path
import threading

_BASE = 'a830c800b24986b279045f0c0ea98be5047472e56e3e0d5e39d15d786f63b32a'
_FIXED = '41116deca9f5c64fa4b5755cdf3fc745685d0fd2c6e10d67293d4611d20f025b'
_LOCK = threading.Lock()


def _with_logical_nz_shapes(original):
    @wraps(original)
    def declared_shape(arg_shape, info):
        if info.layout == 'NZ':
            # Preserve the original dynamic-batch rank rejection before the
            # additional logical-shape acceptance, exactly as in the overlay.
            if info.shape and info.shape[0] < 0 and len(arg_shape) != 3:
                return original(arg_shape, info)
            if info.shape is not None and len(info.shape) == len(arg_shape) and all(
                expected < 0 or expected == actual
                for expected, actual in zip(info.shape, arg_shape, strict=True)
            ):
                return list(arg_shape)
        return original(arg_shape, info)
    declared_shape._oneserving_logical_nz = True
    return declared_shape


def install_nz_bindings():
    module = importlib.import_module('pypto.ir.compiled_program')
    path = Path(module.__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest not in (_BASE, _FIXED):
        raise RuntimeError(
            'The optimized backend requires the pinned PyPTO 98752784548b777b2377a9084411c3716eb341d6 runtime; '
            'the installed compiled_program.py differs from the validated binding.')
    with _LOCK:
        if digest == _BASE and not getattr(module._arg_shape_as_declared, '_oneserving_logical_nz', False):
            module._arg_shape_as_declared = _with_logical_nz_shapes(module._arg_shape_as_declared)
    return dict(nz_binding_path=str(path), nz_binding_sha256=digest,
                nz_binding_adapter='process-local-logical-shape' if digest == _BASE else 'existing-logical-shape',
                installed_library_files_modified=False)


def runtime_provenance():
    import pypto
    import simpler
    result = install_nz_bindings()
    root = Path(__file__).resolve().parent
    source = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        source.update(str(path.relative_to(root)).encode() + b'\0')
        source.update(path.read_bytes())
    result.update(source_tree_sha256=source.hexdigest(),
                  pypto_path=str(Path(pypto.__file__).resolve()),
                  simpler_path=str(Path(simpler.__file__).resolve()),
                  backend='optimized', context=2048, device_feedback_enabled=False)
    return result
