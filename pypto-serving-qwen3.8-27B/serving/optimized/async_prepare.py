"""One-forward CPU lookahead, isolated from the installed PyPTO runtime.

This prototype stages metadata, not TaskArgs serialization or H2D. Callers
allocate both mutable host banks before worker construction, prepare future
values only in the standby bank, and publish after the previous result/commit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

RUNTIME_SHA256 = '59864f560b63f4eb2ced5ddb07af3e06fc78e7affd6e8d12b5a3b6f3d6a4312e'


def _span(tensor):
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.data_ptr() + storage.nbytes()


def _fingerprint(arg):
    if isinstance(arg, torch.Tensor):
        return ('host', id(arg), arg.data_ptr(), tuple(arg.shape), tuple(arg.stride()),
                arg.storage_offset(), arg.dtype, arg.is_shared(), _span(arg))
    if hasattr(arg, 'shards'):
        return ('stacked', id(arg), tuple(arg.shape), tuple(arg.worker_ids),
                tuple(_fingerprint(shard) for shard in arg.shards))
    if hasattr(arg, 'data_ptr') and hasattr(arg, 'buffer'):
        return ('resident', id(arg), arg.data_ptr, tuple(arg.shape), arg.dtype,
                id(arg.buffer), getattr(arg.buffer, 'base', None))
    if type(arg) in (int, float, bool):
        return (type(arg).__name__, arg)
    raise TypeError('Lookahead supports shared torch, owned resident tensors and primitive scalars')


class AsyncPreparation:
    """Own at most one unpublished frame and cancel it before worker teardown.

    The worker must have a single dispatch producer. Publication enforces zero
    outstanding native forwards because state, KV and outputs are shared.
    """

    def __init__(self, worker, runtime_module, mutable_host_names):
        source = Path(runtime_module.__file__)
        if hashlib.sha256(source.read_bytes()).hexdigest() != RUNTIME_SHA256:
            raise RuntimeError('Pinned PyPTO runtime changed')
        self.worker, self.runtime = worker, runtime_module
        self.mutable_names = frozenset(mutable_host_names)
        self.pending = None
        self.closed = False
        self.original_close = worker.close
        worker.close = self._close_worker

    def _close_worker(self):
        self.close()
        return self.original_close()

    def close(self):
        with self.worker._dispatch_submit_mu:
            if self.closed:
                return
            self.closed = True
            if self.pending is not None:
                self.pending._cancel_locked()
            self.worker.close = self.original_close

    def prepare(self, key, compiled, *args, config=None):
        worker = self.worker
        with worker._dispatch_submit_mu:
            if self.closed:
                raise RuntimeError('Lookahead session closed')
            if self.pending is not None:
                raise RuntimeError('One unpublished lookahead frame is already owned')
            if config is not None and config.dfx_options().any():
                raise ValueError('Lookahead excludes profiling; use the original synchronous path')
            worker._require_open('submit')
            state = worker._states.get(compiled)
            if state is None:
                raise ValueError('Lookahead program must be registered on this worker')
            infos = state['param_infos']
            if len(args) != len(infos):
                raise TypeError('Lookahead argument count differs from the prepared program')
            with worker._dispatch_cv:
                active = tuple(worker._active_dispatch_handles)
                if len(active) > 1:
                    raise RuntimeError('Lookahead permits one native forward in flight')
                if not any(not f.in_use for f in worker._dispatch_frames):
                    raise RuntimeError('Lookahead must never wait for metadata-frame capacity')
                for handle in active:
                    current = handle._frame.tensors
                    for info, arg in zip(infos, args, strict=True):
                        if info.name not in self.mutable_names:
                            continue
                        if not isinstance(arg, torch.Tensor):
                            raise TypeError('Mutable lookahead host input must be a shared torch tensor')
                        left, right = _span(arg)
                        for old in current.values():
                            if isinstance(old, torch.Tensor):
                                start, end = _span(old)
                                if left < end and start < right:
                                    raise ValueError('Standby mutable host storage aliases an in-flight argument')
            frame, dispatch_id = worker._acquire_dispatch_frame()
            try:
                call_config, _, two_pass = worker._prepare_dispatch_config(compiled, config)
                if two_pass:
                    raise ValueError('Two-pass diagnostics cannot be prepared speculatively')
                # Same data construction and every original argument validator.
                tensors = frame.tensors
                tensors.clear()
                tensors.update(state['base_tensor_frames'][frame.slot_id])
                frame.keepalive.clear()
                frame.keepalive.extend((compiled, call_config, config, *args))
                for info, arg in zip(infos, args, strict=True):
                    if info.shape is not None:
                        worker._validate_prepared_dispatch_arg(info, arg)
                    tensors[info.name] = arg
                ticket = PreparedMetadata(self, key, compiled, state, frame, dispatch_id,
                                          args, call_config, tuple(map(_fingerprint, args)))
                self.pending = ticket
                return ticket
            except BaseException:
                if frame.handle is None and frame.in_use:
                    worker._release_unpublished_dispatch_frame(frame)
                raise


class PreparedMetadata:
    def __init__(self, session, key, compiled, state, frame, dispatch_id, args, config, fingerprints):
        self.session, self.key, self.compiled, self.state = session, key, compiled, state
        self.frame, self.dispatch_id, self.args = frame, dispatch_id, args
        self.config, self.fingerprints = config, fingerprints
        self.status = 'prepared'

    def _cancel_locked(self):
        if self.status != 'prepared':
            return
        self.status = 'cancelled'
        self.session.pending = None
        self.session.worker._release_unpublished_dispatch_frame(self.frame)

    def cancel(self):
        with self.session.worker._dispatch_submit_mu:
            self._cancel_locked()

    def publish(self, actual_key):
        session, worker = self.session, self.session.worker
        with worker._dispatch_submit_mu:
            if self.status != 'prepared' or session.closed:
                raise RuntimeError('Lookahead ticket is no longer publishable')
            if actual_key != self.key:
                self._cancel_locked()
                raise ValueError('Lookahead request/order/position key changed; rebuild the batch')
            worker._require_open('submit')
            with worker._dispatch_cv:
                if worker._active_dispatch_handles:
                    raise RuntimeError('Wait for the current forward and commit before publication')
            try:
                if worker._states.get(self.compiled) is not self.state:
                    raise RuntimeError('Prepared program registration changed')
                if tuple(map(_fingerprint, self.args)) != self.fingerprints:
                    raise ValueError('Prepared argument metadata changed')
                # Retain the original live-owner/ABA check at publication.
                for info, arg in zip(self.state['param_infos'], self.args, strict=True):
                    if hasattr(arg, 'shards'):
                        for shard, wid in zip(arg.shards, arg.worker_ids, strict=True):
                            worker._require_owned_resident_tensor(shard, info.name, worker_id=wid)
                    elif hasattr(arg, 'buffer') and hasattr(arg, 'data_ptr'):
                        worker._require_owned_resident_tensor(arg, info.name)
                handle = session.runtime.DistributedRunHandle(worker, None, self.frame, self.dispatch_id, None)
                try:
                    self.frame.handle = handle
                    with worker._dispatch_cv:
                        worker._active_dispatch_handles.add(handle)
                except BaseException:
                    worker._discard_unsubmitted_dispatch_handle(handle, self.frame)
                    raise
                self.status = 'published'
                session.pending = None
                # The original method retains accepted handles after interrupted submission.
                worker._submit_native_dispatch(handle, self.frame, self.state, self.frame.tensors, self.config)
                return handle
            except BaseException:
                if self.frame.handle is None and self.frame.in_use:
                    worker._release_unpublished_dispatch_frame(self.frame)
                if self.status == 'prepared':
                    self.status = 'cancelled'
                    session.pending = None
                raise
