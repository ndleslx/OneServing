"""Source-compiled single-card runner for the qualified Qwen3.8 state ABI.

The default path uses large-prefill/decode rotation and device greedy decode.
The newer mixed FFN model is opt-in; no speculative device-feedback graph is
registered. All model weights come from TextWeights(model_root).
"""
from __future__ import annotations

from collections import Counter
import importlib
import json
from pathlib import Path
import time

import torch
from pypto.runtime import DeviceTensor, DistributedWorker

from . import runtime as base
from .kv_cache import CacheCapacityError
from .model import BATCH, BLOCK_SIZE, CONTEXT, HIDDEN, POOL_BLOCKS, VOCAB
from .schedule import RequestCapacityError
from .scheduling_policy import resident_permutation
from .page_table import fill_table
from .weight import TextWeights

FIELDS = ('ids', 'cos', 'sin', 'masks', 'counts', 'offsets', 'tables')
FIELD_INDICES = (0, 24, 25, 26, 27, 28, 33)
DTYPES = {'int32': torch.int32, 'fp32': torch.float32,
          'bfloat16': torch.bfloat16, 'index': torch.int64}


def logical_shape(param):
    """Compile with logical NZ dimensions, retaining physical manifest metadata."""
    shape = tuple(param['shape'])
    if param.get('layout') == 'NZ':
        if len(shape) < 4 or shape[-2:] != (16, 16):
            raise ValueError('Invalid complete NZ tile shape')
        shape = (*shape[:-4], shape[-3] * 16, shape[-4] * 16)
    return shape


def compile_graph(name, specs, config):
    spec = specs[name]
    module_name = spec['module']
    if not module_name.startswith(__package__ + '.'):
        raise ValueError('Graph module must belong to the optimized package')
    module = importlib.import_module(module_name)
    samples = [torch.empty(logical_shape(p), dtype=DTYPES[p['dtype']], device='meta')
               if p['shape'] is not None else 1 for p in spec['params']]
    return getattr(module, spec['entrypoint']).compile(*samples, config=config)


class ModelRunner(base.ModelRunner):
    def __init__(self, *args, enable_mixed=False, async_prepare=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_prefix_hits = False
        self.enable_mixed_prefill = bool(enable_mixed)
        self.enable_async_prepare = bool(async_prepare)
        self.serving_policy = 'compound'
        self._allocated = []
        self._decode_page_rows = {}
        self.executed_graphs = Counter()
        self.preserved_local_compactions = 0
        self.async281 = None

    def _alloc(self, shape, dtype, init=None):
        value = self.worker.alloc_tensor(tuple(shape), dtype, init=init)
        self._allocated.append(value)
        return value

    @staticmethod
    def _host_fields(spec):
        result = {}
        for name, index in zip(FIELDS, FIELD_INDICES, strict=True):
            p = spec['params'][index]
            result[name] = torch.zeros(tuple(p['shape']), dtype=DTYPES[p['dtype']]).share_memory_()
        return result

    def start(self):
        if self.worker is not None:
            raise RuntimeError('Model runner is already started')
        if self.allow_prefix_hits:
            raise ValueError('The optimized runner requires prefix_cache=False')
        if self.config.any_dfx_enabled():
            raise ValueError('Use the base runner for profiling modes')
        specs_path = Path(__file__).with_name('kernels') / 'graph_specs.json'
        self.graph_specs = json.loads(specs_path.read_text())
        names = ['prefill_small16', 'prefill_wide64', 'prefill_chunk512',
                 'decode_greedy', 'zero_state', 'compact_local']
        if self.enable_mixed_prefill:
            names += ['prefill_single128', 'mixed_ffn']
        self.programs = {n: compile_graph(n, self.graph_specs, self.config) for n in names}
        self.prefill = self.programs['prefill_small16']
        self.wide_prefill = self.programs['prefill_wide64']
        self.chunk512 = self.programs['prefill_chunk512']
        self.decode = self.compound_decode = self.programs['decode_greedy']
        self.zero_state_graph = self.programs['zero_state']
        self.local_compact_graph = self.programs['compact_local']
        self.rope = base.rotary_tables(self.model_root)
        if self.rope['cos'].shape != (CONTEXT, 64) or self.rope['sin'].shape != (CONTEXT, 64):
            raise ValueError('Invalid RoPE table')
        self.host_fields = {n: self._host_fields(self.graph_specs[n]) for n in names
                            if n.startswith('prefill_') or n == 'decode_greedy'}
        for name, tensor in self.host_fields['decode_greedy'].items():
            setattr(self, 'decode_' + name, tensor)
        self.local_plan_tensor = torch.full((12, 3), -1, dtype=torch.int32).share_memory_()
        self.greedy_host = torch.empty((8, 8), dtype=torch.int32).share_memory_()
        inherited = [*(t for bank in self.host_fields.values() for t in bank.values()),
                     self.local_plan_tensor, self.greedy_host]
        if self.enable_async_prepare:
            from .async_lookahead import allocate_host_banks
            inherited.extend(allocate_host_banks(self))
        self.worker = DistributedWorker(list(self.programs.values()), config=self.config,
                                        inherited_host_tensors=tuple(inherited))
        self.banks, self.resident, self.state_scratch = [], [], []
        try:
            for _, host in TextWeights(self.model_root).banks():
                self.banks.append(self._alloc(host.shape, host.dtype, host))
            for p in self.graph_specs['prefill_small16']['params'][29:33]:
                shape, dtype = tuple(p['shape']), DTYPES[p['dtype']]
                self.resident.append(self._alloc(shape, dtype, torch.zeros(shape, dtype=dtype)))
            self.conv_words = DeviceTensor(self.resident[0].data_ptr, (8, 983040), torch.int32,
                                           buffer=self.resident[0].buffer)
            self.recurrent_words = DeviceTensor(self.resident[1].data_ptr, (8, 37748736), torch.int32,
                                                buffer=self.resident[1].buffer)
            if (self.conv_words.nbytes != self.resident[0].nbytes or
                    self.recurrent_words.nbytes != self.resident[1].nbytes):
                raise RuntimeError('Local compactor views lost full resident byte coverage')
            self.outputs = {}
            for name in names:
                if name.startswith('prefill_') or name == 'decode_greedy':
                    if name == 'prefill_chunk512':
                        # Original8x64 and compact1x512 share complete output owners.
                        trace, logits = self.outputs['prefill_wide64']
                        shapes = [self.graph_specs[name]['params'][i]['shape'] for i in (34, 35)]
                        views = tuple(DeviceTensor(h.data_ptr, tuple(shape), h.dtype, buffer=h.buffer)
                                      for h, shape in zip((trace, logits), shapes, strict=True))
                        if any(v.nbytes != h.nbytes for v, h in zip(views, (trace, logits), strict=True)):
                            raise RuntimeError('Compact prefill output byte coverage changed')
                        self.outputs[name] = views
                    else:
                        self.outputs[name] = tuple(self._alloc(p['shape'], DTYPES[p['dtype']])
                                                   for p in self.graph_specs[name]['params'][34:36])
            self.traces, self.logits = self.outputs['prefill_small16']
            self.wide_traces, self.wide_logits = self.outputs['prefill_wide64']
            self.decode_traces, self.decode_logits = self.outputs['decode_greedy']
            self.greedy_buffers = [self._alloc(p['shape'], DTYPES[p['dtype']])
                                   for p in self.graph_specs['decode_greedy']['params'][36:40]]
            if self.enable_async_prepare:
                import pypto.runtime.distributed_runner as distributed_runtime
                from .async_cache_observer import DecodeLookaheadWithCacheObserver
                self.async281 = DecodeLookaheadWithCacheObserver(self, distributed_runtime)
            self._sample_memory()
        except BaseException:
            self.close()
            raise

    def close(self):
        worker, self.worker = self.worker, None
        if worker is None:
            return
        try:
            if self.async281 is not None:
                self.async281.cancel('close')
                self.async281.session.close()
            for value in reversed(self._allocated):
                worker.free_tensor(value)
        finally:
            worker.close()
            self._allocated.clear()
            self.banks.clear()
            self.resident.clear()

    def admit(self, request):
        if self.async281 is not None:
            self.async281.cancel('admit')
        return super().admit(request)

    def release(self, request):
        if self.async281 is not None:
            self.async281.cancel('release')
        self._decode_page_rows.pop(request.request_id, None)
        return super().release(request)

    def _compact(self, slots):
        from .kernels.copy_plan import make_plan
        order, moves = resident_permutation(self.slot_ids, self.requests,
            [s.request.request_id for s in slots])
        if not moves:
            return
        if self.async281 is not None:
            self.async281.cancel('compact')
        plan = tuple(make_plan([self.requests[key].slot for key in order if key is not None]))
        if not 0 < len(plan) <= self.local_plan_tensor.shape[0]:
            raise RuntimeError('Resident permutation exceeds the compiled local copy plan')
        self.local_plan_tensor.fill_(-1)
        self.local_plan_tensor[:len(plan)] = torch.tensor(plan, dtype=torch.int32)
        self.worker.run(self.local_compact_graph, self.conv_words, self.recurrent_words,
                        self.local_plan_tensor, len(plan), config=self.config)
        self.slot_ids = list(order)
        for index, key in enumerate(order):
            if key is not None:
                self.requests[key].slot = index
        self.preserved_local_compactions += 1

    def _validate_slots(self, slots):
        if self.worker is None or not 1 <= len(slots) <= 8:
            raise RuntimeError('Invalid model batch')
        if len({s.request.request_id for s in slots}) != len(slots):
            raise RuntimeError('Duplicate request in model batch')
        for s in slots:
            q = s.request
            if q.request_id not in self.requests or q.runtime_state is not self.requests[q.request_id]:
                raise RuntimeError('Scheduled request lost resident ownership')
            if s.phase == 'prefill':
                if (s.position != q.prefill_offset or not s.token_ids or
                    tuple(q.prompt_ids[s.position:s.position + len(s.token_ids)]) != s.token_ids or
                    s.completes_prefill != (s.position + len(s.token_ids) == len(q.prompt_ids))):
                    raise RuntimeError('Invalid prefill frontier/count/offset')
            elif (s.phase != 'decode' or len(s.token_ids) != 1 or not q.generated
                  or q.prefill_offset != len(q.prompt_ids)
                  or s.position != len(q.prompt_ids) + len(q.generated) - 1
                  or s.token_ids != (q.generated[-1],)):
                raise RuntimeError('Invalid decode frontier/input')
            if s.position < 0 or s.position + len(s.token_ids) > CONTEXT:
                raise RuntimeError('Scheduled context exceeds the physical ABI')
        if any(len(s.token_ids) > 128 for s in slots) and len(slots) != 1:
            raise RuntimeError('Large prefill must be submitted in isolation')

    def _resources(self, slots):
        decode_only = all(s.phase == 'decode' for s in slots)
        maximum = max(len(s.token_ids) for s in slots)
        if decode_only:
            name = 'decode_greedy'
        elif maximum > 128:
            name = 'prefill_chunk512'
        elif maximum > 64:
            if len(slots) == 1 and self.enable_mixed_prefill:
                name = 'prefill_single128'
            else:
                name = 'prefill_chunk512'
        else:
            name = 'prefill_wide64' if maximum > 16 else 'prefill_small16'
        if name.startswith('prefill_') and name in ('prefill_single128', 'prefill_chunk512') and len(slots) != 1:
            raise RuntimeError('Compact prefill requires one selected request')
        return name, self.host_fields[name], self.programs[name], self.outputs[name]

    def _prepare(self, slots, fields, rows, *, decode_only=False):
        tensors = tuple(fields[n] for n in FIELDS)
        if decode_only and self.async281 is not None and self.async281.consume_fields(slots, *tensors):
            return
        ids, cos, sin, masks, counts, offsets, tables = tensors
        ids.zero_(); cos.fill_(1); sin.zero_(); masks.fill_(-10000)
        counts.zero_(); offsets.zero_(); tables.zero_()
        for index, slot in enumerate(slots):
            count, position = len(slot.token_ids), slot.position
            resident = self.requests[slot.request.request_id]
            if resident.pages.length != position:
                raise RuntimeError('Request KV length and scheduled position disagree')
            try:
                self.cache.reserve(resident.pages, position + count)
            except CacheCapacityError as error:
                raise RequestCapacityError(slot.request, str(error)) from error
            if decode_only:
                ids[index, 0] = slot.token_ids[0]
                masks[index, :, :position + 1].zero_()
                self._decode_page_rows[slot.request.request_id] = fill_table(
                    self.cache, resident.pages, tables[index], self._decode_page_rows.get(slot.request.request_id))
            else:
                ids[index, :count] = torch.tensor(slot.token_ids, dtype=torch.int32)
                masks[index] = self._mask(count, position, rows)
                tables[index] = self.cache.table(resident.pages)
            cos[index, :count] = self.rope['cos'][position:position + count]
            sin[index, :count] = self.rope['sin'][position:position + count]
            counts[index], offsets[index] = count, position

    def _arguments(self, fields, outputs, count, *, decode=False):
        values = (fields['ids'], *self.banks, *(fields[n] for n in FIELDS[1:6]),
                  *self.resident, fields['tables'], *outputs)
        return (*values, *self.greedy_buffers, count) if decode else (*values, count)

    def _activation(self, kind, slots):
        if self.on_activation is not None:
            fields = dict(time_ns=time.monotonic_ns(), device=self.device_id)
            if kind == 'graph_start':
                fields['phases'] = [s.phase for s in slots]
            self.on_activation(kind, fields)

    def run_batch(self, slots):
        self._validate_slots(slots)
        if self.enable_mixed_prefill:
            from .mixed_prefill import mixed_order, run_mixed
            if mixed_order(slots) is not None:
                if self.async281 is not None:
                    self.async281.cancel('mixed')
                return run_mixed(self, slots)
        self._compact(slots)
        name, fields, graph, outputs = self._resources(slots)
        decode_only = name == 'decode_greedy'
        if self.async281 is not None:
            self.async281.begin(slots)
        rows = int(fields['ids'].shape[1])
        self._prepare(slots, fields, rows, decode_only=decode_only)
        self._activation('graph_start', slots)
        self.worker.run(graph, *self._arguments(fields, outputs, len(slots), decode=decode_only), config=self.config)
        self.executed_graphs[name] += 1
        self._activation('graph_end', slots)
        tokens = None
        if decode_only:
            self._copy_offset(self.greedy_host, self.greedy_buffers[-1], 0, False)
            tokens = self.greedy_host[:, 0].tolist()
        result = []
        for index, slot in enumerate(slots):
            resident = self.requests[slot.request.request_id]
            self.cache.commit(resident.pages, slot.position + len(slot.token_ids))
            if slot.phase == 'prefill' and not slot.completes_prefill:
                result.append(None)
            elif tokens is not None:
                result.append(int(tokens[index]))
            else:
                logits = self._logit_row(index, len(slot.token_ids) - 1, rows=rows, logits=outputs[1])
                result.append(int(logits.float().argmax()))
        self._sample_memory()
        return result
