"""Resident TP2 PyPTO execution, paged state and MTP verification."""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import time
from pathlib import Path
from typing import Any

import pypto
import simpler
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import DistributedWorker, RunConfig, StackedDeviceTensor
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from .distributed import ServingConfig, probabilities, draw, verify_drafts
from .kv_cache import KVCache
from .schedule import BatchSlot, RequestCapacityError, RequestState
from .weight import TextWeights, TP_BANK_SPECS

SLOTS, ROWS, BLOCK_SIZE, HIDDEN, TP_VOCAB = 32, 16, 128, 5120, 124160
CONV_SHAPE = (48, 4, 5120)
RECURRENT_SHAPE = (48, 24, 128, 128)


def runtime_provenance():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.name.encode() + b'\0' + path.read_bytes())
    return {'source_tree_sha256': digest.hexdigest(), 'pypto_path': pypto.__file__,
            'simpler_path': simpler.__file__,
            'packages': {name: importlib.metadata.version(name)
                         for name in ('pypto', 'simpler', 'torch', 'transformers')}}


def shared(shape, dtype=torch.int32, fill=0):
    return torch.full(shape, fill, dtype=dtype).share_memory_()


class ResidentRows:
    def __init__(self,owner,request_id,slot,generation,kind,version,output_key,rows):
        self.owner=owner;self.request_id=request_id;self.slot=slot;self.generation=generation
        self.kind=kind;self.version=version;self.output_key=output_key;self.rows=tuple(rows)
    @property
    def shape(self):return (2,len(self.rows),HIDDEN)
    def __getitem__(self,key):
        if not isinstance(key,tuple) or len(key)!=2 or key[0]!=slice(None) or not isinstance(key[1],slice):
            raise TypeError('Resident hidden supports explicit all-rank row slices only')
        return ResidentRows(self.owner,self.request_id,self.slot,self.generation,self.kind,self.version,self.output_key,self.rows[key[1]])

class ResidentPrevious:
    def __init__(self,owner,buffer,counts,sources,version):
        self.owner=owner;self.buffer=buffer;self.counts=counts;self.sources=sources;self.version=version

class TensorParallelRunner:
    def __init__(self, config: ServingConfig, on_cache_event=None):
        self.settings = config
        self.device_ids = config.devices
        self.pool_blocks = max(2304, math.ceil(config.max_model_len / BLOCK_SIZE) + 2)
        self.table_blocks = math.ceil(config.max_model_len / BLOCK_SIZE)
        self.cache = KVCache('qwen3.8-27b-unified-tp2', self.pool_blocks,
                             config.max_model_len, BLOCK_SIZE, on_event=on_cache_event)
        self.allow_prefix_hits = config.prefix_cache
        self.provenance = runtime_provenance()
        self.worker = None
        self._terminal_device_error = None
        self.banks = []
        self.resident = []
        self.requests: dict[str, dict[str, Any]] = {}
        self.free_slots = list(reversed(range(config.max_num_seqs)))
        self.memory_stats = []
        self.invocations = []
        self.spec_stats = {'proposed': 0, 'accepted': 0, 'rejected_rounds': 0,
                           'draft_ns': 0, 'verify_ns': 0}
        self.calls = 0
        if not 0 < config.max_num_batched_tokens <= 16384:
            raise ValueError('This model supports max_num_batched_tokens in 1..16384')
        self.chunk_threshold = config.long_prefill_token_threshold
        if type(self.chunk_threshold) is not int or not 0 <= self.chunk_threshold <= config.max_model_len:
            raise ValueError('long_prefill_token_threshold must be a nonnegative integer <= max_model_len')
        self._last_layout = None
        self._scratch_key = None
        self.execution_totals = {'prefill_calls': 0, 'prefill_tokens': 0, 'max_prefill_chunk': 0,
                                 'mixed_target_calls': 0, 'max_actual_rows': 0, 'max_allocated_rows': 0,
                                 'target_batches': {str(b): 0 for b in (1, 4, 16, 32)}}
        # Dispatch host pointers are inherited by DistributedWorker's forked
        # children. Allocate backing storage before constructing any worker;
        # shape changes below create views only, never a new shared mapping.
        self._host_capacity_rows = max(self._row_buckets(b)[-1] for b in (1, 4, 16, 32))
        self.resource_policy = self._resource_policy()
        self.runconfig = RunConfig(platform='a2a3', distributed_config=DistributedConfig(
            device_ids=list(config.devices), num_sub_workers=0),
            ring_heap=tuple(self.resource_policy['ring_heap_bytes']))
        self.admission = []
        self._host_slabs = self._allocate_host_slabs()

    @property
    def state_specs(self):
        pool = (17, 2, self.pool_blocks, BLOCK_SIZE, 256)
        return [((32, *CONV_SHAPE), torch.bfloat16),
                ((32, *RECURRENT_SHAPE), torch.float32),
                (pool, torch.bfloat16), (pool, torch.bfloat16),
                ((4, 32, *CONV_SHAPE), torch.bfloat16),
                ((4, 32, 48, 24 * 128, 128), torch.float32)]

    def _row_buckets(self, batch):
        # Zero removes the extra per-request cap, never the finite global budget.
        prefill_cap = self.chunk_threshold or self.settings.max_model_len
        limit = min(self.settings.max_num_batched_tokens,
                    batch * max(prefill_cap, 1 + self.settings.speculative_tokens))
        ceiling = max(16, (limit + 15) // 16 * 16)
        # Tile-aligned runtime views, not compiled graph specializations.
        return range(16, ceiling + 1, 16)

    def _resource_policy(self):
        # Explicit AUTO model scopes retain target scratch for one invocation.
        # Size root storage from the generated target allocations, including
        # its final compact-logit pipe. Preserve the conservative other arenas
        # until their complete device lifecycle has been independently checked.
        def capacity(n):
            return 1 << (max(256 << 20, n + (64 << 10)) - 1).bit_length()

        shapes = [(b, self._row_buckets(b)[-1]) for b in (1, 4, 16, 32)]
        root, layer, branch, legacy_root, target_root = 0, 0, 0, 0, 0
        c16_workspace_bound, target_c16_root = 0, 0
        for batch, rows in shapes:
            logits = max(16, 4 * batch)
            # Keep the previous conservative drafter/root bound. The target
            # now has 29 persistent scratch tensors: 242432 bytes per packed
            # row plus 10240 per selected-logit row, and at most a 4 MiB pipe (M64, N128 FP32, depth2, 64blocks).
            legacy_root = max(legacy_root, 185344 * rows + 506880 * logits + (4 << 20))
            target_root = max(target_root, 242432 * rows + 10240 * logits + (4 << 20))
            c16_workspace = 14208 * (rows + 16 * batch) + 524288 * batch + 4 * (batch + 1) + 13312
            c16_workspace_bound = max(c16_workspace_bound, c16_workspace)
            target_c16_root = max(target_c16_root, 242432 * rows + 10240 * logits + (4 << 20) + c16_workspace)
            root = max(legacy_root, target_c16_root)
            layer = max(layer, 92160 * rows, 121856 * rows + 786432 * batch + (4 << 20))
            gdn = 98048 * rows + 6291456 * batch
            mlp = 121856 * rows
            attention = 62464 * rows + 786432 * batch
            branch = max(branch, gdn + max(64256 * rows, 6291456 * batch),
                         mlp + 34816 * rows,
                         attention + max(24576 * rows, 786432 * batch))
        # a5 generated target/draft use only root/depth2 AUTO allocation;
        # restore/zero/cache-copy have no heap outputs. All five generated
        # entries must be re-audited after model/compiler changes. Keep the
        # unused ring3 at the SDK minimum; preserve other heap and task pools.
        # Packed sampler is another L2 phase in each producer L3. Conservatively
        # retain its full GM temporaries alongside the producer root, not a live
        # peak claim: 3 partial arrays[L*128,16], 3 result arrays[L,16],
        # one pack tile per row, plus seven 64-byte alignment/header allowances.
        baseline_root_bound = root
        decision_rows = max(16, 4 * SLOTS)
        packed_sampler_workspace = (3 * decision_rows * 128 * 16 * 4 +
                                    4 * decision_rows * 16 * 4 + 7 * 64)
        teacher_packet_workspace = (1 + decision_rows + 32) * 16 * 4 + 3 * 64
        root += packed_sampler_workspace + teacher_packet_workspace
        rings = [256 << 20, capacity(root), capacity(layer), 256 << 20]
        kinds = 2 if self.settings.speculative_tokens else 1
        largest_window = 20480 * 16384 + 32
        retained = kinds * largest_window
        # Public SDK policy: two dynamic model programs have fixed 335544352-byte
        # communication windows, bounded by a 1 GiB per-rank retention budget.
        # Both retained windows plus conservative reset staging are charged
        # below before loading weights/state; never lower the admission margin.
        # Never change arena sizes or persistence during a worker's lifetime.
        persistent = retained <= (1 << 30)
        communication = retained if persistent else largest_window
        rows = self._host_capacity_rows
        one_kind_scratch = rows * HIDDEN * 2 + max(16, 4 * SLOTS) * TP_VOCAB * 2
        # Both producer pairs + independent gathered previous + 32-slot tail bank.
        # Conservative status buffers are charged; old host staging remains charged.
        scratch = kinds * one_kind_scratch + rows * HIDDEN * 2 + SLOTS * HIDDEN * 2
        scratch += rows * 16 * 4 + SLOTS * 16 * 4 + 3 * 128 * 16 * 4
        # Conservative simultaneous HBM staging bound for all HOST tensor args,
        # including the drafter hidden input, tables and control copies.
        staging = rows * (4 + 4 + 2 * 64 * 2 + HIDDEN * 2) + 16 * SLOTS
        staging += 32 * self.table_blocks * 4 + 2 * 32 * 4
        # Keep the old standalone-greedy reserve; add both producer decisions
        # and masks explicitly. No existing reserve/margin is reclaimed here.
        scratch += kinds * decision_rows * 16 * 4
        staging += kinds * decision_rows * (16 * 4 + 4)
        teacher_packet_bytes = (1 + decision_rows + 32) * 16 * 4
        scratch += teacher_packet_bytes
        staging += teacher_packet_bytes
        # SDK reset staging is host-owned today; charge another device window
        # conservatively so reset implementation details cannot consume margin.
        reset = largest_window if persistent else 0
        # Reserve for post-prepare HCCL/native metadata and allocator rounding.
        # This is explicit headroom, not a claim to measure unknown counters.
        margin = 1 << 30
        future = communication + scratch + staging + reset + margin
        return {'version': 'teacher-safe-gather-tail-packet-source2',
                'ring_heap_bytes': rings, 'max_rows': rows,
                'scope_payload_bounds_bytes': {'draft_root': legacy_root,
                                               'target_root_with_pipe': target_root,
                                               'target_root_with_c16': target_c16_root,
                                               'gdn_c16_declared_workspace': c16_workspace_bound,
                                               'root_bound': root, 'baseline_root_bound': baseline_root_bound,
                                               'packed_sampler_workspace': packed_sampler_workspace,
                                               'teacher_packet_workspace': teacher_packet_workspace, 'layer': layer,
                                               'legacy_branch_with_wrap': branch,
                                               'generated_ring3_payload': 0},
                'registered_model_programs': kinds,
                'all_visited_windows_bytes': retained,
                'persistent_l3_domains': persistent,
                'reset_persistent_windows': False,
                'communication_epoch_policy': 'program-owned-128-target-3-draft-retirement',
                'domain_policy': 'retain-all' if persistent else 'release-per-call',
                'retention_budget_bytes': 1 << 30,
                'communication_live_bytes': communication,
                'maximum_window_bytes': largest_window,
                'teacher_packet_bytes_per_rank': teacher_packet_bytes,
                'teacher_packet_host_bytes': 2 * teacher_packet_bytes,
                'scratch_bytes': scratch, 'host_argument_staging_bytes': staging,
                'reset_staging_reserve_bytes': reset,
                'native_future_margin_bytes': margin,
                'future_bytes_after_prepare': future}

    def _check_memory_admission(self):
        expected = sum((math.prod(shape) * torch.empty((), dtype=getattr(torch, dtype)).element_size() for _, shape, dtype in TP_BANK_SPECS))
        expected += sum((math.prod(shape) * torch.empty((), dtype=dtype).element_size() for shape, dtype in self.state_specs))
        required = expected + self.resource_policy['future_bytes_after_prepare']
        self.admission = []
        for rank in range(2):
            free, total = self.worker.device_memory_info(rank)
            self.admission.append({'rank': rank, 'free_after_prepare_bytes': free, 'total_bytes': total, 'weight_state_bytes': expected, 'required_future_bytes': required, 'headroom_bytes': free - required, 'arena_already_prepared': True})
            if required > free:
                raise RuntimeError(f'Rank {rank} needs {required} bytes after arena preparation, has {free}; resource_policy={self.resource_policy}')
        self._prefix_device_budget()

    def _allocate_host_slabs(self):
        rows = self._host_capacity_rows
        # One maximum-capacity slab per field, bounded by the global token
        # budget, rather than one full allocation for every B/R specialization.
        return {'ids': shared((2 * rows,)), 'counts': shared((2 * SLOTS,)),
                'row_offsets': shared((2 * SLOTS,)), 'row_sequences': shared((2 * rows,), fill=-1),
                'positions': shared((2 * SLOTS,)), 'slots': shared((2 * SLOTS,)),
                'cos': shared((2 * rows * 64,), torch.bfloat16, 1),
                'sin': shared((2 * rows * 64,), torch.bfloat16),
                'previous': shared((2 * rows * HIDDEN,), torch.bfloat16),
                'greedy_row_valid': shared((2 * 4 * SLOTS,), torch.int32),
                'resident_source_rows': shared((2 * rows,), torch.int32, -1),
                'resident_slot_rows': shared((2 * SLOTS,), torch.int32, -1)}

    def _host_buffers(self, batch, rows):
        if batch not in (1, 4, 16, 32) or rows not in self._row_buckets(batch):
            raise ValueError('Host view does not match a registered B/R shape')
        shapes = {'ids': (2, rows), 'counts': (2, batch),
                  'row_offsets': (2, batch), 'row_sequences': (2, rows),
                  'positions': (2, batch), 'slots': (2, batch),
                  'cos': (2, rows, 64), 'sin': (2, rows, 64),
                  'previous': (2, rows, HIDDEN)}
        return {name: self._host_slabs[name][:math.prod(shape)].view(shape)
                for name, shape in shapes.items()}

    def _meta_args(self, batch, rows, drafter=False):
        def meta(shape, dtype): return torch.empty((2, *shape), dtype=dtype, device='meta')
        args = [meta((rows,), torch.int32)]
        if drafter: args.append(meta((rows, HIDDEN), torch.bfloat16))
        args += [meta(shape, getattr(torch, dtype)) for _, shape, dtype in TP_BANK_SPECS]
        args += [meta((batch,), torch.int32), meta((batch,), torch.int32),
                 meta((rows,), torch.int32), meta((batch,), torch.int32), meta((batch,), torch.int32)]
        args += [meta((rows, 64), torch.bfloat16) for _ in range(2)]
        args += [meta((32, self.table_blocks), torch.int32)]
        args += [meta(shape, dtype) for shape, dtype in self.state_specs]
        args += [meta((rows, HIDDEN), torch.bfloat16), meta((max(16, 4 * batch), TP_VOCAB), torch.bfloat16), 0]
        if not drafter:
            args.append(0)  # Runtime INT32 scalar, not a shape-specialization constant.
        args += [meta((max(16, 4 * batch),), torch.int32),
                 meta((max(16, 4 * batch), 16), torch.float32)]
        if drafter:
            args += [meta((32,), torch.int32), meta((32, HIDDEN), torch.bfloat16),
                     meta((32, 16), torch.int32),
                     meta((1 + max(16, 4 * batch) + 32, 16), torch.float32)]
        return args

    def compile_programs(self):
        from .model import prefix_restore_host, model_host, drafter_host, restore_host, zero_state_host, cache_copy_pages_host, resident_hidden_gather_host, resident_tail_update_host
        self.programs = {}
        self.programs['target'] = model_host.compile(*self._meta_args(1, 16), config=self.runconfig)
        if self.settings.speculative_tokens:
            self.programs['draft'] = drafter_host.compile(*self._meta_args(1, 16, True), config=self.runconfig)

        def meta(shape, dtype):
            return torch.empty((2, *shape), dtype=dtype, device='meta')
        ids = meta((32,), torch.int32)
        states = [meta(shape, dtype) for shape, dtype in self.state_specs]
        self.restore = restore_host.compile(ids, ids, states[4], states[5], states[0], states[1], config=self.runconfig)
        self.zero = zero_state_host.compile(ids, states[0], states[1], config=self.runconfig)
        self.copy_pages = cache_copy_pages_host.compile(ids, ids, states[2], states[3], config=self.runconfig)
        self.programs['resident_gather'] = resident_hidden_gather_host.compile(meta((16, HIDDEN), torch.bfloat16), meta((16,), torch.int32), meta((16, HIDDEN), torch.bfloat16), meta((16, 16), torch.int32), config=self.runconfig)
        self.programs['resident_tail_update'] = resident_tail_update_host.compile(meta((16, HIDDEN), torch.bfloat16), meta((32,), torch.int32), meta((32, HIDDEN), torch.bfloat16), meta((32, 16), torch.int32), config=self.runconfig)
        self.programs['prefix_restore'] = prefix_restore_host.compile(ids, meta(CONV_SHAPE, torch.bfloat16), meta(RECURRENT_SHAPE, torch.float32), states[0], states[1], config=self.runconfig)
        return list(self.programs.values()) + [self.restore, self.zero, self.copy_pages]

    def _allocate(self, shape, dtype, hosts=None):
        shards = tuple(self.worker.alloc_tensor(shape, dtype, init=hosts[r] if hosts else None,
                                               worker_id=r) for r in range(2))
        return StackedDeviceTensor(shards, (2, *shape), (0, 1))

    def start(self):
        if self._terminal_device_error is not None:
            raise RuntimeError('Failed runner is terminal; construct a fresh runner after cleanup: ' + self._terminal_device_error)
        if self.worker is not None: raise RuntimeError('Runner already started')
        programs = self.compile_programs()
        self.host = {}
        self.outputs = {}
        self.tables = shared((2, 32, self.table_blocks))
        self.control_ids = shared((2, 32), fill=-1)
        self.control_values = shared((2, 32), fill=-1)
        config = AutoConfig.from_pretrained(self.settings.model_root, local_files_only=True).text_config
        rotary = Qwen3_5TextRotaryEmbedding(config)
        positions = torch.arange(self.settings.max_model_len).view(1, 1, -1).expand(3, 1, -1)
        dummy = torch.empty((1, 1, HIDDEN), dtype=torch.bfloat16)
        cos, sin = rotary(dummy, positions)
        self.cos, self.sin = cos[0].contiguous(), sin[0].contiguous()
        # Keep one prepared L3 worker. The constructor selected either bounded
        # retained domains with reset, or public per-call domain release.
        self.worker = DistributedWorker(
            programs, config=self.runconfig,
            inherited_host_tensors=[*self._host_slabs.values(), self.tables, self.control_ids, self.control_values],
            persistent=self.resource_policy['persistent_l3_domains'],
            reset_persistent_windows=self.resource_policy['reset_persistent_windows'])
        try:
            self.provenance['compiled_programs'] = []
            for program in programs:
                directory = Path(program.output_dir)
                files = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(directory.rglob('*'))
                         if path.is_file() and path.suffix in ('.cpp', '.so', '.o', '.json')}
                self.provenance['compiled_programs'].append(
                    {'output_dir': str(directory), 'files_sha256': files})
            self._check_memory_admission()
            loaders = [TextWeights(self.settings.model_root).sharded_banks(r) for r in range(2)]
            for spec, first, second in zip(TP_BANK_SPECS, *loaders, strict=True):
                name, shape, dtype = spec
                if first[0] != name or second[0] != name: raise RuntimeError('Weight ABI order mismatch')
                if tuple(first[1].shape) != shape or tuple(second[1].shape) != shape:
                    raise RuntimeError(f'Weight ABI shape mismatch: {name}')
                self.banks.append(self._allocate(shape, getattr(torch, dtype), (first[1], second[1])))
            self.resident = [self._allocate(shape, dtype) for shape, dtype in self.state_specs]
            # Scratch is allocated for the current (B,R) only, after the previous
            # synchronous run/D2H completes. All retained teacher hidden is CPU-owned.
            self._scratch_key = None
            self._sample_memory()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                primary.add_note('Runner startup cleanup failed: ' + repr(cleanup))
            raise

    @property
    def terminal_device_error(self):
        return self._terminal_device_error

    def _require_healthy_worker(self):
        if self._terminal_device_error is not None:
            raise RuntimeError('Failed runner cannot reuse communication epochs: ' + self._terminal_device_error)
        if self.worker is None:
            raise RuntimeError('Runner has no live worker')

    def _worker_call(self, method, *args, **kwargs):
        self._require_healthy_worker()
        try:
            return getattr(self.worker, method)(*args, **kwargs)
        except BaseException as error:
            # No retry or in-executor teardown. Scheduler propagates to its
            # init-owner thread, whose finally closes the whole worker/domains.
            self._terminal_device_error = repr(error)
            raise

    def close(self):
        if getattr(self, '_teacher_tail_route', None) is not None:
            raise RuntimeError('Cannot close during teacher tail consumption')
        if getattr(self, '_prefix_device_inflight', False):
            raise RuntimeError('Cannot release/close during prefix restore')
        if getattr(self, '_greedy_inflight', False):
            raise RuntimeError('Cannot close during packed decision read')
        worker = self.worker
        if worker is not None:
            try:
                worker.close()
            except BaseException as error:
                if self._terminal_device_error is None:
                    self._terminal_device_error = repr(error)
                raise
            self.worker = None
        self._kind_scratch_keys = {}
        self.host = {}
        self.outputs = {}
        self.banks.clear()
        self.resident.clear()
        self._greedy_result = None
        self._greedy_inflight = False
        self._last_layout = None
        self._resident_previous = None
        self._resident_tails = None
        self._resident_previous_rows = None
        self._resident_prev_busy = False
        self._resident_readers = {}
        if hasattr(self, '_kind_epochs'):
            for kind in self._kind_epochs:
                self._kind_epochs[kind] += 1
            self._resident_generations = [g + 1 for g in self._resident_generations]
        self._prefix_device_mirrors = {}
        self._prefix_device_capacity = 0

    def _sample_memory(self):
        self.memory_stats = []
        for rank in range(2):
            free, total = self.worker.device_memory_info(rank)
            self.memory_stats.append({'rank': rank, 'committed_bytes': self.worker.committed_device_memory(rank),
                                      'hbm_used_bytes': total - free, 'hbm_total_bytes': total})

    def _copy_page(self, source, destination):
        self.control_ids.fill_(-1); self.control_values.fill_(-1)
        self.control_ids[:, 0] = source; self.control_values[:, 0] = destination
        self._worker_call('run', self.copy_pages, self.control_ids, self.control_values,
                        self.resident[2], self.resident[3], config=self.runconfig)

    def _state_copy(self, slot, index, host, to_device):
        self._require_healthy_worker()
        for rank in range(2):
            tensor = host[rank].contiguous()
            device = self.resident[index].shards[rank]
            offset = slot * tensor.nbytes
            if to_device:
                self._worker_call('copy_to', device.data_ptr, tensor.data_ptr(), tensor.nbytes,
                                    dst_offset=offset, worker_id=rank)
            else:
                self._worker_call('copy_from', tensor.data_ptr(), device.data_ptr, tensor.nbytes,
                                      src_offset=offset, worker_id=rank)
                host[rank].copy_(tensor)

    def admit(self, request: RequestState):
        if getattr(self, '_teacher_tail_route', None) is not None:
            raise RuntimeError('Cannot admit during teacher tail consumption')
        if getattr(self, '_prefix_device_inflight', False):
            raise RuntimeError('Admit during prefix restore is forbidden')
        self._require_healthy_worker()
        if not self.free_slots:
            raise RuntimeError('No free device request slot')
        slot = self.free_slots.pop()
        entry = self.cache.find_prefix(request.prompt_ids) if self.allow_prefix_hits else None
        pages = self.cache.attach_prefix(entry) if entry else self.cache.new_request()
        record = {'slot': slot, 'pages': pages, 'draft_probs': None, 'draft_hidden': None, 'mtp_seed': entry.hidden if entry and self.settings.speculative_tokens else None}
        self._resident_state()
        self._resident_generations[slot] += 1
        record['resident_generation'] = self._resident_generations[slot]
        self.requests[request.request_id] = record
        try:
            if entry:
                self._prefix_device_restore(entry, slot)
            else:
                self.control_ids.fill_(-1)
                self.control_ids[:, 0] = slot
                self._worker_call('run', self.zero, self.control_ids, self.resident[0], self.resident[1], config=self.runconfig)
            self.tables[:, slot].copy_(self.cache.table(pages))
            return pages.length
        except BaseException:
            self.release(request)
            raise

    def release(self, request):
        if getattr(self, '_teacher_tail_route', None) is not None:
            raise RuntimeError('Cannot release during teacher tail consumption')
        if getattr(self, '_prefix_device_inflight', False):
            raise RuntimeError('Cannot release/close during prefix restore')
        self._resident_state()
        if self._resident_prev_busy or any(self._resident_readers.values()):
            raise RuntimeError('Release must follow blocking resident consumers')
        record = self.requests.pop(request.request_id, None)
        if record is not None:
            self._resident_generations[record['slot']] += 1
            self.cache.release(record['pages'])
            self.tables[:, record['slot']].zero_()
            self.free_slots.append(record['slot'])

    @staticmethod
    def _bucket(count): return next(b for b in (1, 4, 16, 32) if count <= b)

    def _scratch(self, kind, batch, rows):
        if kind not in ('target', 'draft'):
            raise ValueError('Unknown scratch owner')
        keys = getattr(self, '_kind_scratch_keys', {})
        key = (kind, batch, rows)
        if keys.get(kind) != key:
            old = keys.get(kind)
            if old is not None and getattr(self, '_resident_readers', {}).get(old, 0):
                raise RuntimeError('Cannot rekey scratch with resident readers')
            try:
                if old is not None:
                    for tensor in self.outputs.pop(old):
                        for rank, shard in enumerate(tensor.shards):
                            self._worker_call('free_tensor', shard, worker_id=rank)
                self.host[kind] = self._host_buffers(batch, rows)
                self.outputs[key] = (self._allocate((rows, HIDDEN), torch.bfloat16),
                                     self._allocate((max(16, 4 * batch), TP_VOCAB), torch.bfloat16),
                                     self._allocate((max(16, 4 * batch), 16), torch.float32))
                if kind == 'draft':
                    self.outputs[key] += (self._allocate((1 + max(16, 4 * batch) + 32, 16), torch.float32),)
                keys[kind] = key
                self._kind_scratch_keys = keys
                self._sample_memory()
            except BaseException as error:
                if self._terminal_device_error is None:
                    self._terminal_device_error = repr(error)
                raise
        return self.host[kind], self.outputs[key]

    def _execute(self, kind, slots, sequences, positions, previous=None, snapshot_mode=0):
        owns_route = kind == 'draft'
        created_route = False
        try:
            if owns_route:
                self._begin_teacher_tail_route(slots, sequences)
                created_route = True
            return self._execute_model(kind, slots, sequences, positions, previous, snapshot_mode)
        except BaseException as error:
            if owns_route and self._terminal_device_error is None:
                self._terminal_device_error = repr(error)
            raise
        finally:
            if created_route:
                self._teacher_tail_route = None

    def _execute_model(self, kind, slots, sequences, positions, previous=None, snapshot_mode=0):
        if kind not in ('target', 'draft') or not slots:
            raise ValueError('Packed model execution requires a supported kind and nonempty request batch')
        if len(slots) != len(sequences) or len(slots) != len(positions):
            raise ValueError('Packed metadata length mismatch')
        batch = self._bucket(len(slots))
        counts = [len(ids) for ids in sequences]
        actual_rows = sum(counts)
        if not 0 < actual_rows <= self.settings.max_num_batched_tokens:
            raise ValueError('Packed model token count exceeds the global budget')
        rows = next((r for r in self._row_buckets(batch) if r >= actual_rows), None)
        if rows is None:
            raise ValueError('Packed rows exceed the configured per-request chunk capacity')
        offsets = []; total = 0
        for slot, count, position in zip(slots, counts, positions, strict=True):
            offsets.append(total); total += count
            limit = (self.chunk_threshold or self.settings.max_model_len) if slot.phase == 'prefill' else 1 + self.settings.speculative_tokens
            if count > limit or position < 0 or position + count > self.settings.max_model_len:
                raise ValueError('Model rows exceed the configured position/chunk contract')
        self._resident_state()
        current = getattr(self, '_kind_scratch_keys', {}).get(kind)
        if self._resident_readers.get(current, 0):
            raise RuntimeError('Producer overwrite would invalidate an in-flight resident reader')
        self._kind_epochs[kind] += 1
        host, output = self._scratch(kind, batch, rows)
        device_previous = isinstance(previous, ResidentPrevious)
        if device_previous:
            if previous.owner is not self or previous.counts != tuple(counts) or not self._resident_prev_busy or previous.version != self._resident_previous_version or previous.buffer is not self._resident_previous[0]:
                raise RuntimeError('Invalid packed Device previous ownership')
            for value in previous.sources:self._validate_resident(value)
        for name in ('ids', 'counts', 'positions', 'slots', 'sin', 'previous'):
            if name != 'previous' or not device_previous:host[name].zero_()
        host['row_offsets'].fill_(actual_rows)
        host['row_sequences'].fill_(-1)
        host['cos'].fill_(1)
        for i, (slot, ids, position, offset) in enumerate(zip(slots, sequences, positions, offsets, strict=True)):
            count = len(ids)
            host['row_offsets'][:, i] = offset
            if not count:
                continue
            record = self.requests[slot.request.request_id]
            host['ids'][:, offset:offset+count] = torch.tensor(ids, dtype=torch.int32)
            host['counts'][:, i] = count
            host['row_sequences'][:, offset:offset+count] = i
            host['positions'][:, i] = position
            host['slots'][:, i] = record['slot']
            host['cos'][:, offset:offset+count] = self.cos[position:position+count]
            host['sin'][:, offset:offset+count] = self.sin[position:position+count]
            if previous is not None and not device_previous:
                if tuple(previous[i].shape) != (2, count, HIDDEN):
                    raise ValueError('Teacher hidden does not match packed tokens')
                host['previous'][:, offset:offset+count] = previous[i]
        args = [host['ids']]
        if kind == 'draft':
            args.append(previous.buffer if device_previous else host['previous'])
        args += self.banks + [host[k] for k in
                             ('counts', 'row_offsets', 'row_sequences', 'positions', 'slots', 'cos', 'sin')]
        # The validity mask is fixed BEFORE the producer runs. Inactive,
        # zero-count, padded and stochastic rows must never scan/sample logits.
        decision_rows = max(16, 4 * batch)
        valid = self._host_slabs['greedy_row_valid'][:2 * decision_rows].view(2, decision_rows)
        valid.zero_()
        greedy_rows = []
        for index, slot in enumerate(slots):
            if slot.request.sampling.temperature == 0:
                keep = min(counts[index], 4)
                valid[:, 4 * index:4 * index + keep] = 1
                greedy_rows.extend(range(4 * index, 4 * index + keep))
        args += [self.tables] + self.resident + list(output[:2]) + [snapshot_mode]
        if kind == 'target':
            args.append(max(counts, default=0))
        args += [valid, output[2]]
        if kind == 'draft':
            args += [self._teacher_tail_route['mapping'], self._resident_tails,
                     self._resident_tail_status, output[3]]
        program = self.programs[kind]
        self._greedy_result = None  # New content version, even when B/R and pointers repeat.
        self._last_layout = None  # A failed producer must not expose prior decisions.
        started = time.monotonic_ns()
        self._worker_call('run', program, *args, config=self.runconfig)
        decisions = self._read_teacher_return(output[3], decision_rows, self._teacher_tail_route['updated'], greedy_rows) if kind == 'draft' else None
        self.calls += 1
        mapping = [{'request_id': s.request.request_id,
                    'slot': self.requests[s.request.request_id]['slot'],
                    'phase': s.phase, 'position': position, 'tokens': count, 'row_offset': offset}
                   for s, position, count, offset in zip(slots, positions, counts, offsets, strict=True)]
        self.invocations.append({'id': self.calls, 'kind': kind, 'bucket': batch,
            'effective_batch': sum(count > 0 for count in counts),
            'allocated_rows': rows, 'actual_rows': actual_rows,
            'max_request_tokens': max(counts, default=0),
            'row_layout': 'packed-prefix-sum-global-tail',
            'program_identity': str(program.output_dir),
            'long_prefill_token_threshold': self.chunk_threshold,
            'prefill_chunk_mode': 'budget' if self.chunk_threshold == 0 else 'capped',
            'elapsed_ns': time.monotonic_ns()-started, 'slots': mapping})
        self.invocations = self.invocations[-128:]
        totals = self.execution_totals
        totals['max_actual_rows'] = max(totals['max_actual_rows'], actual_rows)
        totals['max_allocated_rows'] = max(totals['max_allocated_rows'], rows)
        if kind == 'target':
            prefill = [count for s, count in zip(slots, counts) if s.phase == 'prefill' and count]
            totals['target_batches'][str(batch)] += 1
            totals['prefill_calls'] += bool(prefill)
            totals['prefill_tokens'] += sum(prefill)
            totals['max_prefill_chunk'] = max(totals['max_prefill_chunk'], max(prefill, default=0))
            totals['mixed_target_calls'] += bool(prefill) and any(
                s.phase == 'decode' and count for s, count in zip(slots, counts))
        self._last_layout = {'key': (batch, rows), 'output_key': (kind, batch, rows), 'counts': counts,
                             'requests': [s.request.request_id for s in slots],
                             'greedy_indices': [i for i, s in enumerate(slots) if s.request.sampling.temperature == 0],
                             'greedy_rows': tuple(greedy_rows), 'decision_epoch': self._kind_epochs[kind]}
        if kind == 'draft':
            version = (self.calls, (kind, batch, rows), self._kind_epochs[kind])
            self._greedy_result = (version, decisions)
            hidden = self._commit_teacher_tails(slots, counts, self._teacher_tail_route['updated'])
        else:
            hidden = []
            for slot,count,offset in zip(slots,counts,offsets,strict=True):
                record=self.requests[slot.request.request_id]
                hidden.append(ResidentRows(self,slot.request.request_id,record['slot'],record['resident_generation'],
                    kind,self._kind_epochs[kind],(kind,batch,rows),range(offset,offset+count)))
        return batch, hidden

    def _distribution(self, batch, index, row, request):
        layout = self._last_layout
        if (layout is None or layout['key'][0] != batch or
                not 0 <= index < len(layout['counts']) or layout['requests'][index] != request.request_id):
            raise RuntimeError('Logit lookup does not match the last packed execution')
        count = layout['counts'][index]
        keep = min(count, 4)
        if not count - keep <= row < count:
            raise ValueError('Requested row was not retained in compact logits')
        logit_row = 4 * index + row - (count - keep)
        if request.sampling.temperature == 0:
            return self._compact_greedy(logit_row)
        parts = []
        for rank in range(2):
            value = torch.empty(TP_VOCAB, dtype=torch.bfloat16)
            self._worker_call('copy_from', value.data_ptr(), self.outputs[layout['output_key']][1].shards[rank].data_ptr,
                                  value.nbytes, src_offset=logit_row*TP_VOCAB*2, worker_id=rank)
            parts.append(value)
        return probabilities(torch.cat(parts), request.sampling)

    def _sample(self, probs, request):
        if type(probs) is int:
            if request.sampling.temperature != 0:
                raise TypeError('Compact greedy decision cannot serve stochastic sampling')
            return probs
        return int(probs.argmax()) if request.sampling.temperature == 0 else draw(probs, request.generator)

    @staticmethod
    def _merge_packed_decisions(host, valid_rows, rows):
        if host.dtype != torch.float32 or tuple(host.shape) != (2, rows, 16):
            raise ValueError('Packed decision dtype/shape mismatch')
        if not bool(torch.isfinite(host).all()):
            raise ValueError('Packed decision contains nonfinite data')
        if not bool((host[:, :, 3:] == 0).all()):
            raise ValueError('Packed decision padding was not written zero')
        if len(set(valid_rows)) != len(valid_rows) or any(type(r) is not int or not 0 <= r < rows for r in valid_rows):
            raise ValueError('Invalid packed decision row ownership')
        valid = set(valid_rows)
        merged = {}
        for row in range(rows):
            candidates = []
            for rank in range(2):
                token, score, status = map(float, host[rank, row, :3])
                if token != math.trunc(token) or status != math.trunc(status) or status not in (0.0, 1.0, 2.0):
                    raise ValueError('Packed token/status must be exact integers')
                if row not in valid:
                    if (token, score, status) != (-1.0, 0.0, 0.0):
                        raise RuntimeError('Inactive/stochastic decision row was not masked')
                else:
                    if status != 1.0:
                        raise ValueError('Packed greedy producer reported invalid logits')
                    if not rank * TP_VOCAB <= token < (rank + 1) * TP_VOCAB:
                        raise RuntimeError('Packed candidate outside actual TP vocabulary shard')
                    candidates.append((score, int(token)))
            if row in valid:
                merged[row] = min(candidates, key=lambda item: (-item[0], item[1]))[1]
        return merged

    def _compact_greedy(self, logit_row):
        self._require_healthy_worker()
        layout = self._last_layout
        if layout is None or logit_row not in layout['greedy_rows']:
            raise RuntimeError('Greedy read is not an authorized current-content row')
        key = layout['output_key']
        kind = key[0]
        if key not in self.outputs or self._kind_epochs[kind] != layout['decision_epoch']:
            raise RuntimeError('Stale packed decision owner/generation')
        version = (self.calls, key, layout['decision_epoch'])
        cached = getattr(self, '_greedy_result', None)
        if cached is not None and cached[0] == version:
            return cached[1][logit_row]
        if getattr(self, '_greedy_inflight', False):
            raise RuntimeError('Packed decision read already in flight')
        rows = max(16, 4 * layout['key'][0])
        self._greedy_inflight = True
        self._resident_readers[key] = self._resident_readers.get(key, 0) + 1
        try:
            device = self.outputs[key][2]
            host = torch.empty((2, rows, 16), dtype=torch.float32)
            # Each rank decision was produced in the completed model L3; no
            # separate sampler run, no full-vocab greedy CPU scan and no replay.
            for rank in range(2):
                self._worker_call('copy_from', host[rank].data_ptr(), device.shards[rank].data_ptr,
                                  host[rank].nbytes, src_offset=0, worker_id=rank)
            merged = self._merge_packed_decisions(host, layout['greedy_rows'], rows)
            self._greedy_result = (version, merged)
            return merged[logit_row]
        except BaseException as error:
            if self._terminal_device_error is None:
                self._terminal_device_error = repr(error)
            raise
        finally:
            self._resident_readers[key] -= 1
            self._greedy_inflight = False

    def _verify_distributions(self, target, proposals, draft_ids, params, generator):
        if params.temperature != 0:
            return verify_drafts(target, proposals, draft_ids, params, generator)
        if len(target) != len(draft_ids) + 1 or len(proposals) != len(draft_ids):
            raise ValueError('Speculative probability layout mismatch')
        if not all(type(token) is int for token in target + proposals):
            raise TypeError('Device greedy requires compact integer decisions')
        accepted = []
        for index, token in enumerate(draft_ids):
            if proposals[index] != token:
                raise RuntimeError('Draft proposal is not its compact greedy decision')
            if token != target[index]:
                return accepted + [target[index]], len(accepted)
            accepted.append(token)
        return accepted + [target[-1]], len(accepted)

    def _reserve(self, slot, end):
        record = self.requests[slot.request.request_id]
        try:
            self.cache.reserve(record['pages'], end, self._copy_page)
            self.tables[:, record['slot']].copy_(self.cache.table(record['pages']))
        except Exception as exc:
            if self._terminal_device_error is not None:
                raise  # A failed devicecopy is not a recoverable capacity rejection.
            raise RequestCapacityError(slot.request, str(exc)) from exc

    def _resident_state(self):
        if not hasattr(self, '_resident_generations'):
            self._resident_generations = [0] * 32
            self._resident_tail_versions = [0] * 32
            self._kind_epochs = {'target': 0, 'draft': 0}
            self._resident_readers = {}
            self._resident_prev_busy = False
            self._resident_previous_version = 0
            self._resident_tail_readers = 0
            self._resident_previous = None
            self._resident_previous_rows = None
            self._resident_tails = None

    def _validate_resident(self, value):
        self._require_healthy_worker()
        if not isinstance(value, ResidentRows) or value.owner is not self:
            raise RuntimeError('Foreign resident hidden owner')
        record = self.requests.get(value.request_id)
        if record is None or record['slot'] != value.slot or record['resident_generation'] != value.generation or self._resident_generations[value.slot] != value.generation:
            raise RuntimeError('Stale request/slot generation')
        if value.kind == 'tails':
            if self._resident_tail_versions[value.slot] != value.version:
                raise RuntimeError('Stale drafter tail content')
        elif self._kind_epochs[value.kind] != value.version or value.output_key not in self.outputs:
            raise RuntimeError('Stale model hidden content epoch')
        return self._resident_tails if value.kind == 'tails' else self.outputs[value.output_key][0]

    def _resident_status(self, device, expected):
        rows = len(expected)
        for rank in range(2):
            status = torch.empty((rows, 16), dtype=torch.int32)
            self._worker_call('copy_from', status.data_ptr(), device.shards[rank].data_ptr,
                              status.nbytes, src_offset=0, worker_id=rank)
            if status[:, 0].tolist() != expected or bool((status[:, 1:] != 0).any()):
                raise RuntimeError('Resident row-copy status mismatch')

    def _resident_gather_previous(self, slots, sequences, values):
        self._resident_state()
        if self._resident_prev_busy:
            raise RuntimeError('Packed previous already has an in-flight consumer')
        counts = tuple(map(len, sequences))
        active = [(i, value) for i, (count, value) in enumerate(zip(counts, values, strict=True)) if count]
        sources = [self._validate_resident(value) for _, value in active]
        if not sources or any(source is not sources[0] for source in sources):
            raise RuntimeError('Teacher requires one owned resident source batch')
        rowmap = []
        for i, value in active:
            if value.request_id != slots[i].request.request_id or len(value.rows) != counts[i]:
                raise ValueError('Teacher resident rows do not match request tokens')
            rowmap.extend(value.rows)
        rows = next(r for r in self._row_buckets(self._bucket(len(slots))) if r >= sum(counts))
        if self._resident_previous_rows != rows:
            if self._resident_previous is not None:
                for tensor in self._resident_previous:
                    for rank, shard in enumerate(tensor.shards):
                        self.worker.free_tensor(shard, worker_id=rank)
            self._resident_previous = []
            self._resident_previous_rows = None
            self._resident_previous.append(self._allocate((rows, HIDDEN), torch.bfloat16))
            self._resident_previous.append(self._allocate((rows, 16), torch.int32))
            self._resident_previous_rows = rows
        mapping = self._host_slabs['resident_source_rows'][:2 * rows].view(2, rows)
        mapping.fill_(-1)
        mapping[:, :len(rowmap)] = torch.tensor(rowmap, dtype=torch.int32)
        # Hold the source until the blocking Device gather has consumed it.
        keys = {value.output_key for _, value in active if value.kind != 'tails'}
        for key in keys:self._resident_readers[key] = self._resident_readers.get(key, 0) + 1
        tail_read = any(value.kind == 'tails' for _, value in active)
        self._resident_tail_readers += int(tail_read)
        self._resident_prev_busy = True
        try:
            self._worker_call('run', self.programs['resident_gather'], sources[0], mapping,
                              *self._resident_previous, config=self.runconfig)
            self._resident_status(self._resident_previous[1], [1] * len(rowmap) + [0] * (rows-len(rowmap)))
        except BaseException:
            self._resident_prev_busy = False
            raise
        finally:
            for key in keys:self._resident_readers[key] -= 1
            self._resident_tail_readers -= int(tail_read)
        self._resident_previous_version += 1
        return ResidentPrevious(self, self._resident_previous[0], counts, tuple(value for _, value in active), self._resident_previous_version)

    def _resident_update_tails(self, slots, counts, offsets, source):
        self._resident_state()
        if self._resident_tail_readers:
            raise RuntimeError('Tail update would overwrite an in-flight gather source')
        if self._resident_tails is None:
            self._resident_tails = self._allocate((32, HIDDEN), torch.bfloat16)
            self._resident_tail_status = self._allocate((32, 16), torch.int32)
        mapping = self._host_slabs['resident_slot_rows'].view(2, 32)
        mapping.fill_(-1); updated=[]
        for i, (slot, count, offset) in enumerate(zip(slots, counts, offsets, strict=True)):
            if not count:continue
            record = self.requests[slot.request.request_id];destination=record['slot']
            if destination in updated:raise RuntimeError('Duplicate resident tail destination')
            updated.append(destination);mapping[:, destination] = offset + count - 1
        self._worker_call('run', self.programs['resident_tail_update'], source, mapping,
                          self._resident_tails, self._resident_tail_status, config=self.runconfig)
        self._resident_status(self._resident_tail_status, [int(i in updated) for i in range(32)])
        for slot in updated:self._resident_tail_versions[slot] += 1
        result=[]
        for slot,count in zip(slots,counts,strict=True):
            record=self.requests[slot.request.request_id];destination=record['slot']
            result.append(ResidentRows(self,slot.request.request_id,destination,record['resident_generation'],
                'tails',self._resident_tail_versions[destination],None,(destination,) if count else ()))
        return result

    def _resident_prefix_cpu(self, value):
        source=self._validate_resident(value)
        if not value.rows:raise ValueError('Cannot cache an empty prefix hidden')
        result=torch.empty((2,HIDDEN),dtype=torch.bfloat16)
        key=value.output_key
        self._resident_readers[key]=self._resident_readers.get(key,0)+1
        try:
            for rank in range(2):
                self._worker_call('copy_from',result[rank].data_ptr(),source.shards[rank].data_ptr,
                    result[rank].nbytes,src_offset=value.rows[-1]*HIDDEN*2,worker_id=rank)
        finally:self._resident_readers[key]-=1
        return result

    def _teacher(self, slots, sequences, positions, hidden):
        if not any(sequences):return
        active=[value for ids,value in zip(sequences,hidden,strict=True) if ids]
        resident=all(isinstance(value,ResidentRows) for value in active)
        if not resident and any(isinstance(value,ResidentRows) for value in active):
            raise TypeError('Mixed CPU prefix seeds/resident rows require explicit routing')
        # Cache prefix seeds remain the explicit old owned-CPU path.
        owns_previous = False
        try:
            previous=self._resident_gather_previous(slots,sequences,hidden) if resident else hidden
            owns_previous = resident
            batch,result=self._execute('draft',slots,sequences,positions,previous=previous)
            for i,ids in enumerate(sequences):
                if ids:
                    record=self.requests[slots[i].request.request_id]
                    record['draft_probs']=self._distribution(batch,i,len(ids)-1,slots[i].request)
                    record['draft_hidden']=result[i]
        except BaseException as error:
            if self._terminal_device_error is None:
                self._terminal_device_error = repr(error)
            raise
        finally:
            if owns_previous:self._resident_prev_busy=False


    def run_batch(self, slots):
        self._require_healthy_worker()
        if not slots: return []
        mtp = bool(self.settings.speculative_tokens)
        sequences = [list(s.token_ids) for s in slots]
        positions = [s.position for s in slots]
        drafts = [[] for _ in slots]; proposals = [[] for _ in slots]
        for i,s in enumerate(slots):
            count = len(s.token_ids) if s.phase == 'prefill' else 1 + min(
                self.settings.speculative_tokens, max(0, s.request.max_tokens-len(s.request.generated)-1))
            self._reserve(s, s.position + count + (1 if mtp else 0))
        if mtp:
            seeds = []; seed_pos = []; seed_hidden = []
            for s in slots:
                r = self.requests[s.request.request_id]
                seeds.append([s.token_ids[0]] if r['mtp_seed'] is not None else [])
                seed_pos.append(s.position)
                seed_hidden.append(r['mtp_seed'].reshape(2,1,HIDDEN) if r['mtp_seed'] is not None else None)
            self._teacher(slots, seeds, seed_pos, seed_hidden)
            for s in slots: self.requests[s.request.request_id]['mtp_seed'] = None
            started = time.monotonic_ns()
            for step in range(3):
                advance = []; advance_pos = []; advance_hidden = []
                for i,s in enumerate(slots):
                    count = min(3, max(0,s.request.max_tokens-len(s.request.generated)-1)) if s.phase=='decode' else 0
                    r = self.requests[s.request.request_id]
                    if step < count:
                        q = r['draft_probs']
                        if q is None: raise RuntimeError('Missing conditioned MTP state')
                        token = self._sample(q,s.request);drafts[i].append(token);proposals[i].append(q)
                        advance.append([token] if step+1<count else [])
                        advance_hidden.append(r['draft_hidden'])
                    else:
                        advance.append([]);advance_hidden.append(None)
                    advance_pos.append(s.position+step+1)
                self._teacher(slots, advance, advance_pos, advance_hidden)
            self.spec_stats['draft_ns'] += time.monotonic_ns()-started
            for i,s in enumerate(slots):
                if s.phase=='decode': sequences[i] += drafts[i]
        started = time.monotonic_ns()
        batch, target_hidden = self._execute('target', slots, sequences, positions,
                                             snapshot_mode=int(mtp and any(s.phase=='decode' for s in slots)))
        self.spec_stats['verify_ns'] += time.monotonic_ns()-started
        outputs = []; accepted_counts = []
        for i,s in enumerate(slots):
            if s.phase=='prefill':
                result = [self._sample(self._distribution(batch,i,len(sequences[i])-1,s.request),s.request)] if s.completes_prefill else None
                accepted_counts.append(len(sequences[i]))
            elif mtp:
                target = [self._distribution(batch,i,t,s.request) for t in range(len(sequences[i]))]
                result, accepted = self._verify_distributions(target,proposals[i],drafts[i],s.request.sampling,s.request.generator)
                accepted_counts.append(accepted+1)
                self.spec_stats['proposed'] += len(drafts[i]);self.spec_stats['accepted'] += accepted
                self.spec_stats['rejected_rounds'] += int(accepted<len(drafts[i]))
            else:
                result = [self._sample(self._distribution(batch,i,0,s.request),s.request)]
                accepted_counts.append(1)
            outputs.append(result)
        self.control_ids.fill_(-1);self.control_values.fill_(-1)
        for i,s in enumerate(slots):
            if s.phase=='decode' and accepted_counts[i]<len(sequences[i]):
                self.control_ids[:,i] = self.requests[s.request.request_id]['slot']
                self.control_values[:,i] = accepted_counts[i]
        if bool((self.control_ids>=0).any()):
            self._worker_call('run', self.restore,self.control_ids,self.control_values,self.resident[4],self.resident[5],
                            self.resident[0],self.resident[1],config=self.runconfig)
        for i,s in enumerate(slots):
            r = self.requests[s.request.request_id];end = s.position+accepted_counts[i]
            self.cache.commit(r['pages'],end)
            if self.allow_prefix_hits and s.phase=='prefill' and end==32:
                key = tuple(s.request.prompt_ids[:32])
                if key not in self.cache.prefixes:
                    conv=torch.empty((2,*CONV_SHAPE),dtype=torch.bfloat16)
                    recurrent=torch.empty((2,*RECURRENT_SHAPE),dtype=torch.float32)
                    self._state_copy(r['slot'],0,conv,False);self._state_copy(r['slot'],1,recurrent,False)
                    self.cache.store_prefix(key,r['pages'],conv,recurrent,self._resident_prefix_cpu(target_hidden[i]))
        if mtp:
            next_ids=[];next_pos=[];previous=[]
            for i,s in enumerate(slots):
                count=accepted_counts[i]
                if s.phase=='prefill':
                    tokens=s.request.prompt_ids[s.position+1:s.position+count+1]
                    if len(tokens)<count: tokens=tokens+[outputs[i][0]]
                else: tokens=outputs[i]
                next_ids.append(tokens);next_pos.append(s.position+1)
                previous.append(target_hidden[i][:,:count])
            self._teacher(slots,next_ids,next_pos,previous)
        return outputs

    def stats(self):
        return {'calls':self.calls,'tp':2,'dp':1,'execution_mode':'unified',
                'long_prefill_token_threshold': self.chunk_threshold,
            'prefill_chunk_mode': 'budget' if self.chunk_threshold == 0 else 'capped',
                'host_dispatch_capacity_rows': self._host_capacity_rows,
                'host_dispatch_storage_bytes': sum(t.numel() * t.element_size() for t in self._host_slabs.values()),
                'execution_totals': self.execution_totals,
                'resource_policy': self.resource_policy, 'memory_admission': self.admission,
                'pool_blocks':self.pool_blocks,'block_size':BLOCK_SIZE,
                'speculative':dict(self.spec_stats),'memory':self.memory_stats,
                'cache':self.cache.stats(),'provenance':self.provenance,
                'recent_invocations':list(self.invocations)}

    def _prefix_device_budget(self):
        entry_bytes = (math.prod(CONV_SHAPE) * 2 + math.prod(RECURRENT_SHAPE) * 4)
        headroom = min(record['headroom_bytes'] for record in self.admission)
        capacity = min(self.cache.max_prefix_entries, max(0, headroom // entry_bytes))
        self._prefix_device_capacity = capacity
        self._prefix_device_inflight = False
        self._prefix_device_mirrors = {}
        for record in self.admission:
            record['prefix_device_reserved_bytes'] = capacity * entry_bytes
            record['prefix_device_capacity'] = capacity
            record['required_future_bytes'] += capacity * entry_bytes
            record['headroom_bytes'] -= capacity * entry_bytes


    def _prefix_device_free(self, key):
        record = self._prefix_device_mirrors[key]
        for tensor in record['buffers']:
            for rank, shard in enumerate(tensor.shards):
                self._worker_call('free_tensor', shard, worker_id=rank)
        del self._prefix_device_mirrors[key]


    def _prefix_device_restore(self, entry, slot):
        self._require_healthy_worker()
        if not 0 <= slot < self.settings.max_num_seqs or self.cache.prefixes.get(entry.token_ids) is not entry:
            raise RuntimeError('Invalid prefix entry or destination slot')
        if self._prefix_device_inflight:
            raise RuntimeError('Prefix mirror already has an in-flight consumer')
        if self._prefix_device_capacity == 0:
            self._state_copy(slot, 0, entry.conv, True)
            self._state_copy(slot, 1, entry.recurrent, True)
            return 'CPU-fallback-capacity-zero'
        self._prefix_device_inflight = True
        try:
            for key, record in list(self._prefix_device_mirrors.items()):
                if self.cache.prefixes.get(key) is not record['entry']:
                    self._prefix_device_free(key)
            key = entry.token_ids
            if key not in self._prefix_device_mirrors:
                while len(self._prefix_device_mirrors) >= self._prefix_device_capacity:
                    self._prefix_device_free(next(iter(self._prefix_device_mirrors)))
                buffers = []
                record = {'entry': entry, 'buffers': buffers, 'ready': False}
                self._prefix_device_mirrors[key] = record
                buffers.append(self._allocate(CONV_SHAPE, torch.bfloat16, (entry.conv[0], entry.conv[1])))
                buffers.append(self._allocate(RECURRENT_SHAPE, torch.float32, (entry.recurrent[0], entry.recurrent[1])))
                record['ready'] = True
            record = self._prefix_device_mirrors.pop(key)
            self._prefix_device_mirrors[key] = record
            if not record['ready'] or record['entry'] is not entry:
                raise RuntimeError('Incomplete or stale prefix device mirror')
            self.control_ids.fill_(-1)
            self.control_ids[:, 0] = slot
            self._worker_call('run', self.programs['prefix_restore'], self.control_ids,
                              *record['buffers'], self.resident[0], self.resident[1], config=self.runconfig)
            return 'Device-mirror'
        except BaseException as error:
            if self._terminal_device_error is None:
                self._terminal_device_error = repr(error)
            raise
        finally:
            self._prefix_device_inflight = False


    def _begin_teacher_tail_route(self, slots, sequences):
        self._require_healthy_worker()
        self._resident_state()
        if getattr(self, '_teacher_tail_route', None) is not None or self._resident_tail_readers:
            raise RuntimeError('Teacher tail route has an in-flight consumer')
        if len(slots) != len(sequences) or not 0 < sum(map(len, sequences)) <= self.settings.max_num_batched_tokens:
            raise ValueError('Invalid teacher tail batch')
        if self._resident_tails is None:
            self._resident_tails = self._allocate((32, HIDDEN), torch.bfloat16)
            self._resident_tail_status = self._allocate((32, 16), torch.int32)
        mapping = self._host_slabs['resident_slot_rows'].view(2, 32)
        mapping.fill_(-1)
        updated = []
        offset = 0
        for slot, ids in zip(slots, sequences, strict=True):
            count = len(ids)
            if count:
                record = self.requests[slot.request.request_id]
                destination = record['slot']
                if type(destination) is not int or not 0 <= destination < 32 or destination in updated:
                    raise RuntimeError('Invalid or duplicate teacher tail destination')
                if record['resident_generation'] != self._resident_generations[destination]:
                    raise RuntimeError('Stale teacher tail destination')
                updated.append(destination)
                mapping[:, destination] = offset + count - 1
            offset += count
        self._teacher_tail_route = {'mapping': mapping, 'updated': updated}

    @staticmethod
    def _validate_teacher_return(packet, logits, updated, greedy_rows):
        if packet.dtype != torch.float32 or tuple(packet.shape) != (2, 1 + logits + 32, 16):
            raise ValueError('Teacher source2 packet shape/dtype mismatch')
        if type(logits) is not int or not 16 <= logits <= 128 or logits % 16:
            raise ValueError('Invalid teacher logit extent')
        if len(set(updated)) != len(updated) or any(type(slot) is not int or not 0 <= slot < 32 for slot in updated):
            raise ValueError('Invalid teacher tail ownership')
        if not bool(torch.isfinite(packet).all()):
            raise ValueError('Nonfinite teacher source2 packet')
        header = torch.zeros((2, 16), dtype=torch.float32)
        header[:, :3] = torch.tensor([100, logits, 32], dtype=torch.float32)
        if not torch.equal(packet[:, 0, :], header):
            raise RuntimeError('Teacher source2 packet ABI/header mismatch')
        tail = packet[:, 1 + logits:, :]
        expected = torch.tensor([int(slot in updated) for slot in range(32)], dtype=torch.float32)
        if not torch.equal(tail[:, :, 0], expected.expand(2, -1)) or bool((tail[:, :, 1:] != 0).any()):
            raise RuntimeError('Teacher source2 full32 tail status mismatch')
        return TensorParallelRunner._merge_packed_decisions(packet[:, 1:1 + logits, :], greedy_rows, logits)

    def _read_teacher_return(self, device, logits, updated, greedy_rows):
        packet = torch.empty((2, 1 + logits + 32, 16), dtype=torch.float32)
        for rank in range(2):
            self._worker_call('copy_from', packet[rank].data_ptr(), device.shards[rank].data_ptr,
                              packet[rank].nbytes, src_offset=0, worker_id=rank)
        return self._validate_teacher_return(packet, logits, updated, greedy_rows)

    def _commit_teacher_tails(self, slots, counts, updated):
        for slot in updated:self._resident_tail_versions[slot] += 1
        result = []
        for slot, count in zip(slots, counts, strict=True):
            record = self.requests[slot.request.request_id]
            destination = record['slot']
            result.append(ResidentRows(self, slot.request.request_id, destination, record['resident_generation'],
                'tails', self._resident_tail_versions[destination], None, (destination,) if count else ()))
        return result
