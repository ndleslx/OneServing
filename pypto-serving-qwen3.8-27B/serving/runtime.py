"""Resident PyPTO runner for mixed prefill and decode batches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import prod
from pathlib import Path
from typing import Any
import time

import pypto
import simpler
import torch
from pypto import ir
from pypto.ir import DistributedConfig
from pypto.runtime import DistributedWorker, RunConfig, StackedDeviceTensor
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from .kv_cache import CacheCapacityError, KVCache, RequestPages
from .model import (BATCH, BLOCK_SIZE, CONTEXT, HIDDEN, POOL_BLOCKS, ROWS,
                    VOCAB, tp_rank_TP_VOCAB, CandidateExchange, decode_fwd,
                    prefill_fwd, tp_decode_fwd, tp_prefill_fwd)
from .schedule import BatchSlot, RequestCapacityError, RequestState
from .weight import TextWeights


CONV_SHAPE = (48, 4, 10240)
RECURRENT_SHAPE = (48, 48, 128, 128)
CONV_BYTES = 48 * 4 * 10240 * 2
RECURRENT_BYTES = 48 * 48 * 128 * 128 * 4


def runtime_provenance() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    source = hashlib.sha256()
    paths = sorted(root.joinpath("serving").glob("*.py"))
    for path in paths:
        source.update(str(path.relative_to(root)).encode() + b"\0")
        source.update(path.read_bytes())
    manifest = root / "manifest.sha256"
    return {
        "source_tree_sha256": source.hexdigest(),
        "pypto_path": str(Path(pypto.__file__).resolve()),
        "pypto_init_sha256": hashlib.sha256(Path(pypto.__file__).read_bytes()).hexdigest(),
        "simpler_path": str(Path(simpler.__file__).resolve()),
        "simpler_init_sha256": hashlib.sha256(Path(simpler.__file__).read_bytes()).hexdigest(),
        "package_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest.exists() else "development-source",
    }


def rotary_tables(model_root: Path) -> dict[str, torch.Tensor]:
    config = AutoConfig.from_pretrained(model_root, local_files_only=True).text_config
    rotary = Qwen3_5TextRotaryEmbedding(config)
    positions = torch.arange(CONTEXT).view(1, 1, CONTEXT).expand(3, 1, CONTEXT)
    dummy = torch.empty((1, CONTEXT, config.hidden_size), dtype=torch.bfloat16)
    cos, sin = rotary(dummy, positions)
    return {"cos": cos[0].contiguous(), "sin": sin[0].contiguous()}


@dataclass
class ResidentRequest:
    slot: int
    pages: RequestPages


class ModelRunner:
    def __init__(self, model_root: Path, device_id: int,
                 on_cache_event=None, on_activation=None):
        self.model_root = model_root
        self.device_id = device_id
        self.provenance = runtime_provenance()
        self.cache = KVCache("qwen3.8-27b-v2", on_event=on_cache_event)
        self.on_activation = on_activation
        self.allow_prefix_hits = True
        self.requests: dict[str, ResidentRequest] = {}
        self.slot_ids: list[str | None] = [None] * BATCH
        self.worker: DistributedWorker | None = None
        self.banks: list[Any] = []
        self.resident: list[Any] = []
        self.zero_conv = torch.zeros(CONV_SHAPE, dtype=torch.bfloat16)
        self.zero_recurrent = torch.zeros(RECURRENT_SHAPE, dtype=torch.float32)
        self.memory_stats: dict[str, int | None] = {
            "committed_bytes": None, "hbm_used_bytes": None,
            "peak_committed_bytes": None, "peak_hbm_used_bytes": None,
        }
        self.config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(
            device_ids=[device_id], num_sub_workers=0), ring_heap=(0, 0, 0, 4 << 30))

    def _buffers(self) -> None:
        self.ids = torch.zeros((BATCH, ROWS), dtype=torch.int32).share_memory_()
        self.cos = torch.ones((BATCH, ROWS, 64), dtype=torch.bfloat16).share_memory_()
        self.sin = torch.zeros_like(self.cos).share_memory_()
        self.masks = torch.full((BATCH, ROWS, CONTEXT), -10000.0).share_memory_()
        self.counts = torch.zeros(BATCH, dtype=torch.int32).share_memory_()
        self.offsets = torch.zeros(BATCH, dtype=torch.int32).share_memory_()
        self.tables = torch.zeros((BATCH, CONTEXT // BLOCK_SIZE), dtype=torch.int32).share_memory_()

    def start(self) -> None:
        if self.worker is not None:
            raise RuntimeError("Model runner is already started")
        samples = single_meta_samples()
        self.prefill = prefill_fwd.compile(*samples, 1, config=self.config)
        self.decode = decode_fwd.compile(*samples, 1, config=self.config)
        self.rope = rotary_tables(self.model_root)
        if self.rope["cos"].shape != (CONTEXT, 64) or self.rope["sin"].shape != (CONTEXT, 64):
            raise ValueError("Invalid RoPE table")
        self._buffers()
        self.worker = DistributedWorker([self.prefill, self.decode], config=self.config)
        try:
            for _, host in TextWeights(self.model_root).banks():
                self.banks.append(self.worker.alloc_tensor(tuple(host.shape), host.dtype, init=host))
            pool_shape = (16, 4, POOL_BLOCKS, BLOCK_SIZE, 256)
            for shape, dtype in (
                ((BATCH, *CONV_SHAPE), torch.bfloat16),
                ((BATCH, *RECURRENT_SHAPE), torch.float32),
                (pool_shape, torch.bfloat16),
                (pool_shape, torch.bfloat16),
            ):
                self.resident.append(self.worker.alloc_tensor(
                    shape, dtype, init=torch.zeros(shape, dtype=dtype)))
            self.traces = self.worker.alloc_tensor((BATCH, 66, ROWS, HIDDEN), torch.bfloat16)
            self.logits = self.worker.alloc_tensor((BATCH, ROWS, VOCAB), torch.bfloat16)
            self._sample_memory()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        worker, self.worker = self.worker, None
        if worker is None:
            return
        try:
            for handle in (getattr(self, "logits", None), getattr(self, "traces", None)):
                if handle is not None:
                    worker.free_tensor(handle)
            for handle in reversed(self.resident):
                worker.free_tensor(handle)
            for handle in reversed(self.banks):
                worker.free_tensor(handle)
        finally:
            worker.close()
            self.resident.clear()
            self.banks.clear()

    def _sample_memory(self) -> None:
        assert self.worker is not None
        committed = self.worker.committed_device_memory()
        free, total = self.worker.device_memory_info()
        used = total - free
        self.memory_stats["committed_bytes"] = committed
        self.memory_stats["hbm_used_bytes"] = used
        self.memory_stats["peak_committed_bytes"] = max(
            self.memory_stats["peak_committed_bytes"] or 0, committed)
        self.memory_stats["peak_hbm_used_bytes"] = max(
            self.memory_stats["peak_hbm_used_bytes"] or 0, used)

    def _copy_offset(self, host: torch.Tensor, device: Any, offset: int,
                     to_device: bool) -> None:
        assert self.worker is not None
        size = host.numel() * host.element_size()
        capacity = prod(device.shape) * torch.empty((), dtype=device.dtype).element_size()
        if not host.is_contiguous() or offset < 0 or offset + size > capacity:
            raise ValueError("Device transfer exceeds the registered tensor")
        if to_device:
            self.worker.copy_to(device.data_ptr, host.data_ptr(), size, dst_offset=offset)
        else:
            self.worker.copy_from(host.data_ptr(), device.data_ptr, size, src_offset=offset)

    def _write_state(self, slot: int, conv: torch.Tensor,
                     recurrent: torch.Tensor) -> None:
        assert self.worker is not None
        for index, host in enumerate((conv, recurrent)):
            stride = CONV_BYTES if index == 0 else RECURRENT_BYTES
            self._copy_offset(host, self.resident[index], slot * stride, True)

    def _read_state(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.worker is not None
        conv = torch.empty(CONV_SHAPE, dtype=torch.bfloat16)
        recurrent = torch.empty(RECURRENT_SHAPE, dtype=torch.float32)
        for index, host in enumerate((conv, recurrent)):
            stride = CONV_BYTES if index == 0 else RECURRENT_BYTES
            self._copy_offset(host, self.resident[index], slot * stride, False)
        return conv, recurrent

    def admit(self, request: RequestState) -> int:
        if self.worker is None:
            raise RuntimeError("Model runner is not started")
        try:
            slot = self.slot_ids.index(None)
        except ValueError as exc:
            raise RuntimeError("No free model batch slot") from exc
        prefix = self.cache.find_prefix(request.prompt_ids) if self.allow_prefix_hits else None
        pages = self.cache.attach_prefix(prefix) if prefix is not None else self.cache.new_request()
        try:
            self._write_state(slot, *(prefix.conv, prefix.recurrent)
                              if prefix is not None else
                              (self.zero_conv, self.zero_recurrent))
            if prefix is not None and self.on_activation is not None:
                restored_conv, restored_recurrent = self._read_state(slot)
                self.on_activation("prefix_restore", {
                    "request_id": request.request_id,
                    "tokens": prefix.length,
                    "pages": list(pages.pages),
                    "kv_pages_match": tuple(pages.pages) == prefix.pages,
                    "conv_exact": torch.equal(restored_conv, prefix.conv),
                    "recurrent_exact": torch.equal(restored_recurrent, prefix.recurrent),
                })
        except BaseException:
            self.cache.release(pages)
            raise
        self.slot_ids[slot] = request.request_id
        self.requests[request.request_id] = ResidentRequest(slot, pages)
        request.runtime_state = self.requests[request.request_id]
        return pages.length

    def release(self, request: RequestState) -> None:
        resident = self.requests.pop(request.request_id, None)
        if resident is None:
            return
        self.slot_ids[resident.slot] = None
        self.cache.release(resident.pages)
        request.runtime_state = None

    def _compact(self, slots: list[BatchSlot]) -> None:
        desired = [slot.request.request_id for slot in slots]
        if self.slot_ids[:len(desired)] == desired:
            return
        moved = [(index, self.requests[request_id].slot) for index, request_id in enumerate(desired)
                 if self.requests[request_id].slot != index]
        snapshots = {source: self._read_state(source) for _, source in moved}
        for destination, source in moved:
            self._write_state(destination, *snapshots[source])
        self.slot_ids = desired + [None] * (BATCH - len(desired))
        for index, request_id in enumerate(desired):
            self.requests[request_id].slot = index

    @staticmethod
    def _mask(count: int, offset: int) -> torch.Tensor:
        mask = torch.full((ROWS, CONTEXT), -10000.0, dtype=torch.float32)
        for row in range(ROWS):
            mask[row, :min(offset + row + 1, offset + count)] = 0.0
        return mask

    def _logit_row(self, slot: int, row: int) -> torch.Tensor:
        assert self.worker is not None
        host = torch.empty(VOCAB, dtype=torch.bfloat16)
        offset = (slot * ROWS + row) * VOCAB * 2
        self._copy_offset(host, self.logits, offset, False)
        return host

    def run_batch(self, slots: list[BatchSlot]) -> list[int | None]:
        if self.worker is None or not (1 <= len(slots) <= BATCH):
            raise RuntimeError("Invalid model batch")
        self._compact(slots)
        ids, cos, sin = self.ids, self.cos, self.sin
        masks, counts, offsets, tables = self.masks, self.counts, self.offsets, self.tables
        ids.zero_()
        cos.fill_(1)
        sin.zero_()
        masks.fill_(-10000)
        counts.zero_()
        offsets.zero_()
        tables.zero_()
        for index, slot in enumerate(slots):
            count, position = len(slot.token_ids), slot.position
            resident = self.requests[slot.request.request_id]
            if resident.pages.length != position:
                raise RuntimeError("Request KV length and scheduled position disagree")
            try:
                self.cache.reserve(resident.pages, position + count)
            except CacheCapacityError as exc:
                raise RequestCapacityError(slot.request, str(exc)) from exc
            ids[index, :count] = torch.tensor(slot.token_ids, dtype=torch.int32)
            cos[index, :count] = self.rope["cos"][position:position + count]
            sin[index, :count] = self.rope["sin"][position:position + count]
            masks[index] = self._mask(count, position)
            counts[index], offsets[index] = count, position
            tables[index] = self.cache.table(resident.pages)
        graph = self.prefill if any(slot.phase == "prefill" for slot in slots) else self.decode
        if self.on_activation is not None:
            self.on_activation("graph_start", {"time_ns": time.monotonic_ns(),
                                               "device": self.device_id,
                                               "phases": [slot.phase for slot in slots]})
        self.worker.run(graph, ids, *self.banks, cos, sin, masks, counts, offsets,
                        *self.resident, tables, self.traces, self.logits, len(slots), config=self.config)
        if self.on_activation is not None:
            self.on_activation("graph_end", {"time_ns": time.monotonic_ns(),
                                             "device": self.device_id})
        output: list[int | None] = []
        for index, slot in enumerate(slots):
            resident = self.requests[slot.request.request_id]
            end = slot.position + len(slot.token_ids)
            self.cache.commit(resident.pages, end)
            if self.allow_prefix_hits and slot.phase == "prefill" and end in (64, 256):
                prefix_tokens = slot.request.prompt_ids[:end]
                if tuple(prefix_tokens) not in self.cache.prefixes:
                    conv, recurrent = self._read_state(index)
                    self.cache.store_prefix(prefix_tokens, resident.pages, conv, recurrent)
            if slot.phase == "prefill" and not slot.completes_prefill:
                output.append(None)
            else:
                row = self._logit_row(index, len(slot.token_ids) - 1)
                output.append(int(row.float().argmax()))
        self._sample_memory()
        return output

TP_VOCAB = tp_rank_TP_VOCAB


def single_meta_samples() -> list[torch.Tensor]:
    samples = tp_meta_samples()
    tensors = [torch.empty(sample.shape[1:], dtype=sample.dtype, device="meta")
               for sample in samples]
    tensors[3] = torch.empty((VOCAB, HIDDEN), dtype=torch.bfloat16, device="meta")
    tensors[-1] = torch.empty((BATCH, ROWS, VOCAB), dtype=torch.bfloat16, device="meta")
    return tensors

def tp_meta_samples() -> list[torch.Tensor]:
    bf16, fp32, int32 = (torch.bfloat16, torch.float32, torch.int32)
    shapes = [((2, 8, 16), int32), ((2, 248320, 5120), bf16), ((2, 1, 5120), fp32), ((2, 124160, 5120), bf16), ((2, 64, 5120), fp32), ((2, 64, 5120), fp32), ((2, 64 * 17408, 5120), bf16), ((2, 64 * 17408, 5120), bf16), ((2, 64 * 5120, 17408), bf16), ((2, 48 * 10240, 5120), bf16), ((2, 48 * 6144, 5120), bf16), ((2, 48 * 128, 5120), bf16), ((2, 48 * 128, 5120), bf16), ((2, 48 * 4, 10240), bf16), ((2, 48, 64), bf16), ((2, 48, 64), bf16), ((2, 48, 128), bf16), ((2, 48 * 5120, 6144), bf16), ((2, 16 * 12288, 5120), bf16), ((2, 16 * 1024, 5120), bf16), ((2, 16 * 1024, 5120), bf16), ((2, 16, 256), fp32), ((2, 16, 256), fp32), ((2, 16 * 5120, 6144), bf16), ((2, 8, 16, 64), bf16), ((2, 8, 16, 64), bf16), ((2, 8, 16, 512), fp32), ((2, 8), int32), ((2, 8), int32), ((2, 8, 48, 4, 10240), bf16), ((2, 8, 48, 48, 128, 128), fp32), ((2, 16, 4, 256, 16, 256), bf16), ((2, 16, 4, 256, 16, 256), bf16), ((2, 8, 32), int32), ((2, 8, 66, 16, 5120), bf16), ((2, 8, 16, 124160), bf16)]
    return [torch.empty(shape, dtype=dtype, device='meta') for shape, dtype in shapes]

class ShardView:

    def __init__(self, owner: TensorParallelRunner, rank: int):
        self.owner = owner
        self.rank = rank
        self.last_logits: dict[int, torch.Tensor] = {}

    @property
    def allow_prefix_hits(self) -> bool:
        return self.owner.allow_prefix_hits

    @allow_prefix_hits.setter
    def allow_prefix_hits(self, value: bool) -> None:
        self.owner.allow_prefix_hits = bool(value)

class TensorParallelRunner:

    def __init__(self, model_root: Path, devices: tuple[int, int], on_cache_event=None):
        if len(devices) != 2 or len(set(devices)) != 2:
            raise ValueError('TP=2 needs two distinct devices')
        self.model_root = model_root
        self.device_ids = devices
        self.provenance = runtime_provenance()
        self.cache = KVCache('qwen3.8-27b-v2-tp', on_event=on_cache_event)
        self.allow_prefix_hits = True
        self.slot_ids: list[str | None] = [None] * BATCH
        self.requests: dict[str, tuple[int, RequestPages]] = {}
        self.zero_conv = torch.zeros(CONV_SHAPE, dtype=torch.bfloat16)
        self.zero_recurrent = torch.zeros(RECURRENT_SHAPE, dtype=torch.float32)
        self.shards = [ShardView(self, rank) for rank in range(2)]
        self.collective_calls = 0
        self.worker: DistributedWorker | None = None
        self.banks: list[StackedDeviceTensor] = []
        self.states: list[StackedDeviceTensor] = []
        self.memory_stats: list[dict[str, int | None]] = [{'committed_bytes': None, 'hbm_used_bytes': None, 'peak_committed_bytes': None, 'peak_hbm_used_bytes': None} for _ in range(2)]
        self.config = RunConfig(platform='a2a3', distributed_config=DistributedConfig(device_ids=list(devices), num_sub_workers=0), ring_heap=(0, 0, 0, 4 << 30))

    @staticmethod
    def _stacked(worker: DistributedWorker, shape: tuple[int, ...], dtype: torch.dtype, hosts: tuple[torch.Tensor, torch.Tensor] | None=None) -> StackedDeviceTensor:
        shards = tuple((worker.alloc_tensor(shape, dtype, init=hosts[rank] if hosts else None, worker_id=rank) for rank in range(2)))
        return StackedDeviceTensor(shards, (2, *shape), (0, 1))

    def _buffers(self) -> None:
        self.ids = torch.zeros((2, BATCH, ROWS), dtype=torch.int32).share_memory_()
        self.cos = torch.ones((2, BATCH, ROWS, 64), dtype=torch.bfloat16).share_memory_()
        self.sin = torch.zeros_like(self.cos).share_memory_()
        self.masks = torch.full((2, BATCH, ROWS, CONTEXT), -10000.0).share_memory_()
        self.counts = torch.zeros((2, BATCH), dtype=torch.int32).share_memory_()
        self.offsets = torch.zeros((2, BATCH), dtype=torch.int32).share_memory_()
        self.tables = torch.zeros((2, BATCH, CONTEXT // BLOCK_SIZE), dtype=torch.int32).share_memory_()
        self.candidates = torch.zeros((2, 1, 8)).share_memory_()
        self.gathered = torch.zeros((2, 1, 16)).share_memory_()

    def start(self) -> None:
        if self.worker is not None:
            raise RuntimeError('TP runner already started')
        samples = tp_meta_samples()
        self.prefill = tp_prefill_fwd.compile(*samples, 1, config=self.config)
        self.decode = tp_decode_fwd.compile(*samples, 1, config=self.config)
        self.exchange = ir.compile(CandidateExchange, platform='a2a3', distributed_config=self.config.distributed_config)
        self.rope = rotary_tables(self.model_root)
        self._buffers()
        self.worker = DistributedWorker([self.prefill, self.decode, self.exchange], config=self.config)
        try:
            for name, host in TextWeights(self.model_root).banks():
                hosts = (host[:TP_VOCAB].contiguous(), host[TP_VOCAB:].contiguous()) if name == 'lm_head' else (host, host)
                self.banks.append(self._stacked(self.worker, tuple(hosts[0].shape), host.dtype, hosts))
            pool_shape = (16, 4, POOL_BLOCKS, BLOCK_SIZE, 256)
            for shape, dtype in (((BATCH, *CONV_SHAPE), torch.bfloat16), ((BATCH, *RECURRENT_SHAPE), torch.float32), (pool_shape, torch.bfloat16), (pool_shape, torch.bfloat16)):
                self.states.append(self._stacked(self.worker, shape, dtype))
            self.traces = self._stacked(self.worker, (BATCH, 66, ROWS, HIDDEN), torch.bfloat16)
            self.logits = self._stacked(self.worker, (BATCH, ROWS, TP_VOCAB), torch.bfloat16)
            self._sample_memory()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        worker, self.worker = (self.worker, None)
        if worker is not None:
            worker.close()
        self.banks.clear()
        self.states.clear()

    def _sample_memory(self) -> None:
        assert self.worker is not None
        for rank in range(2):
            committed = self.worker.committed_device_memory(rank)
            free, total = self.worker.device_memory_info(rank)
            values = self.memory_stats[rank]
            values['committed_bytes'] = committed
            values['hbm_used_bytes'] = total - free
            values['peak_committed_bytes'] = max(values['peak_committed_bytes'] or 0, committed)
            values['peak_hbm_used_bytes'] = max(values['peak_hbm_used_bytes'] or 0, total - free)

    def _copy_state(self, rank: int, slot: int, state_index: int, host: torch.Tensor, to_device: bool) -> None:
        assert self.worker is not None
        state = self.states[state_index].shards[rank]
        offset = slot * host.nbytes
        if to_device:
            self.worker.copy_to(state.data_ptr, host.data_ptr(), host.nbytes, dst_offset=offset, worker_id=rank)
        else:
            self.worker.copy_from(host.data_ptr(), state.data_ptr, host.nbytes, src_offset=offset, worker_id=rank)

    def _write_state(self, slot: int, conv: torch.Tensor, recurrent: torch.Tensor) -> None:
        for rank in range(2):
            self._copy_state(rank, slot, 0, conv, True)
            self._copy_state(rank, slot, 1, recurrent, True)

    def _read_state(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        conv = torch.empty_like(self.zero_conv)
        recurrent = torch.empty_like(self.zero_recurrent)
        self._copy_state(0, slot, 0, conv, False)
        self._copy_state(0, slot, 1, recurrent, False)
        return (conv, recurrent)

    def admit(self, request: RequestState) -> int:
        if self.worker is None:
            raise RuntimeError('TP runner not started')
        try:
            slot = self.slot_ids.index(None)
        except ValueError as exc:
            raise RuntimeError('No free TP batch slot') from exc
        prefix = self.cache.find_prefix(request.prompt_ids) if self.allow_prefix_hits else None
        pages = self.cache.attach_prefix(prefix) if prefix is not None else self.cache.new_request()
        try:
            self._write_state(slot, *((prefix.conv, prefix.recurrent) if prefix is not None else (self.zero_conv, self.zero_recurrent)))
        except BaseException:
            self.cache.release(pages)
            raise
        self.slot_ids[slot] = request.request_id
        self.requests[request.request_id] = (slot, pages)
        request.runtime_state = self.requests[request.request_id]
        return pages.length

    def release(self, request: RequestState) -> None:
        resident = self.requests.pop(request.request_id, None)
        if resident is None:
            return
        slot, pages = resident
        self.slot_ids[slot] = None
        self.cache.release(pages)
        request.runtime_state = None

    def _compact(self, slots: list[BatchSlot]) -> None:
        desired = [slot.request.request_id for slot in slots]
        if self.slot_ids[:len(desired)] == desired:
            return
        moved = [(index, self.requests[request_id][0]) for index, request_id in enumerate(desired) if self.requests[request_id][0] != index]
        snapshots = {source: self._read_state(source) for _, source in moved}
        for destination, source in moved:
            self._write_state(destination, *snapshots[source])
        self.slot_ids = desired + [None] * (BATCH - len(desired))
        for index, request_id in enumerate(desired):
            _, pages = self.requests[request_id]
            self.requests[request_id] = (index, pages)

    def run_batch(self, slots: list[BatchSlot]) -> list[int | None]:
        if self.worker is None or not 1 <= len(slots) <= 2:
            raise RuntimeError('Invalid TP batch')
        self._compact(slots)
        self.ids.zero_()
        self.cos.fill_(1)
        self.sin.zero_()
        self.masks.fill_(-10000)
        self.counts.zero_()
        self.offsets.zero_()
        self.tables.zero_()
        for index, slot in enumerate(slots):
            count, position = (len(slot.token_ids), slot.position)
            _, pages = self.requests[slot.request.request_id]
            if pages.length != position:
                raise RuntimeError('TP KV length and scheduled position disagree')
            try:
                self.cache.reserve(pages, position + count)
            except CacheCapacityError as exc:
                raise RequestCapacityError(slot.request, str(exc)) from exc
            ids = torch.tensor(slot.token_ids, dtype=torch.int32)
            mask = ModelRunner._mask(count, position)
            table = self.cache.table(pages)
            for rank in range(2):
                self.ids[rank, index, :count] = ids
                self.cos[rank, index, :count] = self.rope['cos'][position:position + count]
                self.sin[rank, index, :count] = self.rope['sin'][position:position + count]
                self.masks[rank, index] = mask
                self.counts[rank, index] = count
                self.offsets[rank, index] = position
                self.tables[rank, index] = table
        graph = self.prefill if any((slot.phase == 'prefill' for slot in slots)) else self.decode
        self.worker.run(graph, self.ids, *self.banks, self.cos, self.sin, self.masks, self.counts, self.offsets, *self.states, self.tables, self.traces, self.logits, len(slots), config=self.config)
        outputs: list[int | None] = []
        for index, slot in enumerate(slots):
            _, pages = self.requests[slot.request.request_id]
            end = slot.position + len(slot.token_ids)
            self.cache.commit(pages, end)
            if self.allow_prefix_hits and slot.phase == 'prefill' and (end in (64, 256)):
                key = tuple(slot.request.prompt_ids[:end])
                if key not in self.cache.prefixes:
                    self.cache.store_prefix(key, pages, *self._read_state(index))
            if slot.phase == 'prefill' and (not slot.completes_prefill):
                outputs.append(None)
                continue
            self.candidates.zero_()
            local = []
            for rank in range(2):
                host = torch.empty(TP_VOCAB, dtype=torch.bfloat16)
                offset = (index * ROWS + len(slot.token_ids) - 1) * TP_VOCAB * 2
                self.worker.copy_from(host.data_ptr(), self.logits.shards[rank].data_ptr, host.nbytes, src_offset=offset, worker_id=rank)
                self.shards[rank].last_logits[index] = host
                local.append(host)
                value, token = host.float().max(dim=0)
                self.candidates[rank, 0, 0] = value
                self.candidates[rank, 0, 1] = int(token) + rank * TP_VOCAB
            self.gathered.zero_()
            self.worker.run(self.exchange, self.candidates, self.gathered)
            self.collective_calls += 1
            if not torch.equal(self.gathered[0], self.gathered[1]):
                raise RuntimeError('TP collective ranks disagree')
            winner = max(((float(self.gathered[0, 0, rank * 8]), int(self.gathered[0, 0, rank * 8 + 1])) for rank in range(2)), key=lambda pair: (pair[0], -pair[1]))[1]
            if winner != int(torch.cat(local).float().argmax()):
                raise RuntimeError('TP collective token differs from sharded logits')
            outputs.append(winner)
        self._sample_memory()
        return outputs

    def stats(self) -> dict[str, Any]:
        files = {}
        for name in ('model', 'runtime'):
            path = Path(__file__).with_name(name + '.py').resolve()
            files[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        return {'devices': list(self.device_ids), 'lm_head_shard_rows': TP_VOCAB, 'collective_calls': self.collective_calls, 'memory': self.memory_stats, 'cache': self.cache.stats(), 'provenance': self.provenance, 'tp_files': files, 'time_ns': time.monotonic_ns()}
