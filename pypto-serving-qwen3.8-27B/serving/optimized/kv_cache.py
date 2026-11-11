"""Physical KV pages and matching DeltaNet prefix snapshots."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable

import torch

from .model import BLOCK_SIZE, CONTEXT, POOL_BLOCKS


class CacheCapacityError(RuntimeError):
    pass


@dataclass
class RequestPages:
    pages: list[int] = field(default_factory=list)
    length: int = 0
    released: bool = False


@dataclass
class PrefixEntry:
    token_ids: tuple[int, ...]
    pages: tuple[int, ...]
    conv: torch.Tensor
    recurrent: torch.Tensor

    @property
    def length(self) -> int:
        return len(self.token_ids)


class KVCache:
    def __init__(self, namespace: str, max_prefix_entries: int = 4,
                 on_event: Callable[[str, dict], None] | None = None):
        if not namespace or max_prefix_entries < 0:
            raise ValueError("Invalid KV cache identity or prefix capacity")
        self.namespace = namespace
        self.refs = [0] * POOL_BLOCKS
        self.refs[0] = 1  # Unallocated logical blocks read the zero page.
        self.free = deque(range(1, POOL_BLOCKS))
        self.prefixes: OrderedDict[tuple[int, ...], PrefixEntry] = OrderedDict()
        self.max_prefix_entries = max_prefix_entries
        self.on_event = on_event
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _emit(self, name: str, **fields: object) -> None:
        if self.on_event is not None:
            self.on_event(name, {"namespace": self.namespace, **fields})

    def _retain(self, page: int) -> None:
        if page <= 0 or page >= POOL_BLOCKS or self.refs[page] <= 0:
            raise ValueError(f"Cannot retain unallocated KV page {page}")
        self.refs[page] += 1
        self._emit("retain", page=page, refs=self.refs[page])

    def _drop(self, page: int) -> None:
        if page <= 0 or page >= POOL_BLOCKS or self.refs[page] <= 0:
            raise ValueError(f"Cannot release unallocated KV page {page}")
        self.refs[page] -= 1
        self._emit("release", page=page, refs=self.refs[page])
        if self.refs[page] == 0:
            self.free.append(page)
            self._emit("reclaim", page=page)

    def _evict_oldest(self) -> bool:
        if not self.prefixes:
            return False
        key, entry = self.prefixes.popitem(last=False)
        for page in entry.pages:
            self._drop(page)
        self.evictions += 1
        self._emit("evict", tokens=len(key))
        return True

    def _allocate(self) -> int:
        while not self.free and self._evict_oldest():
            pass
        if not self.free:
            raise CacheCapacityError("No free physical KV blocks")
        page = self.free.popleft()
        if self.refs[page] != 0:
            raise AssertionError("Free list contains a referenced page")
        self.refs[page] = 1
        self._emit("allocate", page=page)
        return page

    def new_request(self) -> RequestPages:
        return RequestPages()

    def reserve(self, request: RequestPages, end_position: int) -> None:
        if request.released or not (request.length <= end_position <= CONTEXT):
            raise ValueError("Invalid KV reservation")
        needed = (end_position + BLOCK_SIZE - 1) // BLOCK_SIZE
        if request.length % BLOCK_SIZE and self.refs[request.pages[request.length // BLOCK_SIZE]] != 1:
            raise ValueError("Writable partial KV block is shared")
        original = len(request.pages)
        try:
            while len(request.pages) < needed:
                request.pages.append(self._allocate())
        except BaseException:
            for page in request.pages[original:]:
                self._drop(page)
            del request.pages[original:]
            raise
        self._emit("reserve", pages=needed, end=end_position)

    def commit(self, request: RequestPages, end_position: int) -> None:
        if request.released or not (request.length <= end_position <= len(request.pages) * BLOCK_SIZE):
            raise ValueError("KV write is outside reserved pages")
        request.length = end_position

    def table(self, request: RequestPages) -> torch.Tensor:
        if request.released or len(request.pages) > CONTEXT // BLOCK_SIZE:
            raise ValueError("Invalid KV page table")
        table = torch.zeros(CONTEXT // BLOCK_SIZE, dtype=torch.int32)
        if request.pages:
            if any(page <= 0 or page >= POOL_BLOCKS or self.refs[page] <= 0
                   for page in request.pages):
                raise ValueError("KV page table references a free or zero page")
            table[:len(request.pages)] = torch.tensor(request.pages, dtype=torch.int32)
        return table

    def release(self, request: RequestPages) -> None:
        if request.released:
            return
        for page in request.pages:
            self._drop(page)
        request.pages.clear()
        request.released = True
        self._emit("request_release")

    def store_prefix(self, token_ids: list[int], request: RequestPages,
                     conv: torch.Tensor, recurrent: torch.Tensor) -> None:
        length = len(token_ids)
        if (self.max_prefix_entries == 0 or request.released or length == 0 or
                length != request.length or length % BLOCK_SIZE):
            raise ValueError("Only a complete block-aligned prefill may be cached")
        key = tuple(token_ids)
        if key in self.prefixes:
            self.prefixes.move_to_end(key)
            return
        while len(self.prefixes) >= self.max_prefix_entries:
            self._evict_oldest()
        pages = tuple(request.pages[:length // BLOCK_SIZE])
        for page in pages:
            self._retain(page)
        self.prefixes[key] = PrefixEntry(key, pages, conv.clone(), recurrent.clone())
        self._emit("prefix_store", tokens=length, pages=len(pages))

    def find_prefix(self, token_ids: list[int]) -> PrefixEntry | None:
        matches = [entry for key, entry in self.prefixes.items()
                   if len(key) < len(token_ids) and tuple(token_ids[:len(key)]) == key]
        if not matches:
            self.misses += 1
            self._emit("prefix_miss")
            return None
        entry = max(matches, key=lambda item: item.length)
        self.prefixes.move_to_end(entry.token_ids)
        self.hits += 1
        self._emit("prefix_hit", tokens=entry.length, pages=len(entry.pages))
        return entry

    def attach_prefix(self, entry: PrefixEntry) -> RequestPages:
        if self.prefixes.get(entry.token_ids) is not entry:
            raise ValueError("Prefix snapshot was evicted")
        for page in entry.pages:
            self._retain(page)
        return RequestPages(list(entry.pages), entry.length)

    def stats(self) -> dict[str, int]:
        return {"free_blocks": len(self.free), "used_blocks": POOL_BLOCKS - 1 - len(self.free),
                "cached_prefixes": len(self.prefixes), "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions}
