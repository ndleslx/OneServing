"""Physical KV pages and matching DeltaNet prefix snapshots."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable

import torch




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
    hidden: torch.Tensor

    @property
    def length(self) -> int:
        return len(self.token_ids)


class KVCache:
    def __init__(self, namespace: str, pool_blocks: int, max_model_len: int,
                 block_size: int = 128, max_prefix_entries: int = 4,
                 on_event: Callable[[str, dict], None] | None = None):
        if not namespace or max_prefix_entries < 0:
            raise ValueError("Invalid KV cache identity or prefix capacity")
        if pool_blocks < 2 or max_model_len < 1 or block_size < 1:
            raise ValueError("Invalid page pool capacity")
        self.pool_blocks = pool_blocks
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.namespace = namespace
        self.refs = [0] * self.pool_blocks
        self.refs[0] = 1  # Unallocated logical blocks read the zero page.
        self.free = deque(range(1, self.pool_blocks))
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
        if page <= 0 or page >= self.pool_blocks or self.refs[page] <= 0:
            raise ValueError(f"Cannot retain unallocated KV page {page}")
        self.refs[page] += 1
        self._emit("retain", page=page, refs=self.refs[page])

    def _drop(self, page: int) -> None:
        if page <= 0 or page >= self.pool_blocks or self.refs[page] <= 0:
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

    def reserve(self, request: RequestPages, end_position: int, copy_page=None) -> None:
        if request.released or not (request.length <= end_position <= self.max_model_len):
            raise ValueError("Invalid KV reservation")
        needed = (end_position + self.block_size - 1) // self.block_size
        if end_position > request.length and request.length % self.block_size:
            index = request.length // self.block_size
            old = request.pages[index]
            if self.refs[old] != 1:
                if copy_page is None:
                    raise ValueError("Shared partial page requires device copy-on-write")
                new = self._allocate()
                try:
                    copy_page(old, new)
                except BaseException:
                    self._drop(new)
                    raise
                request.pages[index] = new
                self._drop(old)
                self._emit("copy_on_write", source=old, destination=new, valid_tokens=request.length)
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
        if request.released or not (request.length <= end_position <= len(request.pages) * self.block_size):
            raise ValueError("KV write is outside reserved pages")
        request.length = end_position

    def table(self, request: RequestPages) -> torch.Tensor:
        if request.released or len(request.pages) > (self.max_model_len + self.block_size - 1) // self.block_size:
            raise ValueError("Invalid KV page table")
        table = torch.zeros((self.max_model_len + self.block_size - 1) // self.block_size, dtype=torch.int32)
        if request.pages:
            if any(page <= 0 or page >= self.pool_blocks or self.refs[page] <= 0
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
                     conv: torch.Tensor, recurrent: torch.Tensor, hidden: torch.Tensor) -> None:
        length = len(token_ids)
        if (self.max_prefix_entries == 0 or request.released or length == 0 or
                length != request.length):
            raise ValueError("Only an exactly committed prefill boundary may be cached")
        key = tuple(token_ids)
        if key in self.prefixes:
            self.prefixes.move_to_end(key)
            return
        while len(self.prefixes) >= self.max_prefix_entries:
            self._evict_oldest()
        pages = tuple(request.pages[:(length + self.block_size - 1) // self.block_size])
        for page in pages:
            self._retain(page)
        self.prefixes[key] = PrefixEntry(key, pages, conv.clone(), recurrent.clone(), hidden.clone())
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
        return {"free_blocks": len(self.free), "used_blocks": self.pool_blocks - 1 - len(self.free),
                "cached_prefixes": len(self.prefixes), "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions}
