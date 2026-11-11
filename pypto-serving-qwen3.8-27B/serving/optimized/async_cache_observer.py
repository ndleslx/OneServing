"""Extend281 field reuse to the sealed serving engine's actual KV observer.

The original engine callback records event data in a deque under its lock. It
does not inspect canonical input tensors or change the cache. Other callbacks
continue to select the exact original preparation/error path.
"""
import sys

from .async_lookahead import DecodeLookahead, FIELD_INDICES


def approved_cache_observer(callback):
    if callback is None:
        return True
    module=sys.modules.get(__package__ + '.engine')
    engine_type=getattr(module,'InferenceEngine',None)
    if engine_type is None or getattr(callback,'__func__',None)is not engine_type._on_cache:
        return False
    owner_type=type(getattr(callback,'__self__',None))
    if owner_type is engine_type:
        return True
    return owner_type is engine_type



# The derived class and its single overridden method are generated before
# qualification from281, with only the callback predicate changed.


class DecodeLookaheadWithCacheObserver(DecodeLookahead):
    def consume_fields(self, slots, ids, cos, sin, masks, counts, offsets, tables):
            """Use prefetched values only when the original reserve/table path cannot fail.

            Allocation boundaries, unusual cache state and callbacks keep the exact
            original preparation block, including its error/partial-write behavior.
            """
            p = self.pending
            if p is None or not self._compatible(slots) or p['key']!=self._key(slots):
                return False
            r = self.runner
            if not approved_cache_observer(r.cache.on_event) or r.cache.prefixes:
                return False
            for s in slots:
                pages = r.requests[s.request.request_id].pages
                if (pages.released or pages.length!=s.position or len(pages.pages)>128 or
                    len(pages.pages)<(s.position+1+15)//16 or
                    any(page<=0 or page>=len(r.cache.refs) or r.cache.refs[page]<=0 for page in pages.pages)):
                    return False
                if s.position%16 and r.cache.refs[pages.pages[s.position//16]]!=1:
                    return False
            canonical = (ids,cos,sin,masks,counts,offsets,tables)
            if any(t is not self.fields[name] for t,(_,name) in zip(canonical,FIELD_INDICES,strict=True)):
                return False
            # No KV/page reservation has happened speculatively. Preserve the exact
            # reserve and fill_table calls, including their ordering and page cache.
            from .page_table import fill_table
            bank = p['bank']
            for t,(_,name) in zip(canonical,FIELD_INDICES,strict=True):
                t.copy_(bank[name])
            cache_rows = r.__dict__.setdefault('_decode_page_rows',{})
            for key in list(cache_rows):
                if key not in r.requests:
                    del cache_rows[key]
            for index,s in enumerate(slots):
                pages = r.requests[s.request.request_id].pages
                r.cache.reserve(pages,s.position+1)
                ids[index,0] = s.token_ids[0]
                cache_rows[s.request.request_id] = fill_table(r.cache,pages,tables[index],cache_rows.get(s.request.request_id))
            self.consumed = True
            self.counts['prefetched_field_hits'] += 1
            return True
