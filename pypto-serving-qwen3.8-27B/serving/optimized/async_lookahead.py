"""Prepare the next decode's host inputs and metadata during one native forward.

Original canonical inputs remain available to the full35-field oracle. Two
additional shared host banks must be allocated before worker construction.
Only one forward may use the model's resident state/KV/outputs at any time.
"""
from __future__ import annotations

from collections import Counter
import torch

from .async_prepare import AsyncPreparation

FIELD_INDICES = ((0,'ids'),(24,'cos'),(25,'sin'),(26,'masks'),(27,'counts'),(28,'offsets'),(33,'tables'))


def allocate_host_banks(runner):
    """Call inside startup, before the actual DistributedWorker constructor."""
    if hasattr(runner, 'async281_banks'):
        raise RuntimeError('Lookahead host banks must be allocated exactly once before fork')
    runner.async281_banks = [{name:torch.empty_like(getattr(runner,'decode_'+name)).share_memory_()
                             for _,name in FIELD_INDICES} for _ in range(2)]
    return tuple(t for bank in runner.async281_banks for t in bank.values())


def _metadata(tensor):
    return (id(tensor),tensor.data_ptr(),tuple(tensor.shape),tuple(tensor.stride()),tensor.dtype)


class DecodeLookahead:
    def __init__(self, runner, runtime_module, *, check_inputs=False):
        self.runner = runner
        self.worker = runner.worker
        self.banks = runner.async281_banks
        self.config = runner.config
        self.check_inputs = check_inputs
        self.pending = None
        self.slots = None
        self.consumed = False
        self.used_bank = -1
        self.counts = Counter()
        self.fields = {name:getattr(runner,'decode_'+name) for _,name in FIELD_INDICES}
        self.field_metadata = {name:_metadata(t) for name,t in self.fields.items()}
        infos = self.worker._states[runner.compound_decode]['param_infos']
        self.param_names = [infos[i].name for i,_ in FIELD_INDICES]
        self.session = AsyncPreparation(self.worker,runtime_module,self.param_names)
        self.original_run = self.worker.run
        self.worker.run = self.run

    def cancel(self, reason):
        if self.pending is not None:
            self.pending['ticket'].cancel()
            self.pending = None
            self.counts['discarded_'+reason] += 1
        self.consumed = False

    def begin(self, slots):
        self.slots = tuple(slots)
        self.consumed = False
        if self.runner.serving_policy != 'compound' or not all(s.phase=='decode' and len(s.token_ids)==1 for s in slots):
            self.cancel('phase')

    def _key(self, slots, delta=0):
        r = self.runner
        if any(_metadata(getattr(r,'decode_'+name))!=self.field_metadata[name] for _,name in FIELD_INDICES):
            return None
        if r.config is not self.config or r.allow_prefix_hits:
            return None
        try:
            rows = tuple((id(s.request),s.request.request_id,id(r.requests[s.request.request_id]),
                          id(r.requests[s.request.request_id].pages),r.requests[s.request.request_id].slot,
                          s.position+delta,s.phase) for s in slots)
        except KeyError:
            return None
        options = None if self.config is None else repr(self.config.run_options())
        baseline = r.compound_decode._distributed_config.aicpu_thread_num
        return (id(r.cache),id(r.rope['cos']),id(r.rope['sin']),
                r.rope['cos']._version,r.rope['sin']._version,options,baseline,tuple(r.slot_ids),rows)

    def _compatible(self, slots):
        r = self.runner
        return (1<=len(slots)<=8 and r.serving_policy=='compound' and
                all(s.phase=='decode' and len(s.token_ids)==1 and type(s.position) is int
                    and type(s.token_ids[0]) is int and -2147483648<=s.token_ids[0]<=2147483647
                    and 0<=s.position<2048
                    and not s.request.cancelled.is_set() for s in slots))

    def consume_fields(self, slots, ids, cos, sin, masks, counts, offsets, tables):
        """Use prefetched values only when the original reserve/table path cannot fail.

        Allocation boundaries, unusual cache state and callbacks keep the exact
        original preparation block, including its error/partial-write behavior.
        """
        p = self.pending
        if p is None or not self._compatible(slots) or p['key']!=self._key(slots):
            return False
        r = self.runner
        if r.cache.on_event is not None or r.cache.prefixes:
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

    def _future(self, args, slots, bank_index):
        r = self.runner
        if not self._compatible(slots) or any(s.position+1>=2048 or len(s.request.generated)+1>=s.request.max_tokens for s in slots):
            return
        key = self._key(slots,1)
        if key is None:
            return
        bank = self.banks[bank_index]
        bank['ids'].zero_();bank['cos'].fill_(1);bank['sin'].zero_()
        bank['masks'].fill_(-10000);bank['counts'].zero_();bank['offsets'].zero_();bank['tables'].zero_()
        for index,s in enumerate(slots):
            position = s.position+1
            bank['cos'][index,:1] = r.rope['cos'][position:position+1]
            bank['sin'][index,:1] = r.rope['sin'][position:position+1]
            bank['masks'][index,:,:position+1].zero_()
            bank['counts'][index],bank['offsets'][index] = 1,position
        future_args = list(args)
        for index,name in FIELD_INDICES:
            future_args[index] = bank[name]
        ticket = self.session.prepare(key,r.compound_decode,*future_args,config=self.config)
        self.pending = dict(key=key,bank=bank,bank_index=bank_index,ticket=ticket,args=tuple(future_args))
        self.counts['next_batch_prepared_during_current_forward'] += 1

    def run(self, compiled, *args, config=None):
        r = self.runner
        slots = self.slots
        if (compiled is not r.compound_decode or slots is None or config is not self.config
                or not self._compatible(slots)):
            self.cancel('non_decode_dispatch')
            return self.original_run(compiled,*args,config=config)
        key = self._key(slots)
        if key is None:
            self.cancel('unsupported_metadata')
            return self.original_run(compiled,*args,config=config)
        p = self.pending
        if p is not None and p['key']==key and all(
            original is future if index!=40 else original==future
            for index,(original,future) in enumerate(zip(args,p['args'],strict=True))
            if index not in {i for i,_ in FIELD_INDICES}):
            bank = p['bank'];bank_index = p['bank_index'];ticket = p['ticket']
            # The current token and the actual page table depend on the previous
            # result/commit. The other five fields were prepared one step ahead.
            if not self.consumed:
                for index,name in FIELD_INDICES:
                    bank[name].copy_(args[index])
            else:
                bank['ids'].copy_(args[0]);bank['tables'].copy_(args[33])
            self.pending = None
            self.counts['metadata_hits'] += 1
        else:
            self.cancel('batch_change')
            bank_index = (self.used_bank+1)%2;bank = self.banks[bank_index]
            current_args = list(args)
            for index,name in FIELD_INDICES:
                bank[name].copy_(args[index]);current_args[index] = bank[name]
            ticket = self.session.prepare(key,compiled,*current_args,config=config)
            self.counts['metadata_misses'] += 1
        if self.check_inputs:
            for index,name in FIELD_INDICES:
                assert torch.equal(bank[name].view(torch.uint8),args[index].view(torch.uint8)),'Actual decode input differs from canonical '+name
            saved = {name:t.clone() for name,t in bank.items()}
        handle = ticket.publish(key)
        self.used_bank = bank_index
        error = None
        try:
            self._future(tuple(ticket.args),slots,1-bank_index)
        except BaseException as exc:
            error = exc
        try:
            result = handle.result()
        except BaseException:
            self.cancel('native_error')
            raise
        if error is not None:
            self.cancel('prepare_error')
            raise error
        if self.check_inputs:
            assert all(torch.equal(t.view(torch.uint8),saved[name].view(torch.uint8)) for name,t in bank.items()),'Actual native host input mutated'
            self.counts['actual_seven_host_inputs_full_readonly_checks'] += 1
        return result


def prepare_decode_fields(runner, slots, *fields):
    lookahead = runner.__dict__.get('async281')
    return lookahead is not None and lookahead.consume_fields(slots,*fields)
