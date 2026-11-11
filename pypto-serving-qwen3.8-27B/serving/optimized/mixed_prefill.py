"""Opt-in host adapter for the precision-verified mixed FFN model.

Reuse the accepted prefill512 and decode host/output owners. Physical state
owner 0 is prefill, 1..N-1 are decode. Return tokens in the caller's order.
This module does not install itself or change any accepted service.
"""
import time


def mixed_order(slots):
    """Return original indices in physical owner order, or require fallback."""
    if not 2 <= len(slots) <= 8:
        return None
    prefill = [i for i, slot in enumerate(slots)
               if slot.phase == 'prefill' and 64 < len(slot.token_ids) <= 128]
    if len(prefill) != 1:
        return None
    first = prefill[0]
    decode = [i for i, slot in enumerate(slots) if i != first]
    if any(slots[i].phase != 'decode' or len(slots[i].token_ids) != 1 for i in decode):
        return None
    return (first, *decode)


def pack_arguments(prefill, decode, active):
    """Project both complete original ABIs onto the compiled mixed ABI50."""
    if len(prefill) != 37 or len(decode) != 41 or not 2 <= active <= 8:
        raise ValueError('Mixed305 requires original prefill37/decode41 and 2..8 owners')
    if any(prefill[i] is not decode[i] for i in (*range(1,24), *range(29,33))):
        raise ValueError('Mixed305 weights and full physical state/KV must share owners')
    return (*prefill[1:24], *prefill[29:33],
            prefill[0], *prefill[24:29], *prefill[33:36],
            decode[0], *decode[24:29], *decode[33:40], active)


def run_mixed(runner, slots):
    """Execute one eligible mixed batch without a serial homogeneous fallback."""
    import torch
    from .kv_cache import CacheCapacityError
    from .schedule import RequestCapacityError
    from .page_table import fill_table

    order = mixed_order(slots)
    if runner.worker is None or order is None:
        raise RuntimeError('Invalid mixed model batch')
    if not runner.enable_mixed_prefill or runner.allow_prefix_hits:
        raise RuntimeError('Mixed305 requires the accepted compound/no-prefix-cache mode')
    keys = [slot.request.request_id for slot in slots]
    if len(set(keys)) != len(keys):
        raise RuntimeError('Duplicate request in mixed batch')
    physical = [slots[i] for i in order]
    prefill, decodes = physical[0], physical[1:]
    runner._validate_slots([prefill])
    runner._validate_slots(decodes)
    for slot in slots:
        if slot.request.request_id not in runner.requests:
            raise RuntimeError('Unknown mixed request owner')
        if slot.position < 0 or slot.position + len(slot.token_ids) > 2048:
            raise RuntimeError('Mixed context exceeds the original physical context')
    runner._compact(physical)
    p = resources(runner, 'prefill_single128')
    d = resources(runner, 'decode_greedy')
    if p[0] or p[1] != 128 or p[-3] is not runner.programs['prefill_single128']:
        raise RuntimeError('Mixed prefill must use accepted modern128 resources')
    if not d[0] or d[1] != 8 or d[-3] is not runner.programs['decode_greedy']:
        raise RuntimeError('Mixed decode must retain all8 physical output rows')
    graph = runner.programs['mixed_ffn']
    for resource in (p, d):
        ids, cos, sin, masks, counts, offsets, tables = resource[2:9]
        ids.zero_(); cos.fill_(1); sin.zero_(); masks.fill_(-10000)
        counts.zero_(); offsets.zero_(); tables.zero_()
    cache_rows = runner.__dict__.setdefault('_decode_page_rows', {})
    for key in list(cache_rows):
        if key not in runner.requests:
            del cache_rows[key]
    physical_index = {original: index for index, original in enumerate(order)}
    # Preserve reserve/error and completion event order from the caller. Only
    # argument rows and physical state-owner packing use the new order.
    for original, slot in enumerate(slots):
        owner = physical_index[original]
        resource, row = (p, 0) if owner == 0 else (d, owner)
        ids, cos, sin, masks, counts, offsets, tables = resource[2:9]
        count, position = len(slot.token_ids), slot.position
        resident = runner.requests[slot.request.request_id]
        if resident.pages.length != position:
            raise RuntimeError('Request KV length and scheduled position disagree')
        try:
            runner.cache.reserve(resident.pages, position + count)
        except CacheCapacityError as error:
            raise RequestCapacityError(slot.request, str(error)) from error
        if owner:
            ids[row, 0] = slot.token_ids[0]
            masks[row, :, :position+1].zero_()
            cache_rows[slot.request.request_id] = fill_table(
                runner.cache, resident.pages, tables[row], cache_rows.get(slot.request.request_id))
        else:
            ids[row, :count] = torch.tensor(slot.token_ids, dtype=torch.int32)
            masks[row] = runner._mask(count, position, 128)
            tables[row] = runner.cache.table(resident.pages)
        cos[row, :count] = runner.rope['cos'][position:position+count]
        sin[row, :count] = runner.rope['sin'][position:position+count]
        counts[row], offsets[row] = count, position
    p_args = (p[2], *runner.banks, *p[3:8], *runner.resident, p[8], p[10], p[11], 1)
    d_args = (d[2], *runner.banks, *d[3:8], *runner.resident, d[8], d[10], d[11],
              *runner.greedy_buffers, len(physical))
    arguments = pack_arguments(p_args, d_args, len(physical))
    if runner.on_activation is not None:
        runner.on_activation('graph_start', {'time_ns':time.monotonic_ns(),
            'device':runner.device_id, 'phases':[slot.phase for slot in slots]})
    runner.worker.run(graph, *arguments, config=runner.config)
    runner.executed_graphs['mixed_ffn'] += 1
    if runner.on_activation is not None:
        runner.on_activation('graph_end', {'time_ns':time.monotonic_ns(), 'device':runner.device_id})
    runner._copy_offset(runner.greedy_host, runner.greedy_buffers[-1], 0, False)
    decode_tokens = runner.greedy_host[:, 0].tolist()
    output = []
    for original, slot in enumerate(slots):
        owner = physical_index[original]
        resident = runner.requests[slot.request.request_id]
        runner.cache.commit(resident.pages, slot.position+len(slot.token_ids))
        if owner:
            output.append(int(decode_tokens[owner]))
        elif not slot.completes_prefill:
            output.append(None)
        else:
            value = runner._logit_row(0, len(slot.token_ids)-1, rows=128, logits=p[11])
            output.append(int(value.float().argmax()))
    runner._sample_memory()
    return output



def resources(runner, name):
    fields = runner.host_fields[name]
    rows = fields['ids'].shape[1]
    return (name == 'decode_greedy', rows, *(fields[n] for n in
            ('ids', 'cos', 'sin', 'masks', 'counts', 'offsets', 'tables')),
            runner.programs[name], *runner.outputs[name])
