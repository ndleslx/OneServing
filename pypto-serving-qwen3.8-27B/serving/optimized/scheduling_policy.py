"""Pure host policy; this module imports no model or runtime packages."""


def resident_permutation(slot_ids, residents, selected_ids, capacity=8):
    """Place selected requests first while preserving every other live owner.

    Moves use the accepted state graph's simultaneous gather/scatter semantics.
    If selected requests already occupy their input slots, retain existing holes.
    """
    if type(capacity) is not int or capacity != 8 or len(slot_ids) != capacity:
        raise ValueError('Only the accepted eight-slot state ABI is supported')
    selected_ids = tuple(selected_ids)
    if not 1 <= len(selected_ids) <= capacity or len(set(selected_ids)) != len(selected_ids):
        raise ValueError('Selected request IDs must be nonempty and unique')
    occupied = [value for value in slot_ids if value is not None]
    if (len(set(occupied)) != len(occupied) or set(occupied) != set(residents)
            or any(value not in residents for value in selected_ids)):
        raise ValueError('Live slot owners and resident request IDs disagree')
    for index, value in enumerate(slot_ids):
        if value is not None and (type(residents[value].slot) is not int or residents[value].slot != index):
            raise ValueError('Resident slot index differs from the owner table')
    if tuple(slot_ids[:len(selected_ids)]) == selected_ids:
        return tuple(slot_ids), ()
    selected_set = set(selected_ids)
    order = selected_ids + tuple(value for value in occupied if value not in selected_set)
    order += (None,) * (capacity - len(order))
    moves = tuple((destination, residents[value].slot)
                  for destination, value in enumerate(order)
                  if value is not None and destination != residents[value].slot)
    return order, moves


class PrefillDecodePolicy:
    """Alternate one prefill chunk with all ready decodes when both exist."""

    def __init__(self):
        self.last_phase = 'decode'

    def choose(self, active, mode, max_active):
        if mode not in ('baseline64', 'de59'):
            raise ValueError('Unsupported source-bound mode')
        if type(max_active) is not int or not 1 <= max_active <= 8 or len(active) > max_active:
            raise ValueError('Invalid active-request capacity')
        current = list(active[:max_active])
        if len({request.request_id for request in current}) != len(current):
            raise ValueError('Duplicate active request ID')
        if mode == 'baseline64':
            self.last_phase = 'decode'
            return current
        prefill = [request for request in current if request.prefill_offset < len(request.prompt_ids)]
        decode = [request for request in current if request.prefill_offset == len(request.prompt_ids)]
        if len(prefill) + len(decode) != len(current):
            raise ValueError('Request prefill frontier exceeds its prompt')
        if prefill and (not decode or self.last_phase != 'prefill'):
            self.last_phase = 'prefill'
            return prefill[:1]
        self.last_phase = 'decode'
        return decode


def select_short_mixed(active, max_active):
    """Use the qualified short348 selection; do not batch packed prefills."""
    if type(max_active) is not int or not 1 <= max_active <= 8 or len(active) > max_active:
        raise ValueError('Invalid active-request capacity')
    current = tuple(active)
    if len({q.request_id for q in current}) != len(current):
        raise ValueError('Duplicate active request ID')
    if any(type(q.prefill_offset) is not int or not 0 <= q.prefill_offset <= len(q.prompt_ids) for q in current):
        raise ValueError('Invalid prefill frontier')
    prefills = [q for q in current if q.prefill_offset < len(q.prompt_ids)]
    if not prefills or len(prefills[0].prompt_ids) - prefills[0].prefill_offset <= 64:
        return None
    decodes = [q for q in current if q.prefill_offset == len(q.prompt_ids)]
    if decodes:
        oldest = prefills[0]
        return 'mixed128', tuple(q for q in current if q is oldest or q in decodes)
    oldest = prefills[0]
    if len(oldest.prompt_ids) - oldest.prefill_offset > 128:
        return None
    return 'single128', (oldest,)
