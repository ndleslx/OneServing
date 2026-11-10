"""Exact greedy selection over the useful row of padded Qwen logits.

FP32 indices are exact for this vocabulary. NaNs win over finite values, then
the smallest index wins; this is the original CPU torch.argmax convention.
Each partial occupies a full 32-byte row to avoid overlapping DMA stores.
"""

import pypto.language as pl

BATCH = 8
VOCAB = 248320
CHUNK = 4096
FULL_CHUNKS = 60
TAIL_CHUNK = 2560
CHUNKS = 64
PARTIAL_COLUMNS = CHUNKS * 8
SENTINEL = 1000000.0
NEGATIVE_FINITE = -1.0e38


@pl.jit.inline
def greedy_partial(values: pl.Tensor, maxima: pl.Tensor, indices: pl.Tensor,
                   nan_indices: pl.Tensor, slot: pl.Scalar[pl.INDEX],
                   part: pl.Scalar[pl.INDEX], base: pl.Scalar[pl.INDEX],
                   row: pl.Scalar[pl.INDEX]):
    raw = pl.load(values, [row, base], [1, CHUNK])
    fp = pl.tile.reshape(pl.tile.cast(raw, pl.FP32), [8, CHUNK // 8])
    local_index = pl.tile.reshape(pl.tile.cast(pl.tile.ci(0, [1, CHUNK], dtype=pl.INT32), pl.FP32), [8, CHUNK // 8])
    global_index = pl.tile.adds(local_index, pl.cast(pl.cast(base, pl.INT32), pl.FP32))
    nan_mask = pl.tile.cmp(fp, fp, cmp_type=1)
    negative_infinity = pl.tile.muls(pl.tile.full([8, CHUNK // 8], dtype=pl.FP32, value=NEGATIVE_FINITE), 4.0)
    safe = pl.tile.select(nan_mask, negative_infinity, fp)
    scratch = pl.tile.create([8, CHUNK // 8], dtype=pl.FP32)
    maximum = pl.tile.row_max(safe, scratch)
    expanded = pl.tile.row_expand(pl.tile.create([8, CHUNK // 8], dtype=pl.FP32), maximum)
    tied_indices = pl.tile.select(pl.tile.cmp(safe, expanded, cmp_type=0), global_index, SENTINEL)
    first_index = pl.tile.row_min(tied_indices, scratch)
    first_nan = pl.tile.row_min(pl.tile.select(nan_mask, global_index, SENTINEL), scratch)
    maximum_row = pl.tile.reshape(maximum, [1, 8])
    index_row = pl.tile.reshape(first_index, [1, 8])
    nan_row = pl.tile.reshape(first_nan, [1, 8])
    pl.store(maximum_row, [slot, part * 8], maxima)
    pl.store(index_row, [slot, part * 8], indices)
    pl.store(nan_row, [slot, part * 8], nan_indices)


@pl.jit.inline
def greedy_partial_tail(values: pl.Tensor, maxima: pl.Tensor, indices: pl.Tensor,
                   nan_indices: pl.Tensor, slot: pl.Scalar[pl.INDEX],
                   part: pl.Scalar[pl.INDEX], base: pl.Scalar[pl.INDEX],
                   row: pl.Scalar[pl.INDEX]):
    raw = pl.load(values, [row, base], [1, TAIL_CHUNK])
    fp = pl.tile.reshape(pl.tile.cast(raw, pl.FP32), [8, TAIL_CHUNK // 8])
    local_index = pl.tile.reshape(pl.tile.cast(pl.tile.ci(0, [1, TAIL_CHUNK], dtype=pl.INT32), pl.FP32), [8, TAIL_CHUNK // 8])
    global_index = pl.tile.adds(local_index, pl.cast(pl.cast(base, pl.INT32), pl.FP32))
    nan_mask = pl.tile.cmp(fp, fp, cmp_type=1)
    negative_infinity = pl.tile.muls(pl.tile.full([8, TAIL_CHUNK // 8], dtype=pl.FP32, value=NEGATIVE_FINITE), 4.0)
    safe = pl.tile.select(nan_mask, negative_infinity, fp)
    scratch = pl.tile.create([8, TAIL_CHUNK // 8], dtype=pl.FP32)
    maximum = pl.tile.row_max(safe, scratch)
    expanded = pl.tile.row_expand(pl.tile.create([8, TAIL_CHUNK // 8], dtype=pl.FP32), maximum)
    tied_indices = pl.tile.select(pl.tile.cmp(safe, expanded, cmp_type=0), global_index, SENTINEL)
    first_index = pl.tile.row_min(tied_indices, scratch)
    first_nan = pl.tile.row_min(pl.tile.select(nan_mask, global_index, SENTINEL), scratch)
    maximum_row = pl.tile.reshape(maximum, [1, 8])
    index_row = pl.tile.reshape(first_index, [1, 8])
    nan_row = pl.tile.reshape(first_nan, [1, 8])
    pl.store(maximum_row, [slot, part * 8], maxima)
    pl.store(index_row, [slot, part * 8], indices)
    pl.store(nan_row, [slot, part * 8], nan_indices)


@pl.jit
def greedy(logits: pl.Tensor, counts: pl.Tensor,
           maxima: pl.InOut[pl.Tensor], indices: pl.InOut[pl.Tensor],
           nan_indices: pl.InOut[pl.Tensor], tokens: pl.InOut[pl.Tensor],
           active: pl.Scalar[pl.INT32]):
    rows = pl.tensor.dim(logits, 1)
    flat = pl.reshape(logits, [BATCH * rows, VOCAB])
    with pl.spmd(BATCH * CHUNKS, name_hint="qwen_greedy_partials") as partials:
        block = pl.tile.get_block_idx()
        slot = block // CHUNKS
        part = block % CHUNKS
        if slot < active:
            row = pl.read(counts, [slot]) - 1
            if part < FULL_CHUNKS:
                greedy_partial(flat, maxima, indices, nan_indices, slot, part, part * CHUNK, slot * rows + row)
            elif part == FULL_CHUNKS:
                greedy_partial_tail(flat, maxima, indices, nan_indices, slot, part, FULL_CHUNKS * CHUNK, slot * rows + row)
            else:
                pl.store(pl.tile.muls(pl.tile.full([1, 8], dtype=pl.FP32, value=NEGATIVE_FINITE), 4.0), [slot, part * 8], maxima)
                pl.store(pl.tile.full([1, 8], dtype=pl.FP32, value=SENTINEL), [slot, part * 8], indices)
                pl.store(pl.tile.full([1, 8], dtype=pl.FP32, value=SENTINEL), [slot, part * 8], nan_indices)
    with pl.spmd(BATCH, name_hint="qwen_greedy_finalize", deps=[partials]):
        final_slot = pl.tile.get_block_idx()
        if final_slot < active:
            part_max = pl.tile.reshape(pl.load(maxima, [final_slot, 0], [1, PARTIAL_COLUMNS]), [8, 64])
            part_index = pl.tile.reshape(pl.load(indices, [final_slot, 0], [1, PARTIAL_COLUMNS]), [8, 64])
            part_nan = pl.tile.reshape(pl.load(nan_indices, [final_slot, 0], [1, PARTIAL_COLUMNS]), [8, 64])
            final_scratch = pl.tile.create([8, 64], dtype=pl.FP32)
            row_maxima = pl.tile.row_max(part_max, final_scratch)
            final_max = pl.tile.col_max(pl.tile.row_expand(pl.tile.create([8, 8], dtype=pl.FP32), row_maxima))
            final_max_column = pl.tile.reshape(final_max, [8, 1])
            final_expanded = pl.tile.row_expand(pl.tile.create([8, 64], dtype=pl.FP32), final_max_column)
            row_indices = pl.tile.row_min(pl.tile.select(
                pl.tile.cmp(part_max, final_expanded, cmp_type=0), part_index, SENTINEL), final_scratch)
            final_index = pl.tile.col_min(pl.tile.row_expand(pl.tile.create([8, 8], dtype=pl.FP32), row_indices))
            row_nan = pl.tile.row_min(part_nan, final_scratch)
            final_nan = pl.tile.col_min(pl.tile.row_expand(pl.tile.create([8, 8], dtype=pl.FP32), row_nan))
            selected = pl.tile.select(pl.tile.cmps(final_nan, SENTINEL, cmp_type=2), final_nan, final_index)
            pl.store(pl.tile.cast(selected, pl.INT32, mode="rint"), [final_slot, 0], tokens)
    return maxima, indices, nan_indices, tokens


@pl.jit.host
def greedy_fwd(logits: pl.Tensor, counts: pl.Tensor,
               maxima: pl.InOut[pl.Tensor], indices: pl.InOut[pl.Tensor],
               nan_indices: pl.InOut[pl.Tensor], tokens: pl.InOut[pl.Tensor],
               active: pl.Scalar[pl.INT32]):
    greedy(logits, counts, maxima, indices, nan_indices, tokens, active, device=0)
