"""Device-resident slot maintenance for the frozen Qwen3.8-27B state ABI."""

import pypto.language as pl

BATCH = 8
CONV_ELEMENTS = 48 * 4 * 10240
RECURRENT_ELEMENTS = 48 * 48 * 128 * 128
TILE = 32768


@pl.jit
def zero_state(conv: pl.InOut[pl.Tensor], recurrent: pl.InOut[pl.Tensor],
               slot: pl.Scalar[pl.INDEX]):
    conv_rows = pl.reshape(conv, [BATCH, CONV_ELEMENTS])
    recurrent_rows = pl.reshape(recurrent, [BATCH, RECURRENT_ELEMENTS])
    with pl.spmd(CONV_ELEMENTS // TILE, name_hint="qwen_zero_conv_state"):
        column = pl.tile.get_block_idx() * TILE
        conv_rows[slot:slot + 1, column:column + TILE] = pl.full([1, TILE], dtype=pl.BF16, value=0.0)
    with pl.spmd(RECURRENT_ELEMENTS // TILE, name_hint="qwen_zero_recurrent_state"):
        column = pl.tile.get_block_idx() * TILE
        recurrent_rows[slot:slot + 1, column:column + TILE] = pl.full([1, TILE], dtype=pl.FP32, value=0.0)
    return conv, recurrent


@pl.jit
def compact_states(conv: pl.InOut[pl.Tensor], recurrent: pl.InOut[pl.Tensor],
                   scratch_conv: pl.InOut[pl.Tensor], scratch_recurrent: pl.InOut[pl.Tensor],
                   sources: pl.Tensor, destinations: pl.Tensor,
                   move_count: pl.Scalar[pl.INT32]):
    conv_rows = pl.reshape(conv, [BATCH, CONV_ELEMENTS])
    recurrent_rows = pl.reshape(recurrent, [BATCH, RECURRENT_ELEMENTS])
    scratch_conv_rows = pl.reshape(scratch_conv, [BATCH, CONV_ELEMENTS])
    scratch_recurrent_rows = pl.reshape(scratch_recurrent, [BATCH, RECURRENT_ELEMENTS])
    with pl.spmd(BATCH * (CONV_ELEMENTS // TILE), name_hint="qwen_gather_conv_state") as gather_conv:
        task = pl.tile.get_block_idx()
        move = task // (CONV_ELEMENTS // TILE)
        column = task % (CONV_ELEMENTS // TILE) * TILE
        if move < move_count:
            source = pl.read(sources, [move])
            scratch_conv_rows[move:move + 1, column:column + TILE] = pl.slice(conv_rows, [1, TILE], [source, column])
    with pl.spmd(BATCH * (RECURRENT_ELEMENTS // TILE), name_hint="qwen_gather_recurrent_state") as gather_recurrent:
        task = pl.tile.get_block_idx()
        move = task // (RECURRENT_ELEMENTS // TILE)
        column = task % (RECURRENT_ELEMENTS // TILE) * TILE
        if move < move_count:
            source = pl.read(sources, [move])
            scratch_recurrent_rows[move:move + 1, column:column + TILE] = pl.slice(recurrent_rows, [1, TILE], [source, column])
    with pl.spmd(BATCH * (CONV_ELEMENTS // TILE), name_hint="qwen_scatter_conv_state", deps=[gather_conv]):
        task = pl.tile.get_block_idx()
        move = task // (CONV_ELEMENTS // TILE)
        column = task % (CONV_ELEMENTS // TILE) * TILE
        if move < move_count:
            destination = pl.read(destinations, [move])
            conv_rows[destination:destination + 1, column:column + TILE] = pl.slice(scratch_conv_rows, [1, TILE], [move, column])
    with pl.spmd(BATCH * (RECURRENT_ELEMENTS // TILE), name_hint="qwen_scatter_recurrent_state", deps=[gather_recurrent]):
        task = pl.tile.get_block_idx()
        move = task // (RECURRENT_ELEMENTS // TILE)
        column = task % (RECURRENT_ELEMENTS // TILE) * TILE
        if move < move_count:
            destination = pl.read(destinations, [move])
            recurrent_rows[destination:destination + 1, column:column + TILE] = pl.slice(scratch_recurrent_rows, [1, TILE], [move, column])
    return conv, recurrent, scratch_conv, scratch_recurrent


@pl.jit.host
def zero_state_fwd(conv: pl.InOut[pl.Tensor], recurrent: pl.InOut[pl.Tensor],
                   slot: pl.Scalar[pl.INDEX]):
    zero_state(conv, recurrent, slot, device=0)


@pl.jit.host
def compact_states_fwd(conv: pl.InOut[pl.Tensor], recurrent: pl.InOut[pl.Tensor],
                       scratch_conv: pl.InOut[pl.Tensor], scratch_recurrent: pl.InOut[pl.Tensor],
                       sources: pl.Tensor, destinations: pl.Tensor,
                       move_count: pl.Scalar[pl.INT32]):
    compact_states(conv, recurrent, scratch_conv, scratch_recurrent, sources,
                   destinations, move_count, device=0)
