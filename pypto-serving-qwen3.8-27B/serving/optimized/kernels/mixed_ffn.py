"""Layer-major prefill with packed projections and original state/attention arithmetic."""
import pypto.language as pl
from ..model import linear_ROWS as _pm334_linear_ROWS, linear_HIDDEN as _pm334_linear_HIDDEN, linear_OUTPUT as _pm334_linear_OUTPUT, linear_INPUT as _pm334_linear_INPUT, linear_OUTPUT_TILE as _pm334_linear_OUTPUT_TILE, linear_K_TILE as _pm334_linear_K_TILE, rmsnorm_ROWS as _pm334_rmsnorm_ROWS, rmsnorm_HIDDEN as _pm334_rmsnorm_HIDDEN, rmsnorm_CHUNK as _pm334_rmsnorm_CHUNK, gated_norm_ROWS as _pm334_gated_norm_ROWS, gated_norm_DIM as _pm334_gated_norm_DIM, gated_norm_ROW_TILE as _pm334_gated_norm_ROW_TILE, causal_conv_TOKENS as _pm334_causal_conv_TOKENS, causal_conv_CHANNELS as _pm334_causal_conv_CHANNELS, causal_conv_KERNEL as _pm334_causal_conv_KERNEL, causal_conv_CHANNEL_TILE as _pm334_causal_conv_CHANNEL_TILE, delta_prepare_TOKENS as _pm334_delta_prepare_TOKENS, delta_prepare_KEY_HEADS as _pm334_delta_prepare_KEY_HEADS, delta_prepare_VALUE_HEADS as _pm334_delta_prepare_VALUE_HEADS, delta_prepare_DIM as _pm334_delta_prepare_DIM, delta_prepare_CHANNELS as _pm334_delta_prepare_CHANNELS, delta_prepare_PAD_ROWS as _pm334_delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS as _pm334_delta_prepare_PAD_HEADS, delta_window16_ROWS as _pm334_delta_window16_ROWS, delta_window16_KEY_HEADS as _pm334_delta_window16_KEY_HEADS, delta_window16_HEADS as _pm334_delta_window16_HEADS, delta_window16_DIM as _pm334_delta_window16_DIM, delta_window16_VALUE_TILE as _pm334_delta_window16_VALUE_TILE, delta_window16_NORM_TILE as _pm334_delta_window16_NORM_TILE, delta_window16_COLUMN_REP as _pm334_delta_window16_COLUMN_REP, delta_window16_VALUE_WIDTH as _pm334_delta_window16_VALUE_WIDTH, attention3_prepare_ROWS as _pm334_attention3_prepare_ROWS, attention3_prepare_HIDDEN as _pm334_attention3_prepare_HIDDEN, attention3_prepare_HEAD_DIM as _pm334_attention3_prepare_HEAD_DIM, attention3_prepare_Q_HEADS as _pm334_attention3_prepare_Q_HEADS, attention3_prepare_KV_HEADS as _pm334_attention3_prepare_KV_HEADS, attention3_prepare_Q_PROJECTION as _pm334_attention3_prepare_Q_PROJECTION, attention3_prepare_KV_WIDTH as _pm334_attention3_prepare_KV_WIDTH, attention3_prepare_Q_WIDTH as _pm334_attention3_prepare_Q_WIDTH, attention3_prepare_ROTARY as _pm334_attention3_prepare_ROTARY, attention512_core_ROWS as _pm334_attention512_core_ROWS, attention512_core_CONTEXT as _pm334_attention512_core_CONTEXT, attention512_core_KEY_BLOCK as _pm334_attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM as _pm334_attention512_core_HEAD_DIM, attention512_core_Q_HEADS as _pm334_attention512_core_Q_HEADS, attention512_core_KV_HEADS as _pm334_attention512_core_KV_HEADS, attention512_core_Q_WIDTH as _pm334_attention512_core_Q_WIDTH, residual_ROWS as _pm334_residual_ROWS, residual_HIDDEN as _pm334_residual_HIDDEN, residual_TILE as _pm334_residual_TILE, swiglu_ROWS as _pm334_swiglu_ROWS, swiglu_INTERMEDIATE as _pm334_swiglu_INTERMEDIATE, swiglu_CHANNEL_TILE as _pm334_swiglu_CHANNEL_TILE, layer0_stem_ROWS as _pm334_layer0_stem_ROWS, layer0_stem_HIDDEN as _pm334_layer0_stem_HIDDEN, layer0_stem_QKV as _pm334_layer0_stem_QKV, layer0_stem_Z as _pm334_layer0_stem_Z, layer0_stem_GATE_PAD as _pm334_layer0_stem_GATE_PAD, ROWS as _pm334_ROWS, BATCH as _pm334_BATCH, CONTEXT as _pm334_CONTEXT, BLOCK_SIZE as _pm334_BLOCK_SIZE, POOL_BLOCKS as _pm334_POOL_BLOCKS, HIDDEN as _pm334_HIDDEN, INTERMEDIATE as _pm334_INTERMEDIATE, QKV as _pm334_QKV, VALUE_WIDTH as _pm334_VALUE_WIDTH, DELTA_HEADS as _pm334_DELTA_HEADS, DELTA_DIM as _pm334_DELTA_DIM, FULL_HEAD_DIM as _pm334_FULL_HEAD_DIM, FULL_Q_WIDTH as _pm334_FULL_Q_WIDTH, FULL_Q_PROJECTION as _pm334_FULL_Q_PROJECTION, FULL_KV_WIDTH as _pm334_FULL_KV_WIDTH, ROTARY as _pm334_ROTARY, VOCAB as _pm334_VOCAB, tp_rank_ROWS as _pm334_tp_rank_ROWS, tp_rank_BATCH as _pm334_tp_rank_BATCH, tp_rank_CONTEXT as _pm334_tp_rank_CONTEXT, tp_rank_BLOCK_SIZE as _pm334_tp_rank_BLOCK_SIZE, tp_rank_POOL_BLOCKS as _pm334_tp_rank_POOL_BLOCKS, tp_rank_HIDDEN as _pm334_tp_rank_HIDDEN, tp_rank_INTERMEDIATE as _pm334_tp_rank_INTERMEDIATE, tp_rank_QKV as _pm334_tp_rank_QKV, tp_rank_VALUE_WIDTH as _pm334_tp_rank_VALUE_WIDTH, tp_rank_DELTA_HEADS as _pm334_tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM as _pm334_tp_rank_DELTA_DIM, tp_rank_FULL_HEAD_DIM as _pm334_tp_rank_FULL_HEAD_DIM, tp_rank_FULL_Q_WIDTH as _pm334_tp_rank_FULL_Q_WIDTH, tp_rank_FULL_Q_PROJECTION as _pm334_tp_rank_FULL_Q_PROJECTION, tp_rank_FULL_KV_WIDTH as _pm334_tp_rank_FULL_KV_WIDTH, tp_rank_ROTARY as _pm334_tp_rank_ROTARY, tp_rank_VOCAB as _pm334_tp_rank_VOCAB, tp_rank_TP_VOCAB as _pm334_tp_rank_TP_VOCAB, causal_conv_causal_conv_silu_inline as _pm334_causal_conv_causal_conv_silu_inline, delta_window16_expand_qkv_window16_inline as _pm334_delta_window16_expand_qkv_window16_inline, delta_prepare_decay_beta_inline as _pm334_delta_prepare_decay_beta_inline, delta_window16_recurrence_window16_inline as _pm334_delta_window16_recurrence_window16_inline, attention3_prepare_normalize_head_inline as _pm334_attention3_prepare_normalize_head_inline, attention3_prepare_rotary_head_inline as _pm334_attention3_prepare_rotary_head_inline
_pm334_STATE_BATCH = 8
_pm334_BATCH = 1
_pm334_ROWS = 128
_pm334_PACKED = _pm334_BATCH * _pm334_ROWS
_pm334_delta_prepare_PAD_ROWS = 128
_pm334_delta_window16_ROWS = _pm334_ROWS
_pm334_attention3_prepare_ROWS = 16
_pm334_ATTENTION_TILE_ROWS = 16
_pm334_PV_TILE_ROWS = 32
_pm334_attention512_core_SCORE_KEY_BLOCK = 128
_pm334_attention512_core_ROWS = _pm334_ROWS
_pm334_delta_window16_VALUE_TILE = 128

@pl.jit.inline
def _pm334_delta_prepare_decay_beta_inline(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor):
    with pl.spmd(_pm334_ROWS // 128, name_hint='qwen_delta_decay_beta'):
        row = pl.tile.get_block_idx() * _pm334_delta_prepare_PAD_ROWS
        a_tile = pl.cast(pl.slice(a, [_pm334_delta_prepare_PAD_ROWS, _pm334_delta_prepare_PAD_HEADS], [row, 0]), pl.FP32)
        b_tile = pl.cast(pl.slice(b, [_pm334_delta_prepare_PAD_ROWS, _pm334_delta_prepare_PAD_HEADS], [row, 0]), pl.FP32)
        a_parameter = pl.cast(pl.slice(a_log, [1, _pm334_delta_prepare_PAD_HEADS], [0, 0]), pl.FP32)
        dt_parameter = pl.cast(pl.slice(dt_bias, [1, _pm334_delta_prepare_PAD_HEADS], [0, 0]), pl.FP32)
        softplus_input = pl.col_expand_add(a_tile, dt_parameter)
        softplus = pl.log(pl.add(pl.exp(softplus_input), 1.0))
        decay = pl.mul(pl.col_expand_mul(softplus, pl.exp(a_parameter)), -1.0)
        beta_value = pl.div(pl.full([_pm334_delta_prepare_PAD_ROWS, _pm334_delta_prepare_PAD_HEADS], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(b_tile)), 1.0))
        log_decay[row:row + _pm334_delta_prepare_PAD_ROWS, 0:_pm334_delta_prepare_PAD_HEADS] = decay
        beta[row:row + _pm334_delta_prepare_PAD_ROWS, 0:_pm334_delta_prepare_PAD_HEADS] = pl.cast(beta_value, pl.BF16, mode='rint')
    return (log_decay, beta)

@pl.jit.inline
def _pm334_attention3_prepare_normalize_head_inline(head_values: pl.Tensor, gamma: pl.Tensor, result: pl.Tensor):
    source = pl.cast(head_values, pl.FP32)
    square = pl.mul(source, source)
    inverse = pl.rsqrt(pl.add(pl.mul(pl.row_sum(square), 1.0 / _pm334_attention3_prepare_HEAD_DIM), 1e-06), high_precision=True)
    normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), gamma)
    result[0:_pm334_attention3_prepare_ROWS, 0:_pm334_attention3_prepare_HEAD_DIM] = pl.cast(normalized, pl.BF16, mode='rint')
    return result

@pl.jit.inline
def _pm334_attention3_prepare_rotary_head_inline(normalized: pl.Tensor[[_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_HEAD_DIM], pl.BF16], cos: pl.Tensor[[_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_ROTARY], pl.BF16], sin: pl.Tensor[[_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_ROTARY], pl.BF16], result: pl.Tensor[[_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_HEAD_DIM], pl.BF16]):
    first = pl.cast(pl.slice(normalized, [_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    second = pl.cast(pl.slice(normalized, [_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_ROTARY // 2], [0, _pm334_attention3_prepare_ROTARY // 2]), pl.FP32)
    cos_first = pl.cast(pl.slice(cos, [_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    cos_second = pl.cast(pl.slice(cos, [_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_ROTARY // 2], [0, _pm334_attention3_prepare_ROTARY // 2]), pl.FP32)
    sin_first = pl.cast(pl.slice(sin, [_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    sin_second = pl.cast(pl.slice(sin, [_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_ROTARY // 2], [0, _pm334_attention3_prepare_ROTARY // 2]), pl.FP32)
    rotated_first = pl.cast(pl.sub(pl.mul(first, cos_first), pl.mul(second, sin_first)), pl.BF16, mode='rint')
    rotated_second = pl.cast(pl.add(pl.mul(second, cos_second), pl.mul(first, sin_second)), pl.BF16, mode='rint')
    rest = pl.slice(normalized, [_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_HEAD_DIM - _pm334_attention3_prepare_ROTARY], [0, _pm334_attention3_prepare_ROTARY])
    result[0:_pm334_attention3_prepare_ROWS, 0:_pm334_attention3_prepare_HEAD_DIM] = pl.concat(pl.concat(rotated_first, rotated_second), rest)
    return result

@pl.jit.inline
def _pm334_prefill_gated_tail_inline(core: pl.Tensor, z_projection: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor, gated_output: pl.Tensor, attention_output: pl.Tensor):
    gate_rows = pl.reshape(z_projection, [_pm334_delta_window16_ROWS * _pm334_delta_window16_HEADS, _pm334_delta_window16_DIM])
    core_rows = pl.reshape(core, [_pm334_delta_window16_ROWS * _pm334_delta_window16_HEADS, _pm334_delta_window16_DIM])
    with pl.spmd(_pm334_delta_window16_ROWS * _pm334_delta_window16_HEADS // 64, name_hint='qwen_window_gated_rmsnorm'):
        row = pl.tile.get_block_idx() * 64
        source = pl.cast(pl.slice(core_rows, [64, _pm334_delta_window16_DIM], [row, 0]), pl.FP32)
        gate_tile = pl.cast(pl.slice(gate_rows, [64, _pm334_delta_window16_DIM], [row, 0]), pl.FP32)
        weight_tile = pl.cast(pl.slice(norm_weight, [1, _pm334_gated_norm_DIM], [0, 0]), pl.FP32)
        variance = pl.mul(pl.row_sum(pl.mul(source, source)), 1.0 / _pm334_delta_window16_DIM)
        inverse = pl.rsqrt(pl.add(variance, 1e-06), high_precision=True)
        normalized = pl.cast(pl.row_expand_mul(source, inverse), pl.BF16, mode='rint')
        weighted = pl.cast(pl.col_expand_mul(pl.cast(normalized, pl.FP32), weight_tile), pl.BF16, mode='rint')
        sigmoid = pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0))
        gated_output[row:row + 64, 0:_pm334_delta_window16_DIM] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), sigmoid), pl.BF16, mode='rint')
    return gated_output

@pl.jit.inline
def _pm334_prefill_attention_original_inline(query: pl.Tensor, key_cache: pl.Tensor, value_cache: pl.Tensor, gate: pl.Tensor, mask: pl.Tensor, output_weight: pl.Tensor, logits_buffer: pl.Tensor, probability_buffer: pl.Tensor, weighted_output: pl.Tensor, gated_output: pl.Tensor, projected_output: pl.Tensor):
    with pl.spmd(_pm334_attention512_core_Q_HEADS * (_pm334_ROWS // _pm334_ATTENTION_TILE_ROWS), name_hint='qwen_attention512_scores'):
        score_task = pl.tile.get_block_idx()
        score_head = score_task // (_pm334_ROWS // _pm334_ATTENTION_TILE_ROWS)
        score_row = score_task % (_pm334_ROWS // _pm334_ATTENTION_TILE_ROWS) * _pm334_ATTENTION_TILE_ROWS
        score_kv_head = score_head // (_pm334_attention512_core_Q_HEADS // _pm334_attention512_core_KV_HEADS)
        score_start = score_head * _pm334_attention512_core_HEAD_DIM
        score_q = pl.slice(query, [_pm334_ATTENTION_TILE_ROWS, _pm334_attention512_core_HEAD_DIM], [score_row, score_start])
        for score_block in pl.range(_pm334_attention512_core_CONTEXT // _pm334_attention512_core_SCORE_KEY_BLOCK):
            score_key_start = score_block * _pm334_attention512_core_SCORE_KEY_BLOCK
            score_k = pl.reshape(pl.slice(key_cache, [1, _pm334_attention512_core_SCORE_KEY_BLOCK, _pm334_attention512_core_HEAD_DIM], [score_kv_head, score_key_start, 0]), [_pm334_attention512_core_SCORE_KEY_BLOCK, _pm334_attention512_core_HEAD_DIM])
            score_raw = pl.matmul(score_q, score_k, pl.FP32, b_trans=True)
            score_rounded = pl.cast(score_raw, pl.BF16, mode='rint')
            score_scaled = pl.cast(pl.mul(pl.cast(score_rounded, pl.FP32), _pm334_attention512_core_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
            score_mask = pl.slice(mask, [_pm334_ATTENTION_TILE_ROWS, _pm334_attention512_core_SCORE_KEY_BLOCK], [score_row, score_key_start])
            logits_buffer[score_head * _pm334_attention512_core_ROWS + score_row:score_head * _pm334_attention512_core_ROWS + score_row + _pm334_ATTENTION_TILE_ROWS, score_key_start:score_key_start + _pm334_attention512_core_SCORE_KEY_BLOCK] = pl.add(pl.cast(score_scaled, pl.FP32), score_mask)
    with pl.spmd(_pm334_attention512_core_Q_HEADS * (_pm334_attention512_core_ROWS // 8), name_hint='qwen_attention512_softmax'):
        softmax_row = pl.tile.get_block_idx() * 8
        softmax_logits = pl.slice(logits_buffer, [8, _pm334_attention512_core_CONTEXT], [softmax_row, 0])
        softmax_exp = pl.exp(pl.row_expand_sub(softmax_logits, pl.row_max(softmax_logits)))
        probability_buffer[softmax_row:softmax_row + 8, 0:_pm334_attention512_core_CONTEXT] = pl.cast(pl.row_expand_div(softmax_exp, pl.row_sum(softmax_exp)), pl.BF16, mode='rint')
    with pl.spmd(_pm334_attention512_core_Q_HEADS * (_pm334_ROWS // _pm334_PV_TILE_ROWS), name_hint='qwen_attention512_weighted'):
        weighted_task = pl.tile.get_block_idx()
        weighted_head = weighted_task // (_pm334_ROWS // _pm334_PV_TILE_ROWS)
        weighted_row = weighted_task % (_pm334_ROWS // _pm334_PV_TILE_ROWS) * _pm334_PV_TILE_ROWS
        weighted_kv_head = weighted_head // (_pm334_attention512_core_Q_HEADS // _pm334_attention512_core_KV_HEADS)
        weighted_start = weighted_head * _pm334_attention512_core_HEAD_DIM
        weighted_sum = pl.full([_pm334_PV_TILE_ROWS, _pm334_attention512_core_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for weighted_block in pl.range(_pm334_attention512_core_CONTEXT // _pm334_attention512_core_KEY_BLOCK):
            weighted_key_start = weighted_block * _pm334_attention512_core_KEY_BLOCK
            weighted_probs = pl.slice(probability_buffer, [_pm334_PV_TILE_ROWS, _pm334_attention512_core_KEY_BLOCK], [weighted_head * _pm334_attention512_core_ROWS + weighted_row, weighted_key_start])
            weighted_values = pl.reshape(pl.slice(value_cache, [1, _pm334_attention512_core_KEY_BLOCK, _pm334_attention512_core_HEAD_DIM], [weighted_kv_head, weighted_key_start, 0]), [_pm334_attention512_core_KEY_BLOCK, _pm334_attention512_core_HEAD_DIM])
            weighted_sum = pl.add(weighted_sum, pl.matmul(weighted_probs, weighted_values, pl.FP32))
        weighted = pl.cast(weighted_sum, pl.BF16, mode='rint')
        weighted_output[weighted_row:weighted_row + _pm334_PV_TILE_ROWS, weighted_start:weighted_start + _pm334_attention512_core_HEAD_DIM] = weighted
        gate_tile = pl.cast(pl.slice(gate, [_pm334_PV_TILE_ROWS, _pm334_attention512_core_HEAD_DIM], [weighted_row, weighted_start]), pl.FP32)
        sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        gated_output[weighted_row:weighted_row + _pm334_PV_TILE_ROWS, weighted_start:weighted_start + _pm334_attention512_core_HEAD_DIM] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint')
    return gated_output

@pl.jit.inline
def _pm334_prefill_attention_zero_tail_inline(query: pl.Tensor, key_cache: pl.Tensor, value_cache: pl.Tensor, gate: pl.Tensor, mask: pl.Tensor, output_weight: pl.Tensor, logits_buffer: pl.Tensor, probability_buffer: pl.Tensor, weighted_output: pl.Tensor, gated_output: pl.Tensor, projected_output: pl.Tensor, valid_history: pl.Scalar[pl.INT32]):
    with pl.spmd(_pm334_attention512_core_Q_HEADS * (_pm334_ROWS // _pm334_ATTENTION_TILE_ROWS), name_hint='qwen_attention512_scores'):
        score_task = pl.tile.get_block_idx()
        score_head = score_task // (_pm334_ROWS // _pm334_ATTENTION_TILE_ROWS)
        score_row = score_task % (_pm334_ROWS // _pm334_ATTENTION_TILE_ROWS) * _pm334_ATTENTION_TILE_ROWS
        score_kv_head = score_head // (_pm334_attention512_core_Q_HEADS // _pm334_attention512_core_KV_HEADS)
        score_start = score_head * _pm334_attention512_core_HEAD_DIM
        score_q = pl.slice(query, [_pm334_ATTENTION_TILE_ROWS, _pm334_attention512_core_HEAD_DIM], [score_row, score_start])
        for score_block in pl.range(_pm334_attention512_core_CONTEXT // _pm334_attention512_core_SCORE_KEY_BLOCK):
            score_key_start = score_block * _pm334_attention512_core_SCORE_KEY_BLOCK
            if score_key_start < (valid_history + 15) // 16 * 16:
                score_k = pl.reshape(pl.slice(key_cache, [1, _pm334_attention512_core_SCORE_KEY_BLOCK, _pm334_attention512_core_HEAD_DIM], [score_kv_head, score_key_start, 0]), [_pm334_attention512_core_SCORE_KEY_BLOCK, _pm334_attention512_core_HEAD_DIM])
                score_raw = pl.matmul(score_q, score_k, pl.FP32, b_trans=True)
                score_rounded = pl.cast(score_raw, pl.BF16, mode='rint')
                score_scaled = pl.cast(pl.mul(pl.cast(score_rounded, pl.FP32), _pm334_attention512_core_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
                score_mask = pl.slice(mask, [_pm334_ATTENTION_TILE_ROWS, _pm334_attention512_core_SCORE_KEY_BLOCK], [score_row, score_key_start])
                logits_buffer[score_head * _pm334_attention512_core_ROWS + score_row:score_head * _pm334_attention512_core_ROWS + score_row + _pm334_ATTENTION_TILE_ROWS, score_key_start:score_key_start + _pm334_attention512_core_SCORE_KEY_BLOCK] = pl.add(pl.cast(score_scaled, pl.FP32), score_mask)
            else:
                score_mask = pl.slice(mask, [_pm334_ATTENTION_TILE_ROWS, _pm334_attention512_core_SCORE_KEY_BLOCK], [score_row, score_key_start])
                logits_buffer[score_head * _pm334_attention512_core_ROWS + score_row:score_head * _pm334_attention512_core_ROWS + score_row + _pm334_ATTENTION_TILE_ROWS, score_key_start:score_key_start + _pm334_attention512_core_SCORE_KEY_BLOCK] = score_mask
    with pl.spmd(_pm334_attention512_core_Q_HEADS * (_pm334_attention512_core_ROWS // 8), name_hint='qwen_attention512_softmax'):
        softmax_row = pl.tile.get_block_idx() * 8
        softmax_logits = pl.slice(logits_buffer, [8, _pm334_attention512_core_CONTEXT], [softmax_row, 0])
        softmax_exp = pl.exp(pl.row_expand_sub(softmax_logits, pl.row_max(softmax_logits)))
        probability_buffer[softmax_row:softmax_row + 8, 0:_pm334_attention512_core_CONTEXT] = pl.cast(pl.row_expand_div(softmax_exp, pl.row_sum(softmax_exp)), pl.BF16, mode='rint')
    with pl.spmd(_pm334_attention512_core_Q_HEADS * (_pm334_ROWS // _pm334_PV_TILE_ROWS), name_hint='qwen_attention512_weighted'):
        weighted_task = pl.tile.get_block_idx()
        weighted_head = weighted_task // (_pm334_ROWS // _pm334_PV_TILE_ROWS)
        weighted_row = weighted_task % (_pm334_ROWS // _pm334_PV_TILE_ROWS) * _pm334_PV_TILE_ROWS
        weighted_kv_head = weighted_head // (_pm334_attention512_core_Q_HEADS // _pm334_attention512_core_KV_HEADS)
        weighted_start = weighted_head * _pm334_attention512_core_HEAD_DIM
        weighted_sum = pl.full([_pm334_PV_TILE_ROWS, _pm334_attention512_core_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for weighted_block in pl.range((valid_history + 15) // 16):
            weighted_key_start = weighted_block * _pm334_attention512_core_KEY_BLOCK
            weighted_probs = pl.slice(probability_buffer, [_pm334_PV_TILE_ROWS, _pm334_attention512_core_KEY_BLOCK], [weighted_head * _pm334_attention512_core_ROWS + weighted_row, weighted_key_start])
            weighted_values = pl.reshape(pl.slice(value_cache, [1, _pm334_attention512_core_KEY_BLOCK, _pm334_attention512_core_HEAD_DIM], [weighted_kv_head, weighted_key_start, 0]), [_pm334_attention512_core_KEY_BLOCK, _pm334_attention512_core_HEAD_DIM])
            weighted_sum = pl.add(weighted_sum, pl.matmul(weighted_probs, weighted_values, pl.FP32))
        if valid_history < 2033:
            weighted_sum = pl.add(weighted_sum, pl.full([_pm334_PV_TILE_ROWS, _pm334_attention512_core_HEAD_DIM], dtype=pl.FP32, value=0.0))
        weighted = pl.cast(weighted_sum, pl.BF16, mode='rint')
        weighted_output[weighted_row:weighted_row + _pm334_PV_TILE_ROWS, weighted_start:weighted_start + _pm334_attention512_core_HEAD_DIM] = weighted
        gate_tile = pl.cast(pl.slice(gate, [_pm334_PV_TILE_ROWS, _pm334_attention512_core_HEAD_DIM], [weighted_row, weighted_start]), pl.FP32)
        sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        gated_output[weighted_row:weighted_row + _pm334_PV_TILE_ROWS, weighted_start:weighted_start + _pm334_attention512_core_HEAD_DIM] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint')
    return gated_output

@pl.jit.inline
def _pm334_prefill_attention_core_inline(query: pl.Tensor, key_cache: pl.Tensor, value_cache: pl.Tensor, gate: pl.Tensor, mask: pl.Tensor, output_weight: pl.Tensor, logits_buffer: pl.Tensor, probability_buffer: pl.Tensor, weighted_output: pl.Tensor, gated_output: pl.Tensor, projected_output: pl.Tensor, valid_history: pl.Scalar[pl.INT32]):
    if valid_history >= 2033:
        result = _pm334_prefill_attention_original_inline(query, key_cache, value_cache, gate, mask, output_weight, logits_buffer, probability_buffer, weighted_output, gated_output, projected_output)
    else:
        result = _pm334_prefill_attention_zero_tail_inline(query, key_cache, value_cache, gate, mask, output_weight, logits_buffer, probability_buffer, weighted_output, gated_output, projected_output, valid_history)
    return result

@pl.jit.inline
def _pm334_prefill_prepare_heads_inline(q_projected: pl.Tensor[[_pm334_ROWS, _pm334_FULL_Q_PROJECTION], pl.BF16], k_projected: pl.Tensor[[_pm334_ROWS, _pm334_FULL_KV_WIDTH], pl.BF16], value: pl.Tensor[[_pm334_ROWS, _pm334_FULL_KV_WIDTH], pl.BF16], q_gamma: pl.Tensor[[1, _pm334_FULL_HEAD_DIM], pl.FP32], k_gamma: pl.Tensor[[1, _pm334_FULL_HEAD_DIM], pl.FP32], cos: pl.Tensor[[_pm334_ROWS, _pm334_ROTARY], pl.BF16], sin: pl.Tensor[[_pm334_ROWS, _pm334_ROTARY], pl.BF16], q_norm: pl.Tensor, k_norm: pl.Tensor, q_rot: pl.Tensor, k_rot: pl.Tensor, gate: pl.Tensor):
    with pl.spmd(_pm334_attention3_prepare_Q_HEADS * (_pm334_ROWS // _pm334_attention3_prepare_ROWS), name_hint='qwen_attention_query_norm'):
        task = pl.tile.get_block_idx()
        head = task // (_pm334_ROWS // _pm334_attention3_prepare_ROWS)
        row = task % (_pm334_ROWS // _pm334_attention3_prepare_ROWS) * _pm334_attention3_prepare_ROWS
        projection_start = head * _pm334_attention3_prepare_HEAD_DIM * 2
        result_start = head * _pm334_attention3_prepare_HEAD_DIM
        raw = pl.slice(q_projected, [_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_HEAD_DIM], [row, projection_start])
        normalized = pl.create_tensor([_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        normalized = _pm334_attention3_prepare_normalize_head_inline(raw, q_gamma, normalized)
        q_norm[row:row + _pm334_attention3_prepare_ROWS, result_start:result_start + _pm334_attention3_prepare_HEAD_DIM] = normalized
        gate[row:row + _pm334_attention3_prepare_ROWS, result_start:result_start + _pm334_attention3_prepare_HEAD_DIM] = pl.slice(q_projected, [_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_HEAD_DIM], [row, projection_start + _pm334_attention3_prepare_HEAD_DIM])
        rotated = pl.create_tensor([_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        tile_cos = pl.slice(cos, [_pm334_attention3_prepare_ROWS, _pm334_ROTARY], [row, 0])
        tile_sin = pl.slice(sin, [_pm334_attention3_prepare_ROWS, _pm334_ROTARY], [row, 0])
        rotated = _pm334_attention3_prepare_rotary_head_inline(normalized, tile_cos, tile_sin, rotated)
        q_rot[row:row + _pm334_attention3_prepare_ROWS, result_start:result_start + _pm334_attention3_prepare_HEAD_DIM] = rotated
    with pl.spmd(_pm334_attention3_prepare_KV_HEADS * (_pm334_ROWS // _pm334_attention3_prepare_ROWS), name_hint='qwen_attention_key_norm'):
        task = pl.tile.get_block_idx()
        head = task // (_pm334_ROWS // _pm334_attention3_prepare_ROWS)
        row = task % (_pm334_ROWS // _pm334_attention3_prepare_ROWS) * _pm334_attention3_prepare_ROWS
        start = head * _pm334_attention3_prepare_HEAD_DIM
        raw = pl.slice(k_projected, [_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_HEAD_DIM], [row, start])
        normalized = pl.create_tensor([_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        normalized = _pm334_attention3_prepare_normalize_head_inline(raw, k_gamma, normalized)
        k_norm[row:row + _pm334_attention3_prepare_ROWS, start:start + _pm334_attention3_prepare_HEAD_DIM] = normalized
        rotated = pl.create_tensor([_pm334_attention3_prepare_ROWS, _pm334_attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        tile_cos = pl.slice(cos, [_pm334_attention3_prepare_ROWS, _pm334_ROTARY], [row, 0])
        tile_sin = pl.slice(sin, [_pm334_attention3_prepare_ROWS, _pm334_ROTARY], [row, 0])
        rotated = _pm334_attention3_prepare_rotary_head_inline(normalized, tile_cos, tile_sin, rotated)
        k_rot[row:row + _pm334_attention3_prepare_ROWS, start:start + _pm334_attention3_prepare_HEAD_DIM] = rotated
    return (q_norm, k_norm, q_rot, k_rot, value, gate)

@pl.jit.inline
def _pm334_prefill_direct_qkv_recurrence_inline(conv_output: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor, state_value_key: pl.Tensor, output: pl.Tensor, token_count: pl.Scalar[pl.INDEX]):
    direct_q_norm = pl.create_tensor([_pm334_delta_window16_ROWS * _pm334_delta_window16_KEY_HEADS, _pm334_delta_window16_DIM], dtype=pl.FP32)
    direct_k_norm = pl.create_tensor([_pm334_delta_window16_ROWS * _pm334_delta_window16_KEY_HEADS, _pm334_delta_window16_DIM], dtype=pl.FP32)
    with pl.spmd(_pm334_delta_window16_ROWS, name_hint='qwen_prefill_compact_qk_norm') as direct_norm_task:
        direct_norm_token = pl.tile.get_block_idx()
        direct_q_tile = pl.cast(pl.reshape(pl.slice(conv_output, [1, _pm334_delta_window16_KEY_HEADS * _pm334_delta_window16_DIM], [direct_norm_token, 0]), [_pm334_delta_window16_KEY_HEADS, _pm334_delta_window16_DIM]), pl.FP32)
        direct_k_tile = pl.cast(pl.reshape(pl.slice(conv_output, [1, _pm334_delta_window16_KEY_HEADS * _pm334_delta_window16_DIM], [direct_norm_token, _pm334_delta_window16_KEY_HEADS * _pm334_delta_window16_DIM]), [_pm334_delta_window16_KEY_HEADS, _pm334_delta_window16_DIM]), pl.FP32)
        direct_q_norm[direct_norm_token * _pm334_delta_window16_KEY_HEADS:(direct_norm_token + 1) * _pm334_delta_window16_KEY_HEADS, 0:_pm334_delta_window16_DIM] = pl.mul(pl.row_expand_mul(direct_q_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(direct_q_tile, direct_q_tile)), 1e-06), high_precision=True)), _pm334_delta_window16_DIM ** (-0.5))
        direct_k_norm[direct_norm_token * _pm334_delta_window16_KEY_HEADS:(direct_norm_token + 1) * _pm334_delta_window16_KEY_HEADS, 0:_pm334_delta_window16_DIM] = pl.row_expand_mul(direct_k_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(direct_k_tile, direct_k_tile)), 1e-06), high_precision=True))
    direct_state_rows = pl.reshape(state_value_key, [_pm334_delta_window16_HEADS * _pm334_delta_window16_DIM, _pm334_delta_window16_DIM])
    direct_output_rows = pl.reshape(output, [_pm334_delta_window16_ROWS, _pm334_delta_window16_HEADS * _pm334_delta_window16_DIM])
    with pl.spmd(_pm334_delta_window16_HEADS * (_pm334_delta_window16_DIM // _pm334_delta_window16_VALUE_TILE), name_hint='qwen_prefill_direct_recurrence', deps=[direct_norm_task], optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=3)]):
        direct_task = pl.tile.get_block_idx()
        direct_head = direct_task // (_pm334_delta_window16_DIM // _pm334_delta_window16_VALUE_TILE)
        direct_value_start = direct_task % (_pm334_delta_window16_DIM // _pm334_delta_window16_VALUE_TILE) * _pm334_delta_window16_VALUE_TILE
        direct_state_start = direct_head * _pm334_delta_window16_DIM + direct_value_start
        direct_state = pl.slice(direct_state_rows, [_pm334_delta_window16_VALUE_TILE, _pm334_delta_window16_DIM], [direct_state_start, 0])
        for direct_pair in pl.range(token_count // 2):
            direct_token0 = direct_pair * 2
            direct_row0 = direct_token0 * _pm334_delta_window16_KEY_HEADS + direct_head // (_pm334_delta_window16_HEADS // _pm334_delta_window16_KEY_HEADS)
            direct_q0 = direct_q_norm[direct_row0:direct_row0 + 1, 0:_pm334_delta_window16_DIM]
            direct_k0 = direct_k_norm[direct_row0:direct_row0 + 1, 0:_pm334_delta_window16_DIM]
            direct_g0 = pl.read(log_decay, [direct_token0, direct_head])
            direct_decay0 = pl.exp(pl.mul(pl.full([1, _pm334_delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_g0))
            direct_b0 = pl.read(beta, [direct_token0, direct_head])
            direct_decayed0 = pl.col_expand_mul(direct_state, direct_decay0)
            direct_remembered0 = pl.row_sum(pl.col_expand_mul(direct_decayed0, direct_k0))
            direct_v_row0 = pl.cast(pl.slice(conv_output, [1, _pm334_delta_window16_VALUE_TILE], [direct_token0, 2 * _pm334_delta_window16_KEY_HEADS * _pm334_delta_window16_DIM + direct_head * _pm334_delta_window16_DIM + direct_value_start]), pl.FP32)
            direct_v_col0 = pl.reshape(direct_v_row0, [_pm334_delta_window16_VALUE_TILE, 1])
            direct_delta0 = pl.mul(pl.sub(direct_v_col0, direct_remembered0), direct_b0)
            direct_state0 = pl.add(direct_decayed0, pl.col_expand_mul(pl.row_expand_mul(pl.full([_pm334_delta_window16_VALUE_TILE, _pm334_delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_delta0), direct_k0))
            direct_product0 = pl.matmul(direct_q0, direct_state0, pl.BF16, b_trans=True)
            direct_token1 = direct_token0 + 1
            direct_row1 = direct_token1 * _pm334_delta_window16_KEY_HEADS + direct_head // (_pm334_delta_window16_HEADS // _pm334_delta_window16_KEY_HEADS)
            direct_q1 = direct_q_norm[direct_row1:direct_row1 + 1, 0:_pm334_delta_window16_DIM]
            direct_k1 = direct_k_norm[direct_row1:direct_row1 + 1, 0:_pm334_delta_window16_DIM]
            direct_g1 = pl.read(log_decay, [direct_token1, direct_head])
            direct_decay1 = pl.exp(pl.mul(pl.full([1, _pm334_delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_g1))
            direct_b1 = pl.read(beta, [direct_token1, direct_head])
            direct_decayed1 = pl.col_expand_mul(direct_state0, direct_decay1)
            direct_remembered1 = pl.row_sum(pl.col_expand_mul(direct_decayed1, direct_k1))
            direct_v_row1 = pl.cast(pl.slice(conv_output, [1, _pm334_delta_window16_VALUE_TILE], [direct_token1, 2 * _pm334_delta_window16_KEY_HEADS * _pm334_delta_window16_DIM + direct_head * _pm334_delta_window16_DIM + direct_value_start]), pl.FP32)
            direct_v_col1 = pl.reshape(direct_v_row1, [_pm334_delta_window16_VALUE_TILE, 1])
            direct_delta1 = pl.mul(pl.sub(direct_v_col1, direct_remembered1), direct_b1)
            direct_state = pl.add(direct_decayed1, pl.col_expand_mul(pl.row_expand_mul(pl.full([_pm334_delta_window16_VALUE_TILE, _pm334_delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_delta1), direct_k1))
            direct_product1 = pl.matmul(direct_q1, direct_state, pl.BF16, b_trans=True)
            direct_output_rows[direct_token0:direct_token0 + 1, direct_head * _pm334_delta_window16_DIM + direct_value_start:direct_head * _pm334_delta_window16_DIM + direct_value_start + _pm334_delta_window16_VALUE_TILE] = pl.set_validshape(direct_product0, 1, _pm334_delta_window16_VALUE_TILE)
            direct_output_rows[direct_token1:direct_token1 + 1, direct_head * _pm334_delta_window16_DIM + direct_value_start:direct_head * _pm334_delta_window16_DIM + direct_value_start + _pm334_delta_window16_VALUE_TILE] = pl.set_validshape(direct_product1, 1, _pm334_delta_window16_VALUE_TILE)
        if token_count % 2 == 1:
            direct_token = token_count - 1
            direct_row = direct_token * _pm334_delta_window16_KEY_HEADS + direct_head // (_pm334_delta_window16_HEADS // _pm334_delta_window16_KEY_HEADS)
            direct_q = direct_q_norm[direct_row:direct_row + 1, 0:_pm334_delta_window16_DIM]
            direct_k = direct_k_norm[direct_row:direct_row + 1, 0:_pm334_delta_window16_DIM]
            direct_g = pl.read(log_decay, [direct_token, direct_head])
            direct_decay = pl.exp(pl.mul(pl.full([1, _pm334_delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_g))
            direct_b = pl.read(beta, [direct_token, direct_head])
            direct_decayed = pl.col_expand_mul(direct_state, direct_decay)
            direct_remembered = pl.row_sum(pl.col_expand_mul(direct_decayed, direct_k))
            direct_v_row = pl.cast(pl.slice(conv_output, [1, _pm334_delta_window16_VALUE_TILE], [direct_token, 2 * _pm334_delta_window16_KEY_HEADS * _pm334_delta_window16_DIM + direct_head * _pm334_delta_window16_DIM + direct_value_start]), pl.FP32)
            direct_v_col = pl.reshape(direct_v_row, [_pm334_delta_window16_VALUE_TILE, 1])
            direct_delta = pl.mul(pl.sub(direct_v_col, direct_remembered), direct_b)
            direct_state = pl.add(direct_decayed, pl.col_expand_mul(pl.row_expand_mul(pl.full([_pm334_delta_window16_VALUE_TILE, _pm334_delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_delta), direct_k))
            direct_output_rows[direct_token:direct_token + 1, direct_head * _pm334_delta_window16_DIM + direct_value_start:direct_head * _pm334_delta_window16_DIM + direct_value_start + _pm334_delta_window16_VALUE_TILE] = pl.set_validshape(pl.matmul(direct_q, direct_state, pl.BF16, b_trans=True), 1, _pm334_delta_window16_VALUE_TILE)
        for direct_token in pl.range(token_count, _pm334_delta_window16_ROWS):
            direct_output_rows[direct_token:direct_token + 1, direct_head * _pm334_delta_window16_DIM + direct_value_start:direct_head * _pm334_delta_window16_DIM + direct_value_start + _pm334_delta_window16_VALUE_TILE] = pl.full([1, _pm334_delta_window16_VALUE_TILE], dtype=pl.BF16, value=0.0)
        direct_state_rows[direct_state_start:direct_state_start + _pm334_delta_window16_VALUE_TILE, 0:_pm334_delta_window16_DIM] = direct_state
    return (output, state_value_key)

@pl.jit.inline
def _pm334_prefill_load_paged_kv_inline(key_pool: pl.Tensor, value_pool: pl.Tensor, block_table: pl.Tensor, key_local: pl.Tensor, value_local: pl.Tensor, full: pl.Scalar[pl.INDEX], cache_offset: pl.Scalar[pl.INT32], token_count: pl.Scalar[pl.INT32]):
    valid_blocks = (cache_offset + token_count + _pm334_BLOCK_SIZE - 1) // _pm334_BLOCK_SIZE
    with pl.spmd(4 * (_pm334_CONTEXT // 128), name_hint='qwen_prefill_zero_local_kv') as zero_task:
        task = pl.tile.get_block_idx()
        head = task // (_pm334_CONTEXT // 128)
        row = task % (_pm334_CONTEXT // 128) * 128
        zero = pl.full([1, 128, _pm334_FULL_HEAD_DIM], dtype=pl.BF16, value=0.0)
        key_local[head:head + 1, row:row + 128, 0:_pm334_FULL_HEAD_DIM] = zero
        value_local[head:head + 1, row:row + 128, 0:_pm334_FULL_HEAD_DIM] = zero
    with pl.spmd(4 * valid_blocks, name_hint='qwen_prefill_load_valid_kv', deps=[zero_task]):
        task = pl.tile.get_block_idx()
        head = task // valid_blocks
        logical_block = task % valid_blocks
        physical = pl.read(block_table, [logical_block])
        row = logical_block * _pm334_BLOCK_SIZE
        key_local[head:head + 1, row:row + _pm334_BLOCK_SIZE, 0:_pm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(key_pool, [1, 1, 1, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM])
        value_local[head:head + 1, row:row + _pm334_BLOCK_SIZE, 0:_pm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_pool, [1, 1, 1, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM])
    return (key_local, value_local)

@pl.jit.inline
def _pm334_prefill_project_inline_prefix512(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(1 * output_tiles, name_hint='prefill_nz_projection_active4'):
        task = pl.tile.get_block_idx()
        row = task % 1 * 128
        col = task // 1 * 128
        accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
            weight_tile = pl.slice(weight, [256, 128], [k_start, col])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
        output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def _pm334_prefill_project_ffn_up_paired_m128(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(1 * output_tiles, name_hint='prefill_ffn_up_paired_m128'):
        task = pl.tile.get_block_idx()
        row = task % 1 * 128
        col = task // 1 * 128
        accumulator_first = pl.create_tensor([128, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_first = pl.slice(hidden, [128, 256], [row, k_start])
            weight_tile = pl.slice(weight, [256, 128], [k_start, col])
            accumulator_first = pl.matmul_acc(accumulator_first, hidden_first, weight_tile, b_trans=False, init_cond=chunk == 0)
        output[row:row + 128, col:col + 128] = pl.cast(accumulator_first, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def _pm334_prefill_norm_inline_prefix512(hidden: pl.Tensor, gamma: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(hidden, 0)
    with pl.spmd(packed_rows // _pm334_rmsnorm_ROWS, name_hint='qwen_prefill_rmsnorm'):
        row_start = pl.tile.get_block_idx() * _pm334_rmsnorm_ROWS
        square_sum = pl.full([1, _pm334_rmsnorm_ROWS], dtype=pl.FP32, value=0.0)
        for chunk_index in pl.range(_pm334_rmsnorm_HIDDEN // _pm334_rmsnorm_CHUNK):
            column = chunk_index * _pm334_rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [_pm334_rmsnorm_ROWS, _pm334_rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(pl.mul(source, source)), [1, _pm334_rmsnorm_ROWS]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / _pm334_rmsnorm_HIDDEN), 1e-06), high_precision=True), [_pm334_rmsnorm_ROWS, 1])
        for chunk_index in pl.range(_pm334_rmsnorm_HIDDEN // _pm334_rmsnorm_CHUNK):
            column = chunk_index * _pm334_rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [_pm334_rmsnorm_ROWS, _pm334_rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            weight = pl.slice(gamma, [1, _pm334_rmsnorm_CHUNK], [0, column])
            normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weight)
            output[row_start:row_start + _pm334_rmsnorm_ROWS, column:column + _pm334_rmsnorm_CHUNK] = pl.cast(normalized, pl.BF16, mode='rint')
    return output
_pm334_FUSED_ROWS = 16

@pl.jit.inline
def _pm334_residual_norm_fused_streaming(left: pl.Tensor, right: pl.Tensor, gamma: pl.Tensor, residual: pl.Tensor, normalized: pl.Tensor):
    with pl.spmd(_pm334_PACKED // _pm334_FUSED_ROWS, name_hint='prefill_residual_norm_fused_streaming'):
        row_start = pl.tile.get_block_idx() * _pm334_FUSED_ROWS
        square_sum = pl.full([1, _pm334_FUSED_ROWS], dtype=pl.FP32, value=0.0)
        for chunk_index in pl.range(_pm334_rmsnorm_HIDDEN // _pm334_rmsnorm_CHUNK):
            column = chunk_index * _pm334_rmsnorm_CHUNK
            a = pl.cast(pl.slice(left, [_pm334_FUSED_ROWS, _pm334_rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            b = pl.cast(pl.slice(right, [_pm334_FUSED_ROWS, _pm334_rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            rounded = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
            residual[row_start:row_start + _pm334_FUSED_ROWS, column:column + _pm334_rmsnorm_CHUNK] = rounded
            source = pl.cast(rounded, pl.FP32)
            square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(pl.mul(source, source)), [1, _pm334_FUSED_ROWS]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / _pm334_rmsnorm_HIDDEN), 1e-06), high_precision=True), [_pm334_FUSED_ROWS, 1])
        for chunk_index in pl.range(_pm334_rmsnorm_HIDDEN // _pm334_rmsnorm_CHUNK):
            column = chunk_index * _pm334_rmsnorm_CHUNK
            source = pl.cast(pl.slice(residual, [_pm334_FUSED_ROWS, _pm334_rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            weight = pl.slice(gamma, [1, _pm334_rmsnorm_CHUNK], [0, column])
            value = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weight)
            normalized[row_start:row_start + _pm334_FUSED_ROWS, column:column + _pm334_rmsnorm_CHUNK] = pl.cast(value, pl.BF16, mode='rint')
    return (residual, normalized)

@pl.jit.inline
def _pm334_residual_next_norm_trace(left: pl.Tensor, right: pl.Tensor, input_gamma: pl.Tensor, final_gamma: pl.Tensor, residual: pl.Tensor, normalized: pl.Tensor, traces: pl.Tensor, layer: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    with pl.spmd(_pm334_PACKED // _pm334_FUSED_ROWS, name_hint='prefill_residual_next_norm_trace'):
        row_start = pl.tile.get_block_idx() * _pm334_FUSED_ROWS
        square_sum = pl.full([1, _pm334_FUSED_ROWS], dtype=pl.FP32, value=0.0)
        for chunk_index in pl.range(_pm334_rmsnorm_HIDDEN // _pm334_rmsnorm_CHUNK):
            column = chunk_index * _pm334_rmsnorm_CHUNK
            a = pl.cast(pl.slice(left, [_pm334_FUSED_ROWS, _pm334_rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            b = pl.cast(pl.slice(right, [_pm334_FUSED_ROWS, _pm334_rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            rounded = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
            residual[row_start:row_start + _pm334_FUSED_ROWS, column:column + _pm334_rmsnorm_CHUNK] = rounded
            if 0 < active_count:
                traces[0:1, layer + 1:layer + 2, row_start:row_start + _pm334_FUSED_ROWS, column:column + _pm334_rmsnorm_CHUNK] = pl.reshape(rounded, [1, 1, _pm334_FUSED_ROWS, _pm334_rmsnorm_CHUNK])
            source = pl.cast(rounded, pl.FP32)
            square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(pl.mul(source, source)), [1, _pm334_FUSED_ROWS]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / _pm334_rmsnorm_HIDDEN), 1e-06), high_precision=True), [_pm334_FUSED_ROWS, 1])
        for chunk_index in pl.range(_pm334_rmsnorm_HIDDEN // _pm334_rmsnorm_CHUNK):
            column = chunk_index * _pm334_rmsnorm_CHUNK
            source = pl.cast(pl.slice(residual, [_pm334_FUSED_ROWS, _pm334_rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            if layer < 63:
                weight = pl.slice(input_gamma, [1, _pm334_rmsnorm_CHUNK], [layer + 1, column])
            else:
                weight = pl.slice(final_gamma, [1, _pm334_rmsnorm_CHUNK], [0, column])
            value = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weight)
            normalized[row_start:row_start + _pm334_FUSED_ROWS, column:column + _pm334_rmsnorm_CHUNK] = pl.cast(value, pl.BF16, mode='rint')
    return (residual, normalized, traces)

@pl.jit.inline
def _pm334_prefill_residual_inline_prefix512(left: pl.Tensor, right: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(left, 0)
    with pl.spmd(packed_rows, name_hint='qwen_prefill_residual'):
        row = pl.tile.get_block_idx()
        a = pl.cast(pl.slice(left, [1, _pm334_HIDDEN], [row, 0]), pl.FP32)
        b = pl.cast(pl.slice(right, [1, _pm334_HIDDEN], [row, 0]), pl.FP32)
        output[row:row + 1, 0:_pm334_HIDDEN] = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def _pm334_prefill_up_swiglu_fused(hidden: pl.Tensor, weight: pl.Tensor, gate: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(1 * output_tiles, name_hint='prefill_up_swiglu_fused', optimizations=[pl.cross_core_slot(slot_num=1)]):
        task = pl.tile.get_block_idx()
        row = task % 1 * 128
        col = task // 1 * 128
        accumulator_first = pl.create_tensor([128, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_first = pl.slice(hidden, [128, 256], [row, k_start])
            weight_tile = pl.slice(weight, [256, 128], [k_start, col])
            accumulator_first = pl.matmul_acc(accumulator_first, hidden_first, weight_tile, b_trans=False, init_cond=chunk == 0)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            first_row = row + aiv_id * 64
            gate_first = pl.cast(pl.slice(gate, [64, 128], [first_row, col]), pl.FP32)
            activated_first = pl.cast(pl.div(gate_first, pl.add(pl.exp(pl.neg(gate_first)), 1.0)), pl.BF16, mode='rint')
            up_first = pl.cast(pl.aiv_shard(accumulator_first), pl.BF16, mode='rint')
            middle_first = pl.cast(pl.mul(pl.cast(activated_first, pl.FP32), pl.cast(up_first, pl.FP32)), pl.BF16, mode='rint')
            output[first_row:first_row + 64, col:col + 128] = middle_first
    return output

@pl.jit.inline
def _pm334_prefill_swiglu_inline_prefix512(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(gate, 0)
    with pl.spmd(packed_rows // 16 * (_pm334_INTERMEDIATE // 1024), name_hint='qwen_prefill_swiglu'):
        task = pl.tile.get_block_idx()
        row = task // (_pm334_INTERMEDIATE // 1024) * 16
        column = task % (_pm334_INTERMEDIATE // 1024) * 1024
        gate_tile = pl.cast(pl.slice(gate, [16, 1024], [row, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        up_tile = pl.cast(pl.slice(up, [16, 1024], [row, column]), pl.FP32)
        output[row:row + 16, column:column + 1024] = pl.cast(pl.mul(pl.cast(activated, pl.FP32), up_tile), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def _pm334_prefill_batched_conv_inline_prefix512(qkv_batch: pl.Tensor, conv_weight: pl.Tensor, conv_states: pl.Tensor, conv_output: pl.Tensor, token_counts: pl.Tensor, linear: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    state_layers = pl.tensor.dim(conv_states, 1)
    conv_rows = pl.reshape(conv_states, [_pm334_STATE_BATCH * state_layers * 4, _pm334_QKV])
    with pl.spmd(packed_rows // _pm334_ROWS * (_pm334_ROWS // 16) * (_pm334_QKV // 512), name_hint='qwen_prefill_batched_conv_rows16') as conv_outputs_done:
        task = pl.tile.get_block_idx()
        batch = task // (_pm334_ROWS // 16 * (_pm334_QKV // 512))
        row = task // (_pm334_QKV // 512) % (_pm334_ROWS // 16) * 16
        column = task % (_pm334_QKV // 512) * 512
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            state_start = (batch * state_layers + linear) * 4
            if row < token_count:
                tap_0 = pl.cast(pl.slice(conv_weight, [1, 512], [0, column]), pl.FP32)
                tap_1 = pl.cast(pl.slice(conv_weight, [1, 512], [1, column]), pl.FP32)
                tap_2 = pl.cast(pl.slice(conv_weight, [1, 512], [2, column]), pl.FP32)
                tap_3 = pl.cast(pl.slice(conv_weight, [1, 512], [3, column]), pl.FP32)
                if row == 0:
                    first_previous_0 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 0, column]), pl.FP32)
                    first_previous_1 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 1, column]), pl.FP32)
                    first_previous_2 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 2, column]), pl.FP32)
                    first_previous_3 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 3, column]), pl.FP32)
                    for first_token in pl.range(16):
                        if first_token < token_count:
                            first_current = pl.cast(pl.slice(qkv_batch, [1, 512], [batch * _pm334_ROWS + first_token, column]), pl.FP32)
                            first_convolved = pl.add(pl.add(pl.mul(first_previous_1, tap_0), pl.mul(first_previous_2, tap_1)), pl.add(pl.mul(first_previous_3, tap_2), pl.mul(first_current, tap_3)))
                            first_convolved = pl.cast(pl.cast(first_convolved, pl.BF16, mode='rint'), pl.FP32)
                            first_activated = pl.div(first_convolved, pl.add(pl.exp(pl.neg(first_convolved)), 1.0), high_precision=True)
                            conv_output[batch * _pm334_ROWS + first_token:batch * _pm334_ROWS + first_token + 1, column:column + 512] = pl.cast(first_activated, pl.BF16, mode='rint')
                            first_previous_0 = first_previous_1
                            first_previous_1 = first_previous_2
                            first_previous_2 = first_previous_3
                            first_previous_3 = first_current
                        else:
                            conv_output[batch * _pm334_ROWS + first_token:batch * _pm334_ROWS + first_token + 1, column:column + 512] = pl.full([1, 512], dtype=pl.BF16, value=0.0)
                elif row + 16 <= token_count:
                    full_previous_1 = pl.cast(pl.slice(qkv_batch, [16, 512], [batch * _pm334_ROWS + row - 3, column]), pl.FP32)
                    full_previous_2 = pl.cast(pl.slice(qkv_batch, [16, 512], [batch * _pm334_ROWS + row - 2, column]), pl.FP32)
                    full_previous_3 = pl.cast(pl.slice(qkv_batch, [16, 512], [batch * _pm334_ROWS + row - 1, column]), pl.FP32)
                    full_current = pl.cast(pl.slice(qkv_batch, [16, 512], [batch * _pm334_ROWS + row, column]), pl.FP32)
                    full_convolved = pl.add(pl.add(pl.col_expand_mul(full_previous_1, tap_0), pl.col_expand_mul(full_previous_2, tap_1)), pl.add(pl.col_expand_mul(full_previous_3, tap_2), pl.col_expand_mul(full_current, tap_3)))
                    full_convolved = pl.cast(pl.cast(full_convolved, pl.BF16, mode='rint'), pl.FP32)
                    full_activated = pl.div(full_convolved, pl.add(pl.exp(pl.neg(full_convolved)), 1.0), high_precision=True)
                    conv_output[batch * _pm334_ROWS + row:batch * _pm334_ROWS + row + 16, column:column + 512] = pl.cast(full_activated, pl.BF16, mode='rint')
                else:
                    for tail_local in pl.range(16):
                        tail_token = row + tail_local
                        if tail_token < token_count:
                            tail_previous_1 = pl.cast(pl.slice(qkv_batch, [1, 512], [batch * _pm334_ROWS + tail_token - 3, column]), pl.FP32)
                            tail_previous_2 = pl.cast(pl.slice(qkv_batch, [1, 512], [batch * _pm334_ROWS + tail_token - 2, column]), pl.FP32)
                            tail_previous_3 = pl.cast(pl.slice(qkv_batch, [1, 512], [batch * _pm334_ROWS + tail_token - 1, column]), pl.FP32)
                            tail_current = pl.cast(pl.slice(qkv_batch, [1, 512], [batch * _pm334_ROWS + tail_token, column]), pl.FP32)
                            tail_convolved = pl.add(pl.add(pl.mul(tail_previous_1, tap_0), pl.mul(tail_previous_2, tap_1)), pl.add(pl.mul(tail_previous_3, tap_2), pl.mul(tail_current, tap_3)))
                            tail_convolved = pl.cast(pl.cast(tail_convolved, pl.BF16, mode='rint'), pl.FP32)
                            tail_activated = pl.div(tail_convolved, pl.add(pl.exp(pl.neg(tail_convolved)), 1.0), high_precision=True)
                            conv_output[batch * _pm334_ROWS + tail_token:batch * _pm334_ROWS + tail_token + 1, column:column + 512] = pl.cast(tail_activated, pl.BF16, mode='rint')
                        else:
                            conv_output[batch * _pm334_ROWS + tail_token:batch * _pm334_ROWS + tail_token + 1, column:column + 512] = pl.full([1, 512], dtype=pl.BF16, value=0.0)
            else:
                conv_output[batch * _pm334_ROWS + row:batch * _pm334_ROWS + row + 16, column:column + 512] = pl.full([16, 512], dtype=pl.BF16, value=0.0)
        else:
            conv_output[batch * _pm334_ROWS + row:batch * _pm334_ROWS + row + 16, column:column + 512] = pl.full([16, 512], dtype=pl.BF16, value=0.0)
    with pl.spmd(packed_rows // _pm334_ROWS * (_pm334_QKV // 512), name_hint='qwen_prefill_batched_conv_final_history', deps=[conv_outputs_done]):
        state_task = pl.tile.get_block_idx()
        state_batch = state_task // (_pm334_QKV // 512)
        state_column = state_task % (_pm334_QKV // 512) * 512
        if state_batch < active_count:
            state_count = pl.read(token_counts, [state_batch])
            state_base = (state_batch * state_layers + linear) * 4
            for history_row in pl.range(4):
                source_row = state_count - 4 + history_row
                if source_row < 0:
                    history_value = pl.cast(pl.slice(conv_rows, [1, 512], [state_base + state_count + history_row, state_column]), pl.FP32)
                else:
                    history_value = pl.cast(pl.slice(qkv_batch, [1, 512], [state_batch * _pm334_ROWS + source_row, state_column]), pl.FP32)
                conv_rows[state_base + history_row:state_base + history_row + 1, state_column:state_column + 512] = pl.cast(history_value, pl.BF16, mode='rint')
    return (conv_states, conv_output)

@pl.jit.inline
def _pm334_prefill_delta_slots_inline_prefix512(qkv_batch: pl.Tensor, z_batch: pl.Tensor, a_batch: pl.Tensor, b_batch: pl.Tensor, conv_weight_layer: pl.Tensor, a_log_layer: pl.Tensor, dt_bias_layer: pl.Tensor, delta_norm_layer: pl.Tensor, delta_out_layer: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, gated_batch: pl.Tensor, linear: pl.Scalar[pl.INDEX], token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    alias_state_layers = pl.tensor.dim(recurrent_states, 1)
    alias_conv_rows = pl.reshape(conv_states, [alias_state_layers * (_pm334_STATE_BATCH * 4), _pm334_QKV])
    alias_recurrent_rows = pl.reshape(recurrent_states, [alias_state_layers * (_pm334_STATE_BATCH * _pm334_DELTA_HEADS * _pm334_DELTA_DIM), _pm334_DELTA_DIM])
    conv_output_storage = pl.create_tensor([packed_rows, _pm334_QKV], dtype=pl.BF16)
    (updated_conv_states, conv_output_batch) = _pm334_prefill_batched_conv_inline_prefix512(qkv_batch, conv_weight_layer, conv_states, conv_output_storage, token_counts, linear, active_count)
    for batch in pl.range(packed_rows // _pm334_ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            qkv = pl.slice(qkv_batch, [_pm334_ROWS, _pm334_QKV], [batch * _pm334_ROWS, 0])
            z = pl.slice(z_batch, [_pm334_ROWS, _pm334_VALUE_WIDTH], [batch * _pm334_ROWS, 0])
            a = pl.slice(a_batch, [_pm334_ROWS, 128], [batch * _pm334_ROWS, 0])
            b = pl.slice(b_batch, [_pm334_ROWS, 128], [batch * _pm334_ROWS, 0])
            conv_source = pl.slice(alias_conv_rows, [4, _pm334_QKV], [(batch * alias_state_layers + linear) * 4, 0])
            recurrent_slice = pl.slice(alias_recurrent_rows, [_pm334_DELTA_HEADS * _pm334_DELTA_DIM, _pm334_DELTA_DIM], [(batch * alias_state_layers + linear) * _pm334_DELTA_HEADS * _pm334_DELTA_DIM, 0])
            conv = pl.reshape(conv_source, [4, _pm334_QKV])
            recurrent = pl.reshape(recurrent_slice, [_pm334_DELTA_HEADS, _pm334_DELTA_DIM, _pm334_DELTA_DIM])
            conv_output = pl.slice(conv_output_batch, [_pm334_ROWS, _pm334_QKV], [batch * _pm334_ROWS, 0])
            log_decay = pl.create_tensor([_pm334_ROWS, _pm334_DELTA_HEADS], dtype=pl.FP32)
            beta = pl.create_tensor([_pm334_ROWS, _pm334_DELTA_HEADS], dtype=pl.FP32)
            core = pl.create_tensor([_pm334_ROWS, _pm334_DELTA_HEADS, _pm334_DELTA_DIM], dtype=pl.BF16)
            gate_values = pl.create_tensor([_pm334_ROWS, 64], dtype=pl.FP32)
            beta_bf16 = pl.create_tensor([_pm334_ROWS, 64], dtype=pl.BF16)
            (gate_values, beta_bf16) = _pm334_delta_prepare_decay_beta_inline(a, b, a_log_layer, dt_bias_layer, gate_values, beta_bf16)
            with pl.spmd(_pm334_ROWS // 16, name_hint='qwen_full_delta_gate_unpad'):
                token = pl.tile.get_block_idx() * 16
                log_decay[token:token + 16, 0:_pm334_DELTA_HEADS] = pl.slice(gate_values, [16, _pm334_DELTA_HEADS], [token, 0])
                beta[token:token + 16, 0:_pm334_DELTA_HEADS] = pl.cast(pl.slice(beta_bf16, [16, _pm334_DELTA_HEADS], [token, 0]), pl.FP32)
            (core, recurrent) = _pm334_prefill_direct_qkv_recurrence_inline(conv_output, log_decay, beta, recurrent, core, token_count)
            gated = pl.create_tensor([_pm334_ROWS * _pm334_DELTA_HEADS, _pm334_DELTA_DIM], dtype=pl.BF16)
            ignored = pl.create_tensor([_pm334_ROWS, _pm334_HIDDEN], dtype=pl.BF16)
            gated = _pm334_prefill_gated_tail_inline(core, z, delta_norm_layer, delta_out_layer, gated, ignored)
            packed = pl.reshape(gated, [_pm334_ROWS, _pm334_VALUE_WIDTH])
            with pl.spmd(_pm334_ROWS // 8, name_hint='qwen_prefill_store_delta_gated'):
                row = pl.tile.get_block_idx() * 8
                gated_batch[batch * _pm334_ROWS + row:batch * _pm334_ROWS + row + 8, 0:_pm334_VALUE_WIDTH] = pl.slice(packed, [8, _pm334_VALUE_WIDTH], [row, 0])
        else:
            with pl.spmd(_pm334_ROWS // 8, name_hint='qwen_prefill_zero_delta_gated'):
                row = pl.tile.get_block_idx() * 8
                gated_batch[batch * _pm334_ROWS + row:batch * _pm334_ROWS + row + 8, 0:_pm334_VALUE_WIDTH] = pl.full([8, _pm334_VALUE_WIDTH], dtype=pl.BF16, value=0.0)
    return (conv_states, recurrent_states, gated_batch)

@pl.jit.inline
def _pm334_prefill_attention_slots_inline_prefix512(queries: pl.Tensor, keys: pl.Tensor, values: pl.Tensor, q_gamma_full: pl.Tensor, k_gamma_full: pl.Tensor, cos_batch: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_ROTARY], pl.BF16], sin_batch: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_ROTARY], pl.BF16], masks: pl.Tensor, out_weight_full: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, cache_offsets: pl.Tensor, token_counts: pl.Tensor, gated_batch: pl.Tensor, full: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(gated_batch, 0)
    for batch in pl.range(packed_rows // _pm334_ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            cache_offset = pl.read(cache_offsets, [batch])
            q_projected = pl.slice(queries, [_pm334_ROWS, _pm334_FULL_Q_PROJECTION], [batch * _pm334_ROWS, 0])
            k_projected = pl.slice(keys, [_pm334_ROWS, _pm334_FULL_KV_WIDTH], [batch * _pm334_ROWS, 0])
            value_full = pl.slice(values, [_pm334_ROWS, _pm334_FULL_KV_WIDTH], [batch * _pm334_ROWS, 0])
            cos_slice = pl.slice(cos_batch, [1, _pm334_ROWS, _pm334_ROTARY], [batch, 0, 0])
            sin_slice = pl.slice(sin_batch, [1, _pm334_ROWS, _pm334_ROTARY], [batch, 0, 0])
            mask_slice = pl.slice(masks, [1, _pm334_ROWS, _pm334_CONTEXT], [batch, 0, 0])
            table_slice = pl.slice(block_tables, [1, _pm334_CONTEXT // _pm334_BLOCK_SIZE], [batch, 0])
            cos = pl.reshape(cos_slice, [_pm334_ROWS, _pm334_ROTARY])
            sin = pl.reshape(sin_slice, [_pm334_ROWS, _pm334_ROTARY])
            mask = pl.reshape(mask_slice, [_pm334_ROWS, _pm334_CONTEXT])
            block_table = pl.reshape(table_slice, [_pm334_CONTEXT // _pm334_BLOCK_SIZE])
            q_norm_full = pl.create_tensor([_pm334_ROWS, _pm334_FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm_full = pl.create_tensor([_pm334_ROWS, _pm334_FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([_pm334_ROWS, _pm334_FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([_pm334_ROWS, _pm334_FULL_KV_WIDTH], dtype=pl.BF16)
            gate_full = pl.create_tensor([_pm334_ROWS, _pm334_FULL_Q_WIDTH], dtype=pl.BF16)
            (q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full) = _pm334_prefill_prepare_heads_inline(q_projected, k_projected, value_full, q_gamma_full, k_gamma_full, cos, sin, q_norm_full, k_norm_full, q_rot, k_rot, gate_full)
            key_local = pl.create_tensor([4, _pm334_CONTEXT, _pm334_FULL_HEAD_DIM], dtype=pl.BF16)
            value_local = pl.create_tensor([4, _pm334_CONTEXT, _pm334_FULL_HEAD_DIM], dtype=pl.BF16)
            (key_local, value_local) = _pm334_prefill_load_paged_kv_inline(key_pool, value_pool, block_table, key_local, value_local, full, cache_offset, token_count)
            with pl.spmd(4, name_hint='qwen_full_update_kv'):
                head = pl.tile.get_block_idx()
                start = head * _pm334_FULL_HEAD_DIM
                for token in pl.range(token_count):
                    key_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:_pm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(k_rot, [1, _pm334_FULL_HEAD_DIM], [token, start]), [1, 1, _pm334_FULL_HEAD_DIM])
                    value_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:_pm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_full, [1, _pm334_FULL_HEAD_DIM], [token, start]), [1, 1, _pm334_FULL_HEAD_DIM])
            scores = pl.create_tensor([24 * _pm334_ROWS, _pm334_CONTEXT], dtype=pl.FP32)
            probabilities = pl.create_tensor([24 * _pm334_ROWS, _pm334_CONTEXT], dtype=pl.BF16)
            weighted = pl.create_tensor([_pm334_ROWS, _pm334_FULL_Q_WIDTH], dtype=pl.BF16)
            full_gated = pl.create_tensor([_pm334_ROWS, _pm334_FULL_Q_WIDTH], dtype=pl.BF16)
            attention_full = pl.create_tensor([_pm334_ROWS, _pm334_HIDDEN], dtype=pl.BF16)
            full_gated = _pm334_prefill_attention_core_inline(q_rot, key_local, value_local, gate_full, mask, out_weight_full, scores, probabilities, weighted, full_gated, attention_full, cache_offset + token_count)
            first_page = cache_offset // _pm334_BLOCK_SIZE
            page_count = (cache_offset % _pm334_BLOCK_SIZE + token_count + _pm334_BLOCK_SIZE - 1) // _pm334_BLOCK_SIZE
            with pl.spmd(4 * (_pm334_ROWS // _pm334_BLOCK_SIZE + 1), name_hint='qwen_full_store_paged_kv_page16'):
                task = pl.tile.get_block_idx()
                head = task // (_pm334_ROWS // _pm334_BLOCK_SIZE + 1)
                page_idx = task % (_pm334_ROWS // _pm334_BLOCK_SIZE + 1)
                if page_idx < page_count and token_count > 0:
                    logical_block = first_page + page_idx
                    position = logical_block * _pm334_BLOCK_SIZE
                    physical = pl.read(block_table, [logical_block])
                    if position >= cache_offset and position + _pm334_BLOCK_SIZE <= cache_offset + token_count:
                        key_pool[full:full + 1, head:head + 1, physical:physical + 1, 0:_pm334_BLOCK_SIZE, 0:_pm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(key_local, [1, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM])
                        value_pool[full:full + 1, head:head + 1, physical:physical + 1, 0:_pm334_BLOCK_SIZE, 0:_pm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_local, [1, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM])
                    else:
                        for page_row in pl.range(_pm334_BLOCK_SIZE):
                            position = logical_block * _pm334_BLOCK_SIZE + page_row
                            if position >= cache_offset and position < cache_offset + token_count:
                                slot = position % _pm334_BLOCK_SIZE
                                key_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:_pm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(key_local, [1, 1, _pm334_FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, _pm334_FULL_HEAD_DIM])
                                value_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:_pm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_local, [1, 1, _pm334_FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, _pm334_FULL_HEAD_DIM])
            with pl.spmd(_pm334_ROWS // 8, name_hint='qwen_prefill_store_attention_gated'):
                row = pl.tile.get_block_idx() * 8
                gated_batch[batch * _pm334_ROWS + row:batch * _pm334_ROWS + row + 8, 0:_pm334_FULL_Q_WIDTH] = pl.slice(full_gated, [8, _pm334_FULL_Q_WIDTH], [row, 0])
        else:
            with pl.spmd(_pm334_ROWS // 8, name_hint='qwen_prefill_zero_attention_gated'):
                row = pl.tile.get_block_idx() * 8
                gated_batch[batch * _pm334_ROWS + row:batch * _pm334_ROWS + row + 8, 0:_pm334_FULL_Q_WIDTH] = pl.full([8, _pm334_FULL_Q_WIDTH], dtype=pl.BF16, value=0.0)
    return (key_pool, value_pool, gated_batch)

@pl.jit.inline
def _pm334_prefill_last_logits_inline_prefix512(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    selected_hidden = pl.create_tensor([16, _pm334_HIDDEN], dtype=pl.BF16)
    with pl.spmd(16, name_hint='prefill_gather_last_hidden'):
        batch = pl.tile.get_block_idx()
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if count > 0:
                source_row = batch * _pm334_ROWS + count - 1
                selected_hidden[batch:batch + 1, 0:_pm334_HIDDEN] = pl.slice(hidden, [1, _pm334_HIDDEN], [source_row, 0])
            else:
                selected_hidden[batch:batch + 1, 0:_pm334_HIDDEN] = pl.full([1, _pm334_HIDDEN], dtype=pl.BF16, value=0.0)
        else:
            selected_hidden[batch:batch + 1, 0:_pm334_HIDDEN] = pl.full([1, _pm334_HIDDEN], dtype=pl.BF16, value=0.0)
    output_tiles = pl.tensor.dim(weight, 0) // 128
    input_columns = pl.tensor.dim(weight, 1)
    with pl.spmd(output_tiles, name_hint='prefill_last_vocab_projection'):
        column = pl.tile.get_block_idx() * 128
        accumulator = pl.create_tensor([16, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_tile = pl.slice(selected_hidden, [16, 256], [0, k_start])
            weight_tile = pl.slice(weight, [128, 256], [column, k_start])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=True, init_cond=chunk == 0)
        projected = pl.cast(accumulator, pl.BF16, mode='rint')
        for batch in pl.range(_pm334_BATCH):
            if batch < active_count:
                count = pl.read(token_counts, [batch])
                if count > 0:
                    destination_row = batch * _pm334_ROWS + count - 1
                    output[destination_row:destination_row + 1, column:column + 128] = pl.slice(projected, [1, 128], [batch, 0])
    return output

@pl.jit.inline
def _pm334_prefill_batch_prefix512_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    running = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
    with pl.spmd(_pm334_PACKED, name_hint='qwen_prefill_embedding'):
        row = pl.tile.get_block_idx()
        batch = row // _pm334_ROWS
        token_row = row % _pm334_ROWS
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if token_row < count:
                token = pl.read(input_ids, [batch, token_row])
                embedded = pl.slice(embedding, [1, _pm334_HIDDEN], [token, 0])
            else:
                embedded = pl.full([1, _pm334_HIDDEN], dtype=pl.BF16, value=0.0)
            running[row:row + 1, 0:_pm334_HIDDEN] = embedded
            traces[batch:batch + 1, 0:1, token_row:token_row + 1, 0:_pm334_HIDDEN] = pl.reshape(embedded, [1, 1, 1, _pm334_HIDDEN])
        else:
            running[row:row + 1, 0:_pm334_HIDDEN] = pl.full([1, _pm334_HIDDEN], dtype=pl.BF16, value=0.0)
    norm = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
    first_gamma: pl.Tensor[[1, _pm334_HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, _pm334_HIDDEN], [0, 0])
    norm = _pm334_prefill_norm_inline_prefix512(running, first_gamma, norm)
    for layer in pl.range(64):
        post_gamma_layer: pl.Tensor[[1, _pm334_HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, _pm334_HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[_pm334_HIDDEN, _pm334_INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        mlp_up_layer: pl.Tensor[[_pm334_HIDDEN, _pm334_INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        mlp_down_layer: pl.Tensor[[_pm334_INTERMEDIATE, _pm334_HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        residual = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
        post_norm = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[_pm334_HIDDEN, _pm334_QKV], pl.BF16, pl.NZ] = delta_qkv_weight[linear]
            z_weight_layer: pl.Tensor[[_pm334_HIDDEN, _pm334_VALUE_WIDTH], pl.BF16, pl.NZ] = delta_z_weight[linear]
            a_weight_layer: pl.Tensor[[_pm334_HIDDEN, 128], pl.BF16, pl.NZ] = delta_a_weight[linear]
            b_weight_layer: pl.Tensor[[_pm334_HIDDEN, 128], pl.BF16, pl.NZ] = delta_b_weight[linear]
            conv_weight_layer: pl.Tensor[[4, _pm334_QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, _pm334_QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, _pm334_DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, _pm334_DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[_pm334_VALUE_WIDTH, _pm334_HIDDEN], pl.BF16, pl.NZ] = delta_out_weight[linear]
            qkv = pl.create_tensor([_pm334_PACKED, _pm334_QKV], dtype=pl.BF16)
            z = pl.create_tensor([_pm334_PACKED, _pm334_VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([_pm334_PACKED, 128], dtype=pl.BF16)
            b = pl.create_tensor([_pm334_PACKED, 128], dtype=pl.BF16)
            qkv = _pm334_prefill_project_ffn_up_paired_m128(norm, qkv_weight_layer, qkv, active_count)
            z = _pm334_prefill_project_ffn_up_paired_m128(norm, z_weight_layer, z, active_count)
            a = _pm334_prefill_project_inline_prefix512(norm, a_weight_layer, a, active_count)
            b = _pm334_prefill_project_inline_prefix512(norm, b_weight_layer, b, active_count)
            gated = pl.create_tensor([_pm334_PACKED, _pm334_VALUE_WIDTH], dtype=pl.BF16)
            _pm334_prefill_delta_slots_inline_prefix512(qkv, z, a, b, conv_weight_layer, a_log_layer, dt_bias_layer, delta_norm_layer, delta_out_layer, conv_states, recurrent_states, gated, linear, token_counts, active_count)
            attention = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
            attention = _pm334_prefill_project_ffn_up_paired_m128(gated, delta_out_layer, attention, active_count)
            (residual, post_norm) = _pm334_residual_norm_fused_streaming(running, attention, post_gamma_layer, residual, post_norm)
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[_pm334_HIDDEN, _pm334_FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[full]
            k_weight_full: pl.Tensor[[_pm334_HIDDEN, _pm334_FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[full]
            v_weight_full: pl.Tensor[[_pm334_HIDDEN, _pm334_FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[full]
            q_gamma_full: pl.Tensor[[1, _pm334_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, _pm334_FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, _pm334_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, _pm334_FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[_pm334_FULL_Q_WIDTH, _pm334_HIDDEN], pl.BF16, pl.NZ] = full_out_weight[full]
            q = pl.create_tensor([_pm334_PACKED, _pm334_FULL_Q_PROJECTION], dtype=pl.BF16)
            k = pl.create_tensor([_pm334_PACKED, _pm334_FULL_KV_WIDTH], dtype=pl.BF16)
            v = pl.create_tensor([_pm334_PACKED, _pm334_FULL_KV_WIDTH], dtype=pl.BF16)
            q = _pm334_prefill_project_ffn_up_paired_m128(norm, q_weight_full, q, active_count)
            k = _pm334_prefill_project_inline_prefix512(norm, k_weight_full, k, active_count)
            v = _pm334_prefill_project_inline_prefix512(norm, v_weight_full, v, active_count)
            gated = pl.create_tensor([_pm334_PACKED, _pm334_FULL_Q_WIDTH], dtype=pl.BF16)
            _pm334_prefill_attention_slots_inline_prefix512(q, k, v, q_gamma_full, k_gamma_full, cos, sin, mask, out_weight_full, key_pool, value_pool, block_tables, cache_offsets, token_counts, gated, full, active_count)
            attention = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
            attention = _pm334_prefill_project_ffn_up_paired_m128(gated, out_weight_full, attention, active_count)
            (residual, post_norm) = _pm334_residual_norm_fused_streaming(running, attention, post_gamma_layer, residual, post_norm)
        middle = pl.create_tensor([_pm334_PACKED, _pm334_INTERMEDIATE], dtype=pl.BF16)
        middle = _ffn369_gate_up_swiglu_shared_activation(post_norm, mlp_gate_layer, mlp_up_layer, middle)
        mlp_output = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
        mlp_output = _pm334_prefill_project_ffn_up_paired_m128(middle, mlp_down_layer, mlp_output, active_count)
        (running, norm, traces) = _pm334_residual_next_norm_trace(residual, mlp_output, input_gamma, final_gamma, running, norm, traces, layer, active_count)
    normalized = norm
    logits_rows = pl.reshape(logits, [_pm334_PACKED, _pm334_VOCAB])
    _pm334_prefill_last_logits_inline_prefix512(normalized, lm_head, logits_rows, token_counts, active_count)
    with pl.spmd(_pm334_PACKED, name_hint='qwen_prefill_store_final_norm'):
        row = pl.tile.get_block_idx()
        batch = row // _pm334_ROWS
        token_row = row % _pm334_ROWS
        if batch < active_count:
            traces[batch:batch + 1, 65:66, token_row:token_row + 1, 0:_pm334_HIDDEN] = pl.reshape(pl.slice(normalized, [1, _pm334_HIDDEN], [row, 0]), [1, 1, 1, _pm334_HIDDEN])
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit.inline
def _pm334_prefill_batch_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    prefix_group = (active_count + 1) // 2
    if prefix_group == 1:
        _pm334_prefill_batch_prefix512_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def _pm334_rank_prefill_shared(input_ids: pl.Tensor[[_pm334_BATCH, _pm334_ROWS], pl.INT32], embedding: pl.Tensor[[_pm334_VOCAB, _pm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _pm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_pm334_VOCAB, _pm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _pm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _pm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _pm334_INTERMEDIATE, _pm334_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _pm334_INTERMEDIATE, _pm334_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _pm334_HIDDEN, _pm334_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _pm334_QKV, _pm334_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _pm334_VALUE_WIDTH, _pm334_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _pm334_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _pm334_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _pm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _pm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _pm334_HIDDEN, _pm334_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _pm334_FULL_Q_PROJECTION, _pm334_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _pm334_FULL_KV_WIDTH, _pm334_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _pm334_FULL_KV_WIDTH, _pm334_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _pm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _pm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _pm334_HIDDEN, _pm334_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_ROTARY], pl.BF16], sin: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_ROTARY], pl.BF16], mask: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_CONTEXT], pl.FP32], token_counts: pl.Tensor[[_pm334_BATCH], pl.INT32], cache_offsets: pl.Tensor[[_pm334_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[_pm334_STATE_BATCH, 48, 4, _pm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_pm334_STATE_BATCH, 48, _pm334_DELTA_HEADS, _pm334_DELTA_DIM, _pm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _pm334_POOL_BLOCKS, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _pm334_POOL_BLOCKS, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[_pm334_BATCH, _pm334_CONTEXT // _pm334_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[_pm334_BATCH, 66, _pm334_ROWS, _pm334_HIDDEN], pl.BF16]], logits: pl.InOut[pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    return _pm334_prefill_batch_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)

@pl.jit.host
def _pm334_prefill_fwd(input_ids: pl.Tensor[[_pm334_BATCH, _pm334_ROWS], pl.INT32], embedding: pl.Tensor[[_pm334_VOCAB, _pm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _pm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_pm334_VOCAB, _pm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _pm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _pm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64, _pm334_HIDDEN, _pm334_INTERMEDIATE], pl.BF16, pl.NZ], mlp_up_weight: pl.Tensor[[64, _pm334_HIDDEN, _pm334_INTERMEDIATE], pl.BF16, pl.NZ], mlp_down_weight: pl.Tensor[[64, _pm334_INTERMEDIATE, _pm334_HIDDEN], pl.BF16, pl.NZ], delta_qkv_weight: pl.Tensor[[48, _pm334_HIDDEN, _pm334_QKV], pl.BF16, pl.NZ], delta_z_weight: pl.Tensor[[48, _pm334_HIDDEN, _pm334_VALUE_WIDTH], pl.BF16, pl.NZ], delta_a_weight: pl.Tensor[[48, _pm334_HIDDEN, 128], pl.BF16, pl.NZ], delta_b_weight: pl.Tensor[[48, _pm334_HIDDEN, 128], pl.BF16, pl.NZ], delta_conv_weight: pl.Tensor[[48 * 4, _pm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _pm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48, _pm334_VALUE_WIDTH, _pm334_HIDDEN], pl.BF16, pl.NZ], full_q_weight: pl.Tensor[[16, _pm334_HIDDEN, _pm334_FULL_Q_PROJECTION], pl.BF16, pl.NZ], full_k_weight: pl.Tensor[[16, _pm334_HIDDEN, _pm334_FULL_KV_WIDTH], pl.BF16, pl.NZ], full_v_weight: pl.Tensor[[16, _pm334_HIDDEN, _pm334_FULL_KV_WIDTH], pl.BF16, pl.NZ], full_q_gamma: pl.Tensor[[16, _pm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _pm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16, _pm334_FULL_Q_WIDTH, _pm334_HIDDEN], pl.BF16, pl.NZ], cos: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_ROTARY], pl.BF16], sin: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_ROTARY], pl.BF16], mask: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_CONTEXT], pl.FP32], token_counts: pl.Tensor[[_pm334_BATCH], pl.INT32], cache_offsets: pl.Tensor[[_pm334_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[_pm334_STATE_BATCH, 48, 4, _pm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_pm334_STATE_BATCH, 48, _pm334_DELTA_HEADS, _pm334_DELTA_DIM, _pm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _pm334_POOL_BLOCKS, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _pm334_POOL_BLOCKS, _pm334_BLOCK_SIZE, _pm334_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[_pm334_BATCH, _pm334_CONTEXT // _pm334_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[_pm334_BATCH, 66, _pm334_ROWS, _pm334_HIDDEN], pl.BF16]], logits: pl.InOut[pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    _pm334_rank_prefill_shared(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, device=0)
'ROWS=8 decode specialization for the Qwen3.8-27B PyPTO kernels.'
import pypto.language as pl
import pypto.language.distributed as pld
_dm334_linear_ROWS = 8
_dm334_linear_HIDDEN = 5120
_dm334_linear_OUTPUT = pl.dynamic('QWEN_LINEAR_OUTPUT')
_dm334_linear_INPUT = pl.dynamic('QWEN_LINEAR_INPUT')
_dm334_linear_OUTPUT_TILE = 128
_dm334_linear_K_TILE = 512

@pl.jit.inline
def _dm334_linear_hidden_projection_inline(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Tensor):
    output_columns = pl.tensor.dim(result, 1)
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(output_columns // 256, name_hint='qwen_nz_hidden_projection'):
        start = pl.tile.get_block_idx() * 256
        accumulator = pl.create_tensor([_dm334_linear_ROWS, 256], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_tile = pl.slice(hidden, [_dm334_linear_ROWS, 256], [0, k_start])
            weight_tile = pl.slice(weight, [256, 256], [k_start, start])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
        result[0:_dm334_linear_ROWS, start:start + 256] = pl.cast(accumulator, pl.BF16, mode='rint')
    return result

@pl.jit
def _dm334_linear_hidden_projection(hidden: pl.Tensor[[_dm334_linear_ROWS, _dm334_linear_INPUT], pl.BF16], weight: pl.Tensor[[_dm334_linear_OUTPUT, _dm334_linear_INPUT], pl.BF16], result: pl.Out[pl.Tensor[[_dm334_linear_ROWS, _dm334_linear_OUTPUT], pl.BF16]]):
    result = _dm334_linear_hidden_projection_inline(hidden, weight, result)
    return result
_dm334_rmsnorm_ROWS = 8
_dm334_rmsnorm_HIDDEN = 5120
_dm334_rmsnorm_CHUNK = 128

@pl.jit.inline
def _dm334_rmsnorm_rmsnorm_inline(hidden: pl.Tensor[[_dm334_rmsnorm_ROWS, _dm334_rmsnorm_HIDDEN], pl.BF16], gamma: pl.Tensor[[1, _dm334_rmsnorm_HIDDEN], pl.FP32], output: pl.Tensor[[_dm334_rmsnorm_ROWS, _dm334_rmsnorm_HIDDEN], pl.BF16]):
    with pl.spmd(1, name_hint='qwen_rmsnorm'):
        row_start = pl.tile.get_block_idx() * 8
        square_sum = pl.full([1, 8], dtype=pl.FP32, value=0.0)
        for load_group in pl.pipeline(0, _dm334_rmsnorm_HIDDEN // 512, stage=2):
            column = load_group * 512
            source_wide = pl.cast(pl.slice(hidden, [8, 512], [row_start, column]), pl.FP32)
            square_wide = pl.mul(source_wide, source_wide)
            for reduction_part in pl.unroll(4):
                square_part = pl.slice(square_wide, [8, 128], [0, reduction_part * 128])
                square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(square_part), [1, 8]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / _dm334_rmsnorm_HIDDEN), 1e-06), high_precision=True), [8, 1])
        for output_group in pl.pipeline(0, _dm334_rmsnorm_HIDDEN // 512, stage=2):
            output_column = output_group * 512
            output_source = pl.cast(pl.slice(hidden, [8, 512], [row_start, output_column]), pl.FP32)
            output_weight = pl.slice(gamma, [1, 512], [0, output_column])
            normalized = pl.col_expand_mul(pl.row_expand_mul(output_source, inverse), output_weight)
            output[row_start:row_start + 8, output_column:output_column + 512] = pl.cast(normalized, pl.BF16, mode='rint')
    return output

@pl.jit
def _dm334_rmsnorm_rmsnorm(hidden: pl.Tensor[[_dm334_rmsnorm_ROWS, _dm334_rmsnorm_HIDDEN], pl.BF16], gamma: pl.Tensor[[1, _dm334_rmsnorm_HIDDEN], pl.FP32], output: pl.Out[pl.Tensor[[_dm334_rmsnorm_ROWS, _dm334_rmsnorm_HIDDEN], pl.BF16]]):
    output = _dm334_rmsnorm_rmsnorm_inline(hidden, gamma, output)
    return output
_dm334_gated_norm_ROWS = 240
_dm334_gated_norm_DIM = 128
_dm334_gated_norm_ROW_TILE = 16

@pl.jit.inline
def _dm334_gated_norm_gated_rmsnorm_inline(core: pl.Tensor, gate: pl.Tensor, weight: pl.Tensor, output: pl.Tensor):
    with pl.spmd(_dm334_gated_norm_ROWS // _dm334_gated_norm_ROW_TILE, name_hint='qwen_gated_rmsnorm'):
        row = pl.tile.get_block_idx() * _dm334_gated_norm_ROW_TILE
        source = pl.cast(pl.slice(core, [_dm334_gated_norm_ROW_TILE, _dm334_gated_norm_DIM], [row, 0]), pl.FP32)
        gate_tile = pl.cast(pl.slice(gate, [_dm334_gated_norm_ROW_TILE, _dm334_gated_norm_DIM], [row, 0]), pl.FP32)
        weight_tile = pl.cast(pl.slice(weight, [1, _dm334_gated_norm_DIM], [0, 0]), pl.FP32)
        variance = pl.mul(pl.row_sum(pl.mul(source, source)), 1.0 / _dm334_gated_norm_DIM)
        inverse = pl.rsqrt(pl.add(variance, 1e-06), high_precision=True)
        normalized = pl.cast(pl.row_expand_mul(source, inverse), pl.BF16, mode='rint')
        weighted = pl.cast(pl.col_expand_mul(pl.cast(normalized, pl.FP32), weight_tile), pl.BF16, mode='rint')
        sigmoid = pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0))
        result = pl.mul(pl.cast(weighted, pl.FP32), sigmoid)
        output[row:row + _dm334_gated_norm_ROW_TILE, 0:_dm334_gated_norm_DIM] = pl.cast(result, pl.BF16, mode='rint')
    return output

@pl.jit
def _dm334_gated_norm_gated_rmsnorm(core: pl.Tensor[[_dm334_gated_norm_ROWS, _dm334_gated_norm_DIM], pl.BF16], gate: pl.Tensor[[_dm334_gated_norm_ROWS, _dm334_gated_norm_DIM], pl.BF16], weight: pl.Tensor[[1, _dm334_gated_norm_DIM], pl.BF16], output: pl.Out[pl.Tensor[[_dm334_gated_norm_ROWS, _dm334_gated_norm_DIM], pl.BF16]]):
    output = _dm334_gated_norm_gated_rmsnorm_inline(core, gate, weight, output)
    return output
_dm334_causal_conv_TOKENS = pl.dynamic('QWEN_CONV_TOKENS')
_dm334_causal_conv_CHANNELS = 10240
_dm334_causal_conv_KERNEL = 4
_dm334_causal_conv_CHANNEL_TILE = 128

@pl.jit.inline
def _dm334_causal_conv_causal_conv_silu_inline(projected: pl.Tensor, weight_tap_channel: pl.Tensor, state_tap_channel: pl.Tensor, output: pl.Tensor, token_count: pl.Scalar[pl.INDEX]):
    with pl.spmd(_dm334_causal_conv_CHANNELS // _dm334_causal_conv_CHANNEL_TILE, name_hint='qwen_causal_conv'):
        column = pl.tile.get_block_idx() * _dm334_causal_conv_CHANNEL_TILE
        previous_0 = pl.cast(pl.slice(state_tap_channel, [1, _dm334_causal_conv_CHANNEL_TILE], [0, column]), pl.FP32)
        previous_1 = pl.cast(pl.slice(state_tap_channel, [1, _dm334_causal_conv_CHANNEL_TILE], [1, column]), pl.FP32)
        previous_2 = pl.cast(pl.slice(state_tap_channel, [1, _dm334_causal_conv_CHANNEL_TILE], [2, column]), pl.FP32)
        previous_3 = pl.cast(pl.slice(state_tap_channel, [1, _dm334_causal_conv_CHANNEL_TILE], [3, column]), pl.FP32)
        tap_0 = pl.cast(pl.slice(weight_tap_channel, [1, _dm334_causal_conv_CHANNEL_TILE], [0, column]), pl.FP32)
        tap_1 = pl.cast(pl.slice(weight_tap_channel, [1, _dm334_causal_conv_CHANNEL_TILE], [1, column]), pl.FP32)
        tap_2 = pl.cast(pl.slice(weight_tap_channel, [1, _dm334_causal_conv_CHANNEL_TILE], [2, column]), pl.FP32)
        tap_3 = pl.cast(pl.slice(weight_tap_channel, [1, _dm334_causal_conv_CHANNEL_TILE], [3, column]), pl.FP32)
        for token in pl.range(token_count):
            current = pl.cast(pl.slice(projected, [1, _dm334_causal_conv_CHANNEL_TILE], [token, column]), pl.FP32)
            convolved = pl.add(pl.add(pl.mul(previous_1, tap_0), pl.mul(previous_2, tap_1)), pl.add(pl.mul(previous_3, tap_2), pl.mul(current, tap_3)))
            convolved = pl.cast(pl.cast(convolved, pl.BF16, mode='rint'), pl.FP32)
            activated = pl.div(convolved, pl.add(pl.exp(pl.neg(convolved)), 1.0), high_precision=True)
            output[token:token + 1, column:column + _dm334_causal_conv_CHANNEL_TILE] = pl.cast(activated, pl.BF16, mode='rint')
            previous_0 = previous_1
            previous_1 = previous_2
            previous_2 = previous_3
            previous_3 = current
        for token in pl.range(token_count, pl.tensor.dim(output, 0)):
            output[token:token + 1, column:column + _dm334_causal_conv_CHANNEL_TILE] = pl.full([1, _dm334_causal_conv_CHANNEL_TILE], dtype=pl.BF16, value=0.0)
        state_tap_channel[0:1, column:column + _dm334_causal_conv_CHANNEL_TILE] = pl.cast(previous_0, pl.BF16, mode='rint')
        state_tap_channel[1:2, column:column + _dm334_causal_conv_CHANNEL_TILE] = pl.cast(previous_1, pl.BF16, mode='rint')
        state_tap_channel[2:3, column:column + _dm334_causal_conv_CHANNEL_TILE] = pl.cast(previous_2, pl.BF16, mode='rint')
        state_tap_channel[3:4, column:column + _dm334_causal_conv_CHANNEL_TILE] = pl.cast(previous_3, pl.BF16, mode='rint')
    return (output, state_tap_channel)

@pl.jit
def _dm334_causal_conv_causal_conv_silu(projected: pl.Tensor[[_dm334_causal_conv_TOKENS, _dm334_causal_conv_CHANNELS], pl.BF16], weight_tap_channel: pl.Tensor[[_dm334_causal_conv_KERNEL, _dm334_causal_conv_CHANNELS], pl.BF16], state_tap_channel: pl.InOut[pl.Tensor[[_dm334_causal_conv_KERNEL, _dm334_causal_conv_CHANNELS], pl.BF16]], output: pl.Out[pl.Tensor[[_dm334_causal_conv_TOKENS, _dm334_causal_conv_CHANNELS], pl.BF16]]):
    (output, state_tap_channel) = _dm334_causal_conv_causal_conv_silu_inline(projected, weight_tap_channel, state_tap_channel, output, pl.tensor.dim(projected, 0))
    return (output, state_tap_channel)
_dm334_delta_prepare_TOKENS = 5
_dm334_delta_prepare_KEY_HEADS = 16
_dm334_delta_prepare_VALUE_HEADS = 48
_dm334_delta_prepare_DIM = 128
_dm334_delta_prepare_CHANNELS = (_dm334_delta_prepare_KEY_HEADS * 2 + _dm334_delta_prepare_VALUE_HEADS) * _dm334_delta_prepare_DIM
_dm334_delta_prepare_PAD_ROWS = 8
_dm334_delta_prepare_PAD_HEADS = 64

@pl.jit.inline
def _dm334_delta_prepare_expand_qkv_inline(conv_output: pl.Tensor, query: pl.Tensor, key: pl.Tensor, value: pl.Tensor):
    query_rows = pl.reshape(query, [_dm334_delta_prepare_TOKENS * _dm334_delta_prepare_VALUE_HEADS, _dm334_delta_prepare_DIM])
    key_rows = pl.reshape(key, [_dm334_delta_prepare_TOKENS * _dm334_delta_prepare_VALUE_HEADS, _dm334_delta_prepare_DIM])
    value_rows = pl.reshape(value, [_dm334_delta_prepare_TOKENS * _dm334_delta_prepare_VALUE_HEADS, _dm334_delta_prepare_DIM])
    with pl.spmd(_dm334_delta_prepare_TOKENS * _dm334_delta_prepare_VALUE_HEADS, name_hint='qwen_delta_expand_qkv'):
        row = pl.tile.get_block_idx()
        token = row // _dm334_delta_prepare_VALUE_HEADS
        head = row % _dm334_delta_prepare_VALUE_HEADS
        key_head = head // (_dm334_delta_prepare_VALUE_HEADS // _dm334_delta_prepare_KEY_HEADS)
        q = pl.slice(conv_output, [1, _dm334_delta_prepare_DIM], [token, key_head * _dm334_delta_prepare_DIM])
        k = pl.slice(conv_output, [1, _dm334_delta_prepare_DIM], [token, _dm334_delta_prepare_KEY_HEADS * _dm334_delta_prepare_DIM + key_head * _dm334_delta_prepare_DIM])
        v = pl.slice(conv_output, [1, _dm334_delta_prepare_DIM], [token, 2 * _dm334_delta_prepare_KEY_HEADS * _dm334_delta_prepare_DIM + head * _dm334_delta_prepare_DIM])
        query_rows[row:row + 1, 0:_dm334_delta_prepare_DIM] = q
        key_rows[row:row + 1, 0:_dm334_delta_prepare_DIM] = k
        value_rows[row:row + 1, 0:_dm334_delta_prepare_DIM] = v
    return (query, key, value)

@pl.jit.inline
def _dm334_delta_prepare_decay_beta_inline(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor):
    with pl.spmd(1, name_hint='qwen_delta_decay_beta'):
        row = pl.tile.get_block_idx() * _dm334_delta_prepare_PAD_ROWS
        a_tile = pl.cast(pl.slice(a, [_dm334_delta_prepare_PAD_ROWS, _dm334_delta_prepare_PAD_HEADS], [row, 0]), pl.FP32)
        b_tile = pl.cast(pl.slice(b, [_dm334_delta_prepare_PAD_ROWS, _dm334_delta_prepare_PAD_HEADS], [row, 0]), pl.FP32)
        a_parameter = pl.cast(pl.slice(a_log, [1, _dm334_delta_prepare_PAD_HEADS], [0, 0]), pl.FP32)
        dt_parameter = pl.cast(pl.slice(dt_bias, [1, _dm334_delta_prepare_PAD_HEADS], [0, 0]), pl.FP32)
        softplus_input = pl.col_expand_add(a_tile, dt_parameter)
        softplus = pl.log(pl.add(pl.exp(softplus_input), 1.0))
        decay = pl.mul(pl.col_expand_mul(softplus, pl.exp(a_parameter)), -1.0)
        beta_value = pl.div(pl.full([_dm334_delta_prepare_PAD_ROWS, _dm334_delta_prepare_PAD_HEADS], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(b_tile)), 1.0))
        log_decay[row:row + _dm334_delta_prepare_PAD_ROWS, 0:_dm334_delta_prepare_PAD_HEADS] = decay
        beta[row:row + _dm334_delta_prepare_PAD_ROWS, 0:_dm334_delta_prepare_PAD_HEADS] = pl.cast(beta_value, pl.BF16, mode='rint')
    return (log_decay, beta)

@pl.jit
def _dm334_delta_prepare_expand_qkv(conv_output: pl.Tensor[[_dm334_delta_prepare_TOKENS, _dm334_delta_prepare_CHANNELS], pl.BF16], query: pl.Out[pl.Tensor[[_dm334_delta_prepare_TOKENS, _dm334_delta_prepare_VALUE_HEADS, _dm334_delta_prepare_DIM], pl.BF16]], key: pl.Out[pl.Tensor[[_dm334_delta_prepare_TOKENS, _dm334_delta_prepare_VALUE_HEADS, _dm334_delta_prepare_DIM], pl.BF16]], value: pl.Out[pl.Tensor[[_dm334_delta_prepare_TOKENS, _dm334_delta_prepare_VALUE_HEADS, _dm334_delta_prepare_DIM], pl.BF16]]):
    (query, key, value) = _dm334_delta_prepare_expand_qkv_inline(conv_output, query, key, value)
    return (query, key, value)

@pl.jit
def _dm334_delta_prepare_decay_beta(a: pl.Tensor[[_dm334_delta_prepare_PAD_ROWS, _dm334_delta_prepare_PAD_HEADS], pl.BF16], b: pl.Tensor[[_dm334_delta_prepare_PAD_ROWS, _dm334_delta_prepare_PAD_HEADS], pl.BF16], a_log: pl.Tensor[[1, _dm334_delta_prepare_PAD_HEADS], pl.BF16], dt_bias: pl.Tensor[[1, _dm334_delta_prepare_PAD_HEADS], pl.BF16], log_decay: pl.Out[pl.Tensor[[_dm334_delta_prepare_PAD_ROWS, _dm334_delta_prepare_PAD_HEADS], pl.FP32]], beta: pl.Out[pl.Tensor[[_dm334_delta_prepare_PAD_ROWS, _dm334_delta_prepare_PAD_HEADS], pl.BF16]]):
    (log_decay, beta) = _dm334_delta_prepare_decay_beta_inline(a, b, a_log, dt_bias, log_decay, beta)
    return (log_decay, beta)
_dm334_delta_window16_ROWS = 8
_dm334_delta_window16_KEY_HEADS = 16
_dm334_delta_window16_HEADS = 48
_dm334_delta_window16_DIM = 128
_dm334_delta_window16_VALUE_TILE = 32
_dm334_delta_window16_NORM_TILE = 16
_dm334_delta_window16_COLUMN_REP = 16
_dm334_delta_window16_VALUE_WIDTH = _dm334_delta_window16_HEADS * _dm334_delta_window16_DIM

@pl.jit.inline
def _dm334_delta_window16_expand_qkv_window16_inline(conv_output: pl.Tensor, query: pl.Tensor, key: pl.Tensor, value: pl.Tensor):
    query_rows = pl.reshape(query, [_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, _dm334_delta_window16_DIM])
    key_rows = pl.reshape(key, [_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, _dm334_delta_window16_DIM])
    value_rows = pl.reshape(value, [_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, _dm334_delta_window16_DIM])
    with pl.spmd(_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, name_hint='qwen_window_delta_expand_qkv'):
        row = pl.tile.get_block_idx()
        token = row // _dm334_delta_window16_HEADS
        head = row % _dm334_delta_window16_HEADS
        key_head = head // (_dm334_delta_window16_HEADS // _dm334_delta_window16_KEY_HEADS)
        query_rows[row:row + 1, 0:_dm334_delta_window16_DIM] = pl.slice(conv_output, [1, _dm334_delta_window16_DIM], [token, key_head * _dm334_delta_window16_DIM])
        key_rows[row:row + 1, 0:_dm334_delta_window16_DIM] = pl.slice(conv_output, [1, _dm334_delta_window16_DIM], [token, _dm334_delta_window16_KEY_HEADS * _dm334_delta_window16_DIM + key_head * _dm334_delta_window16_DIM])
        value_rows[row:row + 1, 0:_dm334_delta_window16_DIM] = pl.slice(conv_output, [1, _dm334_delta_window16_DIM], [token, 2 * _dm334_delta_window16_KEY_HEADS * _dm334_delta_window16_DIM + head * _dm334_delta_window16_DIM])
    return (query, key, value)

@pl.jit.inline
def _dm334_delta_window16_recurrence_window16_inline(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor, state_value_key: pl.Tensor, output: pl.Tensor, token_count: pl.Scalar[pl.INDEX]):
    q_rows = pl.reshape(query, [_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, _dm334_delta_window16_DIM])
    k_rows = pl.reshape(key, [_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, _dm334_delta_window16_DIM])
    q_norm = pl.create_tensor([_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, _dm334_delta_window16_DIM], dtype=pl.FP32)
    k_norm = pl.create_tensor([_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, _dm334_delta_window16_DIM], dtype=pl.FP32)
    with pl.spmd(_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS // _dm334_delta_window16_NORM_TILE, name_hint='qwen_window_delta_norm') as norm_task:
        row = pl.tile.get_block_idx() * _dm334_delta_window16_NORM_TILE
        q_tile = pl.cast(pl.slice(q_rows, [_dm334_delta_window16_NORM_TILE, _dm334_delta_window16_DIM], [row, 0]), pl.FP32)
        k_tile = pl.cast(pl.slice(k_rows, [_dm334_delta_window16_NORM_TILE, _dm334_delta_window16_DIM], [row, 0]), pl.FP32)
        q_norm[row:row + _dm334_delta_window16_NORM_TILE, 0:_dm334_delta_window16_DIM] = pl.mul(pl.row_expand_mul(q_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(q_tile, q_tile)), 1e-06), high_precision=True)), _dm334_delta_window16_DIM ** (-0.5))
        k_norm[row:row + _dm334_delta_window16_NORM_TILE, 0:_dm334_delta_window16_DIM] = pl.row_expand_mul(k_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(k_tile, k_tile)), 1e-06), high_precision=True))
    state_rows = pl.reshape(state_value_key, [_dm334_delta_window16_HEADS * _dm334_delta_window16_DIM, _dm334_delta_window16_DIM])
    output_rows = pl.reshape(output, [_dm334_delta_window16_ROWS, _dm334_delta_window16_HEADS * _dm334_delta_window16_DIM])
    with pl.spmd(_dm334_delta_window16_HEADS * (_dm334_delta_window16_DIM // _dm334_delta_window16_VALUE_TILE), name_hint='qwen_window_delta_recurrence', deps=[norm_task]):
        task = pl.tile.get_block_idx()
        head = task // (_dm334_delta_window16_DIM // _dm334_delta_window16_VALUE_TILE)
        value_start = task % (_dm334_delta_window16_DIM // _dm334_delta_window16_VALUE_TILE) * _dm334_delta_window16_VALUE_TILE
        state_start = head * _dm334_delta_window16_DIM + value_start
        state = pl.slice(state_rows, [_dm334_delta_window16_VALUE_TILE, _dm334_delta_window16_DIM], [state_start, 0])
        for token in pl.range(token_count):
            row = token * _dm334_delta_window16_HEADS + head
            q = q_norm[row:row + 1, 0:_dm334_delta_window16_DIM]
            k = k_norm[row:row + 1, 0:_dm334_delta_window16_DIM]
            g = pl.read(log_decay, [token, head])
            decay = pl.exp(pl.mul(pl.full([1, _dm334_delta_window16_DIM], dtype=pl.FP32, value=1.0), g))
            b = pl.read(beta, [token, head])
            decayed = pl.col_expand_mul(state, decay)
            remembered = pl.row_sum(pl.col_expand_mul(decayed, k))
            v_row = pl.cast(pl.reshape(pl.slice(value, [1, 1, _dm334_delta_window16_VALUE_TILE], [token, head, value_start], drop_dims=[0]), [1, _dm334_delta_window16_VALUE_TILE]), pl.FP32)
            v_col = pl.mul(pl.row_sum(pl.transpose(pl.col_expand_mul(pl.full([_dm334_delta_window16_COLUMN_REP, _dm334_delta_window16_VALUE_TILE], dtype=pl.FP32, value=1.0), v_row), 0, 1)), 1.0 / _dm334_delta_window16_COLUMN_REP)
            delta = pl.mul(pl.sub(v_col, remembered), b)
            state = pl.add(decayed, pl.col_expand_mul(pl.row_expand_mul(pl.full([_dm334_delta_window16_VALUE_TILE, _dm334_delta_window16_DIM], dtype=pl.FP32, value=1.0), delta), k))
            output_rows[token:token + 1, head * _dm334_delta_window16_DIM + value_start:head * _dm334_delta_window16_DIM + value_start + _dm334_delta_window16_VALUE_TILE] = pl.set_validshape(pl.cast(pl.matmul(q, state, pl.FP32, b_trans=True), pl.BF16, mode='rint'), 1, _dm334_delta_window16_VALUE_TILE)
        for token in pl.range(token_count, _dm334_delta_window16_ROWS):
            output_rows[token:token + 1, head * _dm334_delta_window16_DIM + value_start:head * _dm334_delta_window16_DIM + value_start + _dm334_delta_window16_VALUE_TILE] = pl.full([1, _dm334_delta_window16_VALUE_TILE], dtype=pl.BF16, value=0.0)
        state_rows[state_start:state_start + _dm334_delta_window16_VALUE_TILE, 0:_dm334_delta_window16_DIM] = state
    return (output, state_value_key)

@pl.jit.inline
def _dm334_delta_window16_attention_tail_window16_inline(core: pl.Tensor, z_projection: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor, gated_output: pl.Tensor, attention_output: pl.Tensor):
    gate_rows = pl.create_tensor([_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, _dm334_delta_window16_DIM], dtype=pl.BF16)
    with pl.spmd(_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, name_hint='qwen_window_z_gate'):
        row = pl.tile.get_block_idx()
        token = row // _dm334_delta_window16_HEADS
        head = row % _dm334_delta_window16_HEADS
        gate_rows[row:row + 1, 0:_dm334_delta_window16_DIM] = pl.slice(z_projection, [1, _dm334_delta_window16_DIM], [token, head * _dm334_delta_window16_DIM])
    core_rows = pl.reshape(core, [_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, _dm334_delta_window16_DIM])
    with pl.spmd(_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS // _dm334_delta_window16_NORM_TILE, name_hint='qwen_window_gated_rmsnorm'):
        row = pl.tile.get_block_idx() * _dm334_delta_window16_NORM_TILE
        source = pl.cast(pl.slice(core_rows, [_dm334_delta_window16_NORM_TILE, _dm334_delta_window16_DIM], [row, 0]), pl.FP32)
        gate_tile = pl.cast(pl.slice(gate_rows, [_dm334_delta_window16_NORM_TILE, _dm334_delta_window16_DIM], [row, 0]), pl.FP32)
        weight_tile = pl.cast(pl.slice(norm_weight, [1, _dm334_gated_norm_DIM], [0, 0]), pl.FP32)
        variance = pl.mul(pl.row_sum(pl.mul(source, source)), 1.0 / _dm334_delta_window16_DIM)
        inverse = pl.rsqrt(pl.add(variance, 1e-06), high_precision=True)
        normalized = pl.cast(pl.row_expand_mul(source, inverse), pl.BF16, mode='rint')
        weighted = pl.cast(pl.col_expand_mul(pl.cast(normalized, pl.FP32), weight_tile), pl.BF16, mode='rint')
        sigmoid = pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0))
        gated_output[row:row + _dm334_delta_window16_NORM_TILE, 0:_dm334_delta_window16_DIM] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), sigmoid), pl.BF16, mode='rint')
    padded = pl.reshape(gated_output, [_dm334_delta_window16_ROWS, _dm334_delta_window16_VALUE_WIDTH])
    attention_output = _dm334_linear_hidden_projection_inline(padded, out_weight, attention_output)
    return (gated_output, attention_output)
_dm334_attention3_prepare_ROWS = 8
_dm334_attention3_prepare_HIDDEN = 5120
_dm334_attention3_prepare_HEAD_DIM = 256
_dm334_attention3_prepare_Q_HEADS = 24
_dm334_attention3_prepare_KV_HEADS = 4
_dm334_attention3_prepare_Q_PROJECTION = _dm334_attention3_prepare_Q_HEADS * _dm334_attention3_prepare_HEAD_DIM * 2
_dm334_attention3_prepare_KV_WIDTH = _dm334_attention3_prepare_KV_HEADS * _dm334_attention3_prepare_HEAD_DIM
_dm334_attention3_prepare_Q_WIDTH = _dm334_attention3_prepare_Q_HEADS * _dm334_attention3_prepare_HEAD_DIM
_dm334_attention3_prepare_ROTARY = 64

@pl.jit.inline
def _dm334_attention3_prepare_normalize_head_inline(head_values: pl.Tensor, gamma: pl.Tensor, result: pl.Tensor):
    source = pl.cast(head_values, pl.FP32)
    square = pl.mul(source, source)
    inverse = pl.rsqrt(pl.add(pl.mul(pl.row_sum(square), 1.0 / _dm334_attention3_prepare_HEAD_DIM), 1e-06), high_precision=True)
    normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), gamma)
    result[0:_dm334_attention3_prepare_ROWS, 0:_dm334_attention3_prepare_HEAD_DIM] = pl.cast(normalized, pl.BF16, mode='rint')
    return result

@pl.jit.inline
def _dm334_attention3_prepare_rotary_head_inline(normalized: pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HEAD_DIM], pl.BF16], cos: pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_ROTARY], pl.BF16], result: pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HEAD_DIM], pl.BF16]):
    first = pl.cast(pl.slice(normalized, [_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    second = pl.cast(pl.slice(normalized, [_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_ROTARY // 2], [0, _dm334_attention3_prepare_ROTARY // 2]), pl.FP32)
    cos_first = pl.cast(pl.slice(cos, [_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    cos_second = pl.cast(pl.slice(cos, [_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_ROTARY // 2], [0, _dm334_attention3_prepare_ROTARY // 2]), pl.FP32)
    sin_first = pl.cast(pl.slice(sin, [_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    sin_second = pl.cast(pl.slice(sin, [_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_ROTARY // 2], [0, _dm334_attention3_prepare_ROTARY // 2]), pl.FP32)
    rotated_first = pl.cast(pl.sub(pl.mul(first, cos_first), pl.mul(second, sin_first)), pl.BF16, mode='rint')
    rotated_second = pl.cast(pl.add(pl.mul(second, cos_second), pl.mul(first, sin_second)), pl.BF16, mode='rint')
    rest = pl.slice(normalized, [_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HEAD_DIM - _dm334_attention3_prepare_ROTARY], [0, _dm334_attention3_prepare_ROTARY])
    result[0:_dm334_attention3_prepare_ROWS, 0:_dm334_attention3_prepare_HEAD_DIM] = pl.concat(pl.concat(rotated_first, rotated_second), rest)
    return result

@pl.jit.inline
def _dm334_attention3_prepare_attention3_prepare_inline(hidden: pl.Tensor, q_weight: pl.Tensor, k_weight: pl.Tensor, v_weight: pl.Tensor, q_gamma: pl.Tensor, k_gamma: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, q_norm: pl.Tensor, k_norm: pl.Tensor, q_rot: pl.Tensor, k_rot: pl.Tensor, value: pl.Tensor, gate: pl.Tensor):
    q_projected = pl.create_tensor([_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_Q_PROJECTION], dtype=pl.BF16)
    k_projected = pl.create_tensor([_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_KV_WIDTH], dtype=pl.BF16)
    q_projected = _dm334_linear_hidden_projection_inline(hidden, q_weight, q_projected)
    k_projected = _dm334_linear_hidden_projection_inline(hidden, k_weight, k_projected)
    value = _dm334_linear_hidden_projection_inline(hidden, v_weight, value)
    with pl.spmd(_dm334_attention3_prepare_Q_HEADS, name_hint='qwen_attention_query_norm'):
        head = pl.tile.get_block_idx()
        projection_start = head * _dm334_attention3_prepare_HEAD_DIM * 2
        result_start = head * _dm334_attention3_prepare_HEAD_DIM
        raw = pl.slice(q_projected, [_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HEAD_DIM], [0, projection_start])
        normalized = pl.create_tensor([_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        normalized = _dm334_attention3_prepare_normalize_head_inline(raw, q_gamma, normalized)
        q_norm[0:_dm334_attention3_prepare_ROWS, result_start:result_start + _dm334_attention3_prepare_HEAD_DIM] = normalized
        gate[0:_dm334_attention3_prepare_ROWS, result_start:result_start + _dm334_attention3_prepare_HEAD_DIM] = pl.slice(q_projected, [_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HEAD_DIM], [0, projection_start + _dm334_attention3_prepare_HEAD_DIM])
        rotated = pl.create_tensor([_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        rotated = _dm334_attention3_prepare_rotary_head_inline(normalized, cos, sin, rotated)
        q_rot[0:_dm334_attention3_prepare_ROWS, result_start:result_start + _dm334_attention3_prepare_HEAD_DIM] = rotated
    with pl.spmd(_dm334_attention3_prepare_KV_HEADS, name_hint='qwen_attention_key_norm'):
        head = pl.tile.get_block_idx()
        start = head * _dm334_attention3_prepare_HEAD_DIM
        raw = pl.slice(k_projected, [_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HEAD_DIM], [0, start])
        normalized = pl.create_tensor([_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        normalized = _dm334_attention3_prepare_normalize_head_inline(raw, k_gamma, normalized)
        k_norm[0:_dm334_attention3_prepare_ROWS, start:start + _dm334_attention3_prepare_HEAD_DIM] = normalized
        rotated = pl.create_tensor([_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        rotated = _dm334_attention3_prepare_rotary_head_inline(normalized, cos, sin, rotated)
        k_rot[0:_dm334_attention3_prepare_ROWS, start:start + _dm334_attention3_prepare_HEAD_DIM] = rotated
    return (q_norm, k_norm, q_rot, k_rot, value, gate)

@pl.jit
def _dm334_attention3_prepare_attention3_prepare(hidden: pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_HIDDEN], pl.BF16], q_weight: pl.Tensor[[_dm334_attention3_prepare_Q_PROJECTION, _dm334_attention3_prepare_HIDDEN], pl.BF16], k_weight: pl.Tensor[[_dm334_attention3_prepare_KV_WIDTH, _dm334_attention3_prepare_HIDDEN], pl.BF16], v_weight: pl.Tensor[[_dm334_attention3_prepare_KV_WIDTH, _dm334_attention3_prepare_HIDDEN], pl.BF16], q_gamma: pl.Tensor[[1, _dm334_attention3_prepare_HEAD_DIM], pl.FP32], k_gamma: pl.Tensor[[1, _dm334_attention3_prepare_HEAD_DIM], pl.FP32], cos: pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_ROTARY], pl.BF16], q_norm: pl.Out[pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_Q_WIDTH], pl.BF16]], k_norm: pl.Out[pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_KV_WIDTH], pl.BF16]], q_rot: pl.Out[pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_Q_WIDTH], pl.BF16]], k_rot: pl.Out[pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_KV_WIDTH], pl.BF16]], value: pl.Out[pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_KV_WIDTH], pl.BF16]], gate: pl.Out[pl.Tensor[[_dm334_attention3_prepare_ROWS, _dm334_attention3_prepare_Q_WIDTH], pl.BF16]]):
    return _dm334_attention3_prepare_attention3_prepare_inline(hidden, q_weight, k_weight, v_weight, q_gamma, k_gamma, cos, sin, q_norm, k_norm, q_rot, k_rot, value, gate)
_dm334_attention512_core_ROWS = 8
_dm334_attention512_core_CONTEXT = 2048
_dm334_attention512_core_KEY_BLOCK = 16
_dm334_attention512_core_HEAD_DIM = 256
_dm334_attention512_core_Q_HEADS = 24
_dm334_attention512_core_KV_HEADS = 4
_dm334_attention512_core_Q_WIDTH = _dm334_attention512_core_Q_HEADS * _dm334_attention512_core_HEAD_DIM

@pl.jit.inline
def _dm334_attention512_core_attention512_core_inline(query: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16], key_cache: pl.Tensor[[_dm334_attention512_core_KV_HEADS, _dm334_attention512_core_CONTEXT, _dm334_attention512_core_HEAD_DIM], pl.BF16], value_cache: pl.Tensor[[_dm334_attention512_core_KV_HEADS, _dm334_attention512_core_CONTEXT, _dm334_attention512_core_HEAD_DIM], pl.BF16], gate: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16], mask: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_CONTEXT], pl.FP32], output_weight: pl.Tensor[[5120, _dm334_attention512_core_Q_WIDTH], pl.BF16], logits_buffer: pl.Tensor[[_dm334_attention512_core_Q_HEADS * _dm334_attention512_core_ROWS, _dm334_attention512_core_CONTEXT], pl.FP32], probability_buffer: pl.Tensor[[_dm334_attention512_core_Q_HEADS * 16, _dm334_attention512_core_CONTEXT], pl.BF16], weighted_output: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16], gated_output: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16], projected_output: pl.Tensor[[_dm334_attention512_core_ROWS, 5120], pl.BF16]):
    with pl.spmd(_dm334_attention512_core_Q_HEADS, name_hint='qwen_attention512_scores'):
        score_head = pl.tile.get_block_idx()
        score_kv_head = score_head // (_dm334_attention512_core_Q_HEADS // _dm334_attention512_core_KV_HEADS)
        score_start = score_head * _dm334_attention512_core_HEAD_DIM
        score_q = pl.slice(query, [_dm334_attention512_core_ROWS, _dm334_attention512_core_HEAD_DIM], [0, score_start])
        for score_block in pl.range(_dm334_attention512_core_CONTEXT // _dm334_attention512_core_KEY_BLOCK):
            score_key_start = score_block * _dm334_attention512_core_KEY_BLOCK
            score_k = pl.reshape(pl.slice(key_cache, [1, _dm334_attention512_core_KEY_BLOCK, _dm334_attention512_core_HEAD_DIM], [score_kv_head, score_key_start, 0]), [_dm334_attention512_core_KEY_BLOCK, _dm334_attention512_core_HEAD_DIM])
            score_raw = pl.matmul(score_q, score_k, pl.FP32, b_trans=True)
            score_rounded = pl.cast(score_raw, pl.BF16, mode='rint')
            score_scaled = pl.cast(pl.mul(pl.cast(score_rounded, pl.FP32), _dm334_attention512_core_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
            score_scaled = pl.slice(score_scaled, [_dm334_attention512_core_ROWS, _dm334_attention512_core_KEY_BLOCK], [0, 0])
            score_mask = pl.slice(mask, [_dm334_attention512_core_ROWS, _dm334_attention512_core_KEY_BLOCK], [0, score_key_start])
            logits_buffer[score_head * _dm334_attention512_core_ROWS:(score_head + 1) * _dm334_attention512_core_ROWS, score_key_start:score_key_start + _dm334_attention512_core_KEY_BLOCK] = pl.add(pl.cast(score_scaled, pl.FP32), score_mask)
    with pl.spmd(_dm334_attention512_core_Q_HEADS, name_hint='qwen_attention512_softmax'):
        softmax_head = pl.tile.get_block_idx()
        softmax_row = softmax_head * 16
        softmax_logits = pl.slice(logits_buffer, [_dm334_attention512_core_ROWS, _dm334_attention512_core_CONTEXT], [softmax_head * _dm334_attention512_core_ROWS, 0])
        softmax_exp = pl.exp(pl.row_expand_sub(softmax_logits, pl.row_max(softmax_logits)))
        probability_buffer[softmax_row:softmax_row + _dm334_attention512_core_ROWS, 0:_dm334_attention512_core_CONTEXT] = pl.cast(pl.row_expand_div(softmax_exp, pl.row_sum(softmax_exp)), pl.BF16, mode='rint')
    with pl.spmd(_dm334_attention512_core_Q_HEADS, name_hint='qwen_attention512_weighted'):
        weighted_head = pl.tile.get_block_idx()
        weighted_kv_head = weighted_head // (_dm334_attention512_core_Q_HEADS // _dm334_attention512_core_KV_HEADS)
        weighted_start = weighted_head * _dm334_attention512_core_HEAD_DIM
        weighted_sum = pl.full([16, _dm334_attention512_core_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for weighted_block in pl.range(_dm334_attention512_core_CONTEXT // _dm334_attention512_core_KEY_BLOCK):
            weighted_key_start = weighted_block * _dm334_attention512_core_KEY_BLOCK
            weighted_probs = pl.slice(probability_buffer, [16, _dm334_attention512_core_KEY_BLOCK], [weighted_head * 16, weighted_key_start])
            weighted_values = pl.reshape(pl.slice(value_cache, [1, _dm334_attention512_core_KEY_BLOCK, _dm334_attention512_core_HEAD_DIM], [weighted_kv_head, weighted_key_start, 0]), [_dm334_attention512_core_KEY_BLOCK, _dm334_attention512_core_HEAD_DIM])
            weighted_full = pl.matmul(weighted_probs, weighted_values, pl.FP32)
            weighted_sum = pl.add(weighted_sum, weighted_full)
        weighted = pl.cast(weighted_sum, pl.BF16, mode='rint')
        weighted_valid = pl.slice(weighted, [_dm334_attention512_core_ROWS, _dm334_attention512_core_HEAD_DIM], [0, 0])
        weighted_output[0:_dm334_attention512_core_ROWS, weighted_start:weighted_start + _dm334_attention512_core_HEAD_DIM] = weighted_valid
        gate_tile = pl.cast(pl.slice(gate, [_dm334_attention512_core_ROWS, _dm334_attention512_core_HEAD_DIM], [0, weighted_start]), pl.FP32)
        sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        gated_output[0:_dm334_attention512_core_ROWS, weighted_start:weighted_start + _dm334_attention512_core_HEAD_DIM] = pl.cast(pl.mul(pl.cast(weighted_valid, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint')
    projected_output = _dm334_linear_hidden_projection_inline(gated_output, output_weight, projected_output)
    return (logits_buffer, probability_buffer, weighted_output, gated_output, projected_output)

@pl.jit
def _dm334_attention512_core_attention512_core(query: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16], key_cache: pl.Tensor[[_dm334_attention512_core_KV_HEADS, _dm334_attention512_core_CONTEXT, _dm334_attention512_core_HEAD_DIM], pl.BF16], value_cache: pl.Tensor[[_dm334_attention512_core_KV_HEADS, _dm334_attention512_core_CONTEXT, _dm334_attention512_core_HEAD_DIM], pl.BF16], gate: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16], mask: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_CONTEXT], pl.FP32], output_weight: pl.Tensor[[5120, _dm334_attention512_core_Q_WIDTH], pl.BF16], logits_buffer: pl.Out[pl.Tensor[[_dm334_attention512_core_Q_HEADS * _dm334_attention512_core_ROWS, _dm334_attention512_core_CONTEXT], pl.FP32]], probability_buffer: pl.Out[pl.Tensor[[_dm334_attention512_core_Q_HEADS * 16, _dm334_attention512_core_CONTEXT], pl.BF16]], weighted_output: pl.Out[pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16]], gated_output: pl.Out[pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16]], projected_output: pl.Out[pl.Tensor[[_dm334_attention512_core_ROWS, 5120], pl.BF16]]):
    return _dm334_attention512_core_attention512_core_inline(query, key_cache, value_cache, gate, mask, output_weight, logits_buffer, probability_buffer, weighted_output, gated_output, projected_output)
_dm334_residual_ROWS = 8
_dm334_residual_HIDDEN = 5120
_dm334_residual_TILE = 5120

@pl.jit.inline
def _dm334_residual_residual_add_inline(left: pl.Tensor, right: pl.Tensor, output: pl.Tensor):
    with pl.spmd(_dm334_residual_ROWS * (_dm334_residual_HIDDEN // _dm334_residual_TILE), name_hint='qwen_residual_add'):
        task = pl.tile.get_block_idx()
        row = task // (_dm334_residual_HIDDEN // _dm334_residual_TILE)
        column = task % (_dm334_residual_HIDDEN // _dm334_residual_TILE) * _dm334_residual_TILE
        a = pl.cast(pl.slice(left, [1, _dm334_residual_TILE], [row, column]), pl.FP32)
        b = pl.cast(pl.slice(right, [1, _dm334_residual_TILE], [row, column]), pl.FP32)
        output[row:row + 1, column:column + _dm334_residual_TILE] = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
    return output
_dm334_swiglu_ROWS = 8
_dm334_swiglu_INTERMEDIATE = 17408
_dm334_swiglu_CHANNEL_TILE = 128

@pl.jit.inline
def _dm334_swiglu_swiglu_inline(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    with pl.spmd(_dm334_swiglu_INTERMEDIATE // _dm334_swiglu_CHANNEL_TILE, name_hint='qwen_swiglu'):
        column = pl.tile.get_block_idx() * _dm334_swiglu_CHANNEL_TILE
        gate_tile = pl.cast(pl.slice(gate, [_dm334_swiglu_ROWS, _dm334_swiglu_CHANNEL_TILE], [0, column]), pl.FP32)
        up_tile = pl.cast(pl.slice(up, [_dm334_swiglu_ROWS, _dm334_swiglu_CHANNEL_TILE], [0, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        multiplied = pl.mul(pl.cast(activated, pl.FP32), up_tile)
        output[0:_dm334_swiglu_ROWS, column:column + _dm334_swiglu_CHANNEL_TILE] = pl.cast(multiplied, pl.BF16, mode='rint')
    return output

@pl.jit
def _dm334_swiglu_swiglu(gate: pl.Tensor[[_dm334_swiglu_ROWS, _dm334_swiglu_INTERMEDIATE], pl.BF16], up: pl.Tensor[[_dm334_swiglu_ROWS, _dm334_swiglu_INTERMEDIATE], pl.BF16], output: pl.Out[pl.Tensor[[_dm334_swiglu_ROWS, _dm334_swiglu_INTERMEDIATE], pl.BF16]]):
    output = _dm334_swiglu_swiglu_inline(gate, up, output)
    return output
_dm334_layer0_stem_ROWS = 8
_dm334_layer0_stem_HIDDEN = 5120
_dm334_layer0_stem_QKV = 10240
_dm334_layer0_stem_Z = 6144
_dm334_layer0_stem_GATE_PAD = 128

@pl.jit.inline
def _dm334_layer0_stem_layer0_stem_inline(hidden: pl.Tensor, gamma: pl.Tensor, qkv_weight: pl.Tensor, z_weight: pl.Tensor, a_weight: pl.Tensor, b_weight: pl.Tensor, norm_output: pl.Tensor, qkv_output: pl.Tensor, z_output: pl.Tensor, a_output: pl.Tensor, b_output: pl.Tensor):
    norm_output = _dm334_rmsnorm_rmsnorm_inline(hidden, gamma, norm_output)
    qkv_output = _dm334_linear_hidden_projection_inline(norm_output, qkv_weight, qkv_output)
    z_output = _dm334_linear_hidden_projection_inline(norm_output, z_weight, z_output)
    a_output = _dm334_linear_hidden_projection_inline(norm_output, a_weight, a_output)
    b_output = _dm334_linear_hidden_projection_inline(norm_output, b_weight, b_output)
    return (norm_output, qkv_output, z_output, a_output, b_output)

@pl.jit
def _dm334_layer0_stem_layer0_stem(hidden: pl.Tensor[[_dm334_layer0_stem_ROWS, _dm334_layer0_stem_HIDDEN], pl.BF16], gamma: pl.Tensor[[1, _dm334_layer0_stem_HIDDEN], pl.FP32], qkv_weight: pl.Tensor[[_dm334_layer0_stem_QKV, _dm334_layer0_stem_HIDDEN], pl.BF16], z_weight: pl.Tensor[[_dm334_layer0_stem_Z, _dm334_layer0_stem_HIDDEN], pl.BF16], a_weight: pl.Tensor[[_dm334_layer0_stem_GATE_PAD, _dm334_layer0_stem_HIDDEN], pl.BF16], b_weight: pl.Tensor[[_dm334_layer0_stem_GATE_PAD, _dm334_layer0_stem_HIDDEN], pl.BF16], norm_output: pl.Out[pl.Tensor[[_dm334_layer0_stem_ROWS, _dm334_layer0_stem_HIDDEN], pl.BF16]], qkv_output: pl.Out[pl.Tensor[[_dm334_layer0_stem_ROWS, _dm334_layer0_stem_QKV], pl.BF16]], z_output: pl.Out[pl.Tensor[[_dm334_layer0_stem_ROWS, _dm334_layer0_stem_Z], pl.BF16]], a_output: pl.Out[pl.Tensor[[_dm334_layer0_stem_ROWS, _dm334_layer0_stem_GATE_PAD], pl.BF16]], b_output: pl.Out[pl.Tensor[[_dm334_layer0_stem_ROWS, _dm334_layer0_stem_GATE_PAD], pl.BF16]]):
    return _dm334_layer0_stem_layer0_stem_inline(hidden, gamma, qkv_weight, z_weight, a_weight, b_weight, norm_output, qkv_output, z_output, a_output, b_output)
_dm334_ROWS = 8
_dm334_BATCH = 8
_dm334_CONTEXT = 2048
_dm334_BLOCK_SIZE = 16
_dm334_POOL_BLOCKS = 1032
_dm334_HIDDEN = 5120
_dm334_INTERMEDIATE = 17408
_dm334_QKV = 10240
_dm334_VALUE_WIDTH = 6144
_dm334_DELTA_HEADS = 48
_dm334_DELTA_DIM = 128
_dm334_FULL_HEAD_DIM = 256
_dm334_FULL_Q_WIDTH = 6144
_dm334_FULL_Q_PROJECTION = 12288
_dm334_FULL_KV_WIDTH = 1024
_dm334_ROTARY = 64
_dm334_VOCAB = 248320

@pl.jit.inline
def _dm334_full_step_inline(input_ids: pl.Tensor[[_dm334_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_QKV, _dm334_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_FULL_Q_PROJECTION, _dm334_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_HIDDEN, _dm334_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_dm334_ROWS, _dm334_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_ROWS, _dm334_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_ROWS, _dm334_CONTEXT], pl.FP32], token_count: pl.Scalar[pl.INT32], cache_offset: pl.Scalar[pl.INT32], conv_states: pl.Tensor[[48, 4, _dm334_QKV], pl.BF16], recurrent_states: pl.Tensor[[48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32], key_pool: pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16], value_pool: pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16], block_table: pl.Tensor[[_dm334_CONTEXT // _dm334_BLOCK_SIZE], pl.INT32], traces: pl.Tensor[[66, _dm334_ROWS, _dm334_HIDDEN], pl.BF16], logits: pl.Tensor[[_dm334_ROWS, _dm334_VOCAB], pl.BF16]):
    running = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
    with pl.spmd(_dm334_ROWS * (_dm334_HIDDEN // 128), name_hint='qwen_full_token_embedding'):
        embed_task = pl.tile.get_block_idx()
        embed_token = embed_task // (_dm334_HIDDEN // 128)
        embed_column = embed_task % (_dm334_HIDDEN // 128) * 128
        if embed_token < token_count:
            embed_token_id = pl.read(input_ids, [embed_token])
            embed_tile = pl.slice(embedding, [1, 128], [embed_token_id, embed_column])
            running[embed_token:embed_token + 1, embed_column:embed_column + 128] = embed_tile
            traces[0:1, embed_token:embed_token + 1, embed_column:embed_column + 128] = pl.reshape(embed_tile, [1, 1, 128])
        else:
            zero_tile = pl.full([1, 128], dtype=pl.BF16, value=0.0)
            running[embed_token:embed_token + 1, embed_column:embed_column + 128] = zero_tile
            traces[0:1, embed_token:embed_token + 1, embed_column:embed_column + 128] = pl.reshape(zero_tile, [1, 1, 128])
    for layer in pl.range(64):
        input_gamma_layer: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, _dm334_HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, _dm334_HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[_dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16] = pl.slice(mlp_gate_weight, [_dm334_INTERMEDIATE, _dm334_HIDDEN], [layer * _dm334_INTERMEDIATE, 0])
        mlp_up_layer: pl.Tensor[[_dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16] = pl.slice(mlp_up_weight, [_dm334_INTERMEDIATE, _dm334_HIDDEN], [layer * _dm334_INTERMEDIATE, 0])
        mlp_down_layer: pl.Tensor[[_dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16] = pl.slice(mlp_down_weight, [_dm334_HIDDEN, _dm334_INTERMEDIATE], [layer * _dm334_HIDDEN, 0])
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[_dm334_QKV, _dm334_HIDDEN], pl.BF16] = pl.slice(delta_qkv_weight, [_dm334_QKV, _dm334_HIDDEN], [linear * _dm334_QKV, 0])
            z_weight_layer: pl.Tensor[[_dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16] = pl.slice(delta_z_weight, [_dm334_VALUE_WIDTH, _dm334_HIDDEN], [linear * _dm334_VALUE_WIDTH, 0])
            a_weight_layer: pl.Tensor[[128, _dm334_HIDDEN], pl.BF16] = pl.slice(delta_a_weight, [128, _dm334_HIDDEN], [linear * 128, 0])
            b_weight_layer: pl.Tensor[[128, _dm334_HIDDEN], pl.BF16] = pl.slice(delta_b_weight, [128, _dm334_HIDDEN], [linear * 128, 0])
            conv_weight_layer: pl.Tensor[[4, _dm334_QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, _dm334_QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, _dm334_DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, _dm334_DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[_dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16] = pl.slice(delta_out_weight, [_dm334_HIDDEN, _dm334_VALUE_WIDTH], [linear * _dm334_HIDDEN, 0])
            conv_source: pl.Tensor[[4, _dm334_QKV], pl.BF16] = pl.reshape(pl.slice(conv_states, [1, 4, _dm334_QKV], [linear, 0, 0]), [4, _dm334_QKV])
            recurrent_source: pl.Tensor[[_dm334_DELTA_HEADS * _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32] = pl.reshape(pl.slice(recurrent_states, [1, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], [linear, 0, 0, 0]), [_dm334_DELTA_HEADS * _dm334_DELTA_DIM, _dm334_DELTA_DIM])
            conv = pl.create_tensor([4, _dm334_QKV], dtype=pl.BF16)
            recurrent = pl.create_tensor([_dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], dtype=pl.FP32)
            recurrent_rows = pl.reshape(recurrent, [_dm334_DELTA_HEADS * _dm334_DELTA_DIM, _dm334_DELTA_DIM])
            with pl.spmd(4 * (_dm334_QKV // 128), name_hint='qwen_full_load_conv'):
                task = pl.tile.get_block_idx()
                tap = task // (_dm334_QKV // 128)
                col = task % (_dm334_QKV // 128) * 128
                conv[tap:tap + 1, col:col + 128] = pl.slice(conv_source, [1, 128], [tap, col])
            with pl.spmd(_dm334_DELTA_HEADS * _dm334_DELTA_DIM // 16, name_hint='qwen_full_load_recurrent'):
                row = pl.tile.get_block_idx() * 16
                recurrent_rows[row:row + 16, 0:_dm334_DELTA_DIM] = pl.slice(recurrent_source, [16, _dm334_DELTA_DIM], [row, 0])
            norm_delta = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            qkv = pl.create_tensor([_dm334_ROWS, _dm334_QKV], dtype=pl.BF16)
            z = pl.create_tensor([_dm334_ROWS, _dm334_VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([_dm334_ROWS, 128], dtype=pl.BF16)
            b = pl.create_tensor([_dm334_ROWS, 128], dtype=pl.BF16)
            (norm_delta, qkv, z, a, b) = _dm334_layer0_stem_layer0_stem_inline(running, input_gamma_layer, qkv_weight_layer, z_weight_layer, a_weight_layer, b_weight_layer, norm_delta, qkv, z, a, b)
            conv_output = pl.create_tensor([_dm334_ROWS, _dm334_QKV], dtype=pl.BF16)
            query = pl.create_tensor([_dm334_ROWS, _dm334_DELTA_HEADS, _dm334_DELTA_DIM], dtype=pl.BF16)
            key = pl.create_tensor([_dm334_ROWS, _dm334_DELTA_HEADS, _dm334_DELTA_DIM], dtype=pl.BF16)
            value = pl.create_tensor([_dm334_ROWS, _dm334_DELTA_HEADS, _dm334_DELTA_DIM], dtype=pl.BF16)
            log_decay = pl.create_tensor([_dm334_ROWS, _dm334_DELTA_HEADS], dtype=pl.FP32)
            beta = pl.create_tensor([_dm334_ROWS, _dm334_DELTA_HEADS], dtype=pl.FP32)
            core = pl.create_tensor([_dm334_ROWS, _dm334_DELTA_HEADS, _dm334_DELTA_DIM], dtype=pl.BF16)
            (conv_output, conv) = _dm334_causal_conv_causal_conv_silu_inline(qkv, conv_weight_layer, conv, conv_output, token_count)
            (query, key, value) = _dm334_delta_window16_expand_qkv_window16_inline(conv_output, query, key, value)
            gate_values = pl.create_tensor([_dm334_ROWS, 64], dtype=pl.FP32)
            beta_bf16 = pl.create_tensor([_dm334_ROWS, 64], dtype=pl.BF16)
            (gate_values, beta_bf16) = _dm334_delta_prepare_decay_beta_inline(a, b, a_log_layer, dt_bias_layer, gate_values, beta_bf16)
            with pl.spmd(_dm334_ROWS, name_hint='qwen_full_delta_gate_unpad'):
                token = pl.tile.get_block_idx()
                log_decay[token:token + 1, 0:_dm334_DELTA_HEADS] = pl.slice(gate_values, [1, _dm334_DELTA_HEADS], [token, 0])
                beta[token:token + 1, 0:_dm334_DELTA_HEADS] = pl.cast(pl.slice(beta_bf16, [1, _dm334_DELTA_HEADS], [token, 0]), pl.FP32)
            (core, recurrent) = _dm334_delta_window16_recurrence_window16_inline(query, key, value, log_decay, beta, recurrent, core, token_count)
            gated = pl.create_tensor([_dm334_ROWS * _dm334_DELTA_HEADS, _dm334_DELTA_DIM], dtype=pl.BF16)
            attention_delta = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            (gated, attention_delta) = _dm334_delta_window16_attention_tail_window16_inline(core, z, delta_norm_layer, delta_out_layer, gated, attention_delta)
            residual_delta = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            residual_delta = _dm334_residual_residual_add_inline(running, attention_delta, residual_delta)
            post_norm_delta = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            post_norm_delta = _dm334_rmsnorm_rmsnorm_inline(residual_delta, post_gamma_layer, post_norm_delta)
            mlp_gate_delta = pl.create_tensor([_dm334_ROWS, _dm334_INTERMEDIATE], dtype=pl.BF16)
            mlp_up_delta = pl.create_tensor([_dm334_ROWS, _dm334_INTERMEDIATE], dtype=pl.BF16)
            mlp_gate_delta = _dm334_linear_hidden_projection_inline(post_norm_delta, mlp_gate_layer, mlp_gate_delta)
            mlp_up_delta = _dm334_linear_hidden_projection_inline(post_norm_delta, mlp_up_layer, mlp_up_delta)
            mlp_middle_delta = pl.create_tensor([_dm334_ROWS, _dm334_INTERMEDIATE], dtype=pl.BF16)
            mlp_middle_delta = _dm334_swiglu_swiglu_inline(mlp_gate_delta, mlp_up_delta, mlp_middle_delta)
            mlp_output_delta = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            mlp_output_delta = _dm334_linear_hidden_projection_inline(mlp_middle_delta, mlp_down_layer, mlp_output_delta)
            following_delta = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            following_delta = _dm334_residual_residual_add_inline(residual_delta, mlp_output_delta, following_delta)
            conv_dest: pl.Tensor[[4, _dm334_QKV], pl.BF16] = pl.reshape(pl.slice(conv_states, [1, 4, _dm334_QKV], [linear, 0, 0]), [4, _dm334_QKV])
            recurrent_dest: pl.Tensor[[_dm334_DELTA_HEADS * _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32] = pl.reshape(pl.slice(recurrent_states, [1, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], [linear, 0, 0, 0]), [_dm334_DELTA_HEADS * _dm334_DELTA_DIM, _dm334_DELTA_DIM])
            recurrent_rows = pl.reshape(recurrent, [_dm334_DELTA_HEADS * _dm334_DELTA_DIM, _dm334_DELTA_DIM])
            with pl.spmd(4 * (_dm334_QKV // 128), name_hint='qwen_full_store_conv'):
                task = pl.tile.get_block_idx()
                tap = task // (_dm334_QKV // 128)
                col = task % (_dm334_QKV // 128) * 128
                conv_dest[tap:tap + 1, col:col + 128] = pl.slice(conv, [1, 128], [tap, col])
            with pl.spmd(_dm334_DELTA_HEADS * _dm334_DELTA_DIM // 16, name_hint='qwen_full_store_recurrent'):
                row = pl.tile.get_block_idx() * 16
                recurrent_dest[row:row + 16, 0:_dm334_DELTA_DIM] = pl.slice(recurrent_rows, [16, _dm334_DELTA_DIM], [row, 0])
            with pl.spmd(_dm334_ROWS, name_hint='qwen_full_store_delta_hidden'):
                row = pl.tile.get_block_idx()
                next_hidden = pl.slice(following_delta, [1, _dm334_HIDDEN], [row, 0])
                running[row:row + 1, 0:_dm334_HIDDEN] = next_hidden
                traces[layer + 1:layer + 2, row:row + 1, 0:_dm334_HIDDEN] = pl.reshape(next_hidden, [1, 1, _dm334_HIDDEN])
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[_dm334_FULL_Q_PROJECTION, _dm334_HIDDEN], pl.BF16] = pl.slice(full_q_weight, [_dm334_FULL_Q_PROJECTION, _dm334_HIDDEN], [full * _dm334_FULL_Q_PROJECTION, 0])
            k_weight_full: pl.Tensor[[_dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16] = pl.slice(full_k_weight, [_dm334_FULL_KV_WIDTH, _dm334_HIDDEN], [full * _dm334_FULL_KV_WIDTH, 0])
            v_weight_full: pl.Tensor[[_dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16] = pl.slice(full_v_weight, [_dm334_FULL_KV_WIDTH, _dm334_HIDDEN], [full * _dm334_FULL_KV_WIDTH, 0])
            q_gamma_full: pl.Tensor[[1, _dm334_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, _dm334_FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, _dm334_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, _dm334_FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[_dm334_HIDDEN, _dm334_FULL_Q_WIDTH], pl.BF16] = pl.slice(full_out_weight, [_dm334_HIDDEN, _dm334_FULL_Q_WIDTH], [full * _dm334_HIDDEN, 0])
            norm_full = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            norm_full = _dm334_rmsnorm_rmsnorm_inline(running, input_gamma_layer, norm_full)
            q_norm_full = pl.create_tensor([_dm334_ROWS, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm_full = pl.create_tensor([_dm334_ROWS, _dm334_FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([_dm334_ROWS, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([_dm334_ROWS, _dm334_FULL_KV_WIDTH], dtype=pl.BF16)
            value_full = pl.create_tensor([_dm334_ROWS, _dm334_FULL_KV_WIDTH], dtype=pl.BF16)
            gate_full = pl.create_tensor([_dm334_ROWS, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            (q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full) = _dm334_attention3_prepare_attention3_prepare_inline(norm_full, q_weight_full, k_weight_full, v_weight_full, q_gamma_full, k_gamma_full, cos, sin, q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full)
            key_local = pl.create_tensor([4, _dm334_CONTEXT, _dm334_FULL_HEAD_DIM], dtype=pl.BF16)
            value_local = pl.create_tensor([4, _dm334_CONTEXT, _dm334_FULL_HEAD_DIM], dtype=pl.BF16)
            with pl.spmd(4 * (_dm334_CONTEXT // _dm334_BLOCK_SIZE), name_hint='qwen_full_load_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // (_dm334_CONTEXT // _dm334_BLOCK_SIZE)
                logical_block = task % (_dm334_CONTEXT // _dm334_BLOCK_SIZE)
                physical = pl.read(block_table, [logical_block])
                row = logical_block * _dm334_BLOCK_SIZE
                key_local[head:head + 1, row:row + _dm334_BLOCK_SIZE, 0:_dm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(key_pool, [1, 1, 1, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM])
                value_local[head:head + 1, row:row + _dm334_BLOCK_SIZE, 0:_dm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_pool, [1, 1, 1, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM])
            with pl.spmd(4, name_hint='qwen_full_update_kv'):
                head = pl.tile.get_block_idx()
                start = head * _dm334_FULL_HEAD_DIM
                for token in pl.range(token_count):
                    key_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:_dm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(k_rot, [1, _dm334_FULL_HEAD_DIM], [token, start]), [1, 1, _dm334_FULL_HEAD_DIM])
                    value_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:_dm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_full, [1, _dm334_FULL_HEAD_DIM], [token, start]), [1, 1, _dm334_FULL_HEAD_DIM])
            scores = pl.create_tensor([24 * _dm334_ROWS, _dm334_CONTEXT], dtype=pl.FP32)
            probabilities = pl.create_tensor([24 * 16, _dm334_CONTEXT], dtype=pl.BF16)
            weighted = pl.create_tensor([_dm334_ROWS, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            full_gated = pl.create_tensor([_dm334_ROWS, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            attention_full = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            (scores, probabilities, weighted, full_gated, attention_full) = _dm334_attention512_core_attention512_core_inline(q_rot, key_local, value_local, gate_full, mask, out_weight_full, scores, probabilities, weighted, full_gated, attention_full)
            residual_full = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            residual_full = _dm334_residual_residual_add_inline(running, attention_full, residual_full)
            post_norm_full = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            post_norm_full = _dm334_rmsnorm_rmsnorm_inline(residual_full, post_gamma_layer, post_norm_full)
            mlp_gate_full = pl.create_tensor([_dm334_ROWS, _dm334_INTERMEDIATE], dtype=pl.BF16)
            mlp_up_full = pl.create_tensor([_dm334_ROWS, _dm334_INTERMEDIATE], dtype=pl.BF16)
            mlp_gate_full = _dm334_linear_hidden_projection_inline(post_norm_full, mlp_gate_layer, mlp_gate_full)
            mlp_up_full = _dm334_linear_hidden_projection_inline(post_norm_full, mlp_up_layer, mlp_up_full)
            mlp_middle_full = pl.create_tensor([_dm334_ROWS, _dm334_INTERMEDIATE], dtype=pl.BF16)
            mlp_middle_full = _dm334_swiglu_swiglu_inline(mlp_gate_full, mlp_up_full, mlp_middle_full)
            mlp_output_full = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            mlp_output_full = _dm334_linear_hidden_projection_inline(mlp_middle_full, mlp_down_layer, mlp_output_full)
            following_full = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
            following_full = _dm334_residual_residual_add_inline(residual_full, mlp_output_full, following_full)
            with pl.spmd(4 * _dm334_ROWS, name_hint='qwen_full_store_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // _dm334_ROWS
                row = task % _dm334_ROWS
                if row < token_count:
                    position = cache_offset + row
                    logical_block = position // _dm334_BLOCK_SIZE
                    physical = pl.read(block_table, [logical_block])
                    slot = position % _dm334_BLOCK_SIZE
                    key_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:_dm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(key_local, [1, 1, _dm334_FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, _dm334_FULL_HEAD_DIM])
                    value_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:_dm334_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_local, [1, 1, _dm334_FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, _dm334_FULL_HEAD_DIM])
            with pl.spmd(_dm334_ROWS, name_hint='qwen_full_store_attention_hidden'):
                row = pl.tile.get_block_idx()
                next_hidden = pl.slice(following_full, [1, _dm334_HIDDEN], [row, 0])
                running[row:row + 1, 0:_dm334_HIDDEN] = next_hidden
                traces[layer + 1:layer + 2, row:row + 1, 0:_dm334_HIDDEN] = pl.reshape(next_hidden, [1, 1, _dm334_HIDDEN])
    normalized = pl.create_tensor([_dm334_ROWS, _dm334_HIDDEN], dtype=pl.BF16)
    normalized = _dm334_rmsnorm_rmsnorm_inline(running, final_gamma, normalized)
    with pl.spmd(_dm334_ROWS, name_hint='qwen_full_store_final_norm'):
        row = pl.tile.get_block_idx()
        traces[65:66, row:row + 1, 0:_dm334_HIDDEN] = pl.reshape(pl.slice(normalized, [1, _dm334_HIDDEN], [row, 0]), [1, 1, _dm334_HIDDEN])
    logits = _dm334_linear_hidden_projection_inline_nd(normalized, lm_head, logits)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def _dm334_full_short_fwd(input_ids: pl.Tensor[[_dm334_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_QKV, _dm334_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_FULL_Q_PROJECTION, _dm334_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_HIDDEN, _dm334_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_dm334_ROWS, _dm334_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_ROWS, _dm334_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_ROWS, _dm334_CONTEXT], pl.FP32], token_count: pl.Scalar[pl.INT32], cache_offset: pl.Scalar[pl.INT32], conv_states: pl.InOut[pl.Tensor[[48, 4, _dm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], block_table: pl.Tensor[[_dm334_CONTEXT // _dm334_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[66, _dm334_ROWS, _dm334_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[_dm334_ROWS, _dm334_VOCAB], pl.BF16]]):
    return _dm334_full_step_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_count, cache_offset, conv_states, recurrent_states, key_pool, value_pool, block_table, traces, logits)

@pl.jit.inline
def _dm334_decode_gated_tail_inline(core: pl.Tensor, z_projection: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor, gated_output: pl.Tensor, attention_output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    gate_rows = pl.reshape(z_projection, [_dm334_BATCH * _dm334_DELTA_HEADS, _dm334_DELTA_DIM])
    core_rows = pl.reshape(core, [_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS, _dm334_delta_window16_DIM])
    with pl.spmd(_dm334_delta_window16_ROWS * _dm334_delta_window16_HEADS // _dm334_delta_window16_NORM_TILE, name_hint='qwen_window_gated_rmsnorm'):
        row = pl.tile.get_block_idx() * _dm334_delta_window16_NORM_TILE
        if row // _dm334_DELTA_HEADS > 0 and row // _dm334_DELTA_HEADS < active_count:
            source = pl.cast(pl.slice(core_rows, [_dm334_delta_window16_NORM_TILE, _dm334_delta_window16_DIM], [row, 0]), pl.FP32)
            gate_tile = pl.cast(pl.slice(gate_rows, [_dm334_delta_window16_NORM_TILE, _dm334_delta_window16_DIM], [row, 0]), pl.FP32)
            weight_tile = pl.cast(pl.slice(norm_weight, [1, _dm334_gated_norm_DIM], [0, 0]), pl.FP32)
            variance = pl.mul(pl.row_sum(pl.mul(source, source)), 1.0 / _dm334_delta_window16_DIM)
            inverse = pl.rsqrt(pl.add(variance, 1e-06), high_precision=True)
            normalized = pl.cast(pl.row_expand_mul(source, inverse), pl.BF16, mode='rint')
            weighted = pl.cast(pl.col_expand_mul(pl.cast(normalized, pl.FP32), weight_tile), pl.BF16, mode='rint')
            sigmoid = pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0))
            gated_output[row:row + _dm334_delta_window16_NORM_TILE, 0:_dm334_delta_window16_DIM] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), sigmoid), pl.BF16, mode='rint')
    return gated_output

@pl.jit.inline
def _dm334_decode_attention_core_inline(query: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16], key_cache: pl.Tensor[[_dm334_attention512_core_KV_HEADS, _dm334_attention512_core_CONTEXT, _dm334_attention512_core_HEAD_DIM], pl.BF16], value_cache: pl.Tensor[[_dm334_attention512_core_KV_HEADS, _dm334_attention512_core_CONTEXT, _dm334_attention512_core_HEAD_DIM], pl.BF16], gate: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16], mask: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_CONTEXT], pl.FP32], output_weight: pl.Tensor[[5120, _dm334_attention512_core_Q_WIDTH], pl.BF16], logits_buffer: pl.Tensor[[_dm334_attention512_core_Q_HEADS * _dm334_attention512_core_ROWS, _dm334_attention512_core_CONTEXT], pl.FP32], probability_buffer: pl.Tensor[[_dm334_attention512_core_Q_HEADS * 16, _dm334_attention512_core_CONTEXT], pl.BF16], weighted_output: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16], gated_output: pl.Tensor[[_dm334_attention512_core_ROWS, _dm334_attention512_core_Q_WIDTH], pl.BF16], projected_output: pl.Tensor[[_dm334_attention512_core_ROWS, 5120], pl.BF16]):
    with pl.spmd(_dm334_attention512_core_Q_HEADS, name_hint='qwen_attention512_scores'):
        score_head = pl.tile.get_block_idx()
        score_kv_head = score_head // (_dm334_attention512_core_Q_HEADS // _dm334_attention512_core_KV_HEADS)
        score_start = score_head * _dm334_attention512_core_HEAD_DIM
        score_q = pl.slice(query, [_dm334_attention512_core_ROWS, _dm334_attention512_core_HEAD_DIM], [0, score_start])
        for score_block in pl.range(_dm334_attention512_core_CONTEXT // _dm334_attention512_core_KEY_BLOCK):
            score_key_start = score_block * _dm334_attention512_core_KEY_BLOCK
            score_k = pl.reshape(pl.slice(key_cache, [1, _dm334_attention512_core_KEY_BLOCK, _dm334_attention512_core_HEAD_DIM], [score_kv_head, score_key_start, 0]), [_dm334_attention512_core_KEY_BLOCK, _dm334_attention512_core_HEAD_DIM])
            score_raw = pl.matmul(score_q, score_k, pl.FP32, b_trans=True)
            score_rounded = pl.cast(score_raw, pl.BF16, mode='rint')
            score_scaled = pl.cast(pl.mul(pl.cast(score_rounded, pl.FP32), _dm334_attention512_core_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
            score_scaled = pl.slice(score_scaled, [_dm334_attention512_core_ROWS, _dm334_attention512_core_KEY_BLOCK], [0, 0])
            score_mask = pl.slice(mask, [_dm334_attention512_core_ROWS, _dm334_attention512_core_KEY_BLOCK], [0, score_key_start])
            logits_buffer[score_head * _dm334_attention512_core_ROWS:(score_head + 1) * _dm334_attention512_core_ROWS, score_key_start:score_key_start + _dm334_attention512_core_KEY_BLOCK] = pl.add(pl.cast(score_scaled, pl.FP32), score_mask)
    with pl.spmd(_dm334_attention512_core_Q_HEADS, name_hint='qwen_attention512_softmax'):
        softmax_head = pl.tile.get_block_idx()
        softmax_row = softmax_head * 16
        softmax_logits = pl.slice(logits_buffer, [_dm334_attention512_core_ROWS, _dm334_attention512_core_CONTEXT], [softmax_head * _dm334_attention512_core_ROWS, 0])
        softmax_exp = pl.exp(pl.row_expand_sub(softmax_logits, pl.row_max(softmax_logits)))
        probability_buffer[softmax_row:softmax_row + _dm334_attention512_core_ROWS, 0:_dm334_attention512_core_CONTEXT] = pl.cast(pl.row_expand_div(softmax_exp, pl.row_sum(softmax_exp)), pl.BF16, mode='rint')
    with pl.spmd(_dm334_attention512_core_Q_HEADS, name_hint='qwen_attention512_weighted'):
        weighted_head = pl.tile.get_block_idx()
        weighted_kv_head = weighted_head // (_dm334_attention512_core_Q_HEADS // _dm334_attention512_core_KV_HEADS)
        weighted_start = weighted_head * _dm334_attention512_core_HEAD_DIM
        weighted_sum = pl.full([16, _dm334_attention512_core_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for weighted_block in pl.range(_dm334_attention512_core_CONTEXT // _dm334_attention512_core_KEY_BLOCK):
            weighted_key_start = weighted_block * _dm334_attention512_core_KEY_BLOCK
            weighted_probs = pl.slice(probability_buffer, [16, _dm334_attention512_core_KEY_BLOCK], [weighted_head * 16, weighted_key_start])
            weighted_values = pl.reshape(pl.slice(value_cache, [1, _dm334_attention512_core_KEY_BLOCK, _dm334_attention512_core_HEAD_DIM], [weighted_kv_head, weighted_key_start, 0]), [_dm334_attention512_core_KEY_BLOCK, _dm334_attention512_core_HEAD_DIM])
            weighted_full = pl.matmul(weighted_probs, weighted_values, pl.FP32)
            weighted_sum = pl.add(weighted_sum, weighted_full)
        weighted = pl.cast(weighted_sum, pl.BF16, mode='rint')
        weighted_valid = pl.slice(weighted, [_dm334_attention512_core_ROWS, _dm334_attention512_core_HEAD_DIM], [0, 0])
        weighted_output[0:_dm334_attention512_core_ROWS, weighted_start:weighted_start + _dm334_attention512_core_HEAD_DIM] = weighted_valid
        gate_tile = pl.cast(pl.slice(gate, [_dm334_attention512_core_ROWS, _dm334_attention512_core_HEAD_DIM], [0, weighted_start]), pl.FP32)
        sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        gated_output[0:_dm334_attention512_core_ROWS, weighted_start:weighted_start + _dm334_attention512_core_HEAD_DIM] = pl.cast(pl.mul(pl.cast(weighted_valid, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint')
    return gated_output

@pl.jit.inline
def _dm334_decode_delta_slots_inline(qkv: pl.Tensor, z: pl.Tensor, a: pl.Tensor, b: pl.Tensor, conv_weight: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, gated_batch: pl.Tensor, linear: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    state_layers = pl.tensor.dim(conv_states, 1)
    conv_rows = pl.reshape(conv_states, [_dm334_BATCH * state_layers * 4, _dm334_QKV])
    state_rows = pl.reshape(recurrent_states, [state_layers * (_dm334_BATCH * _dm334_DELTA_HEADS * _dm334_DELTA_DIM), _dm334_DELTA_DIM])
    conv_output = pl.create_tensor([_dm334_BATCH, _dm334_QKV], dtype=pl.BF16)
    with pl.spmd(_dm334_BATCH * (_dm334_QKV // 5120), name_hint='qwen_decode_batched_conv'):
        task = pl.tile.get_block_idx()
        batch = task // (_dm334_QKV // 5120)
        col = task % (_dm334_QKV // 5120) * 5120
        if batch > 0 and batch < active_count:
            state_start = (batch * state_layers + linear) * 4
            previous_1 = pl.cast(pl.slice(conv_rows, [1, 5120], [state_start + 1, col]), pl.FP32)
            previous_2 = pl.cast(pl.slice(conv_rows, [1, 5120], [state_start + 2, col]), pl.FP32)
            previous_3 = pl.cast(pl.slice(conv_rows, [1, 5120], [state_start + 3, col]), pl.FP32)
            current = pl.cast(pl.slice(qkv, [1, 5120], [batch, col]), pl.FP32)
            tap_0 = pl.cast(pl.slice(conv_weight, [1, 5120], [0, col]), pl.FP32)
            tap_1 = pl.cast(pl.slice(conv_weight, [1, 5120], [1, col]), pl.FP32)
            tap_2 = pl.cast(pl.slice(conv_weight, [1, 5120], [2, col]), pl.FP32)
            tap_3 = pl.cast(pl.slice(conv_weight, [1, 5120], [3, col]), pl.FP32)
            convolved = pl.add(pl.add(pl.mul(previous_1, tap_0), pl.mul(previous_2, tap_1)), pl.add(pl.mul(previous_3, tap_2), pl.mul(current, tap_3)))
            convolved = pl.cast(pl.cast(convolved, pl.BF16, mode='rint'), pl.FP32)
            activated = pl.div(convolved, pl.add(pl.exp(pl.neg(convolved)), 1.0), high_precision=True)
            conv_output[batch:batch + 1, col:col + 5120] = pl.cast(activated, pl.BF16, mode='rint')
            conv_rows[state_start:state_start + 1, col:col + 5120] = pl.cast(previous_1, pl.BF16, mode='rint')
            conv_rows[state_start + 1:state_start + 2, col:col + 5120] = pl.cast(previous_2, pl.BF16, mode='rint')
            conv_rows[state_start + 2:state_start + 3, col:col + 5120] = pl.cast(previous_3, pl.BF16, mode='rint')
            conv_rows[state_start + 3:state_start + 4, col:col + 5120] = pl.cast(current, pl.BF16, mode='rint')
        else:
            conv_output[batch:batch + 1, col:col + 5120] = pl.full([1, 5120], dtype=pl.BF16, value=0.0)
    gate_values = pl.create_tensor([_dm334_BATCH, 64], dtype=pl.FP32)
    beta_bf16 = pl.create_tensor([_dm334_BATCH, 64], dtype=pl.BF16)
    (gate_values, beta_bf16) = _dm334_delta_prepare_decay_beta_inline(a, b, a_log, dt_bias, gate_values, beta_bf16)
    beta_values = pl.create_tensor([_dm334_BATCH, 64], dtype=pl.FP32)
    with pl.spmd(1, name_hint='qwen_decode_batched_beta'):
        beta_row = pl.tile.get_block_idx() * _dm334_BATCH
        beta_values[beta_row:beta_row + _dm334_BATCH, 0:64] = pl.cast(pl.slice(beta_bf16, [_dm334_BATCH, 64], [beta_row, 0]), pl.FP32)
    q_norm = pl.create_tensor([_dm334_BATCH * _dm334_DELTA_HEADS, _dm334_DELTA_DIM], dtype=pl.FP32)
    k_norm = pl.create_tensor([_dm334_BATCH * _dm334_DELTA_HEADS, _dm334_DELTA_DIM], dtype=pl.FP32)
    with pl.spmd(_dm334_BATCH * 2, name_hint='qwen_decode_direct_qk_norm') as norm_task:
        direct_norm_task = pl.tile.get_block_idx()
        direct_norm_batch = direct_norm_task // 2
        direct_norm_half = direct_norm_task % 2
        direct_norm_q = pl.cast(pl.reshape(pl.slice(conv_output, [1, 1024], [direct_norm_batch, direct_norm_half * 1024]), [8, _dm334_DELTA_DIM]), pl.FP32)
        direct_norm_k = pl.cast(pl.reshape(pl.slice(conv_output, [1, 1024], [direct_norm_batch, 2048 + direct_norm_half * 1024]), [8, _dm334_DELTA_DIM]), pl.FP32)
        direct_norm_q_unit = pl.mul(pl.row_expand_mul(direct_norm_q, pl.rsqrt(pl.add(pl.row_sum(pl.mul(direct_norm_q, direct_norm_q)), 1e-06), high_precision=True)), _dm334_DELTA_DIM ** (-0.5))
        direct_norm_k_unit = pl.row_expand_mul(direct_norm_k, pl.rsqrt(pl.add(pl.row_sum(pl.mul(direct_norm_k, direct_norm_k)), 1e-06), high_precision=True))
        direct_norm_heads = pl.cast(pl.mul(pl.cast(pl.arange(0, [1, 24], dtype=pl.INT32), pl.FP32), 1.0 / 3.0), pl.INT32, mode='floor')
        direct_norm_indices = pl.row_expand(pl.create_tensor([24, _dm334_DELTA_DIM], dtype=pl.INT32), pl.reshape(direct_norm_heads, [24, 1]))
        direct_norm_flat_indices = pl.col_expand_add(pl.mul(direct_norm_indices, 128), pl.arange(0, [1, 128], dtype=pl.INT32))
        direct_norm_start = direct_norm_batch * _dm334_DELTA_HEADS + direct_norm_half * 24
        q_norm[direct_norm_start:direct_norm_start + 24, 0:_dm334_DELTA_DIM] = pl.gather(direct_norm_q_unit, index=direct_norm_flat_indices)
        k_norm[direct_norm_start:direct_norm_start + 24, 0:_dm334_DELTA_DIM] = pl.gather(direct_norm_k_unit, index=direct_norm_flat_indices)
    core = pl.create_tensor([_dm334_BATCH, _dm334_DELTA_HEADS, _dm334_DELTA_DIM], dtype=pl.BF16)
    core_rows = pl.reshape(core, [_dm334_BATCH, _dm334_VALUE_WIDTH])
    with pl.spmd(40, name_hint='qwen_decode_delta_state_update', deps=[norm_task]) as update_task:
        reuse_state_block = pl.tile.get_block_idx()
        for reuse_state_head_part in pl.range(10):
            reuse_state_head_task = reuse_state_block + reuse_state_head_part * 40
            reuse_state_batch = reuse_state_head_task // _dm334_DELTA_HEADS
            reuse_state_head = reuse_state_head_task % _dm334_DELTA_HEADS
            if reuse_state_batch > 0 and reuse_state_batch < active_count:
                reuse_state_row = reuse_state_batch * _dm334_DELTA_HEADS + reuse_state_head
                reuse_state_k = pl.slice(k_norm, [1, _dm334_DELTA_DIM], [reuse_state_row, 0])
                reuse_state_g = pl.read(gate_values, [reuse_state_batch, reuse_state_head])
                reuse_state_decay = pl.exp(pl.mul(pl.full([1, _dm334_DELTA_DIM], dtype=pl.FP32, value=1.0), reuse_state_g))
                reuse_state_beta_value = pl.read(beta_values, [reuse_state_batch, reuse_state_head])
                for reuse_state_half in pl.range(2):
                    reuse_state_value_start = reuse_state_half * 64
                    reuse_state_output_start = reuse_state_head * _dm334_DELTA_DIM + reuse_state_value_start
                    reuse_state_state_start = (reuse_state_batch * state_layers + linear) * _dm334_DELTA_HEADS * _dm334_DELTA_DIM + reuse_state_output_start
                    reuse_state_state = pl.slice(state_rows, [64, _dm334_DELTA_DIM], [reuse_state_state_start, 0])
                    reuse_state_decayed = pl.col_expand_mul(reuse_state_state, reuse_state_decay)
                    reuse_state_remembered = pl.row_sum(pl.col_expand_mul(reuse_state_decayed, reuse_state_k))
                    reuse_state_v_row = pl.cast(pl.slice(conv_output, [1, 64], [reuse_state_batch, 4096 + reuse_state_head * _dm334_DELTA_DIM + reuse_state_value_start]), pl.FP32)
                    reuse_state_v_col = pl.reshape(reuse_state_v_row, [64, 1])
                    reuse_state_delta = pl.mul(pl.sub(reuse_state_v_col, reuse_state_remembered), reuse_state_beta_value)
                    reuse_state_following = pl.add(reuse_state_decayed, pl.col_expand_mul(pl.row_expand_mul(pl.full([64, _dm334_DELTA_DIM], dtype=pl.FP32, value=1.0), reuse_state_delta), reuse_state_k))
                    state_rows[reuse_state_state_start:reuse_state_state_start + 64, 0:_dm334_DELTA_DIM] = reuse_state_following
    with pl.spmd(20, name_hint='qwen_decode_delta_output_group4', deps=[update_task]):
        reuse_output_coarse_block = pl.tile.get_block_idx()
        for reuse_output_coarse_part in pl.range(5):
            reuse_output_coarse_task = reuse_output_coarse_block + reuse_output_coarse_part * 20
            if reuse_output_coarse_task < _dm334_BATCH * (_dm334_DELTA_HEADS // 4):
                reuse_output_coarse_batch = reuse_output_coarse_task // (_dm334_DELTA_HEADS // 4)
                reuse_output_coarse_head = reuse_output_coarse_task % (_dm334_DELTA_HEADS // 4) * 4
                reuse_output_coarse_output_start = reuse_output_coarse_head * _dm334_DELTA_DIM
                if reuse_output_coarse_batch > 0 and reuse_output_coarse_batch < active_count:
                    reuse_output_coarse_state_start = (reuse_output_coarse_batch * state_layers + linear) * _dm334_DELTA_HEADS * _dm334_DELTA_DIM + reuse_output_coarse_output_start
                    reuse_output_coarse_output_following = pl.slice(state_rows, [4 * _dm334_DELTA_DIM, _dm334_DELTA_DIM], [reuse_output_coarse_state_start, 0])
                    reuse_output_coarse_q = pl.slice(q_norm, [4, _dm334_DELTA_DIM], [reuse_output_coarse_batch * _dm334_DELTA_HEADS + reuse_output_coarse_head, 0])
                    reuse_output_coarse_product = pl.cast(pl.matmul(reuse_output_coarse_q, reuse_output_coarse_output_following, pl.FP32, b_trans=True), pl.BF16, mode='rint')
                    for reuse_output_coarse_local in pl.unroll(4):
                        reuse_output_coarse_col = reuse_output_coarse_output_start + reuse_output_coarse_local * _dm334_DELTA_DIM
                        core_rows[reuse_output_coarse_batch:reuse_output_coarse_batch + 1, reuse_output_coarse_col:reuse_output_coarse_col + _dm334_DELTA_DIM] = pl.slice(reuse_output_coarse_product, [1, _dm334_DELTA_DIM], [reuse_output_coarse_local, reuse_output_coarse_local * _dm334_DELTA_DIM])
                else:
                    core_rows[reuse_output_coarse_batch:reuse_output_coarse_batch + 1, reuse_output_coarse_output_start:reuse_output_coarse_output_start + 4 * _dm334_DELTA_DIM] = pl.full([1, 4 * _dm334_DELTA_DIM], dtype=pl.BF16, value=0.0)
    gated_rows = pl.reshape(gated_batch, [_dm334_BATCH * _dm334_DELTA_HEADS, _dm334_DELTA_DIM])
    ignored_projection = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
    _dm334_decode_gated_tail_inline(core, z, norm_weight, out_weight, gated_rows, ignored_projection, active_count)
    return (conv_states, recurrent_states, gated_batch)

@pl.jit.inline
def _dm334_decode_attention_slots_inline(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor, mask: pl.Tensor, out_weight: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, offsets: pl.Tensor, gated_batch: pl.Tensor, full: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    with pl.spmd(_dm334_BATCH * 4, name_hint='qwen_decode_batched_update_kv') as update_task:
        task = pl.tile.get_block_idx()
        batch = task // 4
        head = task % 4
        if batch > 0 and batch < active_count:
            position = pl.read(offsets, [batch])
            col = head * _dm334_FULL_HEAD_DIM
            physical = pl.read(block_tables, [batch, position // _dm334_BLOCK_SIZE])
            slot = position % _dm334_BLOCK_SIZE
            new_key = pl.slice(key, [1, _dm334_FULL_HEAD_DIM], [batch, col])
            new_value = pl.slice(value, [1, _dm334_FULL_HEAD_DIM], [batch, col])
            key_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:_dm334_FULL_HEAD_DIM] = pl.reshape(new_key, [1, 1, 1, 1, _dm334_FULL_HEAD_DIM])
            value_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:_dm334_FULL_HEAD_DIM] = pl.reshape(new_value, [1, 1, 1, 1, _dm334_FULL_HEAD_DIM])
    query_rows = pl.create_tensor([_dm334_BATCH * 4 * 8, _dm334_FULL_HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(_dm334_BATCH * 4, name_hint='qwen_decode_group6_query') as grouped_query_task:
        task = pl.tile.get_block_idx()
        batch = task // 4
        kv_head = task % 4
        if batch > 0 and batch < active_count:
            query_rows[task * 8:task * 8 + 8, 0:_dm334_FULL_HEAD_DIM] = pl.full([8, _dm334_FULL_HEAD_DIM], dtype=pl.BF16, value=0.0)
            six_queries = pl.reshape(pl.slice(query, [1, 6 * _dm334_FULL_HEAD_DIM], [batch, kv_head * 6 * _dm334_FULL_HEAD_DIM]), [6, _dm334_FULL_HEAD_DIM])
            query_rows[task * 8:task * 8 + 6, 0:_dm334_FULL_HEAD_DIM] = six_queries
    scores = pl.create_tensor([_dm334_BATCH * 4 * 8, _dm334_CONTEXT], dtype=pl.FP32)
    probabilities = pl.create_tensor([_dm334_BATCH * 4 * 16, _dm334_CONTEXT], dtype=pl.BF16)
    with pl.spmd(_dm334_BATCH * 4, name_hint='qwen_decode_group6_init_scores') as init_scores_task:
        task = pl.tile.get_block_idx()
        batch = task // 4
        if batch > 0 and batch < active_count:
            initial = pl.full([8, _dm334_CONTEXT], dtype=pl.FP32, value=0.0)
            initial_mask = pl.reshape(pl.slice(mask, [1, 1, _dm334_CONTEXT], [batch, 0, 0]), [1, _dm334_CONTEXT])
            scores[task * 8:task * 8 + 8, 0:_dm334_CONTEXT] = pl.col_expand_add(initial, initial_mask)
    if active_count == 1:
        with pl.spmd(16, name_hint='qwen_decode_group6_scores_C1_split4', deps=[update_task, init_scores_task, grouped_query_task]):
            lane = pl.tile.get_block_idx()
            task = lane // 4
            batch = 0
            kv_head = task
            partition = lane % 4
            q = pl.slice(query_rows, [8, _dm334_FULL_HEAD_DIM], [task * 8, 0])
            position = pl.read(offsets, [batch])
            for block in pl.range(partition, position // _dm334_BLOCK_SIZE + 1, 4):
                key_start = block * _dm334_BLOCK_SIZE
                physical = pl.read(block_tables, [batch, block])
                k = pl.reshape(pl.slice(key_pool, [1, 1, 1, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], [full, kv_head, physical, 0, 0]), [_dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM])
                raw = pl.matmul(q, k, pl.FP32, b_trans=True)
                rounded = pl.cast(raw, pl.BF16, mode='rint')
                scaled = pl.cast(pl.mul(pl.cast(rounded, pl.FP32), _dm334_FULL_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
                score_mask = pl.reshape(pl.slice(mask, [1, 1, _dm334_BLOCK_SIZE], [batch, 0, key_start]), [1, _dm334_BLOCK_SIZE])
                score_rows = pl.slice(scaled, [8, _dm334_BLOCK_SIZE], [0, 0])
                scores[task * 8:task * 8 + 8, key_start:key_start + _dm334_BLOCK_SIZE] = pl.col_expand_add(pl.cast(score_rows, pl.FP32), score_mask)
    elif active_count == 8:
        with pl.spmd(_dm334_BATCH * 4 * 4, name_hint='qwen_decode_group6_scores_C8_split4', deps=[update_task, init_scores_task, grouped_query_task]):
            lane = pl.tile.get_block_idx()
            task = lane // 4
            partition = lane % 4
            batch = task // 4
            kv_head = task % 4
            if batch > 0 and batch < active_count:
                q = pl.slice(query_rows, [8, _dm334_FULL_HEAD_DIM], [task * 8, 0])
                position = pl.read(offsets, [batch])
                for block in pl.range(partition, position // _dm334_BLOCK_SIZE + 1, 4):
                    key_start = block * _dm334_BLOCK_SIZE
                    physical = pl.read(block_tables, [batch, block])
                    k = pl.reshape(pl.slice(key_pool, [1, 1, 1, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], [full, kv_head, physical, 0, 0]), [_dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM])
                    raw = pl.matmul(q, k, pl.FP32, b_trans=True)
                    rounded = pl.cast(raw, pl.BF16, mode='rint')
                    scaled = pl.cast(pl.mul(pl.cast(rounded, pl.FP32), _dm334_FULL_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
                    score_mask = pl.reshape(pl.slice(mask, [1, 1, _dm334_BLOCK_SIZE], [batch, 0, key_start]), [1, _dm334_BLOCK_SIZE])
                    score_rows = pl.slice(scaled, [8, _dm334_BLOCK_SIZE], [0, 0])
                    scores[task * 8:task * 8 + 8, key_start:key_start + _dm334_BLOCK_SIZE] = pl.col_expand_add(pl.cast(score_rows, pl.FP32), score_mask)
    else:
        with pl.spmd(_dm334_BATCH * 4, name_hint='qwen_decode_group6_scores', deps=[update_task, init_scores_task, grouped_query_task]):
            task = pl.tile.get_block_idx()
            batch = task // 4
            kv_head = task % 4
            if batch > 0 and batch < active_count:
                q = pl.slice(query_rows, [8, _dm334_FULL_HEAD_DIM], [task * 8, 0])
                position = pl.read(offsets, [batch])
                for block in pl.range(position // _dm334_BLOCK_SIZE + 1):
                    key_start = block * _dm334_BLOCK_SIZE
                    physical = pl.read(block_tables, [batch, block])
                    k = pl.reshape(pl.slice(key_pool, [1, 1, 1, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], [full, kv_head, physical, 0, 0]), [_dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM])
                    raw = pl.matmul(q, k, pl.FP32, b_trans=True)
                    rounded = pl.cast(raw, pl.BF16, mode='rint')
                    scaled = pl.cast(pl.mul(pl.cast(rounded, pl.FP32), _dm334_FULL_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
                    score_mask = pl.reshape(pl.slice(mask, [1, 1, _dm334_BLOCK_SIZE], [batch, 0, key_start]), [1, _dm334_BLOCK_SIZE])
                    score_rows = pl.slice(scaled, [8, _dm334_BLOCK_SIZE], [0, 0])
                    scores[task * 8:task * 8 + 8, key_start:key_start + _dm334_BLOCK_SIZE] = pl.col_expand_add(pl.cast(score_rows, pl.FP32), score_mask)
    with pl.spmd(_dm334_BATCH * 4, name_hint='qwen_decode_group6_softmax'):
        task = pl.tile.get_block_idx()
        batch = task // 4
        if batch > 0 and batch < active_count:
            logits = pl.slice(scores, [8, _dm334_CONTEXT], [task * 8, 0])
            exps = pl.exp(pl.row_expand_sub(logits, pl.row_max(logits)))
            probabilities[task * 16:task * 16 + 16, 0:_dm334_CONTEXT] = pl.full([16, _dm334_CONTEXT], dtype=pl.BF16, value=0.0)
            probabilities[task * 16:task * 16 + 8, 0:_dm334_CONTEXT] = pl.cast(pl.row_expand_div(exps, pl.row_sum(exps)), pl.BF16, mode='rint')
    with pl.spmd(_dm334_BATCH * 4, name_hint='qwen_decode_group6_weighted'):
        task = pl.tile.get_block_idx()
        batch = task // 4
        kv_head = task % 4
        if batch > 0 and batch < active_count:
            weighted_sum = pl.full([16, _dm334_FULL_HEAD_DIM], dtype=pl.FP32, value=0.0)
            position = pl.read(offsets, [batch])
            for block in pl.range(position // _dm334_BLOCK_SIZE + 1):
                key_start = block * _dm334_BLOCK_SIZE
                probs = pl.slice(probabilities, [16, _dm334_BLOCK_SIZE], [task * 16, key_start])
                physical = pl.read(block_tables, [batch, block])
                values = pl.reshape(pl.slice(value_pool, [1, 1, 1, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], [full, kv_head, physical, 0, 0]), [_dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM])
                weighted_sum = pl.add(weighted_sum, pl.matmul(probs, values, pl.FP32))
            weighted_rounded = pl.cast(weighted_sum, pl.BF16, mode='rint')
            weighted = pl.slice(weighted_rounded, [6, _dm334_FULL_HEAD_DIM], [0, 0])
            col = kv_head * 6 * _dm334_FULL_HEAD_DIM
            gate_values = pl.cast(pl.reshape(pl.slice(gate, [1, 6 * _dm334_FULL_HEAD_DIM], [batch, col]), [6, _dm334_FULL_HEAD_DIM]), pl.FP32)
            sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(gate_values)), 1.0)), pl.BF16, mode='rint')
            result = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint')
            gated_batch[batch:batch + 1, col:col + 6 * _dm334_FULL_HEAD_DIM] = pl.reshape(result, [1, 6 * _dm334_FULL_HEAD_DIM])
    return (key_pool, value_pool, gated_batch)

@pl.jit.inline
def _dm334_decode_io_swiglu_inline(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    with pl.spmd(_dm334_swiglu_INTERMEDIATE // 1024, name_hint='qwen_decode_wide_swiglu'):
        column = pl.tile.get_block_idx() * 1024
        gate_tile = pl.cast(pl.slice(gate, [_dm334_swiglu_ROWS, 1024], [0, column]), pl.FP32)
        up_tile = pl.cast(pl.slice(up, [_dm334_swiglu_ROWS, 1024], [0, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        multiplied = pl.mul(pl.cast(activated, pl.FP32), up_tile)
        output[0:_dm334_swiglu_ROWS, column:column + 1024] = pl.cast(multiplied, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def _dm334_fused_wide_inline(left: pl.Tensor, right: pl.Tensor, gamma: pl.Tensor, residual: pl.Tensor, normalized: pl.Tensor):
    with pl.spmd(8 // 8, name_hint='decode_residual_norm_full8_pipeline'):
        row_start = pl.tile.get_block_idx() * 8
        square_sum = pl.full([1, 8], dtype=pl.FP32, value=0.0)
        for load_group in pl.pipeline(0, 5120 // 512, stage=2):
            column = load_group * 512
            a = pl.cast(pl.slice(left, [8, 512], [row_start, column]), pl.FP32)
            b = pl.cast(pl.slice(right, [8, 512], [row_start, column]), pl.FP32)
            rounded = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
            residual[row_start:row_start + 8, column:column + 512] = rounded
            source_wide = pl.cast(rounded, pl.FP32)
            square_wide = pl.mul(source_wide, source_wide)
            for reduction_part in pl.unroll(4):
                square_part = pl.slice(square_wide, [8, 128], [0, reduction_part * 128])
                square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(square_part), [1, 8]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / 5120), 1e-06), high_precision=True), [8, 1])
        for output_group in pl.pipeline(0, 5120 // 512, stage=2):
            output_column = output_group * 512
            output_source = pl.cast(pl.slice(residual, [8, 512], [row_start, output_column]), pl.FP32)
            output_weight = pl.slice(gamma, [1, 512], [0, output_column])
            value = pl.col_expand_mul(pl.row_expand_mul(output_source, inverse), output_weight)
            normalized[row_start:row_start + 8, output_column:output_column + 512] = pl.cast(value, pl.BF16, mode='rint')
    return (residual, normalized)

@pl.jit.inline
def _dm334_batch_step_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    running = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
    cos_batch = pl.create_tensor([_dm334_BATCH, _dm334_ROTARY], dtype=pl.BF16)
    sin_batch = pl.create_tensor([_dm334_BATCH, _dm334_ROTARY], dtype=pl.BF16)
    with pl.spmd(_dm334_BATCH, name_hint='qwen_decode_pack_rope'):
        batch = pl.tile.get_block_idx()
        cos_batch[batch:batch + 1, 0:_dm334_ROTARY] = pl.reshape(pl.slice(cos, [1, 1, _dm334_ROTARY], [batch, 0, 0]), [1, _dm334_ROTARY])
        sin_batch[batch:batch + 1, 0:_dm334_ROTARY] = pl.reshape(pl.slice(sin, [1, 1, _dm334_ROTARY], [batch, 0, 0]), [1, _dm334_ROTARY])
    with pl.spmd(_dm334_BATCH * (_dm334_HIDDEN // 256), name_hint='qwen_decode_token_embedding'):
        task = pl.tile.get_block_idx()
        batch = task // (_dm334_HIDDEN // 256)
        col = task % (_dm334_HIDDEN // 256) * 256
        if batch > 0 and batch < active_count:
            token = pl.read(input_ids, [batch, 0])
            embedded = pl.slice(embedding, [1, 256], [token, col])
            running[batch:batch + 1, col:col + 256] = embedded
            traces[batch:batch + 1, 0:1, 0:1, col:col + 256] = pl.reshape(embedded, [1, 1, 1, 256])
        else:
            running[batch:batch + 1, col:col + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    for layer in pl.range(64):
        input_gamma_layer: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, _dm334_HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, _dm334_HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[_dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        mlp_up_layer: pl.Tensor[[_dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        mlp_down_layer: pl.Tensor[[_dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        norm = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
        norm = _dm334_rmsnorm_rmsnorm_inline(running, input_gamma_layer, norm)
        residual = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
        post_norm = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[_dm334_HIDDEN, _dm334_QKV], pl.BF16, pl.NZ] = delta_qkv_weight[linear]
            z_weight_layer: pl.Tensor[[_dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16, pl.NZ] = delta_z_weight[linear]
            a_weight_layer: pl.Tensor[[_dm334_HIDDEN, 128], pl.BF16, pl.NZ] = delta_a_weight[linear]
            b_weight_layer: pl.Tensor[[_dm334_HIDDEN, 128], pl.BF16, pl.NZ] = delta_b_weight[linear]
            conv_weight_layer: pl.Tensor[[4, _dm334_QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, _dm334_QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, _dm334_DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, _dm334_DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[_dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16, pl.NZ] = delta_out_weight[linear]
            qkv = pl.create_tensor([_dm334_BATCH, _dm334_QKV], dtype=pl.BF16)
            z = pl.create_tensor([_dm334_BATCH, _dm334_VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([_dm334_BATCH, 128], dtype=pl.BF16)
            b = pl.create_tensor([_dm334_BATCH, 128], dtype=pl.BF16)
            qkv = _dm334_linear_hidden_projection_inline(norm, qkv_weight_layer, qkv)
            z = _dm334_linear_hidden_projection_inline(norm, z_weight_layer, z)
            a = _dm334_linear_hidden_projection_gate_inline(norm, a_weight_layer, a)
            b = _dm334_linear_hidden_projection_gate_inline(norm, b_weight_layer, b)
            gated_delta = pl.create_tensor([_dm334_BATCH, _dm334_VALUE_WIDTH], dtype=pl.BF16)
            _dm334_decode_delta_slots_inline(qkv, z, a, b, conv_weight_layer, a_log_layer, dt_bias_layer, delta_norm_layer, delta_out_layer, conv_states, recurrent_states, gated_delta, linear, active_count)
            attention_delta = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
            attention_delta = _dm334_linear_hidden_projection_inline(gated_delta, delta_out_layer, attention_delta)
            (residual, post_norm) = _dm334_fused_wide_inline(running, attention_delta, post_gamma_layer, residual, post_norm)
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[_dm334_HIDDEN, _dm334_FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[full]
            k_weight_full: pl.Tensor[[_dm334_HIDDEN, _dm334_FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[full]
            v_weight_full: pl.Tensor[[_dm334_HIDDEN, _dm334_FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[full]
            q_gamma_full: pl.Tensor[[1, _dm334_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, _dm334_FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, _dm334_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, _dm334_FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[_dm334_FULL_Q_WIDTH, _dm334_HIDDEN], pl.BF16, pl.NZ] = full_out_weight[full]
            q_norm = pl.create_tensor([_dm334_BATCH, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm = pl.create_tensor([_dm334_BATCH, _dm334_FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([_dm334_BATCH, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([_dm334_BATCH, _dm334_FULL_KV_WIDTH], dtype=pl.BF16)
            value = pl.create_tensor([_dm334_BATCH, _dm334_FULL_KV_WIDTH], dtype=pl.BF16)
            gate = pl.create_tensor([_dm334_BATCH, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            (q_norm, k_norm, q_rot, k_rot, value, gate) = _dm334_attention3_prepare_attention3_prepare_inline(norm, q_weight_full, k_weight_full, v_weight_full, q_gamma_full, k_gamma_full, cos_batch, sin_batch, q_norm, k_norm, q_rot, k_rot, value, gate)
            gated_full = pl.create_tensor([_dm334_BATCH, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            _dm334_decode_attention_slots_inline(q_rot, k_rot, value, gate, mask, out_weight_full, key_pool, value_pool, block_tables, cache_offsets, gated_full, full, active_count)
            attention_full = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
            attention_full = _dm334_linear_hidden_projection_inline(gated_full, out_weight_full, attention_full)
            (residual, post_norm) = _dm334_fused_wide_inline(running, attention_full, post_gamma_layer, residual, post_norm)
        mlp_gate = pl.create_tensor([_dm334_BATCH, _dm334_INTERMEDIATE], dtype=pl.BF16)
        mlp_up = pl.create_tensor([_dm334_BATCH, _dm334_INTERMEDIATE], dtype=pl.BF16)
        mlp_gate = _dm334_linear_hidden_projection_inline(post_norm, mlp_gate_layer, mlp_gate)
        mlp_up = _dm334_linear_hidden_projection_inline(post_norm, mlp_up_layer, mlp_up)
        mlp_middle = pl.create_tensor([_dm334_BATCH, _dm334_INTERMEDIATE], dtype=pl.BF16)
        mlp_middle = _dm334_decode_io_swiglu_inline(mlp_gate, mlp_up, mlp_middle)
        mlp_output = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
        mlp_output = _dm334_linear_hidden_projection_inline(mlp_middle, mlp_down_layer, mlp_output)
        with pl.spmd(_dm334_BATCH, name_hint='qwen_decode_residual_store'):
            io_batch = pl.tile.get_block_idx()
            io_left = pl.cast(pl.slice(residual, [1, _dm334_HIDDEN], [io_batch, 0]), pl.FP32)
            io_right = pl.cast(pl.slice(mlp_output, [1, _dm334_HIDDEN], [io_batch, 0]), pl.FP32)
            io_following = pl.cast(pl.add(io_left, io_right), pl.BF16, mode='rint')
            running[io_batch:io_batch + 1, 0:_dm334_HIDDEN] = io_following
            if io_batch > 0 and io_batch < active_count:
                traces[io_batch:io_batch + 1, layer + 1:layer + 2, 0:1, 0:_dm334_HIDDEN] = pl.reshape(io_following, [1, 1, 1, _dm334_HIDDEN])
    normalized = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
    normalized = _dm334_rmsnorm_rmsnorm_inline(running, final_gamma, normalized)
    packed_logits = pl.create_tensor([_dm334_BATCH, _dm334_VOCAB], dtype=pl.BF16)
    _dm334_linear_hidden_projection_inline_nd(normalized, lm_head, packed_logits)
    with pl.spmd(_dm334_BATCH * 61, name_hint='qwen_decode_coarse_logits'):
        io_task = pl.tile.get_block_idx()
        io_batch = io_task // 61
        io_part = io_task % 61
        io_column = io_part * 4096
        if io_batch > 0 and io_batch < active_count:
            if io_part < 60:
                logits[io_batch:io_batch + 1, 0:1, io_column:io_column + 4096] = pl.reshape(pl.slice(packed_logits, [1, 4096], [io_batch, io_column]), [1, 1, 4096])
            else:
                logits[io_batch:io_batch + 1, 0:1, 245760:248320] = pl.reshape(pl.slice(packed_logits, [1, 2560], [io_batch, 245760]), [1, 1, 2560])
    with pl.spmd(_dm334_BATCH, name_hint='qwen_decode_store_final_norm'):
        batch = pl.tile.get_block_idx()
        if batch > 0 and batch < active_count:
            traces[batch:batch + 1, 65:66, 0:1, 0:_dm334_HIDDEN] = pl.reshape(pl.slice(normalized, [1, _dm334_HIDDEN], [batch, 0]), [1, 1, 1, _dm334_HIDDEN])
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def _dm334_rank_prefill_fwd(input_ids: pl.Tensor[[_dm334_BATCH, _dm334_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_QKV, _dm334_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_FULL_Q_PROJECTION, _dm334_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_HIDDEN, _dm334_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_CONTEXT], pl.FP32], token_counts: pl.Tensor[[_dm334_BATCH], pl.INT32], cache_offsets: pl.Tensor[[_dm334_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, 4, _dm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[_dm334_BATCH, _dm334_CONTEXT // _dm334_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[_dm334_BATCH, 66, _dm334_ROWS, _dm334_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    for batch in pl.range(_dm334_BATCH):
        if batch < active_count:
            ids_slice = pl.slice(input_ids, [1, _dm334_ROWS], [batch, 0])
            ids = pl.reshape(ids_slice, [_dm334_ROWS])
            cos_slice = pl.slice(cos, [1, _dm334_ROWS, _dm334_ROTARY], [batch, 0, 0])
            cos_slot = pl.reshape(cos_slice, [_dm334_ROWS, _dm334_ROTARY])
            sin_slice = pl.slice(sin, [1, _dm334_ROWS, _dm334_ROTARY], [batch, 0, 0])
            sin_slot = pl.reshape(sin_slice, [_dm334_ROWS, _dm334_ROTARY])
            mask_slice = pl.slice(mask, [1, _dm334_ROWS, _dm334_CONTEXT], [batch, 0, 0])
            mask_slot = pl.reshape(mask_slice, [_dm334_ROWS, _dm334_CONTEXT])
            conv_slice = pl.slice(conv_states, [1, 48, 4, _dm334_QKV], [batch, 0, 0, 0])
            conv_slot = pl.reshape(conv_slice, [48, 4, _dm334_QKV])
            recurrent_slice = pl.slice(recurrent_states, [1, 48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], [batch, 0, 0, 0, 0])
            recurrent_slot = pl.reshape(recurrent_slice, [48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM])
            table_slice = pl.slice(block_tables, [1, _dm334_CONTEXT // _dm334_BLOCK_SIZE], [batch, 0])
            table_slot = pl.reshape(table_slice, [_dm334_CONTEXT // _dm334_BLOCK_SIZE])
            trace_slice = pl.slice(traces, [1, 66, _dm334_ROWS, _dm334_HIDDEN], [batch, 0, 0, 0])
            trace_slot = pl.reshape(trace_slice, [66, _dm334_ROWS, _dm334_HIDDEN])
            logits_slice = pl.slice(logits, [1, _dm334_ROWS, _dm334_VOCAB], [batch, 0, 0])
            logits_slot = pl.reshape(logits_slice, [_dm334_ROWS, _dm334_VOCAB])
            (conv_slot, recurrent_slot, key_pool, value_pool, trace_slot, logits_slot) = _dm334_full_step_inline(ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos_slot, sin_slot, mask_slot, pl.read(token_counts, [batch]), pl.read(cache_offsets, [batch]), conv_slot, recurrent_slot, key_pool, value_pool, table_slot, trace_slot, logits_slot)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def _dm334_rank_decode_fwd(input_ids: pl.Tensor[[_dm334_BATCH, _dm334_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_QKV, _dm334_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_FULL_Q_PROJECTION, _dm334_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_HIDDEN, _dm334_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_CONTEXT], pl.FP32], token_counts: pl.Tensor[[_dm334_BATCH], pl.INT32], cache_offsets: pl.Tensor[[_dm334_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, 4, _dm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[_dm334_BATCH, _dm334_CONTEXT // _dm334_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[_dm334_BATCH, 66, _dm334_ROWS, _dm334_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    (conv_states, recurrent_states, key_pool, value_pool, traces, logits) = _dm334_batch_step_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)
_dm334_tp_rank_ROWS = 8
_dm334_tp_rank_BATCH = 8
_dm334_tp_rank_CONTEXT = 2048
_dm334_tp_rank_BLOCK_SIZE = 16
_dm334_tp_rank_POOL_BLOCKS = 1032
_dm334_tp_rank_HIDDEN = 5120
_dm334_tp_rank_INTERMEDIATE = 17408
_dm334_tp_rank_QKV = 10240
_dm334_tp_rank_VALUE_WIDTH = 6144
_dm334_tp_rank_DELTA_HEADS = 48
_dm334_tp_rank_DELTA_DIM = 128
_dm334_tp_rank_FULL_HEAD_DIM = 256
_dm334_tp_rank_FULL_Q_WIDTH = 6144
_dm334_tp_rank_FULL_Q_PROJECTION = 12288
_dm334_tp_rank_FULL_KV_WIDTH = 1024
_dm334_tp_rank_ROTARY = 64
_dm334_tp_rank_VOCAB = 248320
_dm334_tp_rank_TP_VOCAB = _dm334_tp_rank_VOCAB // 2

@pl.jit.inline
def _dm334_tp_rank_full_step_inline(input_ids: pl.Tensor[[_dm334_tp_rank_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_tp_rank_VOCAB, _dm334_tp_rank_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_tp_rank_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_tp_rank_TP_VOCAB, _dm334_tp_rank_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_tp_rank_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_tp_rank_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_tp_rank_QKV, _dm334_tp_rank_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_tp_rank_VALUE_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_tp_rank_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_tp_rank_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_tp_rank_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_tp_rank_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_Q_PROJECTION, _dm334_tp_rank_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_tp_rank_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_tp_rank_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_tp_rank_ROWS, _dm334_tp_rank_CONTEXT], pl.FP32], token_count: pl.Scalar[pl.INT32], cache_offset: pl.Scalar[pl.INT32], conv_states: pl.Tensor[[48, 4, _dm334_tp_rank_QKV], pl.BF16], recurrent_states: pl.Tensor[[48, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], pl.FP32], key_pool: pl.Tensor[[16, 4, _dm334_tp_rank_POOL_BLOCKS, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM], pl.BF16], value_pool: pl.Tensor[[16, 4, _dm334_tp_rank_POOL_BLOCKS, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM], pl.BF16], block_table: pl.Tensor[[_dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE], pl.INT32], traces: pl.Tensor[[66, _dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], pl.BF16], logits: pl.Tensor[[_dm334_tp_rank_ROWS, _dm334_tp_rank_TP_VOCAB], pl.BF16]):
    running = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
    with pl.spmd(_dm334_tp_rank_ROWS * (_dm334_tp_rank_HIDDEN // 128), name_hint='qwen_full_token_embedding'):
        embed_task = pl.tile.get_block_idx()
        embed_token = embed_task // (_dm334_tp_rank_HIDDEN // 128)
        embed_column = embed_task % (_dm334_tp_rank_HIDDEN // 128) * 128
        if embed_token < token_count:
            embed_token_id = pl.read(input_ids, [embed_token])
            embed_tile = pl.slice(embedding, [1, 128], [embed_token_id, embed_column])
            running[embed_token:embed_token + 1, embed_column:embed_column + 128] = embed_tile
            traces[0:1, embed_token:embed_token + 1, embed_column:embed_column + 128] = pl.reshape(embed_tile, [1, 1, 128])
        else:
            zero_tile = pl.full([1, 128], dtype=pl.BF16, value=0.0)
            running[embed_token:embed_token + 1, embed_column:embed_column + 128] = zero_tile
            traces[0:1, embed_token:embed_token + 1, embed_column:embed_column + 128] = pl.reshape(zero_tile, [1, 1, 128])
    for layer in pl.range(64):
        input_gamma_layer: pl.Tensor[[1, _dm334_tp_rank_HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, _dm334_tp_rank_HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, _dm334_tp_rank_HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, _dm334_tp_rank_HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[_dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], pl.BF16] = pl.slice(mlp_gate_weight, [_dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], [layer * _dm334_tp_rank_INTERMEDIATE, 0])
        mlp_up_layer: pl.Tensor[[_dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], pl.BF16] = pl.slice(mlp_up_weight, [_dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], [layer * _dm334_tp_rank_INTERMEDIATE, 0])
        mlp_down_layer: pl.Tensor[[_dm334_tp_rank_HIDDEN, _dm334_tp_rank_INTERMEDIATE], pl.BF16] = pl.slice(mlp_down_weight, [_dm334_tp_rank_HIDDEN, _dm334_tp_rank_INTERMEDIATE], [layer * _dm334_tp_rank_HIDDEN, 0])
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[_dm334_tp_rank_QKV, _dm334_tp_rank_HIDDEN], pl.BF16] = pl.slice(delta_qkv_weight, [_dm334_tp_rank_QKV, _dm334_tp_rank_HIDDEN], [linear * _dm334_tp_rank_QKV, 0])
            z_weight_layer: pl.Tensor[[_dm334_tp_rank_VALUE_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16] = pl.slice(delta_z_weight, [_dm334_tp_rank_VALUE_WIDTH, _dm334_tp_rank_HIDDEN], [linear * _dm334_tp_rank_VALUE_WIDTH, 0])
            a_weight_layer: pl.Tensor[[128, _dm334_tp_rank_HIDDEN], pl.BF16] = pl.slice(delta_a_weight, [128, _dm334_tp_rank_HIDDEN], [linear * 128, 0])
            b_weight_layer: pl.Tensor[[128, _dm334_tp_rank_HIDDEN], pl.BF16] = pl.slice(delta_b_weight, [128, _dm334_tp_rank_HIDDEN], [linear * 128, 0])
            conv_weight_layer: pl.Tensor[[4, _dm334_tp_rank_QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, _dm334_tp_rank_QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, _dm334_tp_rank_DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, _dm334_tp_rank_DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[_dm334_tp_rank_HIDDEN, _dm334_tp_rank_VALUE_WIDTH], pl.BF16] = pl.slice(delta_out_weight, [_dm334_tp_rank_HIDDEN, _dm334_tp_rank_VALUE_WIDTH], [linear * _dm334_tp_rank_HIDDEN, 0])
            conv_source: pl.Tensor[[4, _dm334_tp_rank_QKV], pl.BF16] = pl.reshape(pl.slice(conv_states, [1, 4, _dm334_tp_rank_QKV], [linear, 0, 0]), [4, _dm334_tp_rank_QKV])
            recurrent_source: pl.Tensor[[_dm334_tp_rank_DELTA_HEADS * _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], pl.FP32] = pl.reshape(pl.slice(recurrent_states, [1, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], [linear, 0, 0, 0]), [_dm334_tp_rank_DELTA_HEADS * _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM])
            conv = pl.create_tensor([4, _dm334_tp_rank_QKV], dtype=pl.BF16)
            recurrent = pl.create_tensor([_dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], dtype=pl.FP32)
            recurrent_rows = pl.reshape(recurrent, [_dm334_tp_rank_DELTA_HEADS * _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM])
            with pl.spmd(4 * (_dm334_tp_rank_QKV // 128), name_hint='qwen_full_load_conv'):
                task = pl.tile.get_block_idx()
                tap = task // (_dm334_tp_rank_QKV // 128)
                col = task % (_dm334_tp_rank_QKV // 128) * 128
                conv[tap:tap + 1, col:col + 128] = pl.slice(conv_source, [1, 128], [tap, col])
            with pl.spmd(_dm334_tp_rank_DELTA_HEADS * _dm334_tp_rank_DELTA_DIM // 16, name_hint='qwen_full_load_recurrent'):
                row = pl.tile.get_block_idx() * 16
                recurrent_rows[row:row + 16, 0:_dm334_tp_rank_DELTA_DIM] = pl.slice(recurrent_source, [16, _dm334_tp_rank_DELTA_DIM], [row, 0])
            norm_delta = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            qkv = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_QKV], dtype=pl.BF16)
            z = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([_dm334_tp_rank_ROWS, 128], dtype=pl.BF16)
            b = pl.create_tensor([_dm334_tp_rank_ROWS, 128], dtype=pl.BF16)
            (norm_delta, qkv, z, a, b) = _dm334_layer0_stem_layer0_stem_inline(running, input_gamma_layer, qkv_weight_layer, z_weight_layer, a_weight_layer, b_weight_layer, norm_delta, qkv, z, a, b)
            conv_output = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_QKV], dtype=pl.BF16)
            query = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM], dtype=pl.BF16)
            key = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM], dtype=pl.BF16)
            value = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM], dtype=pl.BF16)
            log_decay = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_DELTA_HEADS], dtype=pl.FP32)
            beta = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_DELTA_HEADS], dtype=pl.FP32)
            core = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM], dtype=pl.BF16)
            (conv_output, conv) = _dm334_causal_conv_causal_conv_silu_inline(qkv, conv_weight_layer, conv, conv_output, token_count)
            (query, key, value) = _dm334_delta_window16_expand_qkv_window16_inline(conv_output, query, key, value)
            gate_values = pl.create_tensor([_dm334_tp_rank_ROWS, 64], dtype=pl.FP32)
            beta_bf16 = pl.create_tensor([_dm334_tp_rank_ROWS, 64], dtype=pl.BF16)
            (gate_values, beta_bf16) = _dm334_delta_prepare_decay_beta_inline(a, b, a_log_layer, dt_bias_layer, gate_values, beta_bf16)
            with pl.spmd(_dm334_tp_rank_ROWS, name_hint='qwen_full_delta_gate_unpad'):
                token = pl.tile.get_block_idx()
                log_decay[token:token + 1, 0:_dm334_tp_rank_DELTA_HEADS] = pl.slice(gate_values, [1, _dm334_tp_rank_DELTA_HEADS], [token, 0])
                beta[token:token + 1, 0:_dm334_tp_rank_DELTA_HEADS] = pl.cast(pl.slice(beta_bf16, [1, _dm334_tp_rank_DELTA_HEADS], [token, 0]), pl.FP32)
            (core, recurrent) = _dm334_delta_window16_recurrence_window16_inline(query, key, value, log_decay, beta, recurrent, core, token_count)
            gated = pl.create_tensor([_dm334_tp_rank_ROWS * _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM], dtype=pl.BF16)
            attention_delta = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            (gated, attention_delta) = _dm334_delta_window16_attention_tail_window16_inline(core, z, delta_norm_layer, delta_out_layer, gated, attention_delta)
            residual_delta = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            residual_delta = _dm334_residual_residual_add_inline(running, attention_delta, residual_delta)
            post_norm_delta = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            post_norm_delta = _dm334_rmsnorm_rmsnorm_inline(residual_delta, post_gamma_layer, post_norm_delta)
            mlp_gate_delta = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_up_delta = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_gate_delta = _dm334_linear_hidden_projection_inline(post_norm_delta, mlp_gate_layer, mlp_gate_delta)
            mlp_up_delta = _dm334_linear_hidden_projection_inline(post_norm_delta, mlp_up_layer, mlp_up_delta)
            mlp_middle_delta = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_middle_delta = _dm334_swiglu_swiglu_inline(mlp_gate_delta, mlp_up_delta, mlp_middle_delta)
            mlp_output_delta = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            mlp_output_delta = _dm334_linear_hidden_projection_inline(mlp_middle_delta, mlp_down_layer, mlp_output_delta)
            following_delta = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            following_delta = _dm334_residual_residual_add_inline(residual_delta, mlp_output_delta, following_delta)
            conv_dest: pl.Tensor[[4, _dm334_tp_rank_QKV], pl.BF16] = pl.reshape(pl.slice(conv_states, [1, 4, _dm334_tp_rank_QKV], [linear, 0, 0]), [4, _dm334_tp_rank_QKV])
            recurrent_dest: pl.Tensor[[_dm334_tp_rank_DELTA_HEADS * _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], pl.FP32] = pl.reshape(pl.slice(recurrent_states, [1, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], [linear, 0, 0, 0]), [_dm334_tp_rank_DELTA_HEADS * _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM])
            recurrent_rows = pl.reshape(recurrent, [_dm334_tp_rank_DELTA_HEADS * _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM])
            with pl.spmd(4 * (_dm334_tp_rank_QKV // 128), name_hint='qwen_full_store_conv'):
                task = pl.tile.get_block_idx()
                tap = task // (_dm334_tp_rank_QKV // 128)
                col = task % (_dm334_tp_rank_QKV // 128) * 128
                conv_dest[tap:tap + 1, col:col + 128] = pl.slice(conv, [1, 128], [tap, col])
            with pl.spmd(_dm334_tp_rank_DELTA_HEADS * _dm334_tp_rank_DELTA_DIM // 16, name_hint='qwen_full_store_recurrent'):
                row = pl.tile.get_block_idx() * 16
                recurrent_dest[row:row + 16, 0:_dm334_tp_rank_DELTA_DIM] = pl.slice(recurrent_rows, [16, _dm334_tp_rank_DELTA_DIM], [row, 0])
            with pl.spmd(_dm334_tp_rank_ROWS, name_hint='qwen_full_store_delta_hidden'):
                row = pl.tile.get_block_idx()
                next_hidden = pl.slice(following_delta, [1, _dm334_tp_rank_HIDDEN], [row, 0])
                running[row:row + 1, 0:_dm334_tp_rank_HIDDEN] = next_hidden
                traces[layer + 1:layer + 2, row:row + 1, 0:_dm334_tp_rank_HIDDEN] = pl.reshape(next_hidden, [1, 1, _dm334_tp_rank_HIDDEN])
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[_dm334_tp_rank_FULL_Q_PROJECTION, _dm334_tp_rank_HIDDEN], pl.BF16] = pl.slice(full_q_weight, [_dm334_tp_rank_FULL_Q_PROJECTION, _dm334_tp_rank_HIDDEN], [full * _dm334_tp_rank_FULL_Q_PROJECTION, 0])
            k_weight_full: pl.Tensor[[_dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16] = pl.slice(full_k_weight, [_dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], [full * _dm334_tp_rank_FULL_KV_WIDTH, 0])
            v_weight_full: pl.Tensor[[_dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16] = pl.slice(full_v_weight, [_dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], [full * _dm334_tp_rank_FULL_KV_WIDTH, 0])
            q_gamma_full: pl.Tensor[[1, _dm334_tp_rank_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, _dm334_tp_rank_FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, _dm334_tp_rank_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, _dm334_tp_rank_FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[_dm334_tp_rank_HIDDEN, _dm334_tp_rank_FULL_Q_WIDTH], pl.BF16] = pl.slice(full_out_weight, [_dm334_tp_rank_HIDDEN, _dm334_tp_rank_FULL_Q_WIDTH], [full * _dm334_tp_rank_HIDDEN, 0])
            norm_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            norm_full = _dm334_rmsnorm_rmsnorm_inline(running, input_gamma_layer, norm_full)
            q_norm_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_FULL_KV_WIDTH], dtype=pl.BF16)
            value_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_FULL_KV_WIDTH], dtype=pl.BF16)
            gate_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_FULL_Q_WIDTH], dtype=pl.BF16)
            (q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full) = _dm334_attention3_prepare_attention3_prepare_inline(norm_full, q_weight_full, k_weight_full, v_weight_full, q_gamma_full, k_gamma_full, cos, sin, q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full)
            key_local = pl.create_tensor([4, _dm334_tp_rank_CONTEXT, _dm334_tp_rank_FULL_HEAD_DIM], dtype=pl.BF16)
            value_local = pl.create_tensor([4, _dm334_tp_rank_CONTEXT, _dm334_tp_rank_FULL_HEAD_DIM], dtype=pl.BF16)
            with pl.spmd(4 * (_dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE), name_hint='qwen_full_load_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // (_dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE)
                logical_block = task % (_dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE)
                physical = pl.read(block_table, [logical_block])
                row = logical_block * _dm334_tp_rank_BLOCK_SIZE
                key_local[head:head + 1, row:row + _dm334_tp_rank_BLOCK_SIZE, 0:_dm334_tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(key_pool, [1, 1, 1, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM])
                value_local[head:head + 1, row:row + _dm334_tp_rank_BLOCK_SIZE, 0:_dm334_tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_pool, [1, 1, 1, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM])
            with pl.spmd(4, name_hint='qwen_full_update_kv'):
                head = pl.tile.get_block_idx()
                start = head * _dm334_tp_rank_FULL_HEAD_DIM
                for token in pl.range(token_count):
                    key_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:_dm334_tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(k_rot, [1, _dm334_tp_rank_FULL_HEAD_DIM], [token, start]), [1, 1, _dm334_tp_rank_FULL_HEAD_DIM])
                    value_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:_dm334_tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_full, [1, _dm334_tp_rank_FULL_HEAD_DIM], [token, start]), [1, 1, _dm334_tp_rank_FULL_HEAD_DIM])
            scores = pl.create_tensor([24 * _dm334_tp_rank_ROWS, _dm334_tp_rank_CONTEXT], dtype=pl.FP32)
            probabilities = pl.create_tensor([24 * _dm334_tp_rank_ROWS, _dm334_tp_rank_CONTEXT], dtype=pl.BF16)
            weighted = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_FULL_Q_WIDTH], dtype=pl.BF16)
            full_gated = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_FULL_Q_WIDTH], dtype=pl.BF16)
            attention_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            (scores, probabilities, weighted, full_gated, attention_full) = _dm334_attention512_core_attention512_core_inline(q_rot, key_local, value_local, gate_full, mask, out_weight_full, scores, probabilities, weighted, full_gated, attention_full)
            residual_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            residual_full = _dm334_residual_residual_add_inline(running, attention_full, residual_full)
            post_norm_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            post_norm_full = _dm334_rmsnorm_rmsnorm_inline(residual_full, post_gamma_layer, post_norm_full)
            mlp_gate_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_up_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_gate_full = _dm334_linear_hidden_projection_inline(post_norm_full, mlp_gate_layer, mlp_gate_full)
            mlp_up_full = _dm334_linear_hidden_projection_inline(post_norm_full, mlp_up_layer, mlp_up_full)
            mlp_middle_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_middle_full = _dm334_swiglu_swiglu_inline(mlp_gate_full, mlp_up_full, mlp_middle_full)
            mlp_output_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            mlp_output_full = _dm334_linear_hidden_projection_inline(mlp_middle_full, mlp_down_layer, mlp_output_full)
            following_full = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
            following_full = _dm334_residual_residual_add_inline(residual_full, mlp_output_full, following_full)
            with pl.spmd(4 * _dm334_tp_rank_ROWS, name_hint='qwen_full_store_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // _dm334_tp_rank_ROWS
                row = task % _dm334_tp_rank_ROWS
                if row < token_count:
                    position = cache_offset + row
                    logical_block = position // _dm334_tp_rank_BLOCK_SIZE
                    physical = pl.read(block_table, [logical_block])
                    slot = position % _dm334_tp_rank_BLOCK_SIZE
                    key_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:_dm334_tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(key_local, [1, 1, _dm334_tp_rank_FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, _dm334_tp_rank_FULL_HEAD_DIM])
                    value_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:_dm334_tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_local, [1, 1, _dm334_tp_rank_FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, _dm334_tp_rank_FULL_HEAD_DIM])
            with pl.spmd(_dm334_tp_rank_ROWS, name_hint='qwen_full_store_attention_hidden'):
                row = pl.tile.get_block_idx()
                next_hidden = pl.slice(following_full, [1, _dm334_tp_rank_HIDDEN], [row, 0])
                running[row:row + 1, 0:_dm334_tp_rank_HIDDEN] = next_hidden
                traces[layer + 1:layer + 2, row:row + 1, 0:_dm334_tp_rank_HIDDEN] = pl.reshape(next_hidden, [1, 1, _dm334_tp_rank_HIDDEN])
    normalized = pl.create_tensor([_dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], dtype=pl.BF16)
    normalized = _dm334_rmsnorm_rmsnorm_inline(running, final_gamma, normalized)
    with pl.spmd(_dm334_tp_rank_ROWS, name_hint='qwen_full_store_final_norm'):
        row = pl.tile.get_block_idx()
        traces[65:66, row:row + 1, 0:_dm334_tp_rank_HIDDEN] = pl.reshape(pl.slice(normalized, [1, _dm334_tp_rank_HIDDEN], [row, 0]), [1, 1, _dm334_tp_rank_HIDDEN])
    logits = _dm334_linear_hidden_projection_inline_nd(normalized, lm_head, logits)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def _dm334_tp_rank_full_short_fwd(input_ids: pl.Tensor[[_dm334_tp_rank_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_tp_rank_VOCAB, _dm334_tp_rank_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_tp_rank_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_tp_rank_TP_VOCAB, _dm334_tp_rank_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_tp_rank_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_tp_rank_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_tp_rank_QKV, _dm334_tp_rank_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_tp_rank_VALUE_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_tp_rank_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_tp_rank_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_tp_rank_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_tp_rank_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_Q_PROJECTION, _dm334_tp_rank_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_tp_rank_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_tp_rank_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_tp_rank_ROWS, _dm334_tp_rank_CONTEXT], pl.FP32], token_count: pl.Scalar[pl.INT32], cache_offset: pl.Scalar[pl.INT32], conv_states: pl.InOut[pl.Tensor[[48, 4, _dm334_tp_rank_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[48, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_tp_rank_POOL_BLOCKS, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_tp_rank_POOL_BLOCKS, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM], pl.BF16]], block_table: pl.Tensor[[_dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[66, _dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[_dm334_tp_rank_ROWS, _dm334_tp_rank_TP_VOCAB], pl.BF16]]):
    return _dm334_tp_rank_full_step_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_count, cache_offset, conv_states, recurrent_states, key_pool, value_pool, block_table, traces, logits)

@pl.jit.inline
def _dm334_tp_rank_batch_step_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    for batch in pl.parallel(_dm334_tp_rank_BATCH):
        if batch < active_count:
            ids_slice = pl.slice(input_ids, [1, _dm334_tp_rank_ROWS], [batch, 0])
            ids = pl.reshape(ids_slice, [_dm334_tp_rank_ROWS])
            cos_slice = pl.slice(cos, [1, _dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], [batch, 0, 0])
            cos_slot = pl.reshape(cos_slice, [_dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY])
            sin_slice = pl.slice(sin, [1, _dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], [batch, 0, 0])
            sin_slot = pl.reshape(sin_slice, [_dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY])
            mask_slice = pl.slice(mask, [1, _dm334_tp_rank_ROWS, _dm334_tp_rank_CONTEXT], [batch, 0, 0])
            mask_slot = pl.reshape(mask_slice, [_dm334_tp_rank_ROWS, _dm334_tp_rank_CONTEXT])
            conv_slice = pl.slice(conv_states, [1, 48, 4, _dm334_tp_rank_QKV], [batch, 0, 0, 0])
            conv_slot = pl.reshape(conv_slice, [48, 4, _dm334_tp_rank_QKV])
            recurrent_slice = pl.slice(recurrent_states, [1, 48, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], [batch, 0, 0, 0, 0])
            recurrent_slot = pl.reshape(recurrent_slice, [48, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM])
            table_slice = pl.slice(block_tables, [1, _dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE], [batch, 0])
            table_slot = pl.reshape(table_slice, [_dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE])
            trace_slice = pl.slice(traces, [1, 66, _dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], [batch, 0, 0, 0])
            trace_slot = pl.reshape(trace_slice, [66, _dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN])
            logits_slice = pl.slice(logits, [1, _dm334_tp_rank_ROWS, _dm334_tp_rank_TP_VOCAB], [batch, 0, 0])
            logits_slot = pl.reshape(logits_slice, [_dm334_tp_rank_ROWS, _dm334_tp_rank_TP_VOCAB])
            _dm334_tp_rank_full_step_inline(ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos_slot, sin_slot, mask_slot, pl.read(token_counts, [batch]), pl.read(cache_offsets, [batch]), conv_slot, recurrent_slot, key_pool, value_pool, table_slot, trace_slot, logits_slot)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def _dm334_tp_rank_prefill_fwd(input_ids: pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_tp_rank_VOCAB, _dm334_tp_rank_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_tp_rank_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_tp_rank_TP_VOCAB, _dm334_tp_rank_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_tp_rank_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_tp_rank_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_tp_rank_QKV, _dm334_tp_rank_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_tp_rank_VALUE_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_tp_rank_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_tp_rank_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_tp_rank_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_tp_rank_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_Q_PROJECTION, _dm334_tp_rank_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_tp_rank_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_tp_rank_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_ROWS, _dm334_tp_rank_CONTEXT], pl.FP32], token_counts: pl.Tensor[[_dm334_tp_rank_BATCH], pl.INT32], cache_offsets: pl.Tensor[[_dm334_tp_rank_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[_dm334_tp_rank_BATCH, 48, 4, _dm334_tp_rank_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_dm334_tp_rank_BATCH, 48, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_tp_rank_POOL_BLOCKS, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_tp_rank_POOL_BLOCKS, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[_dm334_tp_rank_BATCH, 66, _dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_ROWS, _dm334_tp_rank_TP_VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    for batch in pl.range(_dm334_tp_rank_BATCH):
        if batch < active_count:
            ids_slice = pl.slice(input_ids, [1, _dm334_tp_rank_ROWS], [batch, 0])
            ids = pl.reshape(ids_slice, [_dm334_tp_rank_ROWS])
            cos_slice = pl.slice(cos, [1, _dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], [batch, 0, 0])
            cos_slot = pl.reshape(cos_slice, [_dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY])
            sin_slice = pl.slice(sin, [1, _dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], [batch, 0, 0])
            sin_slot = pl.reshape(sin_slice, [_dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY])
            mask_slice = pl.slice(mask, [1, _dm334_tp_rank_ROWS, _dm334_tp_rank_CONTEXT], [batch, 0, 0])
            mask_slot = pl.reshape(mask_slice, [_dm334_tp_rank_ROWS, _dm334_tp_rank_CONTEXT])
            conv_slice = pl.slice(conv_states, [1, 48, 4, _dm334_tp_rank_QKV], [batch, 0, 0, 0])
            conv_slot = pl.reshape(conv_slice, [48, 4, _dm334_tp_rank_QKV])
            recurrent_slice = pl.slice(recurrent_states, [1, 48, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], [batch, 0, 0, 0, 0])
            recurrent_slot = pl.reshape(recurrent_slice, [48, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM])
            table_slice = pl.slice(block_tables, [1, _dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE], [batch, 0])
            table_slot = pl.reshape(table_slice, [_dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE])
            trace_slice = pl.slice(traces, [1, 66, _dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], [batch, 0, 0, 0])
            trace_slot = pl.reshape(trace_slice, [66, _dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN])
            logits_slice = pl.slice(logits, [1, _dm334_tp_rank_ROWS, _dm334_tp_rank_TP_VOCAB], [batch, 0, 0])
            logits_slot = pl.reshape(logits_slice, [_dm334_tp_rank_ROWS, _dm334_tp_rank_TP_VOCAB])
            (conv_slot, recurrent_slot, key_pool, value_pool, trace_slot, logits_slot) = _dm334_tp_rank_full_step_inline(ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos_slot, sin_slot, mask_slot, pl.read(token_counts, [batch]), pl.read(cache_offsets, [batch]), conv_slot, recurrent_slot, key_pool, value_pool, table_slot, trace_slot, logits_slot)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def _dm334_tp_rank_decode_fwd(input_ids: pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_tp_rank_VOCAB, _dm334_tp_rank_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_tp_rank_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_tp_rank_TP_VOCAB, _dm334_tp_rank_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_tp_rank_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_tp_rank_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_tp_rank_INTERMEDIATE, _dm334_tp_rank_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_tp_rank_QKV, _dm334_tp_rank_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_tp_rank_VALUE_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_tp_rank_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_tp_rank_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_tp_rank_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_tp_rank_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_Q_PROJECTION, _dm334_tp_rank_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_tp_rank_FULL_KV_WIDTH, _dm334_tp_rank_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_tp_rank_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_tp_rank_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_tp_rank_HIDDEN, _dm334_tp_rank_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_ROWS, _dm334_tp_rank_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_ROWS, _dm334_tp_rank_CONTEXT], pl.FP32], token_counts: pl.Tensor[[_dm334_tp_rank_BATCH], pl.INT32], cache_offsets: pl.Tensor[[_dm334_tp_rank_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[_dm334_tp_rank_BATCH, 48, 4, _dm334_tp_rank_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_dm334_tp_rank_BATCH, 48, _dm334_tp_rank_DELTA_HEADS, _dm334_tp_rank_DELTA_DIM, _dm334_tp_rank_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_tp_rank_POOL_BLOCKS, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_tp_rank_POOL_BLOCKS, _dm334_tp_rank_BLOCK_SIZE, _dm334_tp_rank_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_CONTEXT // _dm334_tp_rank_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[_dm334_tp_rank_BATCH, 66, _dm334_tp_rank_ROWS, _dm334_tp_rank_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[_dm334_tp_rank_BATCH, _dm334_tp_rank_ROWS, _dm334_tp_rank_TP_VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    (conv_states, recurrent_states, key_pool, value_pool, traces, logits) = _dm334_tp_rank_batch_step_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit.host
def _dm334_tp_prefill_fwd(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.InOut[pl.Tensor], recurrent_states: pl.InOut[pl.Tensor], key_pool: pl.InOut[pl.Tensor], value_pool: pl.InOut[pl.Tensor], block_tables: pl.Tensor, traces: pl.Out[pl.Tensor], logits: pl.Out[pl.Tensor], active_count: pl.Scalar[pl.INT32]):
    for rank in pl.range(pld.world_size()):
        _dm334_tp_rank_prefill_fwd(input_ids[rank], embedding[rank], final_gamma[rank], lm_head[rank], input_gamma[rank], post_gamma[rank], mlp_gate_weight[rank], mlp_up_weight[rank], mlp_down_weight[rank], delta_qkv_weight[rank], delta_z_weight[rank], delta_a_weight[rank], delta_b_weight[rank], delta_conv_weight[rank], delta_a_log[rank], delta_dt_bias[rank], delta_norm_weight[rank], delta_out_weight[rank], full_q_weight[rank], full_k_weight[rank], full_v_weight[rank], full_q_gamma[rank], full_k_gamma[rank], full_out_weight[rank], cos[rank], sin[rank], mask[rank], token_counts[rank], cache_offsets[rank], conv_states[rank], recurrent_states[rank], key_pool[rank], value_pool[rank], block_tables[rank], traces[rank], logits[rank], active_count, device=rank)

@pl.jit.host
def _dm334_tp_decode_fwd(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.InOut[pl.Tensor], recurrent_states: pl.InOut[pl.Tensor], key_pool: pl.InOut[pl.Tensor], value_pool: pl.InOut[pl.Tensor], block_tables: pl.Tensor, traces: pl.Out[pl.Tensor], logits: pl.Out[pl.Tensor], active_count: pl.Scalar[pl.INT32]):
    for rank in pl.range(pld.world_size()):
        _dm334_tp_rank_decode_fwd(input_ids[rank], embedding[rank], final_gamma[rank], lm_head[rank], input_gamma[rank], post_gamma[rank], mlp_gate_weight[rank], mlp_up_weight[rank], mlp_down_weight[rank], delta_qkv_weight[rank], delta_z_weight[rank], delta_a_weight[rank], delta_b_weight[rank], delta_conv_weight[rank], delta_a_log[rank], delta_dt_bias[rank], delta_norm_weight[rank], delta_out_weight[rank], full_q_weight[rank], full_k_weight[rank], full_v_weight[rank], full_q_gamma[rank], full_k_gamma[rank], full_out_weight[rank], cos[rank], sin[rank], mask[rank], token_counts[rank], cache_offsets[rank], conv_states[rank], recurrent_states[rank], key_pool[rank], value_pool[rank], block_tables[rank], traces[rank], logits[rank], active_count, device=rank)

@pl.program
class CandidateExchange:

    @pl.function(type=pl.FunctionType.InCore)
    def exchange(self, local: pl.Tensor[[1, 8], pl.FP32], result: pl.Out[pl.Tensor[[1, 16], pl.FP32]], data: pl.InOut[pld.DistributedTensor[[2, 8], pl.FP32]], signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]]) -> pl.Tensor[[1, 16], pl.FP32]:
        data = pld.tensor.allgather(local, data, signal)
        for rank in pl.range(2):
            chunk = pl.load(data, [rank, 0], [1, 8])
            result = pl.store(chunk, [0, rank * 8], result)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def rank_orch(self, local: pl.Tensor[[1, 8], pl.FP32], result: pl.Out[pl.Tensor[[1, 16], pl.FP32]], data: pl.InOut[pld.DistributedTensor[[2, 8], pl.FP32]], signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]]) -> pl.Tensor[[1, 16], pl.FP32]:
        return self.exchange(local, result, data, signal)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(self, candidates: pl.Tensor[[2, 1, 8], pl.FP32], gathered: pl.Out[pl.Tensor[[2, 1, 16], pl.FP32]]) -> pl.Tensor[[2, 1, 16], pl.FP32]:
        data_buffer = pld.alloc_window_buffer(2 * 8 * pl.FP32.get_byte())
        signal_buffer = pld.alloc_window_buffer(2 * pl.INT32.get_byte())
        for rank in pl.range(pld.world_size()):
            data = pld.window(data_buffer, [2, 8], dtype=pl.FP32)
            signal = pld.window(signal_buffer, [2, 1], dtype=pl.INT32)
            self.rank_orch(candidates[rank], gathered[rank], data, signal, device=rank)
        return gathered

@pl.jit.host
def _dm334_prefill_fwd(input_ids: pl.Tensor[[_dm334_BATCH, _dm334_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_QKV, _dm334_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_FULL_Q_PROJECTION, _dm334_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_HIDDEN, _dm334_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_CONTEXT], pl.FP32], token_counts: pl.Tensor[[_dm334_BATCH], pl.INT32], cache_offsets: pl.Tensor[[_dm334_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, 4, _dm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[_dm334_BATCH, _dm334_CONTEXT // _dm334_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[_dm334_BATCH, 66, _dm334_ROWS, _dm334_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    _dm334_rank_prefill_fwd(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, device=0)

@pl.jit.host
def _dm334_decode_fwd(input_ids: pl.Tensor[[_dm334_BATCH, _dm334_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64, _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16, pl.NZ], mlp_up_weight: pl.Tensor[[64, _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16, pl.NZ], mlp_down_weight: pl.Tensor[[64, _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16, pl.NZ], delta_qkv_weight: pl.Tensor[[48, _dm334_HIDDEN, _dm334_QKV], pl.BF16, pl.NZ], delta_z_weight: pl.Tensor[[48, _dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16, pl.NZ], delta_a_weight: pl.Tensor[[48, _dm334_HIDDEN, 128], pl.BF16, pl.NZ], delta_b_weight: pl.Tensor[[48, _dm334_HIDDEN, 128], pl.BF16, pl.NZ], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48, _dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16, pl.NZ], full_q_weight: pl.Tensor[[16, _dm334_HIDDEN, _dm334_FULL_Q_PROJECTION], pl.BF16, pl.NZ], full_k_weight: pl.Tensor[[16, _dm334_HIDDEN, _dm334_FULL_KV_WIDTH], pl.BF16, pl.NZ], full_v_weight: pl.Tensor[[16, _dm334_HIDDEN, _dm334_FULL_KV_WIDTH], pl.BF16, pl.NZ], full_q_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16, _dm334_FULL_Q_WIDTH, _dm334_HIDDEN], pl.BF16, pl.NZ], cos: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_CONTEXT], pl.FP32], token_counts: pl.Tensor[[_dm334_BATCH], pl.INT32], cache_offsets: pl.Tensor[[_dm334_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, 4, _dm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[_dm334_BATCH, _dm334_CONTEXT // _dm334_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[_dm334_BATCH, 66, _dm334_ROWS, _dm334_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    _dm334_rank_decode_fwd(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, device=0)

@pl.jit.inline
def _dm334_linear_hidden_projection_inline_nd(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Tensor):
    output_columns = pl.tensor.dim(weight, 0)
    input_columns = pl.tensor.dim(weight, 1)
    with pl.spmd(output_columns // _dm334_linear_OUTPUT_TILE, name_hint='qwen_hidden_projection'):
        start = pl.tile.get_block_idx() * _dm334_linear_OUTPUT_TILE
        accumulator = pl.create_tensor([_dm334_linear_ROWS, _dm334_linear_OUTPUT_TILE], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // _dm334_linear_K_TILE, stage=2):
            k_start = chunk * _dm334_linear_K_TILE
            hidden_tile = pl.slice(hidden, [_dm334_linear_ROWS, _dm334_linear_K_TILE], [0, k_start])
            weight_tile = pl.slice(weight, [_dm334_linear_OUTPUT_TILE, _dm334_linear_K_TILE], [start, k_start])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=True, init_cond=chunk == 0)
        result[0:_dm334_linear_ROWS, start:start + _dm334_linear_OUTPUT_TILE] = pl.cast(accumulator, pl.BF16, mode='rint')
    return result

@pl.jit.inline
def _dm334_linear_hidden_projection_gate_inline(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Tensor):
    output_columns = pl.tensor.dim(result, 1)
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(output_columns // 128, name_hint='qwen_nz_gate_projection'):
        start = pl.tile.get_block_idx() * 128
        accumulator = pl.create_tensor([_dm334_linear_ROWS, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            hidden_tile = pl.slice(hidden, [_dm334_linear_ROWS, 256], [0, k_start])
            weight_tile = pl.slice(weight, [256, 128], [k_start, start])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
        result[0:_dm334_linear_ROWS, start:start + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return result
from ..sampling import BATCH as _dm334_combined_sampler_BATCH, VOCAB as _dm334_combined_sampler_VOCAB, CHUNKS as _dm334_combined_sampler_CHUNKS, FULL_CHUNKS as _dm334_combined_sampler_FULL_CHUNKS, CHUNK as _dm334_combined_sampler_CHUNK, PARTIAL_COLUMNS as _dm334_combined_sampler_PARTIAL_COLUMNS, SENTINEL as _dm334_combined_sampler_SENTINEL, NEGATIVE_FINITE as _dm334_combined_sampler_NEGATIVE_FINITE, greedy_partial as _dm334_combined_sampler_greedy_partial, greedy_partial_tail as _dm334_combined_sampler_greedy_partial_tail

@pl.jit.inline
def _dm334_combined_greedy_inline(logits: pl.Tensor, counts: pl.Tensor, maxima: pl.InOut[pl.Tensor], indices: pl.InOut[pl.Tensor], nan_indices: pl.InOut[pl.Tensor], tokens: pl.InOut[pl.Tensor], active: pl.Scalar[pl.INT32]):
    rows = pl.tensor.dim(logits, 1)
    flat = pl.reshape(logits, [_dm334_combined_sampler_BATCH * rows, _dm334_combined_sampler_VOCAB])
    with pl.spmd(_dm334_combined_sampler_BATCH * _dm334_combined_sampler_CHUNKS, name_hint='qwen_greedy_partials') as partials:
        block = pl.tile.get_block_idx()
        slot = block // _dm334_combined_sampler_CHUNKS
        part = block % _dm334_combined_sampler_CHUNKS
        if slot > 0 and slot < active:
            row = pl.read(counts, [slot]) - 1
            if part < _dm334_combined_sampler_FULL_CHUNKS:
                _dm334_combined_sampler_greedy_partial(flat, maxima, indices, nan_indices, slot, part, part * _dm334_combined_sampler_CHUNK, slot * rows + row)
            elif part == _dm334_combined_sampler_FULL_CHUNKS:
                _dm334_combined_sampler_greedy_partial_tail(flat, maxima, indices, nan_indices, slot, part, _dm334_combined_sampler_FULL_CHUNKS * _dm334_combined_sampler_CHUNK, slot * rows + row)
            else:
                pl.store(pl.tile.muls(pl.tile.full([1, 8], dtype=pl.FP32, value=_dm334_combined_sampler_NEGATIVE_FINITE), 4.0), [slot, part * 8], maxima)
                pl.store(pl.tile.full([1, 8], dtype=pl.FP32, value=_dm334_combined_sampler_SENTINEL), [slot, part * 8], indices)
                pl.store(pl.tile.full([1, 8], dtype=pl.FP32, value=_dm334_combined_sampler_SENTINEL), [slot, part * 8], nan_indices)
    with pl.spmd(_dm334_combined_sampler_BATCH, name_hint='qwen_greedy_finalize', deps=[partials]):
        final_slot = pl.tile.get_block_idx()
        if final_slot > 0 and final_slot < active:
            part_max = pl.tile.reshape(pl.load(maxima, [final_slot, 0], [1, _dm334_combined_sampler_PARTIAL_COLUMNS]), [8, 64])
            part_index = pl.tile.reshape(pl.load(indices, [final_slot, 0], [1, _dm334_combined_sampler_PARTIAL_COLUMNS]), [8, 64])
            part_nan = pl.tile.reshape(pl.load(nan_indices, [final_slot, 0], [1, _dm334_combined_sampler_PARTIAL_COLUMNS]), [8, 64])
            final_scratch = pl.tile.create([8, 64], dtype=pl.FP32)
            row_maxima = pl.tile.row_max(part_max, final_scratch)
            final_max = pl.tile.col_max(pl.tile.row_expand(pl.tile.create([8, 8], dtype=pl.FP32), row_maxima))
            final_max_column = pl.tile.reshape(final_max, [8, 1])
            final_expanded = pl.tile.row_expand(pl.tile.create([8, 64], dtype=pl.FP32), final_max_column)
            row_indices = pl.tile.row_min(pl.tile.select(pl.tile.cmp(part_max, final_expanded, cmp_type=0), part_index, _dm334_combined_sampler_SENTINEL), final_scratch)
            final_index = pl.tile.col_min(pl.tile.row_expand(pl.tile.create([8, 8], dtype=pl.FP32), row_indices))
            row_nan = pl.tile.row_min(part_nan, final_scratch)
            final_nan = pl.tile.col_min(pl.tile.row_expand(pl.tile.create([8, 8], dtype=pl.FP32), row_nan))
            selected = pl.tile.select(pl.tile.cmps(final_nan, _dm334_combined_sampler_SENTINEL, cmp_type=2), final_nan, final_index)
            pl.store(pl.tile.cast(selected, pl.INT32, mode='rint'), [final_slot, 0], tokens)
    return (maxima, indices, nan_indices, tokens)

@pl.jit
def _dm334_rank_decode_greedy_fwd(input_ids: pl.Tensor[[_dm334_BATCH, _dm334_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_QKV, _dm334_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_FULL_Q_PROJECTION, _dm334_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_HIDDEN, _dm334_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_CONTEXT], pl.FP32], token_counts: pl.Tensor[[_dm334_BATCH], pl.INT32], cache_offsets: pl.Tensor[[_dm334_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, 4, _dm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[_dm334_BATCH, _dm334_CONTEXT // _dm334_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[_dm334_BATCH, 66, _dm334_ROWS, _dm334_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_VOCAB], pl.BF16]], maxima: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], indices: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], nan_indices: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], selected_tokens: pl.InOut[pl.Tensor[[_dm334_BATCH, 8], pl.INT32]], active_count: pl.Scalar[pl.INT32]):
    (conv_states, recurrent_states, key_pool, value_pool, traces, logits) = _dm334_batch_step_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    _dm334_combined_greedy_inline(logits, token_counts, maxima, indices, nan_indices, selected_tokens, active_count)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits, maxima, indices, nan_indices, selected_tokens)

@pl.jit.host
def _dm334_decode_greedy_fwd(input_ids: pl.Tensor[[_dm334_BATCH, _dm334_ROWS], pl.INT32], embedding: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64, _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16, pl.NZ], mlp_up_weight: pl.Tensor[[64, _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16, pl.NZ], mlp_down_weight: pl.Tensor[[64, _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16, pl.NZ], delta_qkv_weight: pl.Tensor[[48, _dm334_HIDDEN, _dm334_QKV], pl.BF16, pl.NZ], delta_z_weight: pl.Tensor[[48, _dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16, pl.NZ], delta_a_weight: pl.Tensor[[48, _dm334_HIDDEN, 128], pl.BF16, pl.NZ], delta_b_weight: pl.Tensor[[48, _dm334_HIDDEN, 128], pl.BF16, pl.NZ], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48, _dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16, pl.NZ], full_q_weight: pl.Tensor[[16, _dm334_HIDDEN, _dm334_FULL_Q_PROJECTION], pl.BF16, pl.NZ], full_k_weight: pl.Tensor[[16, _dm334_HIDDEN, _dm334_FULL_KV_WIDTH], pl.BF16, pl.NZ], full_v_weight: pl.Tensor[[16, _dm334_HIDDEN, _dm334_FULL_KV_WIDTH], pl.BF16, pl.NZ], full_q_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16, _dm334_FULL_Q_WIDTH, _dm334_HIDDEN], pl.BF16, pl.NZ], cos: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], sin: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], mask: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_CONTEXT], pl.FP32], token_counts: pl.Tensor[[_dm334_BATCH], pl.INT32], cache_offsets: pl.Tensor[[_dm334_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, 4, _dm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[_dm334_BATCH, _dm334_CONTEXT // _dm334_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[_dm334_BATCH, 66, _dm334_ROWS, _dm334_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_VOCAB], pl.BF16]], maxima: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], indices: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], nan_indices: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], selected_tokens: pl.InOut[pl.Tensor[[_dm334_BATCH, 8], pl.INT32]], active_count: pl.Scalar[pl.INT32]):
    _dm334_rank_decode_greedy_fwd(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, maxima, indices, nan_indices, selected_tokens, active_count, device=0)

@pl.jit
def rank_mixed334(embedding: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * _dm334_QKV, _dm334_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * _dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, _dm334_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * _dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * _dm334_FULL_Q_PROJECTION, _dm334_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * _dm334_FULL_KV_WIDTH, _dm334_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * _dm334_HIDDEN, _dm334_FULL_Q_WIDTH], pl.BF16], conv_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, 4, _dm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], pm_input_ids: pl.Tensor[[_pm334_BATCH, _pm334_ROWS], pl.INT32], pm_cos: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_ROTARY], pl.BF16], pm_sin: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_ROTARY], pl.BF16], pm_mask: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_CONTEXT], pl.FP32], pm_token_counts: pl.Tensor[[_pm334_BATCH], pl.INT32], pm_cache_offsets: pl.Tensor[[_pm334_BATCH], pl.INT32], pm_block_tables: pl.Tensor[[_pm334_BATCH, _pm334_CONTEXT // _pm334_BLOCK_SIZE], pl.INT32], pm_traces: pl.Out[pl.Tensor[[_pm334_BATCH, 66, _pm334_ROWS, _pm334_HIDDEN], pl.BF16]], pm_logits: pl.InOut[pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_VOCAB], pl.BF16]], dm_input_ids: pl.Tensor[[_dm334_BATCH, _dm334_ROWS], pl.INT32], dm_cos: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], dm_sin: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], dm_mask: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_CONTEXT], pl.FP32], dm_token_counts: pl.Tensor[[_dm334_BATCH], pl.INT32], dm_cache_offsets: pl.Tensor[[_dm334_BATCH], pl.INT32], dm_block_tables: pl.Tensor[[_dm334_BATCH, _dm334_CONTEXT // _dm334_BLOCK_SIZE], pl.INT32], dm_traces: pl.Out[pl.Tensor[[_dm334_BATCH, 66, _dm334_ROWS, _dm334_HIDDEN], pl.BF16]], dm_logits: pl.Out[pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_VOCAB], pl.BF16]], dm_maxima: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], dm_indices: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], dm_nan_indices: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], dm_selected_tokens: pl.InOut[pl.Tensor[[_dm334_BATCH, 8], pl.INT32]], active_count: pl.Scalar[pl.INT32]):
    pm_running = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
    with pl.spmd(_pm334_PACKED, name_hint='qwen_prefill_embedding'):
        pm_row = pl.tile.get_block_idx()
        pm_batch = pm_row // _pm334_ROWS
        pm_token_row = pm_row % _pm334_ROWS
        if pm_batch < 1:
            pm_count = pl.read(pm_token_counts, [pm_batch])
            if pm_token_row < pm_count:
                pm_token = pl.read(pm_input_ids, [pm_batch, pm_token_row])
                pm_embedded = pl.slice(embedding, [1, _pm334_HIDDEN], [pm_token, 0])
            else:
                pm_embedded = pl.full([1, _pm334_HIDDEN], dtype=pl.BF16, value=0.0)
            pm_running[pm_row:pm_row + 1, 0:_pm334_HIDDEN] = pm_embedded
            pm_traces[pm_batch:pm_batch + 1, 0:1, pm_token_row:pm_token_row + 1, 0:_pm334_HIDDEN] = pl.reshape(pm_embedded, [1, 1, 1, _pm334_HIDDEN])
        else:
            pm_running[pm_row:pm_row + 1, 0:_pm334_HIDDEN] = pl.full([1, _pm334_HIDDEN], dtype=pl.BF16, value=0.0)
    pm_norm = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
    pm_first_gamma: pl.Tensor[[1, _pm334_HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, _pm334_HIDDEN], [0, 0])
    pm_norm = _pm334_prefill_norm_inline_prefix512(pm_running, pm_first_gamma, pm_norm)
    dm_running = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
    dm_cos_batch = pl.create_tensor([_dm334_BATCH, _dm334_ROTARY], dtype=pl.BF16)
    dm_sin_batch = pl.create_tensor([_dm334_BATCH, _dm334_ROTARY], dtype=pl.BF16)
    with pl.spmd(_dm334_BATCH, name_hint='qwen_decode_pack_rope'):
        dm_batch = pl.tile.get_block_idx()
        dm_cos_batch[dm_batch:dm_batch + 1, 0:_dm334_ROTARY] = pl.reshape(pl.slice(dm_cos, [1, 1, _dm334_ROTARY], [dm_batch, 0, 0]), [1, _dm334_ROTARY])
        dm_sin_batch[dm_batch:dm_batch + 1, 0:_dm334_ROTARY] = pl.reshape(pl.slice(dm_sin, [1, 1, _dm334_ROTARY], [dm_batch, 0, 0]), [1, _dm334_ROTARY])
    with pl.spmd(_dm334_BATCH * (_dm334_HIDDEN // 256), name_hint='qwen_decode_token_embedding'):
        dm_task = pl.tile.get_block_idx()
        dm_batch = dm_task // (_dm334_HIDDEN // 256)
        dm_col = dm_task % (_dm334_HIDDEN // 256) * 256
        if dm_batch > 0 and dm_batch < active_count:
            dm_token = pl.read(dm_input_ids, [dm_batch, 0])
            dm_embedded = pl.slice(embedding, [1, 256], [dm_token, dm_col])
            dm_running[dm_batch:dm_batch + 1, dm_col:dm_col + 256] = dm_embedded
            dm_traces[dm_batch:dm_batch + 1, 0:1, 0:1, dm_col:dm_col + 256] = pl.reshape(dm_embedded, [1, 1, 1, 256])
        else:
            dm_running[dm_batch:dm_batch + 1, dm_col:dm_col + 256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    for layer in pl.range(64):
        pm_post_gamma_layer: pl.Tensor[[1, _pm334_HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, _pm334_HIDDEN], [layer, 0])
        pm_mlp_gate_layer: pl.Tensor[[_pm334_HIDDEN, _pm334_INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        pm_mlp_up_layer: pl.Tensor[[_pm334_HIDDEN, _pm334_INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        pm_mlp_down_layer: pl.Tensor[[_pm334_INTERMEDIATE, _pm334_HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        pm_residual = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
        pm_post_norm = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
        if layer % 4 < 3:
            pm_linear = layer - layer // 4
            pm_qkv_weight_layer: pl.Tensor[[_pm334_HIDDEN, _pm334_QKV], pl.BF16, pl.NZ] = delta_qkv_weight[pm_linear]
            pm_z_weight_layer: pl.Tensor[[_pm334_HIDDEN, _pm334_VALUE_WIDTH], pl.BF16, pl.NZ] = delta_z_weight[pm_linear]
            pm_a_weight_layer: pl.Tensor[[_pm334_HIDDEN, 128], pl.BF16, pl.NZ] = delta_a_weight[pm_linear]
            pm_b_weight_layer: pl.Tensor[[_pm334_HIDDEN, 128], pl.BF16, pl.NZ] = delta_b_weight[pm_linear]
            pm_conv_weight_layer: pl.Tensor[[4, _pm334_QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, _pm334_QKV], [pm_linear * 4, 0])
            pm_a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [pm_linear, 0])
            pm_dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [pm_linear, 0])
            pm_delta_norm_layer: pl.Tensor[[1, _pm334_DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, _pm334_DELTA_DIM], [pm_linear, 0])
            pm_delta_out_layer: pl.Tensor[[_pm334_VALUE_WIDTH, _pm334_HIDDEN], pl.BF16, pl.NZ] = delta_out_weight[pm_linear]
            pm_qkv = pl.create_tensor([_pm334_PACKED, _pm334_QKV], dtype=pl.BF16)
            pm_z = pl.create_tensor([_pm334_PACKED, _pm334_VALUE_WIDTH], dtype=pl.BF16)
            pm_a = pl.create_tensor([_pm334_PACKED, 128], dtype=pl.BF16)
            pm_b = pl.create_tensor([_pm334_PACKED, 128], dtype=pl.BF16)
            pm_qkv = _pm334_prefill_project_ffn_up_paired_m128(pm_norm, pm_qkv_weight_layer, pm_qkv, 1)
            pm_z = _pm334_prefill_project_ffn_up_paired_m128(pm_norm, pm_z_weight_layer, pm_z, 1)
            pm_a = _pm334_prefill_project_inline_prefix512(pm_norm, pm_a_weight_layer, pm_a, 1)
            pm_b = _pm334_prefill_project_inline_prefix512(pm_norm, pm_b_weight_layer, pm_b, 1)
            pm_gated = pl.create_tensor([_pm334_PACKED, _pm334_VALUE_WIDTH], dtype=pl.BF16)
            _pm334_prefill_delta_slots_inline_prefix512(pm_qkv, pm_z, pm_a, pm_b, pm_conv_weight_layer, pm_a_log_layer, pm_dt_bias_layer, pm_delta_norm_layer, pm_delta_out_layer, conv_states, recurrent_states, pm_gated, pm_linear, pm_token_counts, 1)
            pm_attention = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
            pm_attention = _pm334_prefill_project_ffn_up_paired_m128(pm_gated, pm_delta_out_layer, pm_attention, 1)
            (pm_residual, pm_post_norm) = _pm334_residual_norm_fused_streaming(pm_running, pm_attention, pm_post_gamma_layer, pm_residual, pm_post_norm)
        else:
            pm_full = layer // 4
            pm_q_weight_full: pl.Tensor[[_pm334_HIDDEN, _pm334_FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[pm_full]
            pm_k_weight_full: pl.Tensor[[_pm334_HIDDEN, _pm334_FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[pm_full]
            pm_v_weight_full: pl.Tensor[[_pm334_HIDDEN, _pm334_FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[pm_full]
            pm_q_gamma_full: pl.Tensor[[1, _pm334_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, _pm334_FULL_HEAD_DIM], [pm_full, 0])
            pm_k_gamma_full: pl.Tensor[[1, _pm334_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, _pm334_FULL_HEAD_DIM], [pm_full, 0])
            pm_out_weight_full: pl.Tensor[[_pm334_FULL_Q_WIDTH, _pm334_HIDDEN], pl.BF16, pl.NZ] = full_out_weight[pm_full]
            pm_q = pl.create_tensor([_pm334_PACKED, _pm334_FULL_Q_PROJECTION], dtype=pl.BF16)
            pm_k = pl.create_tensor([_pm334_PACKED, _pm334_FULL_KV_WIDTH], dtype=pl.BF16)
            pm_v = pl.create_tensor([_pm334_PACKED, _pm334_FULL_KV_WIDTH], dtype=pl.BF16)
            pm_q = _pm334_prefill_project_ffn_up_paired_m128(pm_norm, pm_q_weight_full, pm_q, 1)
            pm_k = _pm334_prefill_project_inline_prefix512(pm_norm, pm_k_weight_full, pm_k, 1)
            pm_v = _pm334_prefill_project_inline_prefix512(pm_norm, pm_v_weight_full, pm_v, 1)
            pm_gated = pl.create_tensor([_pm334_PACKED, _pm334_FULL_Q_WIDTH], dtype=pl.BF16)
            _pm334_prefill_attention_slots_inline_prefix512(pm_q, pm_k, pm_v, pm_q_gamma_full, pm_k_gamma_full, pm_cos, pm_sin, pm_mask, pm_out_weight_full, key_pool, value_pool, pm_block_tables, pm_cache_offsets, pm_token_counts, pm_gated, pm_full, 1)
            pm_attention = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
            pm_attention = _pm334_prefill_project_ffn_up_paired_m128(pm_gated, pm_out_weight_full, pm_attention, 1)
            (pm_residual, pm_post_norm) = _pm334_residual_norm_fused_streaming(pm_running, pm_attention, pm_post_gamma_layer, pm_residual, pm_post_norm)
        pm_middle = pl.create_tensor([_pm334_PACKED, _pm334_INTERMEDIATE], dtype=pl.BF16)
        pm_middle = _ffn369_gate_up_swiglu_shared_activation(pm_post_norm, pm_mlp_gate_layer, pm_mlp_up_layer, pm_middle)
        pm_mlp_output = pl.create_tensor([_pm334_PACKED, _pm334_HIDDEN], dtype=pl.BF16)
        pm_mlp_output = _pm334_prefill_project_ffn_up_paired_m128(pm_middle, pm_mlp_down_layer, pm_mlp_output, 1)
        (pm_running, pm_norm, pm_traces) = _pm334_residual_next_norm_trace(pm_residual, pm_mlp_output, input_gamma, final_gamma, pm_running, pm_norm, pm_traces, layer, 1)
        dm_input_gamma_layer: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, _dm334_HIDDEN], [layer, 0])
        dm_post_gamma_layer: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, _dm334_HIDDEN], [layer, 0])
        dm_mlp_gate_layer: pl.Tensor[[_dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        dm_mlp_up_layer: pl.Tensor[[_dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        dm_mlp_down_layer: pl.Tensor[[_dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        dm_norm = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
        dm_norm = _dm334_rmsnorm_rmsnorm_inline(dm_running, dm_input_gamma_layer, dm_norm)
        dm_residual = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
        dm_post_norm = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
        if layer % 4 < 3:
            dm_linear = layer - layer // 4
            dm_qkv_weight_layer: pl.Tensor[[_dm334_HIDDEN, _dm334_QKV], pl.BF16, pl.NZ] = delta_qkv_weight[dm_linear]
            dm_z_weight_layer: pl.Tensor[[_dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16, pl.NZ] = delta_z_weight[dm_linear]
            dm_a_weight_layer: pl.Tensor[[_dm334_HIDDEN, 128], pl.BF16, pl.NZ] = delta_a_weight[dm_linear]
            dm_b_weight_layer: pl.Tensor[[_dm334_HIDDEN, 128], pl.BF16, pl.NZ] = delta_b_weight[dm_linear]
            dm_conv_weight_layer: pl.Tensor[[4, _dm334_QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, _dm334_QKV], [dm_linear * 4, 0])
            dm_a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [dm_linear, 0])
            dm_dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [dm_linear, 0])
            dm_delta_norm_layer: pl.Tensor[[1, _dm334_DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, _dm334_DELTA_DIM], [dm_linear, 0])
            dm_delta_out_layer: pl.Tensor[[_dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16, pl.NZ] = delta_out_weight[dm_linear]
            dm_qkv = pl.create_tensor([_dm334_BATCH, _dm334_QKV], dtype=pl.BF16)
            dm_z = pl.create_tensor([_dm334_BATCH, _dm334_VALUE_WIDTH], dtype=pl.BF16)
            dm_a = pl.create_tensor([_dm334_BATCH, 128], dtype=pl.BF16)
            dm_b = pl.create_tensor([_dm334_BATCH, 128], dtype=pl.BF16)
            dm_qkv = _dm334_linear_hidden_projection_inline(dm_norm, dm_qkv_weight_layer, dm_qkv)
            dm_z = _dm334_linear_hidden_projection_inline(dm_norm, dm_z_weight_layer, dm_z)
            dm_a = _dm334_linear_hidden_projection_gate_inline(dm_norm, dm_a_weight_layer, dm_a)
            dm_b = _dm334_linear_hidden_projection_gate_inline(dm_norm, dm_b_weight_layer, dm_b)
            dm_gated_delta = pl.create_tensor([_dm334_BATCH, _dm334_VALUE_WIDTH], dtype=pl.BF16)
            _dm334_decode_delta_slots_inline(dm_qkv, dm_z, dm_a, dm_b, dm_conv_weight_layer, dm_a_log_layer, dm_dt_bias_layer, dm_delta_norm_layer, dm_delta_out_layer, conv_states, recurrent_states, dm_gated_delta, dm_linear, active_count)
            dm_attention_delta = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
            dm_attention_delta = _dm334_linear_hidden_projection_inline(dm_gated_delta, dm_delta_out_layer, dm_attention_delta)
            (dm_residual, dm_post_norm) = _dm334_fused_wide_inline(dm_running, dm_attention_delta, dm_post_gamma_layer, dm_residual, dm_post_norm)
        else:
            dm_full = layer // 4
            dm_q_weight_full: pl.Tensor[[_dm334_HIDDEN, _dm334_FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[dm_full]
            dm_k_weight_full: pl.Tensor[[_dm334_HIDDEN, _dm334_FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[dm_full]
            dm_v_weight_full: pl.Tensor[[_dm334_HIDDEN, _dm334_FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[dm_full]
            dm_q_gamma_full: pl.Tensor[[1, _dm334_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, _dm334_FULL_HEAD_DIM], [dm_full, 0])
            dm_k_gamma_full: pl.Tensor[[1, _dm334_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, _dm334_FULL_HEAD_DIM], [dm_full, 0])
            dm_out_weight_full: pl.Tensor[[_dm334_FULL_Q_WIDTH, _dm334_HIDDEN], pl.BF16, pl.NZ] = full_out_weight[dm_full]
            dm_q_norm = pl.create_tensor([_dm334_BATCH, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            dm_k_norm = pl.create_tensor([_dm334_BATCH, _dm334_FULL_KV_WIDTH], dtype=pl.BF16)
            dm_q_rot = pl.create_tensor([_dm334_BATCH, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            dm_k_rot = pl.create_tensor([_dm334_BATCH, _dm334_FULL_KV_WIDTH], dtype=pl.BF16)
            dm_value = pl.create_tensor([_dm334_BATCH, _dm334_FULL_KV_WIDTH], dtype=pl.BF16)
            dm_gate = pl.create_tensor([_dm334_BATCH, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            (dm_q_norm, dm_k_norm, dm_q_rot, dm_k_rot, dm_value, dm_gate) = _dm334_attention3_prepare_attention3_prepare_inline(dm_norm, dm_q_weight_full, dm_k_weight_full, dm_v_weight_full, dm_q_gamma_full, dm_k_gamma_full, dm_cos_batch, dm_sin_batch, dm_q_norm, dm_k_norm, dm_q_rot, dm_k_rot, dm_value, dm_gate)
            dm_gated_full = pl.create_tensor([_dm334_BATCH, _dm334_FULL_Q_WIDTH], dtype=pl.BF16)
            _dm334_decode_attention_slots_inline(dm_q_rot, dm_k_rot, dm_value, dm_gate, dm_mask, dm_out_weight_full, key_pool, value_pool, dm_block_tables, dm_cache_offsets, dm_gated_full, dm_full, active_count)
            dm_attention_full = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
            dm_attention_full = _dm334_linear_hidden_projection_inline(dm_gated_full, dm_out_weight_full, dm_attention_full)
            (dm_residual, dm_post_norm) = _dm334_fused_wide_inline(dm_running, dm_attention_full, dm_post_gamma_layer, dm_residual, dm_post_norm)
        dm_mlp_gate = pl.create_tensor([_dm334_BATCH, _dm334_INTERMEDIATE], dtype=pl.BF16)
        dm_mlp_up = pl.create_tensor([_dm334_BATCH, _dm334_INTERMEDIATE], dtype=pl.BF16)
        dm_mlp_gate = _dm334_linear_hidden_projection_inline(dm_post_norm, dm_mlp_gate_layer, dm_mlp_gate)
        dm_mlp_up = _dm334_linear_hidden_projection_inline(dm_post_norm, dm_mlp_up_layer, dm_mlp_up)
        dm_mlp_middle = pl.create_tensor([_dm334_BATCH, _dm334_INTERMEDIATE], dtype=pl.BF16)
        dm_mlp_middle = _dm334_decode_io_swiglu_inline(dm_mlp_gate, dm_mlp_up, dm_mlp_middle)
        dm_mlp_output = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
        dm_mlp_output = _dm334_linear_hidden_projection_inline(dm_mlp_middle, dm_mlp_down_layer, dm_mlp_output)
        with pl.spmd(_dm334_BATCH, name_hint='qwen_decode_residual_store'):
            dm_io_batch = pl.tile.get_block_idx()
            dm_io_left = pl.cast(pl.slice(dm_residual, [1, _dm334_HIDDEN], [dm_io_batch, 0]), pl.FP32)
            dm_io_right = pl.cast(pl.slice(dm_mlp_output, [1, _dm334_HIDDEN], [dm_io_batch, 0]), pl.FP32)
            dm_io_following = pl.cast(pl.add(dm_io_left, dm_io_right), pl.BF16, mode='rint')
            dm_running[dm_io_batch:dm_io_batch + 1, 0:_dm334_HIDDEN] = dm_io_following
            if dm_io_batch > 0 and dm_io_batch < active_count:
                dm_traces[dm_io_batch:dm_io_batch + 1, layer + 1:layer + 2, 0:1, 0:_dm334_HIDDEN] = pl.reshape(dm_io_following, [1, 1, 1, _dm334_HIDDEN])
    pm_normalized = pm_norm
    pm_logits_rows = pl.reshape(pm_logits, [_pm334_PACKED, _pm334_VOCAB])
    _pm334_prefill_last_logits_inline_prefix512(pm_normalized, lm_head, pm_logits_rows, pm_token_counts, 1)
    with pl.spmd(_pm334_PACKED, name_hint='qwen_prefill_store_final_norm'):
        pm_row = pl.tile.get_block_idx()
        pm_batch = pm_row // _pm334_ROWS
        pm_token_row = pm_row % _pm334_ROWS
        if pm_batch < 1:
            pm_traces[pm_batch:pm_batch + 1, 65:66, pm_token_row:pm_token_row + 1, 0:_pm334_HIDDEN] = pl.reshape(pl.slice(pm_normalized, [1, _pm334_HIDDEN], [pm_row, 0]), [1, 1, 1, _pm334_HIDDEN])
    dm_normalized = pl.create_tensor([_dm334_BATCH, _dm334_HIDDEN], dtype=pl.BF16)
    dm_normalized = _dm334_rmsnorm_rmsnorm_inline(dm_running, final_gamma, dm_normalized)
    dm_packed_logits = pl.create_tensor([_dm334_BATCH, _dm334_VOCAB], dtype=pl.BF16)
    _dm334_linear_hidden_projection_inline_nd(dm_normalized, lm_head, dm_packed_logits)
    with pl.spmd(_dm334_BATCH * 61, name_hint='qwen_decode_coarse_logits'):
        dm_io_task = pl.tile.get_block_idx()
        dm_io_batch = dm_io_task // 61
        dm_io_part = dm_io_task % 61
        dm_io_column = dm_io_part * 4096
        if dm_io_batch > 0 and dm_io_batch < active_count:
            if dm_io_part < 60:
                dm_logits[dm_io_batch:dm_io_batch + 1, 0:1, dm_io_column:dm_io_column + 4096] = pl.reshape(pl.slice(dm_packed_logits, [1, 4096], [dm_io_batch, dm_io_column]), [1, 1, 4096])
            else:
                dm_logits[dm_io_batch:dm_io_batch + 1, 0:1, 245760:248320] = pl.reshape(pl.slice(dm_packed_logits, [1, 2560], [dm_io_batch, 245760]), [1, 1, 2560])
    with pl.spmd(_dm334_BATCH, name_hint='qwen_decode_store_final_norm'):
        dm_batch = pl.tile.get_block_idx()
        if dm_batch > 0 and dm_batch < active_count:
            dm_traces[dm_batch:dm_batch + 1, 65:66, 0:1, 0:_dm334_HIDDEN] = pl.reshape(pl.slice(dm_normalized, [1, _dm334_HIDDEN], [dm_batch, 0]), [1, 1, 1, _dm334_HIDDEN])
    _dm334_combined_greedy_inline(dm_logits, dm_token_counts, dm_maxima, dm_indices, dm_nan_indices, dm_selected_tokens, active_count)
    return (conv_states, recurrent_states, key_pool, value_pool, pm_traces, pm_logits, dm_traces, dm_logits, dm_maxima, dm_indices, dm_nan_indices, dm_selected_tokens)

@pl.jit.host
def mixed334_fwd(embedding: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, _dm334_HIDDEN], pl.FP32], lm_head: pl.Tensor[[_dm334_VOCAB, _dm334_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, _dm334_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64, _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16, pl.NZ], mlp_up_weight: pl.Tensor[[64, _dm334_HIDDEN, _dm334_INTERMEDIATE], pl.BF16, pl.NZ], mlp_down_weight: pl.Tensor[[64, _dm334_INTERMEDIATE, _dm334_HIDDEN], pl.BF16, pl.NZ], delta_qkv_weight: pl.Tensor[[48, _dm334_HIDDEN, _dm334_QKV], pl.BF16, pl.NZ], delta_z_weight: pl.Tensor[[48, _dm334_HIDDEN, _dm334_VALUE_WIDTH], pl.BF16, pl.NZ], delta_a_weight: pl.Tensor[[48, _dm334_HIDDEN, 128], pl.BF16, pl.NZ], delta_b_weight: pl.Tensor[[48, _dm334_HIDDEN, 128], pl.BF16, pl.NZ], delta_conv_weight: pl.Tensor[[48 * 4, _dm334_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, _dm334_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48, _dm334_VALUE_WIDTH, _dm334_HIDDEN], pl.BF16, pl.NZ], full_q_weight: pl.Tensor[[16, _dm334_HIDDEN, _dm334_FULL_Q_PROJECTION], pl.BF16, pl.NZ], full_k_weight: pl.Tensor[[16, _dm334_HIDDEN, _dm334_FULL_KV_WIDTH], pl.BF16, pl.NZ], full_v_weight: pl.Tensor[[16, _dm334_HIDDEN, _dm334_FULL_KV_WIDTH], pl.BF16, pl.NZ], full_q_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, _dm334_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16, _dm334_FULL_Q_WIDTH, _dm334_HIDDEN], pl.BF16, pl.NZ], conv_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, 4, _dm334_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[_dm334_BATCH, 48, _dm334_DELTA_HEADS, _dm334_DELTA_DIM, _dm334_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, _dm334_POOL_BLOCKS, _dm334_BLOCK_SIZE, _dm334_FULL_HEAD_DIM], pl.BF16]], pm_input_ids: pl.Tensor[[_pm334_BATCH, _pm334_ROWS], pl.INT32], pm_cos: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_ROTARY], pl.BF16], pm_sin: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_ROTARY], pl.BF16], pm_mask: pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_CONTEXT], pl.FP32], pm_token_counts: pl.Tensor[[_pm334_BATCH], pl.INT32], pm_cache_offsets: pl.Tensor[[_pm334_BATCH], pl.INT32], pm_block_tables: pl.Tensor[[_pm334_BATCH, _pm334_CONTEXT // _pm334_BLOCK_SIZE], pl.INT32], pm_traces: pl.Out[pl.Tensor[[_pm334_BATCH, 66, _pm334_ROWS, _pm334_HIDDEN], pl.BF16]], pm_logits: pl.InOut[pl.Tensor[[_pm334_BATCH, _pm334_ROWS, _pm334_VOCAB], pl.BF16]], dm_input_ids: pl.Tensor[[_dm334_BATCH, _dm334_ROWS], pl.INT32], dm_cos: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], dm_sin: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_ROTARY], pl.BF16], dm_mask: pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_CONTEXT], pl.FP32], dm_token_counts: pl.Tensor[[_dm334_BATCH], pl.INT32], dm_cache_offsets: pl.Tensor[[_dm334_BATCH], pl.INT32], dm_block_tables: pl.Tensor[[_dm334_BATCH, _dm334_CONTEXT // _dm334_BLOCK_SIZE], pl.INT32], dm_traces: pl.Out[pl.Tensor[[_dm334_BATCH, 66, _dm334_ROWS, _dm334_HIDDEN], pl.BF16]], dm_logits: pl.Out[pl.Tensor[[_dm334_BATCH, _dm334_ROWS, _dm334_VOCAB], pl.BF16]], dm_maxima: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], dm_indices: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], dm_nan_indices: pl.InOut[pl.Tensor[[_dm334_BATCH, 512], pl.FP32]], dm_selected_tokens: pl.InOut[pl.Tensor[[_dm334_BATCH, 8], pl.INT32]], active_count: pl.Scalar[pl.INT32]):
    rank_mixed334(embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, conv_states, recurrent_states, key_pool, value_pool, pm_input_ids, pm_cos, pm_sin, pm_mask, pm_token_counts, pm_cache_offsets, pm_block_tables, pm_traces, pm_logits, dm_input_ids, dm_cos, dm_sin, dm_mask, dm_token_counts, dm_cache_offsets, dm_block_tables, dm_traces, dm_logits, dm_maxima, dm_indices, dm_nan_indices, dm_selected_tokens, active_count, device=0)


@pl.jit.inline
def _ffn369_gate_up_swiglu_shared_activation(hidden: pl.Tensor, gate_weight: pl.Tensor, up_weight: pl.Tensor, output: pl.Tensor):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    with pl.spmd(output_tiles, name_hint='ffn369_gate_up_shared_activation', optimizations=[pl.cross_core_slot(slot_num=1)]):
        col = pl.tile.get_block_idx() * 128
        gate_acc = pl.create_tensor([128, 128], dtype=pl.FP32)
        up_acc = pl.create_tensor([128, 128], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // 256, stage=2):
            k_start = chunk * 256
            shared_activation = pl.slice(hidden, [128, 256], [0, k_start])
            gate_rhs = pl.slice(gate_weight, [256, 128], [k_start, col])
            up_rhs = pl.slice(up_weight, [256, 128], [k_start, col])
            gate_acc = pl.matmul_acc(gate_acc, shared_activation, gate_rhs, b_trans=False, init_cond=chunk == 0)
            up_acc = pl.matmul_acc(up_acc, shared_activation, up_rhs, b_trans=False, init_cond=chunk == 0)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            row = aiv_id * 64
            gate_bf16 = pl.cast(pl.aiv_shard(gate_acc), pl.BF16, mode='rint')
            gate_fp32 = pl.cast(gate_bf16, pl.FP32)
            activated_bf16 = pl.cast(pl.div(gate_fp32, pl.add(pl.exp(pl.neg(gate_fp32)), 1.0)), pl.BF16, mode='rint')
            up_bf16 = pl.cast(pl.aiv_shard(up_acc), pl.BF16, mode='rint')
            middle_bf16 = pl.cast(pl.mul(pl.cast(activated_bf16, pl.FP32), pl.cast(up_bf16, pl.FP32)), pl.BF16, mode='rint')
            output[row:row + 64, col:col + 128] = middle_bf16
    return output
