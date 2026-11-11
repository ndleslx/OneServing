"""Layer-major prefill with packed projections and original state/attention arithmetic."""
import pypto.language as pl
from ..model import linear_ROWS, linear_HIDDEN, linear_OUTPUT, linear_INPUT, linear_OUTPUT_TILE, linear_K_TILE, rmsnorm_ROWS, rmsnorm_HIDDEN, rmsnorm_CHUNK, gated_norm_ROWS, gated_norm_DIM, gated_norm_ROW_TILE, causal_conv_TOKENS, causal_conv_CHANNELS, causal_conv_KERNEL, causal_conv_CHANNEL_TILE, delta_prepare_TOKENS, delta_prepare_KEY_HEADS, delta_prepare_VALUE_HEADS, delta_prepare_DIM, delta_prepare_CHANNELS, delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS, delta_window16_ROWS, delta_window16_KEY_HEADS, delta_window16_HEADS, delta_window16_DIM, delta_window16_VALUE_TILE, delta_window16_NORM_TILE, delta_window16_COLUMN_REP, delta_window16_VALUE_WIDTH, attention3_prepare_ROWS, attention3_prepare_HIDDEN, attention3_prepare_HEAD_DIM, attention3_prepare_Q_HEADS, attention3_prepare_KV_HEADS, attention3_prepare_Q_PROJECTION, attention3_prepare_KV_WIDTH, attention3_prepare_Q_WIDTH, attention3_prepare_ROTARY, attention512_core_ROWS, attention512_core_CONTEXT, attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM, attention512_core_Q_HEADS, attention512_core_KV_HEADS, attention512_core_Q_WIDTH, residual_ROWS, residual_HIDDEN, residual_TILE, swiglu_ROWS, swiglu_INTERMEDIATE, swiglu_CHANNEL_TILE, layer0_stem_ROWS, layer0_stem_HIDDEN, layer0_stem_QKV, layer0_stem_Z, layer0_stem_GATE_PAD, ROWS, BATCH, CONTEXT, BLOCK_SIZE, POOL_BLOCKS, HIDDEN, INTERMEDIATE, QKV, VALUE_WIDTH, DELTA_HEADS, DELTA_DIM, FULL_HEAD_DIM, FULL_Q_WIDTH, FULL_Q_PROJECTION, FULL_KV_WIDTH, ROTARY, VOCAB, tp_rank_ROWS, tp_rank_BATCH, tp_rank_CONTEXT, tp_rank_BLOCK_SIZE, tp_rank_POOL_BLOCKS, tp_rank_HIDDEN, tp_rank_INTERMEDIATE, tp_rank_QKV, tp_rank_VALUE_WIDTH, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_FULL_HEAD_DIM, tp_rank_FULL_Q_WIDTH, tp_rank_FULL_Q_PROJECTION, tp_rank_FULL_KV_WIDTH, tp_rank_ROTARY, tp_rank_VOCAB, tp_rank_TP_VOCAB, causal_conv_causal_conv_silu_inline, delta_window16_expand_qkv_window16_inline, delta_prepare_decay_beta_inline, delta_window16_recurrence_window16_inline, attention3_prepare_normalize_head_inline, attention3_prepare_rotary_head_inline
ROWS = 64
PACKED = BATCH * ROWS
delta_prepare_PAD_ROWS = ROWS
delta_window16_ROWS = ROWS
attention3_prepare_ROWS = 16
ATTENTION_TILE_ROWS = 16
attention512_core_ROWS = ROWS
delta_window16_VALUE_TILE = 128

@pl.jit.inline
def delta_prepare_decay_beta_inline(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor):
    with pl.spmd(1, name_hint='qwen_delta_decay_beta'):
        row = pl.tile.get_block_idx() * delta_prepare_PAD_ROWS
        a_tile = pl.cast(pl.slice(a, [delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS], [row, 0]), pl.FP32)
        b_tile = pl.cast(pl.slice(b, [delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS], [row, 0]), pl.FP32)
        a_parameter = pl.cast(pl.slice(a_log, [1, delta_prepare_PAD_HEADS], [0, 0]), pl.FP32)
        dt_parameter = pl.cast(pl.slice(dt_bias, [1, delta_prepare_PAD_HEADS], [0, 0]), pl.FP32)
        softplus_input = pl.col_expand_add(a_tile, dt_parameter)
        softplus = pl.log(pl.add(pl.exp(softplus_input), 1.0))
        decay = pl.mul(pl.col_expand_mul(softplus, pl.exp(a_parameter)), -1.0)
        beta_value = pl.div(pl.full([delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(b_tile)), 1.0))
        log_decay[row:row + delta_prepare_PAD_ROWS, 0:delta_prepare_PAD_HEADS] = decay
        beta[row:row + delta_prepare_PAD_ROWS, 0:delta_prepare_PAD_HEADS] = pl.cast(beta_value, pl.BF16, mode='rint')
    return (log_decay, beta)

@pl.jit.inline
def attention3_prepare_normalize_head_inline(head_values: pl.Tensor, gamma: pl.Tensor, result: pl.Tensor):
    source = pl.cast(head_values, pl.FP32)
    square = pl.mul(source, source)
    inverse = pl.rsqrt(pl.add(pl.mul(pl.row_sum(square), 1.0 / attention3_prepare_HEAD_DIM), 1e-06), high_precision=True)
    normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), gamma)
    result[0:attention3_prepare_ROWS, 0:attention3_prepare_HEAD_DIM] = pl.cast(normalized, pl.BF16, mode='rint')
    return result

@pl.jit.inline
def attention3_prepare_rotary_head_inline(normalized: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], pl.BF16], cos: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_ROTARY], pl.BF16], sin: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_ROTARY], pl.BF16], result: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], pl.BF16]):
    first = pl.cast(pl.slice(normalized, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    second = pl.cast(pl.slice(normalized, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, attention3_prepare_ROTARY // 2]), pl.FP32)
    cos_first = pl.cast(pl.slice(cos, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    cos_second = pl.cast(pl.slice(cos, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, attention3_prepare_ROTARY // 2]), pl.FP32)
    sin_first = pl.cast(pl.slice(sin, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, 0]), pl.FP32)
    sin_second = pl.cast(pl.slice(sin, [attention3_prepare_ROWS, attention3_prepare_ROTARY // 2], [0, attention3_prepare_ROTARY // 2]), pl.FP32)
    rotated_first = pl.cast(pl.sub(pl.mul(first, cos_first), pl.mul(second, sin_first)), pl.BF16, mode='rint')
    rotated_second = pl.cast(pl.add(pl.mul(second, cos_second), pl.mul(first, sin_second)), pl.BF16, mode='rint')
    rest = pl.slice(normalized, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM - attention3_prepare_ROTARY], [0, attention3_prepare_ROTARY])
    result[0:attention3_prepare_ROWS, 0:attention3_prepare_HEAD_DIM] = pl.concat(pl.concat(rotated_first, rotated_second), rest)
    return result

@pl.jit.inline
def prefill_project_inline(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    if active_count <= 2:
        with pl.spmd(1 * output_tiles, name_hint='prefill_nz_projection_active1'):
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
    elif active_count <= 4:
        with pl.spmd(2 * output_tiles, name_hint='prefill_nz_projection_active2'):
            task = pl.tile.get_block_idx()
            row = task % 2 * 128
            col = task // 2 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    elif active_count <= 6:
        with pl.spmd(3 * output_tiles, name_hint='prefill_nz_projection_active3'):
            task = pl.tile.get_block_idx()
            row = task % 3 * 128
            col = task // 3 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    else:
        with pl.spmd(4 * output_tiles, name_hint='prefill_nz_projection_active4'):
            task = pl.tile.get_block_idx()
            row = task % 4 * 128
            col = task // 4 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_norm_inline(hidden: pl.Tensor, gamma: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(hidden, 0)
    with pl.spmd(packed_rows // rmsnorm_ROWS, name_hint='qwen_prefill_rmsnorm'):
        row_start = pl.tile.get_block_idx() * rmsnorm_ROWS
        square_sum = pl.full([1, rmsnorm_ROWS], dtype=pl.FP32, value=0.0)
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(pl.mul(source, source)), [1, rmsnorm_ROWS]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / rmsnorm_HIDDEN), 1e-06), high_precision=True), [rmsnorm_ROWS, 1])
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            weight = pl.slice(gamma, [1, rmsnorm_CHUNK], [0, column])
            normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weight)
            output[row_start:row_start + rmsnorm_ROWS, column:column + rmsnorm_CHUNK] = pl.cast(normalized, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_residual_inline(left: pl.Tensor, right: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(left, 0)
    with pl.spmd(packed_rows, name_hint='qwen_prefill_residual'):
        row = pl.tile.get_block_idx()
        a = pl.cast(pl.slice(left, [1, HIDDEN], [row, 0]), pl.FP32)
        b = pl.cast(pl.slice(right, [1, HIDDEN], [row, 0]), pl.FP32)
        output[row:row + 1, 0:HIDDEN] = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_swiglu_inline(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(gate, 0)
    with pl.spmd(packed_rows // 16 * (INTERMEDIATE // 1024), name_hint='qwen_prefill_swiglu'):
        task = pl.tile.get_block_idx()
        row = task // (INTERMEDIATE // 1024) * 16
        column = task % (INTERMEDIATE // 1024) * 1024
        gate_tile = pl.cast(pl.slice(gate, [16, 1024], [row, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        up_tile = pl.cast(pl.slice(up, [16, 1024], [row, column]), pl.FP32)
        output[row:row + 16, column:column + 1024] = pl.cast(pl.mul(pl.cast(activated, pl.FP32), up_tile), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_gated_tail_inline(core: pl.Tensor, z_projection: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor, gated_output: pl.Tensor, attention_output: pl.Tensor):
    gate_rows = pl.reshape(z_projection, [delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM])
    core_rows = pl.reshape(core, [delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM])
    with pl.spmd(delta_window16_ROWS * delta_window16_HEADS // delta_window16_NORM_TILE, name_hint='qwen_window_gated_rmsnorm'):
        row = pl.tile.get_block_idx() * delta_window16_NORM_TILE
        source = pl.cast(pl.slice(core_rows, [delta_window16_NORM_TILE, delta_window16_DIM], [row, 0]), pl.FP32)
        gate_tile = pl.cast(pl.slice(gate_rows, [delta_window16_NORM_TILE, delta_window16_DIM], [row, 0]), pl.FP32)
        weight_tile = pl.cast(pl.slice(norm_weight, [1, gated_norm_DIM], [0, 0]), pl.FP32)
        variance = pl.mul(pl.row_sum(pl.mul(source, source)), 1.0 / delta_window16_DIM)
        inverse = pl.rsqrt(pl.add(variance, 1e-06), high_precision=True)
        normalized = pl.cast(pl.row_expand_mul(source, inverse), pl.BF16, mode='rint')
        weighted = pl.cast(pl.col_expand_mul(pl.cast(normalized, pl.FP32), weight_tile), pl.BF16, mode='rint')
        sigmoid = pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0))
        gated_output[row:row + delta_window16_NORM_TILE, 0:delta_window16_DIM] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), sigmoid), pl.BF16, mode='rint')
    return gated_output

@pl.jit.inline
def prefill_attention_core_inline(query: pl.Tensor, key_cache: pl.Tensor, value_cache: pl.Tensor, gate: pl.Tensor, mask: pl.Tensor, output_weight: pl.Tensor, logits_buffer: pl.Tensor, probability_buffer: pl.Tensor, weighted_output: pl.Tensor, gated_output: pl.Tensor, projected_output: pl.Tensor):
    with pl.spmd(attention512_core_Q_HEADS * (ROWS // ATTENTION_TILE_ROWS), name_hint='qwen_attention512_scores'):
        score_task = pl.tile.get_block_idx()
        score_head = score_task // (ROWS // ATTENTION_TILE_ROWS)
        score_row = score_task % (ROWS // ATTENTION_TILE_ROWS) * ATTENTION_TILE_ROWS
        score_kv_head = score_head // (attention512_core_Q_HEADS // attention512_core_KV_HEADS)
        score_start = score_head * attention512_core_HEAD_DIM
        score_q = pl.slice(query, [ATTENTION_TILE_ROWS, attention512_core_HEAD_DIM], [score_row, score_start])
        for score_block in pl.range(attention512_core_CONTEXT // attention512_core_KEY_BLOCK):
            score_key_start = score_block * attention512_core_KEY_BLOCK
            score_k = pl.reshape(pl.slice(key_cache, [1, attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM], [score_kv_head, score_key_start, 0]), [attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM])
            score_raw = pl.matmul(score_q, score_k, pl.FP32, b_trans=True)
            score_rounded = pl.cast(score_raw, pl.BF16, mode='rint')
            score_scaled = pl.cast(pl.mul(pl.cast(score_rounded, pl.FP32), attention512_core_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
            score_mask = pl.slice(mask, [ATTENTION_TILE_ROWS, attention512_core_KEY_BLOCK], [score_row, score_key_start])
            logits_buffer[score_head * attention512_core_ROWS + score_row:score_head * attention512_core_ROWS + score_row + ATTENTION_TILE_ROWS, score_key_start:score_key_start + attention512_core_KEY_BLOCK] = pl.add(pl.cast(score_scaled, pl.FP32), score_mask)
    with pl.spmd(attention512_core_Q_HEADS * (attention512_core_ROWS // 8), name_hint='qwen_attention512_softmax'):
        softmax_row = pl.tile.get_block_idx() * 8
        softmax_logits = pl.slice(logits_buffer, [8, attention512_core_CONTEXT], [softmax_row, 0])
        softmax_exp = pl.exp(pl.row_expand_sub(softmax_logits, pl.row_max(softmax_logits)))
        probability_buffer[softmax_row:softmax_row + 8, 0:attention512_core_CONTEXT] = pl.cast(pl.row_expand_div(softmax_exp, pl.row_sum(softmax_exp)), pl.BF16, mode='rint')
    with pl.spmd(attention512_core_Q_HEADS * (ROWS // ATTENTION_TILE_ROWS), name_hint='qwen_attention512_weighted'):
        weighted_task = pl.tile.get_block_idx()
        weighted_head = weighted_task // (ROWS // ATTENTION_TILE_ROWS)
        weighted_row = weighted_task % (ROWS // ATTENTION_TILE_ROWS) * ATTENTION_TILE_ROWS
        weighted_kv_head = weighted_head // (attention512_core_Q_HEADS // attention512_core_KV_HEADS)
        weighted_start = weighted_head * attention512_core_HEAD_DIM
        weighted_sum = pl.full([ATTENTION_TILE_ROWS, attention512_core_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for weighted_block in pl.range(attention512_core_CONTEXT // attention512_core_KEY_BLOCK):
            weighted_key_start = weighted_block * attention512_core_KEY_BLOCK
            weighted_probs = pl.slice(probability_buffer, [ATTENTION_TILE_ROWS, attention512_core_KEY_BLOCK], [weighted_head * attention512_core_ROWS + weighted_row, weighted_key_start])
            weighted_values = pl.reshape(pl.slice(value_cache, [1, attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM], [weighted_kv_head, weighted_key_start, 0]), [attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM])
            weighted_sum = pl.add(weighted_sum, pl.matmul(weighted_probs, weighted_values, pl.FP32))
        weighted = pl.cast(weighted_sum, pl.BF16, mode='rint')
        weighted_output[weighted_row:weighted_row + ATTENTION_TILE_ROWS, weighted_start:weighted_start + attention512_core_HEAD_DIM] = weighted
        gate_tile = pl.cast(pl.slice(gate, [ATTENTION_TILE_ROWS, attention512_core_HEAD_DIM], [weighted_row, weighted_start]), pl.FP32)
        sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        gated_output[weighted_row:weighted_row + ATTENTION_TILE_ROWS, weighted_start:weighted_start + attention512_core_HEAD_DIM] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint')
    return gated_output

@pl.jit.inline
def prefill_prepare_heads_inline(q_projected: pl.Tensor[[ROWS, FULL_Q_PROJECTION], pl.BF16], k_projected: pl.Tensor[[ROWS, FULL_KV_WIDTH], pl.BF16], value: pl.Tensor[[ROWS, FULL_KV_WIDTH], pl.BF16], q_gamma: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32], k_gamma: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32], cos: pl.Tensor[[ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[ROWS, ROTARY], pl.BF16], q_norm: pl.Tensor, k_norm: pl.Tensor, q_rot: pl.Tensor, k_rot: pl.Tensor, gate: pl.Tensor):
    with pl.spmd(attention3_prepare_Q_HEADS * (ROWS // attention3_prepare_ROWS), name_hint='qwen_attention_query_norm'):
        task = pl.tile.get_block_idx()
        head = task // (ROWS // attention3_prepare_ROWS)
        row = task % (ROWS // attention3_prepare_ROWS) * attention3_prepare_ROWS
        projection_start = head * attention3_prepare_HEAD_DIM * 2
        result_start = head * attention3_prepare_HEAD_DIM
        raw = pl.slice(q_projected, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], [row, projection_start])
        normalized = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        normalized = attention3_prepare_normalize_head_inline(raw, q_gamma, normalized)
        q_norm[row:row + attention3_prepare_ROWS, result_start:result_start + attention3_prepare_HEAD_DIM] = normalized
        gate[row:row + attention3_prepare_ROWS, result_start:result_start + attention3_prepare_HEAD_DIM] = pl.slice(q_projected, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], [row, projection_start + attention3_prepare_HEAD_DIM])
        rotated = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        tile_cos = pl.slice(cos, [attention3_prepare_ROWS, ROTARY], [row, 0])
        tile_sin = pl.slice(sin, [attention3_prepare_ROWS, ROTARY], [row, 0])
        rotated = attention3_prepare_rotary_head_inline(normalized, tile_cos, tile_sin, rotated)
        q_rot[row:row + attention3_prepare_ROWS, result_start:result_start + attention3_prepare_HEAD_DIM] = rotated
    with pl.spmd(attention3_prepare_KV_HEADS * (ROWS // attention3_prepare_ROWS), name_hint='qwen_attention_key_norm'):
        task = pl.tile.get_block_idx()
        head = task // (ROWS // attention3_prepare_ROWS)
        row = task % (ROWS // attention3_prepare_ROWS) * attention3_prepare_ROWS
        start = head * attention3_prepare_HEAD_DIM
        raw = pl.slice(k_projected, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], [row, start])
        normalized = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        normalized = attention3_prepare_normalize_head_inline(raw, k_gamma, normalized)
        k_norm[row:row + attention3_prepare_ROWS, start:start + attention3_prepare_HEAD_DIM] = normalized
        rotated = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        tile_cos = pl.slice(cos, [attention3_prepare_ROWS, ROTARY], [row, 0])
        tile_sin = pl.slice(sin, [attention3_prepare_ROWS, ROTARY], [row, 0])
        rotated = attention3_prepare_rotary_head_inline(normalized, tile_cos, tile_sin, rotated)
        k_rot[row:row + attention3_prepare_ROWS, start:start + attention3_prepare_HEAD_DIM] = rotated
    return (q_norm, k_norm, q_rot, k_rot, value, gate)

@pl.jit.inline
def prefill_direct_qkv_recurrence_inline(conv_output: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor, state_value_key: pl.Tensor, output: pl.Tensor, token_count: pl.Scalar[pl.INDEX]):
    direct_q_norm = pl.create_tensor([delta_window16_ROWS * delta_window16_KEY_HEADS, delta_window16_DIM], dtype=pl.FP32)
    direct_k_norm = pl.create_tensor([delta_window16_ROWS * delta_window16_KEY_HEADS, delta_window16_DIM], dtype=pl.FP32)
    with pl.spmd(delta_window16_ROWS, name_hint='qwen_prefill_compact_qk_norm') as direct_norm_task:
        direct_norm_token = pl.tile.get_block_idx()
        direct_q_tile = pl.cast(pl.reshape(pl.slice(conv_output, [1, delta_window16_KEY_HEADS * delta_window16_DIM], [direct_norm_token, 0]), [delta_window16_KEY_HEADS, delta_window16_DIM]), pl.FP32)
        direct_k_tile = pl.cast(pl.reshape(pl.slice(conv_output, [1, delta_window16_KEY_HEADS * delta_window16_DIM], [direct_norm_token, delta_window16_KEY_HEADS * delta_window16_DIM]), [delta_window16_KEY_HEADS, delta_window16_DIM]), pl.FP32)
        direct_q_norm[direct_norm_token * delta_window16_KEY_HEADS:(direct_norm_token + 1) * delta_window16_KEY_HEADS, 0:delta_window16_DIM] = pl.mul(pl.row_expand_mul(direct_q_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(direct_q_tile, direct_q_tile)), 1e-06), high_precision=True)), delta_window16_DIM ** (-0.5))
        direct_k_norm[direct_norm_token * delta_window16_KEY_HEADS:(direct_norm_token + 1) * delta_window16_KEY_HEADS, 0:delta_window16_DIM] = pl.row_expand_mul(direct_k_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(direct_k_tile, direct_k_tile)), 1e-06), high_precision=True))
    direct_state_rows = pl.reshape(state_value_key, [delta_window16_HEADS * delta_window16_DIM, delta_window16_DIM])
    direct_output_rows = pl.reshape(output, [delta_window16_ROWS, delta_window16_HEADS * delta_window16_DIM])
    with pl.spmd(delta_window16_HEADS * (delta_window16_DIM // delta_window16_VALUE_TILE), name_hint='qwen_prefill_direct_recurrence', deps=[direct_norm_task], optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=3)]):
        direct_task = pl.tile.get_block_idx()
        direct_head = direct_task // (delta_window16_DIM // delta_window16_VALUE_TILE)
        direct_value_start = direct_task % (delta_window16_DIM // delta_window16_VALUE_TILE) * delta_window16_VALUE_TILE
        direct_state_start = direct_head * delta_window16_DIM + direct_value_start
        direct_state = pl.slice(direct_state_rows, [delta_window16_VALUE_TILE, delta_window16_DIM], [direct_state_start, 0])
        for direct_pair in pl.range(token_count // 2):
            direct_token0 = direct_pair * 2
            direct_row0 = direct_token0 * delta_window16_KEY_HEADS + direct_head // (delta_window16_HEADS // delta_window16_KEY_HEADS)
            direct_q0 = direct_q_norm[direct_row0:direct_row0 + 1, 0:delta_window16_DIM]
            direct_k0 = direct_k_norm[direct_row0:direct_row0 + 1, 0:delta_window16_DIM]
            direct_g0 = pl.read(log_decay, [direct_token0, direct_head])
            direct_decay0 = pl.exp(pl.mul(pl.full([1, delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_g0))
            direct_b0 = pl.read(beta, [direct_token0, direct_head])
            direct_decayed0 = pl.col_expand_mul(direct_state, direct_decay0)
            direct_remembered0 = pl.row_sum(pl.col_expand_mul(direct_decayed0, direct_k0))
            direct_v_row0 = pl.cast(pl.slice(conv_output, [1, delta_window16_VALUE_TILE], [direct_token0, 2 * delta_window16_KEY_HEADS * delta_window16_DIM + direct_head * delta_window16_DIM + direct_value_start]), pl.FP32)
            direct_v_col0 = pl.reshape(direct_v_row0, [delta_window16_VALUE_TILE, 1])
            direct_delta0 = pl.mul(pl.sub(direct_v_col0, direct_remembered0), direct_b0)
            direct_state0 = pl.add(direct_decayed0, pl.col_expand_mul(pl.row_expand_mul(pl.full([delta_window16_VALUE_TILE, delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_delta0), direct_k0))
            direct_product0 = pl.matmul(direct_q0, direct_state0, pl.BF16, b_trans=True)
            direct_token1 = direct_token0 + 1
            direct_row1 = direct_token1 * delta_window16_KEY_HEADS + direct_head // (delta_window16_HEADS // delta_window16_KEY_HEADS)
            direct_q1 = direct_q_norm[direct_row1:direct_row1 + 1, 0:delta_window16_DIM]
            direct_k1 = direct_k_norm[direct_row1:direct_row1 + 1, 0:delta_window16_DIM]
            direct_g1 = pl.read(log_decay, [direct_token1, direct_head])
            direct_decay1 = pl.exp(pl.mul(pl.full([1, delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_g1))
            direct_b1 = pl.read(beta, [direct_token1, direct_head])
            direct_decayed1 = pl.col_expand_mul(direct_state0, direct_decay1)
            direct_remembered1 = pl.row_sum(pl.col_expand_mul(direct_decayed1, direct_k1))
            direct_v_row1 = pl.cast(pl.slice(conv_output, [1, delta_window16_VALUE_TILE], [direct_token1, 2 * delta_window16_KEY_HEADS * delta_window16_DIM + direct_head * delta_window16_DIM + direct_value_start]), pl.FP32)
            direct_v_col1 = pl.reshape(direct_v_row1, [delta_window16_VALUE_TILE, 1])
            direct_delta1 = pl.mul(pl.sub(direct_v_col1, direct_remembered1), direct_b1)
            direct_state = pl.add(direct_decayed1, pl.col_expand_mul(pl.row_expand_mul(pl.full([delta_window16_VALUE_TILE, delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_delta1), direct_k1))
            direct_product1 = pl.matmul(direct_q1, direct_state, pl.BF16, b_trans=True)
            direct_output_rows[direct_token0:direct_token0 + 1, direct_head * delta_window16_DIM + direct_value_start:direct_head * delta_window16_DIM + direct_value_start + delta_window16_VALUE_TILE] = pl.set_validshape(direct_product0, 1, delta_window16_VALUE_TILE)
            direct_output_rows[direct_token1:direct_token1 + 1, direct_head * delta_window16_DIM + direct_value_start:direct_head * delta_window16_DIM + direct_value_start + delta_window16_VALUE_TILE] = pl.set_validshape(direct_product1, 1, delta_window16_VALUE_TILE)
        if token_count % 2 == 1:
            direct_token = token_count - 1
            direct_row = direct_token * delta_window16_KEY_HEADS + direct_head // (delta_window16_HEADS // delta_window16_KEY_HEADS)
            direct_q = direct_q_norm[direct_row:direct_row + 1, 0:delta_window16_DIM]
            direct_k = direct_k_norm[direct_row:direct_row + 1, 0:delta_window16_DIM]
            direct_g = pl.read(log_decay, [direct_token, direct_head])
            direct_decay = pl.exp(pl.mul(pl.full([1, delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_g))
            direct_b = pl.read(beta, [direct_token, direct_head])
            direct_decayed = pl.col_expand_mul(direct_state, direct_decay)
            direct_remembered = pl.row_sum(pl.col_expand_mul(direct_decayed, direct_k))
            direct_v_row = pl.cast(pl.slice(conv_output, [1, delta_window16_VALUE_TILE], [direct_token, 2 * delta_window16_KEY_HEADS * delta_window16_DIM + direct_head * delta_window16_DIM + direct_value_start]), pl.FP32)
            direct_v_col = pl.reshape(direct_v_row, [delta_window16_VALUE_TILE, 1])
            direct_delta = pl.mul(pl.sub(direct_v_col, direct_remembered), direct_b)
            direct_state = pl.add(direct_decayed, pl.col_expand_mul(pl.row_expand_mul(pl.full([delta_window16_VALUE_TILE, delta_window16_DIM], dtype=pl.FP32, value=1.0), direct_delta), direct_k))
            direct_output_rows[direct_token:direct_token + 1, direct_head * delta_window16_DIM + direct_value_start:direct_head * delta_window16_DIM + direct_value_start + delta_window16_VALUE_TILE] = pl.set_validshape(pl.matmul(direct_q, direct_state, pl.BF16, b_trans=True), 1, delta_window16_VALUE_TILE)
        for direct_token in pl.range(token_count, delta_window16_ROWS):
            direct_output_rows[direct_token:direct_token + 1, direct_head * delta_window16_DIM + direct_value_start:direct_head * delta_window16_DIM + direct_value_start + delta_window16_VALUE_TILE] = pl.full([1, delta_window16_VALUE_TILE], dtype=pl.BF16, value=0.0)
        direct_state_rows[direct_state_start:direct_state_start + delta_window16_VALUE_TILE, 0:delta_window16_DIM] = direct_state
    return (output, state_value_key)

@pl.jit.inline
def prefill_batched_conv_inline(qkv_batch: pl.Tensor, conv_weight: pl.Tensor, conv_states: pl.Tensor, conv_output: pl.Tensor, token_counts: pl.Tensor, linear: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    state_layers = pl.tensor.dim(conv_states, 1)
    conv_rows = pl.reshape(conv_states, [BATCH * state_layers * 4, QKV])
    with pl.spmd(packed_rows // ROWS * (QKV // 512), name_hint='qwen_prefill_batched_conv'):
        task = pl.tile.get_block_idx()
        batch = task // (QKV // 512)
        column = task % (QKV // 512) * 512
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            state_start = (batch * state_layers + linear) * 4
            previous_0 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 0, column]), pl.FP32)
            previous_1 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 1, column]), pl.FP32)
            previous_2 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 2, column]), pl.FP32)
            previous_3 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 3, column]), pl.FP32)
            tap_0 = pl.cast(pl.slice(conv_weight, [1, 512], [0, column]), pl.FP32)
            tap_1 = pl.cast(pl.slice(conv_weight, [1, 512], [1, column]), pl.FP32)
            tap_2 = pl.cast(pl.slice(conv_weight, [1, 512], [2, column]), pl.FP32)
            tap_3 = pl.cast(pl.slice(conv_weight, [1, 512], [3, column]), pl.FP32)
            for token in pl.range(token_count):
                current = pl.cast(pl.slice(qkv_batch, [1, 512], [batch * ROWS + token, column]), pl.FP32)
                convolved = pl.add(pl.add(pl.mul(previous_1, tap_0), pl.mul(previous_2, tap_1)), pl.add(pl.mul(previous_3, tap_2), pl.mul(current, tap_3)))
                convolved = pl.cast(pl.cast(convolved, pl.BF16, mode='rint'), pl.FP32)
                activated = pl.div(convolved, pl.add(pl.exp(pl.neg(convolved)), 1.0), high_precision=True)
                conv_output[batch * ROWS + token:batch * ROWS + token + 1, column:column + 512] = pl.cast(activated, pl.BF16, mode='rint')
                previous_0 = previous_1
                previous_1 = previous_2
                previous_2 = previous_3
                previous_3 = current
            for token in pl.range(token_count, ROWS):
                conv_output[batch * ROWS + token:batch * ROWS + token + 1, column:column + 512] = pl.full([1, 512], dtype=pl.BF16, value=0.0)
            conv_rows[state_start + 0:state_start + 1, column:column + 512] = pl.cast(previous_0, pl.BF16, mode='rint')
            conv_rows[state_start + 1:state_start + 2, column:column + 512] = pl.cast(previous_1, pl.BF16, mode='rint')
            conv_rows[state_start + 2:state_start + 3, column:column + 512] = pl.cast(previous_2, pl.BF16, mode='rint')
            conv_rows[state_start + 3:state_start + 4, column:column + 512] = pl.cast(previous_3, pl.BF16, mode='rint')
        else:
            conv_output[batch * ROWS:(batch + 1) * ROWS, column:column + 512] = pl.full([ROWS, 512], dtype=pl.BF16, value=0.0)
    return (conv_states, conv_output)

@pl.jit.inline
def prefill_delta_slots_inline(qkv_batch: pl.Tensor, z_batch: pl.Tensor, a_batch: pl.Tensor, b_batch: pl.Tensor, conv_weight_layer: pl.Tensor, a_log_layer: pl.Tensor, dt_bias_layer: pl.Tensor, delta_norm_layer: pl.Tensor, delta_out_layer: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, gated_batch: pl.Tensor, linear: pl.Scalar[pl.INDEX], token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    alias_state_layers = pl.tensor.dim(recurrent_states, 1)
    alias_conv_rows = pl.reshape(conv_states, [alias_state_layers * (BATCH * 4), QKV])
    alias_recurrent_rows = pl.reshape(recurrent_states, [alias_state_layers * (BATCH * DELTA_HEADS * DELTA_DIM), DELTA_DIM])
    conv_output_storage = pl.create_tensor([packed_rows, QKV], dtype=pl.BF16)
    (updated_conv_states, conv_output_batch) = prefill_batched_conv_inline(qkv_batch, conv_weight_layer, conv_states, conv_output_storage, token_counts, linear, active_count)
    for batch in pl.range(packed_rows // ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            qkv = pl.slice(qkv_batch, [ROWS, QKV], [batch * ROWS, 0])
            z = pl.slice(z_batch, [ROWS, VALUE_WIDTH], [batch * ROWS, 0])
            a = pl.slice(a_batch, [ROWS, 128], [batch * ROWS, 0])
            b = pl.slice(b_batch, [ROWS, 128], [batch * ROWS, 0])
            conv_source = pl.slice(alias_conv_rows, [4, QKV], [(batch * alias_state_layers + linear) * 4, 0])
            recurrent_slice = pl.slice(alias_recurrent_rows, [DELTA_HEADS * DELTA_DIM, DELTA_DIM], [(batch * alias_state_layers + linear) * DELTA_HEADS * DELTA_DIM, 0])
            conv = pl.reshape(conv_source, [4, QKV])
            recurrent = pl.reshape(recurrent_slice, [DELTA_HEADS, DELTA_DIM, DELTA_DIM])
            conv_output = pl.slice(conv_output_batch, [ROWS, QKV], [batch * ROWS, 0])
            log_decay = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            beta = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            core = pl.create_tensor([ROWS, DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            gate_values = pl.create_tensor([ROWS, 64], dtype=pl.FP32)
            beta_bf16 = pl.create_tensor([ROWS, 64], dtype=pl.BF16)
            (gate_values, beta_bf16) = delta_prepare_decay_beta_inline(a, b, a_log_layer, dt_bias_layer, gate_values, beta_bf16)
            with pl.spmd(ROWS, name_hint='qwen_full_delta_gate_unpad'):
                token = pl.tile.get_block_idx()
                log_decay[token:token + 1, 0:DELTA_HEADS] = pl.slice(gate_values, [1, DELTA_HEADS], [token, 0])
                beta[token:token + 1, 0:DELTA_HEADS] = pl.cast(pl.slice(beta_bf16, [1, DELTA_HEADS], [token, 0]), pl.FP32)
            (core, recurrent) = prefill_direct_qkv_recurrence_inline(conv_output, log_decay, beta, recurrent, core, token_count)
            gated = pl.create_tensor([ROWS * DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            ignored = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            gated = prefill_gated_tail_inline(core, z, delta_norm_layer, delta_out_layer, gated, ignored)
            packed = pl.reshape(gated, [ROWS, VALUE_WIDTH])
            with pl.spmd(ROWS, name_hint='qwen_prefill_store_delta_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:VALUE_WIDTH] = pl.slice(packed, [1, VALUE_WIDTH], [row, 0])
        else:
            with pl.spmd(ROWS, name_hint='qwen_prefill_zero_delta_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:VALUE_WIDTH] = pl.full([1, VALUE_WIDTH], dtype=pl.BF16, value=0.0)
    return (conv_states, recurrent_states, gated_batch)

@pl.jit.inline
def prefill_load_paged_kv_inline(key_pool: pl.Tensor, value_pool: pl.Tensor, block_table: pl.Tensor, key_local: pl.Tensor, value_local: pl.Tensor, full: pl.Scalar[pl.INDEX], cache_offset: pl.Scalar[pl.INT32], token_count: pl.Scalar[pl.INT32]):
    valid_blocks = (cache_offset + token_count + BLOCK_SIZE - 1) // BLOCK_SIZE
    with pl.spmd(4 * (CONTEXT // 128), name_hint='qwen_prefill_zero_local_kv') as zero_task:
        task = pl.tile.get_block_idx()
        head = task // (CONTEXT // 128)
        row = task % (CONTEXT // 128) * 128
        zero = pl.full([1, 128, FULL_HEAD_DIM], dtype=pl.BF16, value=0.0)
        key_local[head:head + 1, row:row + 128, 0:FULL_HEAD_DIM] = zero
        value_local[head:head + 1, row:row + 128, 0:FULL_HEAD_DIM] = zero
    with pl.spmd(4 * valid_blocks, name_hint='qwen_prefill_load_valid_kv', deps=[zero_task]):
        task = pl.tile.get_block_idx()
        head = task // valid_blocks
        logical_block = task % valid_blocks
        physical = pl.read(block_table, [logical_block])
        row = logical_block * BLOCK_SIZE
        key_local[head:head + 1, row:row + BLOCK_SIZE, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(key_pool, [1, 1, 1, BLOCK_SIZE, FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, BLOCK_SIZE, FULL_HEAD_DIM])
        value_local[head:head + 1, row:row + BLOCK_SIZE, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_pool, [1, 1, 1, BLOCK_SIZE, FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, BLOCK_SIZE, FULL_HEAD_DIM])
    return (key_local, value_local)

@pl.jit.inline
def prefill_attention_slots_inline(queries: pl.Tensor, keys: pl.Tensor, values: pl.Tensor, q_gamma_full: pl.Tensor, k_gamma_full: pl.Tensor, cos_batch: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin_batch: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], masks: pl.Tensor, out_weight_full: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, cache_offsets: pl.Tensor, token_counts: pl.Tensor, gated_batch: pl.Tensor, full: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(gated_batch, 0)
    for batch in pl.range(packed_rows // ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            cache_offset = pl.read(cache_offsets, [batch])
            q_projected = pl.slice(queries, [ROWS, FULL_Q_PROJECTION], [batch * ROWS, 0])
            k_projected = pl.slice(keys, [ROWS, FULL_KV_WIDTH], [batch * ROWS, 0])
            value_full = pl.slice(values, [ROWS, FULL_KV_WIDTH], [batch * ROWS, 0])
            cos_slice = pl.slice(cos_batch, [1, ROWS, ROTARY], [batch, 0, 0])
            sin_slice = pl.slice(sin_batch, [1, ROWS, ROTARY], [batch, 0, 0])
            mask_slice = pl.slice(masks, [1, ROWS, CONTEXT], [batch, 0, 0])
            table_slice = pl.slice(block_tables, [1, CONTEXT // BLOCK_SIZE], [batch, 0])
            cos = pl.reshape(cos_slice, [ROWS, ROTARY])
            sin = pl.reshape(sin_slice, [ROWS, ROTARY])
            mask = pl.reshape(mask_slice, [ROWS, CONTEXT])
            block_table = pl.reshape(table_slice, [CONTEXT // BLOCK_SIZE])
            q_norm_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm_full = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            gate_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            (q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full) = prefill_prepare_heads_inline(q_projected, k_projected, value_full, q_gamma_full, k_gamma_full, cos, sin, q_norm_full, k_norm_full, q_rot, k_rot, gate_full)
            key_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            value_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            (key_local, value_local) = prefill_load_paged_kv_inline(key_pool, value_pool, block_table, key_local, value_local, full, cache_offset, token_count)
            with pl.spmd(4, name_hint='qwen_full_update_kv'):
                head = pl.tile.get_block_idx()
                start = head * FULL_HEAD_DIM
                for token in pl.range(token_count):
                    key_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(k_rot, [1, FULL_HEAD_DIM], [token, start]), [1, 1, FULL_HEAD_DIM])
                    value_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_full, [1, FULL_HEAD_DIM], [token, start]), [1, 1, FULL_HEAD_DIM])
            scores = pl.create_tensor([24 * ROWS, CONTEXT], dtype=pl.FP32)
            probabilities = pl.create_tensor([24 * ROWS, CONTEXT], dtype=pl.BF16)
            weighted = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            full_gated = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            attention_full = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            full_gated = prefill_attention_core_inline(q_rot, key_local, value_local, gate_full, mask, out_weight_full, scores, probabilities, weighted, full_gated, attention_full)
            with pl.spmd(4 * ROWS, name_hint='qwen_full_store_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // ROWS
                row = task % ROWS
                if row < token_count:
                    position = cache_offset + row
                    logical_block = position // BLOCK_SIZE
                    physical = pl.read(block_table, [logical_block])
                    slot = position % BLOCK_SIZE
                    key_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(key_local, [1, 1, FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, FULL_HEAD_DIM])
                    value_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_local, [1, 1, FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, FULL_HEAD_DIM])
            with pl.spmd(ROWS, name_hint='qwen_prefill_store_attention_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:FULL_Q_WIDTH] = pl.slice(full_gated, [1, FULL_Q_WIDTH], [row, 0])
        else:
            with pl.spmd(ROWS, name_hint='qwen_prefill_zero_attention_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:FULL_Q_WIDTH] = pl.full([1, FULL_Q_WIDTH], dtype=pl.BF16, value=0.0)
    return (key_pool, value_pool, gated_batch)

@pl.jit.inline
def prefill_project_inline_prefix128(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    if active_count <= 2:
        with pl.spmd(1 * output_tiles, name_hint='prefill_nz_projection_active1'):
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
    elif active_count <= 4:
        with pl.spmd(2 * output_tiles, name_hint='prefill_nz_projection_active2'):
            task = pl.tile.get_block_idx()
            row = task % 2 * 128
            col = task // 2 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    elif active_count <= 6:
        with pl.spmd(3 * output_tiles, name_hint='prefill_nz_projection_active3'):
            task = pl.tile.get_block_idx()
            row = task % 3 * 128
            col = task // 3 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    else:
        with pl.spmd(4 * output_tiles, name_hint='prefill_nz_projection_active4'):
            task = pl.tile.get_block_idx()
            row = task % 4 * 128
            col = task // 4 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_norm_inline_prefix128(hidden: pl.Tensor, gamma: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(hidden, 0)
    with pl.spmd(packed_rows // rmsnorm_ROWS, name_hint='qwen_prefill_rmsnorm'):
        row_start = pl.tile.get_block_idx() * rmsnorm_ROWS
        square_sum = pl.full([1, rmsnorm_ROWS], dtype=pl.FP32, value=0.0)
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(pl.mul(source, source)), [1, rmsnorm_ROWS]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / rmsnorm_HIDDEN), 1e-06), high_precision=True), [rmsnorm_ROWS, 1])
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            weight = pl.slice(gamma, [1, rmsnorm_CHUNK], [0, column])
            normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weight)
            output[row_start:row_start + rmsnorm_ROWS, column:column + rmsnorm_CHUNK] = pl.cast(normalized, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_residual_inline_prefix128(left: pl.Tensor, right: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(left, 0)
    with pl.spmd(packed_rows, name_hint='qwen_prefill_residual'):
        row = pl.tile.get_block_idx()
        a = pl.cast(pl.slice(left, [1, HIDDEN], [row, 0]), pl.FP32)
        b = pl.cast(pl.slice(right, [1, HIDDEN], [row, 0]), pl.FP32)
        output[row:row + 1, 0:HIDDEN] = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_swiglu_inline_prefix128(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(gate, 0)
    with pl.spmd(packed_rows // 16 * (INTERMEDIATE // 1024), name_hint='qwen_prefill_swiglu'):
        task = pl.tile.get_block_idx()
        row = task // (INTERMEDIATE // 1024) * 16
        column = task % (INTERMEDIATE // 1024) * 1024
        gate_tile = pl.cast(pl.slice(gate, [16, 1024], [row, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        up_tile = pl.cast(pl.slice(up, [16, 1024], [row, column]), pl.FP32)
        output[row:row + 16, column:column + 1024] = pl.cast(pl.mul(pl.cast(activated, pl.FP32), up_tile), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_batched_conv_inline_prefix128(qkv_batch: pl.Tensor, conv_weight: pl.Tensor, conv_states: pl.Tensor, conv_output: pl.Tensor, token_counts: pl.Tensor, linear: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    state_layers = pl.tensor.dim(conv_states, 1)
    conv_rows = pl.reshape(conv_states, [BATCH * state_layers * 4, QKV])
    with pl.spmd(packed_rows // ROWS * (QKV // 512), name_hint='qwen_prefill_batched_conv'):
        task = pl.tile.get_block_idx()
        batch = task // (QKV // 512)
        column = task % (QKV // 512) * 512
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            state_start = (batch * state_layers + linear) * 4
            previous_0 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 0, column]), pl.FP32)
            previous_1 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 1, column]), pl.FP32)
            previous_2 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 2, column]), pl.FP32)
            previous_3 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 3, column]), pl.FP32)
            tap_0 = pl.cast(pl.slice(conv_weight, [1, 512], [0, column]), pl.FP32)
            tap_1 = pl.cast(pl.slice(conv_weight, [1, 512], [1, column]), pl.FP32)
            tap_2 = pl.cast(pl.slice(conv_weight, [1, 512], [2, column]), pl.FP32)
            tap_3 = pl.cast(pl.slice(conv_weight, [1, 512], [3, column]), pl.FP32)
            for token in pl.range(token_count):
                current = pl.cast(pl.slice(qkv_batch, [1, 512], [batch * ROWS + token, column]), pl.FP32)
                convolved = pl.add(pl.add(pl.mul(previous_1, tap_0), pl.mul(previous_2, tap_1)), pl.add(pl.mul(previous_3, tap_2), pl.mul(current, tap_3)))
                convolved = pl.cast(pl.cast(convolved, pl.BF16, mode='rint'), pl.FP32)
                activated = pl.div(convolved, pl.add(pl.exp(pl.neg(convolved)), 1.0), high_precision=True)
                conv_output[batch * ROWS + token:batch * ROWS + token + 1, column:column + 512] = pl.cast(activated, pl.BF16, mode='rint')
                previous_0 = previous_1
                previous_1 = previous_2
                previous_2 = previous_3
                previous_3 = current
            for token in pl.range(token_count, ROWS):
                conv_output[batch * ROWS + token:batch * ROWS + token + 1, column:column + 512] = pl.full([1, 512], dtype=pl.BF16, value=0.0)
            conv_rows[state_start + 0:state_start + 1, column:column + 512] = pl.cast(previous_0, pl.BF16, mode='rint')
            conv_rows[state_start + 1:state_start + 2, column:column + 512] = pl.cast(previous_1, pl.BF16, mode='rint')
            conv_rows[state_start + 2:state_start + 3, column:column + 512] = pl.cast(previous_2, pl.BF16, mode='rint')
            conv_rows[state_start + 3:state_start + 4, column:column + 512] = pl.cast(previous_3, pl.BF16, mode='rint')
        else:
            conv_output[batch * ROWS:(batch + 1) * ROWS, column:column + 512] = pl.full([ROWS, 512], dtype=pl.BF16, value=0.0)
    return (conv_states, conv_output)

@pl.jit.inline
def prefill_delta_slots_inline_prefix128(qkv_batch: pl.Tensor, z_batch: pl.Tensor, a_batch: pl.Tensor, b_batch: pl.Tensor, conv_weight_layer: pl.Tensor, a_log_layer: pl.Tensor, dt_bias_layer: pl.Tensor, delta_norm_layer: pl.Tensor, delta_out_layer: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, gated_batch: pl.Tensor, linear: pl.Scalar[pl.INDEX], token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    alias_state_layers = pl.tensor.dim(recurrent_states, 1)
    alias_conv_rows = pl.reshape(conv_states, [alias_state_layers * (BATCH * 4), QKV])
    alias_recurrent_rows = pl.reshape(recurrent_states, [alias_state_layers * (BATCH * DELTA_HEADS * DELTA_DIM), DELTA_DIM])
    conv_output_storage = pl.create_tensor([packed_rows, QKV], dtype=pl.BF16)
    (updated_conv_states, conv_output_batch) = prefill_batched_conv_inline_prefix128(qkv_batch, conv_weight_layer, conv_states, conv_output_storage, token_counts, linear, active_count)
    for batch in pl.range(packed_rows // ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            qkv = pl.slice(qkv_batch, [ROWS, QKV], [batch * ROWS, 0])
            z = pl.slice(z_batch, [ROWS, VALUE_WIDTH], [batch * ROWS, 0])
            a = pl.slice(a_batch, [ROWS, 128], [batch * ROWS, 0])
            b = pl.slice(b_batch, [ROWS, 128], [batch * ROWS, 0])
            conv_source = pl.slice(alias_conv_rows, [4, QKV], [(batch * alias_state_layers + linear) * 4, 0])
            recurrent_slice = pl.slice(alias_recurrent_rows, [DELTA_HEADS * DELTA_DIM, DELTA_DIM], [(batch * alias_state_layers + linear) * DELTA_HEADS * DELTA_DIM, 0])
            conv = pl.reshape(conv_source, [4, QKV])
            recurrent = pl.reshape(recurrent_slice, [DELTA_HEADS, DELTA_DIM, DELTA_DIM])
            conv_output = pl.slice(conv_output_batch, [ROWS, QKV], [batch * ROWS, 0])
            log_decay = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            beta = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            core = pl.create_tensor([ROWS, DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            gate_values = pl.create_tensor([ROWS, 64], dtype=pl.FP32)
            beta_bf16 = pl.create_tensor([ROWS, 64], dtype=pl.BF16)
            (gate_values, beta_bf16) = delta_prepare_decay_beta_inline(a, b, a_log_layer, dt_bias_layer, gate_values, beta_bf16)
            with pl.spmd(ROWS, name_hint='qwen_full_delta_gate_unpad'):
                token = pl.tile.get_block_idx()
                log_decay[token:token + 1, 0:DELTA_HEADS] = pl.slice(gate_values, [1, DELTA_HEADS], [token, 0])
                beta[token:token + 1, 0:DELTA_HEADS] = pl.cast(pl.slice(beta_bf16, [1, DELTA_HEADS], [token, 0]), pl.FP32)
            (core, recurrent) = prefill_direct_qkv_recurrence_inline(conv_output, log_decay, beta, recurrent, core, token_count)
            gated = pl.create_tensor([ROWS * DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            ignored = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            gated = prefill_gated_tail_inline(core, z, delta_norm_layer, delta_out_layer, gated, ignored)
            packed = pl.reshape(gated, [ROWS, VALUE_WIDTH])
            with pl.spmd(ROWS, name_hint='qwen_prefill_store_delta_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:VALUE_WIDTH] = pl.slice(packed, [1, VALUE_WIDTH], [row, 0])
        else:
            with pl.spmd(ROWS, name_hint='qwen_prefill_zero_delta_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:VALUE_WIDTH] = pl.full([1, VALUE_WIDTH], dtype=pl.BF16, value=0.0)
    return (conv_states, recurrent_states, gated_batch)

@pl.jit.inline
def prefill_attention_slots_inline_prefix128(queries: pl.Tensor, keys: pl.Tensor, values: pl.Tensor, q_gamma_full: pl.Tensor, k_gamma_full: pl.Tensor, cos_batch: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin_batch: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], masks: pl.Tensor, out_weight_full: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, cache_offsets: pl.Tensor, token_counts: pl.Tensor, gated_batch: pl.Tensor, full: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(gated_batch, 0)
    for batch in pl.range(packed_rows // ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            cache_offset = pl.read(cache_offsets, [batch])
            q_projected = pl.slice(queries, [ROWS, FULL_Q_PROJECTION], [batch * ROWS, 0])
            k_projected = pl.slice(keys, [ROWS, FULL_KV_WIDTH], [batch * ROWS, 0])
            value_full = pl.slice(values, [ROWS, FULL_KV_WIDTH], [batch * ROWS, 0])
            cos_slice = pl.slice(cos_batch, [1, ROWS, ROTARY], [batch, 0, 0])
            sin_slice = pl.slice(sin_batch, [1, ROWS, ROTARY], [batch, 0, 0])
            mask_slice = pl.slice(masks, [1, ROWS, CONTEXT], [batch, 0, 0])
            table_slice = pl.slice(block_tables, [1, CONTEXT // BLOCK_SIZE], [batch, 0])
            cos = pl.reshape(cos_slice, [ROWS, ROTARY])
            sin = pl.reshape(sin_slice, [ROWS, ROTARY])
            mask = pl.reshape(mask_slice, [ROWS, CONTEXT])
            block_table = pl.reshape(table_slice, [CONTEXT // BLOCK_SIZE])
            q_norm_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm_full = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            gate_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            (q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full) = prefill_prepare_heads_inline(q_projected, k_projected, value_full, q_gamma_full, k_gamma_full, cos, sin, q_norm_full, k_norm_full, q_rot, k_rot, gate_full)
            key_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            value_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            (key_local, value_local) = prefill_load_paged_kv_inline(key_pool, value_pool, block_table, key_local, value_local, full, cache_offset, token_count)
            with pl.spmd(4, name_hint='qwen_full_update_kv'):
                head = pl.tile.get_block_idx()
                start = head * FULL_HEAD_DIM
                for token in pl.range(token_count):
                    key_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(k_rot, [1, FULL_HEAD_DIM], [token, start]), [1, 1, FULL_HEAD_DIM])
                    value_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_full, [1, FULL_HEAD_DIM], [token, start]), [1, 1, FULL_HEAD_DIM])
            scores = pl.create_tensor([24 * ROWS, CONTEXT], dtype=pl.FP32)
            probabilities = pl.create_tensor([24 * ROWS, CONTEXT], dtype=pl.BF16)
            weighted = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            full_gated = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            attention_full = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            full_gated = prefill_attention_core_inline(q_rot, key_local, value_local, gate_full, mask, out_weight_full, scores, probabilities, weighted, full_gated, attention_full)
            with pl.spmd(4 * ROWS, name_hint='qwen_full_store_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // ROWS
                row = task % ROWS
                if row < token_count:
                    position = cache_offset + row
                    logical_block = position // BLOCK_SIZE
                    physical = pl.read(block_table, [logical_block])
                    slot = position % BLOCK_SIZE
                    key_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(key_local, [1, 1, FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, FULL_HEAD_DIM])
                    value_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_local, [1, 1, FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, FULL_HEAD_DIM])
            with pl.spmd(ROWS, name_hint='qwen_prefill_store_attention_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:FULL_Q_WIDTH] = pl.slice(full_gated, [1, FULL_Q_WIDTH], [row, 0])
        else:
            with pl.spmd(ROWS, name_hint='qwen_prefill_zero_attention_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:FULL_Q_WIDTH] = pl.full([1, FULL_Q_WIDTH], dtype=pl.BF16, value=0.0)
    return (key_pool, value_pool, gated_batch)

@pl.jit.inline
def prefill_last_logits_inline_prefix128(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    selected_hidden = pl.create_tensor([16, HIDDEN], dtype=pl.BF16)
    with pl.spmd(16, name_hint='prefill_gather_last_hidden'):
        batch = pl.tile.get_block_idx()
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if count > 0:
                source_row = batch * ROWS + count - 1
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.slice(hidden, [1, HIDDEN], [source_row, 0])
            else:
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
        else:
            selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
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
        for batch in pl.range(BATCH):
            if batch < active_count:
                count = pl.read(token_counts, [batch])
                if count > 0:
                    destination_row = batch * ROWS + count - 1
                    output[destination_row:destination_row + 1, column:column + 128] = pl.slice(projected, [1, 128], [batch, 0])
    return output

@pl.jit.inline
def prefill_project_inline_prefix256(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    if active_count <= 2:
        with pl.spmd(1 * output_tiles, name_hint='prefill_nz_projection_active1'):
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
    elif active_count <= 4:
        with pl.spmd(2 * output_tiles, name_hint='prefill_nz_projection_active2'):
            task = pl.tile.get_block_idx()
            row = task % 2 * 128
            col = task // 2 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    elif active_count <= 6:
        with pl.spmd(3 * output_tiles, name_hint='prefill_nz_projection_active3'):
            task = pl.tile.get_block_idx()
            row = task % 3 * 128
            col = task // 3 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    else:
        with pl.spmd(4 * output_tiles, name_hint='prefill_nz_projection_active4'):
            task = pl.tile.get_block_idx()
            row = task % 4 * 128
            col = task // 4 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_norm_inline_prefix256(hidden: pl.Tensor, gamma: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(hidden, 0)
    with pl.spmd(packed_rows // rmsnorm_ROWS, name_hint='qwen_prefill_rmsnorm'):
        row_start = pl.tile.get_block_idx() * rmsnorm_ROWS
        square_sum = pl.full([1, rmsnorm_ROWS], dtype=pl.FP32, value=0.0)
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(pl.mul(source, source)), [1, rmsnorm_ROWS]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / rmsnorm_HIDDEN), 1e-06), high_precision=True), [rmsnorm_ROWS, 1])
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            weight = pl.slice(gamma, [1, rmsnorm_CHUNK], [0, column])
            normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weight)
            output[row_start:row_start + rmsnorm_ROWS, column:column + rmsnorm_CHUNK] = pl.cast(normalized, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_residual_inline_prefix256(left: pl.Tensor, right: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(left, 0)
    with pl.spmd(packed_rows, name_hint='qwen_prefill_residual'):
        row = pl.tile.get_block_idx()
        a = pl.cast(pl.slice(left, [1, HIDDEN], [row, 0]), pl.FP32)
        b = pl.cast(pl.slice(right, [1, HIDDEN], [row, 0]), pl.FP32)
        output[row:row + 1, 0:HIDDEN] = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_swiglu_inline_prefix256(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(gate, 0)
    with pl.spmd(packed_rows // 16 * (INTERMEDIATE // 1024), name_hint='qwen_prefill_swiglu'):
        task = pl.tile.get_block_idx()
        row = task // (INTERMEDIATE // 1024) * 16
        column = task % (INTERMEDIATE // 1024) * 1024
        gate_tile = pl.cast(pl.slice(gate, [16, 1024], [row, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        up_tile = pl.cast(pl.slice(up, [16, 1024], [row, column]), pl.FP32)
        output[row:row + 16, column:column + 1024] = pl.cast(pl.mul(pl.cast(activated, pl.FP32), up_tile), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_batched_conv_inline_prefix256(qkv_batch: pl.Tensor, conv_weight: pl.Tensor, conv_states: pl.Tensor, conv_output: pl.Tensor, token_counts: pl.Tensor, linear: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    state_layers = pl.tensor.dim(conv_states, 1)
    conv_rows = pl.reshape(conv_states, [BATCH * state_layers * 4, QKV])
    with pl.spmd(packed_rows // ROWS * (QKV // 512), name_hint='qwen_prefill_batched_conv'):
        task = pl.tile.get_block_idx()
        batch = task // (QKV // 512)
        column = task % (QKV // 512) * 512
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            state_start = (batch * state_layers + linear) * 4
            previous_0 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 0, column]), pl.FP32)
            previous_1 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 1, column]), pl.FP32)
            previous_2 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 2, column]), pl.FP32)
            previous_3 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 3, column]), pl.FP32)
            tap_0 = pl.cast(pl.slice(conv_weight, [1, 512], [0, column]), pl.FP32)
            tap_1 = pl.cast(pl.slice(conv_weight, [1, 512], [1, column]), pl.FP32)
            tap_2 = pl.cast(pl.slice(conv_weight, [1, 512], [2, column]), pl.FP32)
            tap_3 = pl.cast(pl.slice(conv_weight, [1, 512], [3, column]), pl.FP32)
            for token in pl.range(token_count):
                current = pl.cast(pl.slice(qkv_batch, [1, 512], [batch * ROWS + token, column]), pl.FP32)
                convolved = pl.add(pl.add(pl.mul(previous_1, tap_0), pl.mul(previous_2, tap_1)), pl.add(pl.mul(previous_3, tap_2), pl.mul(current, tap_3)))
                convolved = pl.cast(pl.cast(convolved, pl.BF16, mode='rint'), pl.FP32)
                activated = pl.div(convolved, pl.add(pl.exp(pl.neg(convolved)), 1.0), high_precision=True)
                conv_output[batch * ROWS + token:batch * ROWS + token + 1, column:column + 512] = pl.cast(activated, pl.BF16, mode='rint')
                previous_0 = previous_1
                previous_1 = previous_2
                previous_2 = previous_3
                previous_3 = current
            for token in pl.range(token_count, ROWS):
                conv_output[batch * ROWS + token:batch * ROWS + token + 1, column:column + 512] = pl.full([1, 512], dtype=pl.BF16, value=0.0)
            conv_rows[state_start + 0:state_start + 1, column:column + 512] = pl.cast(previous_0, pl.BF16, mode='rint')
            conv_rows[state_start + 1:state_start + 2, column:column + 512] = pl.cast(previous_1, pl.BF16, mode='rint')
            conv_rows[state_start + 2:state_start + 3, column:column + 512] = pl.cast(previous_2, pl.BF16, mode='rint')
            conv_rows[state_start + 3:state_start + 4, column:column + 512] = pl.cast(previous_3, pl.BF16, mode='rint')
        else:
            conv_output[batch * ROWS:(batch + 1) * ROWS, column:column + 512] = pl.full([ROWS, 512], dtype=pl.BF16, value=0.0)
    return (conv_states, conv_output)

@pl.jit.inline
def prefill_delta_slots_inline_prefix256(qkv_batch: pl.Tensor, z_batch: pl.Tensor, a_batch: pl.Tensor, b_batch: pl.Tensor, conv_weight_layer: pl.Tensor, a_log_layer: pl.Tensor, dt_bias_layer: pl.Tensor, delta_norm_layer: pl.Tensor, delta_out_layer: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, gated_batch: pl.Tensor, linear: pl.Scalar[pl.INDEX], token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    alias_state_layers = pl.tensor.dim(recurrent_states, 1)
    alias_conv_rows = pl.reshape(conv_states, [alias_state_layers * (BATCH * 4), QKV])
    alias_recurrent_rows = pl.reshape(recurrent_states, [alias_state_layers * (BATCH * DELTA_HEADS * DELTA_DIM), DELTA_DIM])
    conv_output_storage = pl.create_tensor([packed_rows, QKV], dtype=pl.BF16)
    (updated_conv_states, conv_output_batch) = prefill_batched_conv_inline_prefix256(qkv_batch, conv_weight_layer, conv_states, conv_output_storage, token_counts, linear, active_count)
    for batch in pl.range(packed_rows // ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            qkv = pl.slice(qkv_batch, [ROWS, QKV], [batch * ROWS, 0])
            z = pl.slice(z_batch, [ROWS, VALUE_WIDTH], [batch * ROWS, 0])
            a = pl.slice(a_batch, [ROWS, 128], [batch * ROWS, 0])
            b = pl.slice(b_batch, [ROWS, 128], [batch * ROWS, 0])
            conv_source = pl.slice(alias_conv_rows, [4, QKV], [(batch * alias_state_layers + linear) * 4, 0])
            recurrent_slice = pl.slice(alias_recurrent_rows, [DELTA_HEADS * DELTA_DIM, DELTA_DIM], [(batch * alias_state_layers + linear) * DELTA_HEADS * DELTA_DIM, 0])
            conv = pl.reshape(conv_source, [4, QKV])
            recurrent = pl.reshape(recurrent_slice, [DELTA_HEADS, DELTA_DIM, DELTA_DIM])
            conv_output = pl.slice(conv_output_batch, [ROWS, QKV], [batch * ROWS, 0])
            log_decay = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            beta = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            core = pl.create_tensor([ROWS, DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            gate_values = pl.create_tensor([ROWS, 64], dtype=pl.FP32)
            beta_bf16 = pl.create_tensor([ROWS, 64], dtype=pl.BF16)
            (gate_values, beta_bf16) = delta_prepare_decay_beta_inline(a, b, a_log_layer, dt_bias_layer, gate_values, beta_bf16)
            with pl.spmd(ROWS, name_hint='qwen_full_delta_gate_unpad'):
                token = pl.tile.get_block_idx()
                log_decay[token:token + 1, 0:DELTA_HEADS] = pl.slice(gate_values, [1, DELTA_HEADS], [token, 0])
                beta[token:token + 1, 0:DELTA_HEADS] = pl.cast(pl.slice(beta_bf16, [1, DELTA_HEADS], [token, 0]), pl.FP32)
            (core, recurrent) = prefill_direct_qkv_recurrence_inline(conv_output, log_decay, beta, recurrent, core, token_count)
            gated = pl.create_tensor([ROWS * DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            ignored = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            gated = prefill_gated_tail_inline(core, z, delta_norm_layer, delta_out_layer, gated, ignored)
            packed = pl.reshape(gated, [ROWS, VALUE_WIDTH])
            with pl.spmd(ROWS, name_hint='qwen_prefill_store_delta_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:VALUE_WIDTH] = pl.slice(packed, [1, VALUE_WIDTH], [row, 0])
        else:
            with pl.spmd(ROWS, name_hint='qwen_prefill_zero_delta_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:VALUE_WIDTH] = pl.full([1, VALUE_WIDTH], dtype=pl.BF16, value=0.0)
    return (conv_states, recurrent_states, gated_batch)

@pl.jit.inline
def prefill_attention_slots_inline_prefix256(queries: pl.Tensor, keys: pl.Tensor, values: pl.Tensor, q_gamma_full: pl.Tensor, k_gamma_full: pl.Tensor, cos_batch: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin_batch: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], masks: pl.Tensor, out_weight_full: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, cache_offsets: pl.Tensor, token_counts: pl.Tensor, gated_batch: pl.Tensor, full: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(gated_batch, 0)
    for batch in pl.range(packed_rows // ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            cache_offset = pl.read(cache_offsets, [batch])
            q_projected = pl.slice(queries, [ROWS, FULL_Q_PROJECTION], [batch * ROWS, 0])
            k_projected = pl.slice(keys, [ROWS, FULL_KV_WIDTH], [batch * ROWS, 0])
            value_full = pl.slice(values, [ROWS, FULL_KV_WIDTH], [batch * ROWS, 0])
            cos_slice = pl.slice(cos_batch, [1, ROWS, ROTARY], [batch, 0, 0])
            sin_slice = pl.slice(sin_batch, [1, ROWS, ROTARY], [batch, 0, 0])
            mask_slice = pl.slice(masks, [1, ROWS, CONTEXT], [batch, 0, 0])
            table_slice = pl.slice(block_tables, [1, CONTEXT // BLOCK_SIZE], [batch, 0])
            cos = pl.reshape(cos_slice, [ROWS, ROTARY])
            sin = pl.reshape(sin_slice, [ROWS, ROTARY])
            mask = pl.reshape(mask_slice, [ROWS, CONTEXT])
            block_table = pl.reshape(table_slice, [CONTEXT // BLOCK_SIZE])
            q_norm_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm_full = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            gate_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            (q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full) = prefill_prepare_heads_inline(q_projected, k_projected, value_full, q_gamma_full, k_gamma_full, cos, sin, q_norm_full, k_norm_full, q_rot, k_rot, gate_full)
            key_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            value_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            (key_local, value_local) = prefill_load_paged_kv_inline(key_pool, value_pool, block_table, key_local, value_local, full, cache_offset, token_count)
            with pl.spmd(4, name_hint='qwen_full_update_kv'):
                head = pl.tile.get_block_idx()
                start = head * FULL_HEAD_DIM
                for token in pl.range(token_count):
                    key_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(k_rot, [1, FULL_HEAD_DIM], [token, start]), [1, 1, FULL_HEAD_DIM])
                    value_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_full, [1, FULL_HEAD_DIM], [token, start]), [1, 1, FULL_HEAD_DIM])
            scores = pl.create_tensor([24 * ROWS, CONTEXT], dtype=pl.FP32)
            probabilities = pl.create_tensor([24 * ROWS, CONTEXT], dtype=pl.BF16)
            weighted = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            full_gated = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            attention_full = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            full_gated = prefill_attention_core_inline(q_rot, key_local, value_local, gate_full, mask, out_weight_full, scores, probabilities, weighted, full_gated, attention_full)
            with pl.spmd(4 * ROWS, name_hint='qwen_full_store_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // ROWS
                row = task % ROWS
                if row < token_count:
                    position = cache_offset + row
                    logical_block = position // BLOCK_SIZE
                    physical = pl.read(block_table, [logical_block])
                    slot = position % BLOCK_SIZE
                    key_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(key_local, [1, 1, FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, FULL_HEAD_DIM])
                    value_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_local, [1, 1, FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, FULL_HEAD_DIM])
            with pl.spmd(ROWS, name_hint='qwen_prefill_store_attention_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:FULL_Q_WIDTH] = pl.slice(full_gated, [1, FULL_Q_WIDTH], [row, 0])
        else:
            with pl.spmd(ROWS, name_hint='qwen_prefill_zero_attention_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:FULL_Q_WIDTH] = pl.full([1, FULL_Q_WIDTH], dtype=pl.BF16, value=0.0)
    return (key_pool, value_pool, gated_batch)

@pl.jit.inline
def prefill_last_logits_inline_prefix256(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    selected_hidden = pl.create_tensor([16, HIDDEN], dtype=pl.BF16)
    with pl.spmd(16, name_hint='prefill_gather_last_hidden'):
        batch = pl.tile.get_block_idx()
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if count > 0:
                source_row = batch * ROWS + count - 1
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.slice(hidden, [1, HIDDEN], [source_row, 0])
            else:
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
        else:
            selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
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
        for batch in pl.range(BATCH):
            if batch < active_count:
                count = pl.read(token_counts, [batch])
                if count > 0:
                    destination_row = batch * ROWS + count - 1
                    output[destination_row:destination_row + 1, column:column + 128] = pl.slice(projected, [1, 128], [batch, 0])
    return output

@pl.jit.inline
def prefill_project_inline_prefix384(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    if active_count <= 2:
        with pl.spmd(1 * output_tiles, name_hint='prefill_nz_projection_active1'):
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
    elif active_count <= 4:
        with pl.spmd(2 * output_tiles, name_hint='prefill_nz_projection_active2'):
            task = pl.tile.get_block_idx()
            row = task % 2 * 128
            col = task // 2 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    elif active_count <= 6:
        with pl.spmd(3 * output_tiles, name_hint='prefill_nz_projection_active3'):
            task = pl.tile.get_block_idx()
            row = task % 3 * 128
            col = task // 3 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    else:
        with pl.spmd(4 * output_tiles, name_hint='prefill_nz_projection_active4'):
            task = pl.tile.get_block_idx()
            row = task % 4 * 128
            col = task // 4 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_norm_inline_prefix384(hidden: pl.Tensor, gamma: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(hidden, 0)
    with pl.spmd(packed_rows // rmsnorm_ROWS, name_hint='qwen_prefill_rmsnorm'):
        row_start = pl.tile.get_block_idx() * rmsnorm_ROWS
        square_sum = pl.full([1, rmsnorm_ROWS], dtype=pl.FP32, value=0.0)
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(pl.mul(source, source)), [1, rmsnorm_ROWS]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / rmsnorm_HIDDEN), 1e-06), high_precision=True), [rmsnorm_ROWS, 1])
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            weight = pl.slice(gamma, [1, rmsnorm_CHUNK], [0, column])
            normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weight)
            output[row_start:row_start + rmsnorm_ROWS, column:column + rmsnorm_CHUNK] = pl.cast(normalized, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_residual_inline_prefix384(left: pl.Tensor, right: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(left, 0)
    with pl.spmd(packed_rows, name_hint='qwen_prefill_residual'):
        row = pl.tile.get_block_idx()
        a = pl.cast(pl.slice(left, [1, HIDDEN], [row, 0]), pl.FP32)
        b = pl.cast(pl.slice(right, [1, HIDDEN], [row, 0]), pl.FP32)
        output[row:row + 1, 0:HIDDEN] = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_swiglu_inline_prefix384(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(gate, 0)
    with pl.spmd(packed_rows // 16 * (INTERMEDIATE // 1024), name_hint='qwen_prefill_swiglu'):
        task = pl.tile.get_block_idx()
        row = task // (INTERMEDIATE // 1024) * 16
        column = task % (INTERMEDIATE // 1024) * 1024
        gate_tile = pl.cast(pl.slice(gate, [16, 1024], [row, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        up_tile = pl.cast(pl.slice(up, [16, 1024], [row, column]), pl.FP32)
        output[row:row + 16, column:column + 1024] = pl.cast(pl.mul(pl.cast(activated, pl.FP32), up_tile), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_batched_conv_inline_prefix384(qkv_batch: pl.Tensor, conv_weight: pl.Tensor, conv_states: pl.Tensor, conv_output: pl.Tensor, token_counts: pl.Tensor, linear: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    state_layers = pl.tensor.dim(conv_states, 1)
    conv_rows = pl.reshape(conv_states, [BATCH * state_layers * 4, QKV])
    with pl.spmd(packed_rows // ROWS * (QKV // 512), name_hint='qwen_prefill_batched_conv'):
        task = pl.tile.get_block_idx()
        batch = task // (QKV // 512)
        column = task % (QKV // 512) * 512
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            state_start = (batch * state_layers + linear) * 4
            previous_0 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 0, column]), pl.FP32)
            previous_1 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 1, column]), pl.FP32)
            previous_2 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 2, column]), pl.FP32)
            previous_3 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 3, column]), pl.FP32)
            tap_0 = pl.cast(pl.slice(conv_weight, [1, 512], [0, column]), pl.FP32)
            tap_1 = pl.cast(pl.slice(conv_weight, [1, 512], [1, column]), pl.FP32)
            tap_2 = pl.cast(pl.slice(conv_weight, [1, 512], [2, column]), pl.FP32)
            tap_3 = pl.cast(pl.slice(conv_weight, [1, 512], [3, column]), pl.FP32)
            for token in pl.range(token_count):
                current = pl.cast(pl.slice(qkv_batch, [1, 512], [batch * ROWS + token, column]), pl.FP32)
                convolved = pl.add(pl.add(pl.mul(previous_1, tap_0), pl.mul(previous_2, tap_1)), pl.add(pl.mul(previous_3, tap_2), pl.mul(current, tap_3)))
                convolved = pl.cast(pl.cast(convolved, pl.BF16, mode='rint'), pl.FP32)
                activated = pl.div(convolved, pl.add(pl.exp(pl.neg(convolved)), 1.0), high_precision=True)
                conv_output[batch * ROWS + token:batch * ROWS + token + 1, column:column + 512] = pl.cast(activated, pl.BF16, mode='rint')
                previous_0 = previous_1
                previous_1 = previous_2
                previous_2 = previous_3
                previous_3 = current
            for token in pl.range(token_count, ROWS):
                conv_output[batch * ROWS + token:batch * ROWS + token + 1, column:column + 512] = pl.full([1, 512], dtype=pl.BF16, value=0.0)
            conv_rows[state_start + 0:state_start + 1, column:column + 512] = pl.cast(previous_0, pl.BF16, mode='rint')
            conv_rows[state_start + 1:state_start + 2, column:column + 512] = pl.cast(previous_1, pl.BF16, mode='rint')
            conv_rows[state_start + 2:state_start + 3, column:column + 512] = pl.cast(previous_2, pl.BF16, mode='rint')
            conv_rows[state_start + 3:state_start + 4, column:column + 512] = pl.cast(previous_3, pl.BF16, mode='rint')
        else:
            conv_output[batch * ROWS:(batch + 1) * ROWS, column:column + 512] = pl.full([ROWS, 512], dtype=pl.BF16, value=0.0)
    return (conv_states, conv_output)

@pl.jit.inline
def prefill_delta_slots_inline_prefix384(qkv_batch: pl.Tensor, z_batch: pl.Tensor, a_batch: pl.Tensor, b_batch: pl.Tensor, conv_weight_layer: pl.Tensor, a_log_layer: pl.Tensor, dt_bias_layer: pl.Tensor, delta_norm_layer: pl.Tensor, delta_out_layer: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, gated_batch: pl.Tensor, linear: pl.Scalar[pl.INDEX], token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    alias_state_layers = pl.tensor.dim(recurrent_states, 1)
    alias_conv_rows = pl.reshape(conv_states, [alias_state_layers * (BATCH * 4), QKV])
    alias_recurrent_rows = pl.reshape(recurrent_states, [alias_state_layers * (BATCH * DELTA_HEADS * DELTA_DIM), DELTA_DIM])
    conv_output_storage = pl.create_tensor([packed_rows, QKV], dtype=pl.BF16)
    (updated_conv_states, conv_output_batch) = prefill_batched_conv_inline_prefix384(qkv_batch, conv_weight_layer, conv_states, conv_output_storage, token_counts, linear, active_count)
    for batch in pl.range(packed_rows // ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            qkv = pl.slice(qkv_batch, [ROWS, QKV], [batch * ROWS, 0])
            z = pl.slice(z_batch, [ROWS, VALUE_WIDTH], [batch * ROWS, 0])
            a = pl.slice(a_batch, [ROWS, 128], [batch * ROWS, 0])
            b = pl.slice(b_batch, [ROWS, 128], [batch * ROWS, 0])
            conv_source = pl.slice(alias_conv_rows, [4, QKV], [(batch * alias_state_layers + linear) * 4, 0])
            recurrent_slice = pl.slice(alias_recurrent_rows, [DELTA_HEADS * DELTA_DIM, DELTA_DIM], [(batch * alias_state_layers + linear) * DELTA_HEADS * DELTA_DIM, 0])
            conv = pl.reshape(conv_source, [4, QKV])
            recurrent = pl.reshape(recurrent_slice, [DELTA_HEADS, DELTA_DIM, DELTA_DIM])
            conv_output = pl.slice(conv_output_batch, [ROWS, QKV], [batch * ROWS, 0])
            log_decay = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            beta = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            core = pl.create_tensor([ROWS, DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            gate_values = pl.create_tensor([ROWS, 64], dtype=pl.FP32)
            beta_bf16 = pl.create_tensor([ROWS, 64], dtype=pl.BF16)
            (gate_values, beta_bf16) = delta_prepare_decay_beta_inline(a, b, a_log_layer, dt_bias_layer, gate_values, beta_bf16)
            with pl.spmd(ROWS, name_hint='qwen_full_delta_gate_unpad'):
                token = pl.tile.get_block_idx()
                log_decay[token:token + 1, 0:DELTA_HEADS] = pl.slice(gate_values, [1, DELTA_HEADS], [token, 0])
                beta[token:token + 1, 0:DELTA_HEADS] = pl.cast(pl.slice(beta_bf16, [1, DELTA_HEADS], [token, 0]), pl.FP32)
            (core, recurrent) = prefill_direct_qkv_recurrence_inline(conv_output, log_decay, beta, recurrent, core, token_count)
            gated = pl.create_tensor([ROWS * DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            ignored = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            gated = prefill_gated_tail_inline(core, z, delta_norm_layer, delta_out_layer, gated, ignored)
            packed = pl.reshape(gated, [ROWS, VALUE_WIDTH])
            with pl.spmd(ROWS, name_hint='qwen_prefill_store_delta_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:VALUE_WIDTH] = pl.slice(packed, [1, VALUE_WIDTH], [row, 0])
        else:
            with pl.spmd(ROWS, name_hint='qwen_prefill_zero_delta_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:VALUE_WIDTH] = pl.full([1, VALUE_WIDTH], dtype=pl.BF16, value=0.0)
    return (conv_states, recurrent_states, gated_batch)

@pl.jit.inline
def prefill_attention_slots_inline_prefix384(queries: pl.Tensor, keys: pl.Tensor, values: pl.Tensor, q_gamma_full: pl.Tensor, k_gamma_full: pl.Tensor, cos_batch: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin_batch: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], masks: pl.Tensor, out_weight_full: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, cache_offsets: pl.Tensor, token_counts: pl.Tensor, gated_batch: pl.Tensor, full: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(gated_batch, 0)
    for batch in pl.range(packed_rows // ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            cache_offset = pl.read(cache_offsets, [batch])
            q_projected = pl.slice(queries, [ROWS, FULL_Q_PROJECTION], [batch * ROWS, 0])
            k_projected = pl.slice(keys, [ROWS, FULL_KV_WIDTH], [batch * ROWS, 0])
            value_full = pl.slice(values, [ROWS, FULL_KV_WIDTH], [batch * ROWS, 0])
            cos_slice = pl.slice(cos_batch, [1, ROWS, ROTARY], [batch, 0, 0])
            sin_slice = pl.slice(sin_batch, [1, ROWS, ROTARY], [batch, 0, 0])
            mask_slice = pl.slice(masks, [1, ROWS, CONTEXT], [batch, 0, 0])
            table_slice = pl.slice(block_tables, [1, CONTEXT // BLOCK_SIZE], [batch, 0])
            cos = pl.reshape(cos_slice, [ROWS, ROTARY])
            sin = pl.reshape(sin_slice, [ROWS, ROTARY])
            mask = pl.reshape(mask_slice, [ROWS, CONTEXT])
            block_table = pl.reshape(table_slice, [CONTEXT // BLOCK_SIZE])
            q_norm_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm_full = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            gate_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            (q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full) = prefill_prepare_heads_inline(q_projected, k_projected, value_full, q_gamma_full, k_gamma_full, cos, sin, q_norm_full, k_norm_full, q_rot, k_rot, gate_full)
            key_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            value_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            (key_local, value_local) = prefill_load_paged_kv_inline(key_pool, value_pool, block_table, key_local, value_local, full, cache_offset, token_count)
            with pl.spmd(4, name_hint='qwen_full_update_kv'):
                head = pl.tile.get_block_idx()
                start = head * FULL_HEAD_DIM
                for token in pl.range(token_count):
                    key_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(k_rot, [1, FULL_HEAD_DIM], [token, start]), [1, 1, FULL_HEAD_DIM])
                    value_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_full, [1, FULL_HEAD_DIM], [token, start]), [1, 1, FULL_HEAD_DIM])
            scores = pl.create_tensor([24 * ROWS, CONTEXT], dtype=pl.FP32)
            probabilities = pl.create_tensor([24 * ROWS, CONTEXT], dtype=pl.BF16)
            weighted = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            full_gated = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            attention_full = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            full_gated = prefill_attention_core_inline(q_rot, key_local, value_local, gate_full, mask, out_weight_full, scores, probabilities, weighted, full_gated, attention_full)
            with pl.spmd(4 * ROWS, name_hint='qwen_full_store_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // ROWS
                row = task % ROWS
                if row < token_count:
                    position = cache_offset + row
                    logical_block = position // BLOCK_SIZE
                    physical = pl.read(block_table, [logical_block])
                    slot = position % BLOCK_SIZE
                    key_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(key_local, [1, 1, FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, FULL_HEAD_DIM])
                    value_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_local, [1, 1, FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, FULL_HEAD_DIM])
            with pl.spmd(ROWS, name_hint='qwen_prefill_store_attention_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:FULL_Q_WIDTH] = pl.slice(full_gated, [1, FULL_Q_WIDTH], [row, 0])
        else:
            with pl.spmd(ROWS, name_hint='qwen_prefill_zero_attention_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:FULL_Q_WIDTH] = pl.full([1, FULL_Q_WIDTH], dtype=pl.BF16, value=0.0)
    return (key_pool, value_pool, gated_batch)

@pl.jit.inline
def prefill_last_logits_inline_prefix384(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    selected_hidden = pl.create_tensor([16, HIDDEN], dtype=pl.BF16)
    with pl.spmd(16, name_hint='prefill_gather_last_hidden'):
        batch = pl.tile.get_block_idx()
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if count > 0:
                source_row = batch * ROWS + count - 1
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.slice(hidden, [1, HIDDEN], [source_row, 0])
            else:
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
        else:
            selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
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
        for batch in pl.range(BATCH):
            if batch < active_count:
                count = pl.read(token_counts, [batch])
                if count > 0:
                    destination_row = batch * ROWS + count - 1
                    output[destination_row:destination_row + 1, column:column + 128] = pl.slice(projected, [1, 128], [batch, 0])
    return output

@pl.jit.inline
def prefill_project_inline_prefix512(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(output, 1) // 128
    input_columns = pl.tensor.dim(hidden, 1)
    if active_count <= 2:
        with pl.spmd(1 * output_tiles, name_hint='prefill_nz_projection_active1'):
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
    elif active_count <= 4:
        with pl.spmd(2 * output_tiles, name_hint='prefill_nz_projection_active2'):
            task = pl.tile.get_block_idx()
            row = task % 2 * 128
            col = task // 2 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    elif active_count <= 6:
        with pl.spmd(3 * output_tiles, name_hint='prefill_nz_projection_active3'):
            task = pl.tile.get_block_idx()
            row = task % 3 * 128
            col = task // 3 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    else:
        with pl.spmd(4 * output_tiles, name_hint='prefill_nz_projection_active4'):
            task = pl.tile.get_block_idx()
            row = task % 4 * 128
            col = task // 4 * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [256, 128], [k_start, col])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=False, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_norm_inline_prefix512(hidden: pl.Tensor, gamma: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(hidden, 0)
    with pl.spmd(packed_rows // rmsnorm_ROWS, name_hint='qwen_prefill_rmsnorm'):
        row_start = pl.tile.get_block_idx() * rmsnorm_ROWS
        square_sum = pl.full([1, rmsnorm_ROWS], dtype=pl.FP32, value=0.0)
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(pl.mul(source, source)), [1, rmsnorm_ROWS]))
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / rmsnorm_HIDDEN), 1e-06), high_precision=True), [rmsnorm_ROWS, 1])
        for chunk_index in pl.range(rmsnorm_HIDDEN // rmsnorm_CHUNK):
            column = chunk_index * rmsnorm_CHUNK
            source = pl.cast(pl.slice(hidden, [rmsnorm_ROWS, rmsnorm_CHUNK], [row_start, column]), pl.FP32)
            weight = pl.slice(gamma, [1, rmsnorm_CHUNK], [0, column])
            normalized = pl.col_expand_mul(pl.row_expand_mul(source, inverse), weight)
            output[row_start:row_start + rmsnorm_ROWS, column:column + rmsnorm_CHUNK] = pl.cast(normalized, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_residual_inline_prefix512(left: pl.Tensor, right: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(left, 0)
    with pl.spmd(packed_rows, name_hint='qwen_prefill_residual'):
        row = pl.tile.get_block_idx()
        a = pl.cast(pl.slice(left, [1, HIDDEN], [row, 0]), pl.FP32)
        b = pl.cast(pl.slice(right, [1, HIDDEN], [row, 0]), pl.FP32)
        output[row:row + 1, 0:HIDDEN] = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_swiglu_inline_prefix512(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    packed_rows = pl.tensor.dim(gate, 0)
    with pl.spmd(packed_rows // 16 * (INTERMEDIATE // 1024), name_hint='qwen_prefill_swiglu'):
        task = pl.tile.get_block_idx()
        row = task // (INTERMEDIATE // 1024) * 16
        column = task % (INTERMEDIATE // 1024) * 1024
        gate_tile = pl.cast(pl.slice(gate, [16, 1024], [row, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        up_tile = pl.cast(pl.slice(up, [16, 1024], [row, column]), pl.FP32)
        output[row:row + 16, column:column + 1024] = pl.cast(pl.mul(pl.cast(activated, pl.FP32), up_tile), pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_batched_conv_inline_prefix512(qkv_batch: pl.Tensor, conv_weight: pl.Tensor, conv_states: pl.Tensor, conv_output: pl.Tensor, token_counts: pl.Tensor, linear: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    state_layers = pl.tensor.dim(conv_states, 1)
    conv_rows = pl.reshape(conv_states, [BATCH * state_layers * 4, QKV])
    with pl.spmd(packed_rows // ROWS * (QKV // 512), name_hint='qwen_prefill_batched_conv'):
        task = pl.tile.get_block_idx()
        batch = task // (QKV // 512)
        column = task % (QKV // 512) * 512
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            state_start = (batch * state_layers + linear) * 4
            previous_0 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 0, column]), pl.FP32)
            previous_1 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 1, column]), pl.FP32)
            previous_2 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 2, column]), pl.FP32)
            previous_3 = pl.cast(pl.slice(conv_rows, [1, 512], [state_start + 3, column]), pl.FP32)
            tap_0 = pl.cast(pl.slice(conv_weight, [1, 512], [0, column]), pl.FP32)
            tap_1 = pl.cast(pl.slice(conv_weight, [1, 512], [1, column]), pl.FP32)
            tap_2 = pl.cast(pl.slice(conv_weight, [1, 512], [2, column]), pl.FP32)
            tap_3 = pl.cast(pl.slice(conv_weight, [1, 512], [3, column]), pl.FP32)
            for token in pl.range(token_count):
                current = pl.cast(pl.slice(qkv_batch, [1, 512], [batch * ROWS + token, column]), pl.FP32)
                convolved = pl.add(pl.add(pl.mul(previous_1, tap_0), pl.mul(previous_2, tap_1)), pl.add(pl.mul(previous_3, tap_2), pl.mul(current, tap_3)))
                convolved = pl.cast(pl.cast(convolved, pl.BF16, mode='rint'), pl.FP32)
                activated = pl.div(convolved, pl.add(pl.exp(pl.neg(convolved)), 1.0), high_precision=True)
                conv_output[batch * ROWS + token:batch * ROWS + token + 1, column:column + 512] = pl.cast(activated, pl.BF16, mode='rint')
                previous_0 = previous_1
                previous_1 = previous_2
                previous_2 = previous_3
                previous_3 = current
            for token in pl.range(token_count, ROWS):
                conv_output[batch * ROWS + token:batch * ROWS + token + 1, column:column + 512] = pl.full([1, 512], dtype=pl.BF16, value=0.0)
            conv_rows[state_start + 0:state_start + 1, column:column + 512] = pl.cast(previous_0, pl.BF16, mode='rint')
            conv_rows[state_start + 1:state_start + 2, column:column + 512] = pl.cast(previous_1, pl.BF16, mode='rint')
            conv_rows[state_start + 2:state_start + 3, column:column + 512] = pl.cast(previous_2, pl.BF16, mode='rint')
            conv_rows[state_start + 3:state_start + 4, column:column + 512] = pl.cast(previous_3, pl.BF16, mode='rint')
        else:
            conv_output[batch * ROWS:(batch + 1) * ROWS, column:column + 512] = pl.full([ROWS, 512], dtype=pl.BF16, value=0.0)
    return (conv_states, conv_output)

@pl.jit.inline
def prefill_delta_slots_inline_prefix512(qkv_batch: pl.Tensor, z_batch: pl.Tensor, a_batch: pl.Tensor, b_batch: pl.Tensor, conv_weight_layer: pl.Tensor, a_log_layer: pl.Tensor, dt_bias_layer: pl.Tensor, delta_norm_layer: pl.Tensor, delta_out_layer: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, gated_batch: pl.Tensor, linear: pl.Scalar[pl.INDEX], token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(qkv_batch, 0)
    alias_state_layers = pl.tensor.dim(recurrent_states, 1)
    alias_conv_rows = pl.reshape(conv_states, [alias_state_layers * (BATCH * 4), QKV])
    alias_recurrent_rows = pl.reshape(recurrent_states, [alias_state_layers * (BATCH * DELTA_HEADS * DELTA_DIM), DELTA_DIM])
    conv_output_storage = pl.create_tensor([packed_rows, QKV], dtype=pl.BF16)
    (updated_conv_states, conv_output_batch) = prefill_batched_conv_inline_prefix512(qkv_batch, conv_weight_layer, conv_states, conv_output_storage, token_counts, linear, active_count)
    for batch in pl.range(packed_rows // ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            qkv = pl.slice(qkv_batch, [ROWS, QKV], [batch * ROWS, 0])
            z = pl.slice(z_batch, [ROWS, VALUE_WIDTH], [batch * ROWS, 0])
            a = pl.slice(a_batch, [ROWS, 128], [batch * ROWS, 0])
            b = pl.slice(b_batch, [ROWS, 128], [batch * ROWS, 0])
            conv_source = pl.slice(alias_conv_rows, [4, QKV], [(batch * alias_state_layers + linear) * 4, 0])
            recurrent_slice = pl.slice(alias_recurrent_rows, [DELTA_HEADS * DELTA_DIM, DELTA_DIM], [(batch * alias_state_layers + linear) * DELTA_HEADS * DELTA_DIM, 0])
            conv = pl.reshape(conv_source, [4, QKV])
            recurrent = pl.reshape(recurrent_slice, [DELTA_HEADS, DELTA_DIM, DELTA_DIM])
            conv_output = pl.slice(conv_output_batch, [ROWS, QKV], [batch * ROWS, 0])
            log_decay = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            beta = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            core = pl.create_tensor([ROWS, DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            gate_values = pl.create_tensor([ROWS, 64], dtype=pl.FP32)
            beta_bf16 = pl.create_tensor([ROWS, 64], dtype=pl.BF16)
            (gate_values, beta_bf16) = delta_prepare_decay_beta_inline(a, b, a_log_layer, dt_bias_layer, gate_values, beta_bf16)
            with pl.spmd(ROWS, name_hint='qwen_full_delta_gate_unpad'):
                token = pl.tile.get_block_idx()
                log_decay[token:token + 1, 0:DELTA_HEADS] = pl.slice(gate_values, [1, DELTA_HEADS], [token, 0])
                beta[token:token + 1, 0:DELTA_HEADS] = pl.cast(pl.slice(beta_bf16, [1, DELTA_HEADS], [token, 0]), pl.FP32)
            (core, recurrent) = prefill_direct_qkv_recurrence_inline(conv_output, log_decay, beta, recurrent, core, token_count)
            gated = pl.create_tensor([ROWS * DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            ignored = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            gated = prefill_gated_tail_inline(core, z, delta_norm_layer, delta_out_layer, gated, ignored)
            packed = pl.reshape(gated, [ROWS, VALUE_WIDTH])
            with pl.spmd(ROWS, name_hint='qwen_prefill_store_delta_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:VALUE_WIDTH] = pl.slice(packed, [1, VALUE_WIDTH], [row, 0])
        else:
            with pl.spmd(ROWS, name_hint='qwen_prefill_zero_delta_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:VALUE_WIDTH] = pl.full([1, VALUE_WIDTH], dtype=pl.BF16, value=0.0)
    return (conv_states, recurrent_states, gated_batch)

@pl.jit.inline
def prefill_attention_slots_inline_prefix512(queries: pl.Tensor, keys: pl.Tensor, values: pl.Tensor, q_gamma_full: pl.Tensor, k_gamma_full: pl.Tensor, cos_batch: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin_batch: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], masks: pl.Tensor, out_weight_full: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, cache_offsets: pl.Tensor, token_counts: pl.Tensor, gated_batch: pl.Tensor, full: pl.Scalar[pl.INDEX], active_count: pl.Scalar[pl.INT32]):
    packed_rows = pl.tensor.dim(gated_batch, 0)
    for batch in pl.range(packed_rows // ROWS):
        if batch < active_count:
            token_count = pl.read(token_counts, [batch])
            cache_offset = pl.read(cache_offsets, [batch])
            q_projected = pl.slice(queries, [ROWS, FULL_Q_PROJECTION], [batch * ROWS, 0])
            k_projected = pl.slice(keys, [ROWS, FULL_KV_WIDTH], [batch * ROWS, 0])
            value_full = pl.slice(values, [ROWS, FULL_KV_WIDTH], [batch * ROWS, 0])
            cos_slice = pl.slice(cos_batch, [1, ROWS, ROTARY], [batch, 0, 0])
            sin_slice = pl.slice(sin_batch, [1, ROWS, ROTARY], [batch, 0, 0])
            mask_slice = pl.slice(masks, [1, ROWS, CONTEXT], [batch, 0, 0])
            table_slice = pl.slice(block_tables, [1, CONTEXT // BLOCK_SIZE], [batch, 0])
            cos = pl.reshape(cos_slice, [ROWS, ROTARY])
            sin = pl.reshape(sin_slice, [ROWS, ROTARY])
            mask = pl.reshape(mask_slice, [ROWS, CONTEXT])
            block_table = pl.reshape(table_slice, [CONTEXT // BLOCK_SIZE])
            q_norm_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm_full = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            gate_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            (q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full) = prefill_prepare_heads_inline(q_projected, k_projected, value_full, q_gamma_full, k_gamma_full, cos, sin, q_norm_full, k_norm_full, q_rot, k_rot, gate_full)
            key_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            value_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            (key_local, value_local) = prefill_load_paged_kv_inline(key_pool, value_pool, block_table, key_local, value_local, full, cache_offset, token_count)
            with pl.spmd(4, name_hint='qwen_full_update_kv'):
                head = pl.tile.get_block_idx()
                start = head * FULL_HEAD_DIM
                for token in pl.range(token_count):
                    key_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(k_rot, [1, FULL_HEAD_DIM], [token, start]), [1, 1, FULL_HEAD_DIM])
                    value_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_full, [1, FULL_HEAD_DIM], [token, start]), [1, 1, FULL_HEAD_DIM])
            scores = pl.create_tensor([24 * ROWS, CONTEXT], dtype=pl.FP32)
            probabilities = pl.create_tensor([24 * ROWS, CONTEXT], dtype=pl.BF16)
            weighted = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            full_gated = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            attention_full = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            full_gated = prefill_attention_core_inline(q_rot, key_local, value_local, gate_full, mask, out_weight_full, scores, probabilities, weighted, full_gated, attention_full)
            with pl.spmd(4 * ROWS, name_hint='qwen_full_store_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // ROWS
                row = task % ROWS
                if row < token_count:
                    position = cache_offset + row
                    logical_block = position // BLOCK_SIZE
                    physical = pl.read(block_table, [logical_block])
                    slot = position % BLOCK_SIZE
                    key_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(key_local, [1, 1, FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, FULL_HEAD_DIM])
                    value_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_local, [1, 1, FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, FULL_HEAD_DIM])
            with pl.spmd(ROWS, name_hint='qwen_prefill_store_attention_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:FULL_Q_WIDTH] = pl.slice(full_gated, [1, FULL_Q_WIDTH], [row, 0])
        else:
            with pl.spmd(ROWS, name_hint='qwen_prefill_zero_attention_gated'):
                row = pl.tile.get_block_idx()
                gated_batch[batch * ROWS + row:batch * ROWS + row + 1, 0:FULL_Q_WIDTH] = pl.full([1, FULL_Q_WIDTH], dtype=pl.BF16, value=0.0)
    return (key_pool, value_pool, gated_batch)

@pl.jit.inline
def prefill_last_logits_inline_prefix512(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    selected_hidden = pl.create_tensor([16, HIDDEN], dtype=pl.BF16)
    with pl.spmd(16, name_hint='prefill_gather_last_hidden'):
        batch = pl.tile.get_block_idx()
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if count > 0:
                source_row = batch * ROWS + count - 1
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.slice(hidden, [1, HIDDEN], [source_row, 0])
            else:
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
        else:
            selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
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
        for batch in pl.range(BATCH):
            if batch < active_count:
                count = pl.read(token_counts, [batch])
                if count > 0:
                    destination_row = batch * ROWS + count - 1
                    output[destination_row:destination_row + 1, column:column + 128] = pl.slice(projected, [1, 128], [batch, 0])
    return output

@pl.jit.inline
def prefill_batch_prefix128_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    running = pl.create_tensor([128, HIDDEN], dtype=pl.BF16)
    with pl.spmd(128, name_hint='qwen_prefill_embedding'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if token_row < count:
                token = pl.read(input_ids, [batch, token_row])
                embedded = pl.slice(embedding, [1, HIDDEN], [token, 0])
            else:
                embedded = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
            running[row:row + 1, 0:HIDDEN] = embedded
            traces[batch:batch + 1, 0:1, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(embedded, [1, 1, 1, HIDDEN])
        else:
            running[row:row + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
    for layer in pl.range(64):
        input_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        mlp_up_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        mlp_down_layer: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        norm = pl.create_tensor([128, HIDDEN], dtype=pl.BF16)
        norm = prefill_norm_inline_prefix128(running, input_gamma_layer, norm)
        residual = pl.create_tensor([128, HIDDEN], dtype=pl.BF16)
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[HIDDEN, QKV], pl.BF16, pl.NZ] = delta_qkv_weight[linear]
            z_weight_layer: pl.Tensor[[HIDDEN, VALUE_WIDTH], pl.BF16, pl.NZ] = delta_z_weight[linear]
            a_weight_layer: pl.Tensor[[HIDDEN, 128], pl.BF16, pl.NZ] = delta_a_weight[linear]
            b_weight_layer: pl.Tensor[[HIDDEN, 128], pl.BF16, pl.NZ] = delta_b_weight[linear]
            conv_weight_layer: pl.Tensor[[4, QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[VALUE_WIDTH, HIDDEN], pl.BF16, pl.NZ] = delta_out_weight[linear]
            qkv = pl.create_tensor([128, QKV], dtype=pl.BF16)
            z = pl.create_tensor([128, VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([128, 128], dtype=pl.BF16)
            b = pl.create_tensor([128, 128], dtype=pl.BF16)
            qkv = prefill_project_inline_prefix128(norm, qkv_weight_layer, qkv, active_count)
            z = prefill_project_inline_prefix128(norm, z_weight_layer, z, active_count)
            a = prefill_project_inline_prefix128(norm, a_weight_layer, a, active_count)
            b = prefill_project_inline_prefix128(norm, b_weight_layer, b, active_count)
            gated = pl.create_tensor([128, VALUE_WIDTH], dtype=pl.BF16)
            prefill_delta_slots_inline_prefix128(qkv, z, a, b, conv_weight_layer, a_log_layer, dt_bias_layer, delta_norm_layer, delta_out_layer, conv_states, recurrent_states, gated, linear, token_counts, active_count)
            attention = pl.create_tensor([128, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_inline_prefix128(gated, delta_out_layer, attention, active_count)
            residual = prefill_residual_inline_prefix128(running, attention, residual)
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[full]
            k_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[full]
            v_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[full]
            q_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ] = full_out_weight[full]
            q = pl.create_tensor([128, FULL_Q_PROJECTION], dtype=pl.BF16)
            k = pl.create_tensor([128, FULL_KV_WIDTH], dtype=pl.BF16)
            v = pl.create_tensor([128, FULL_KV_WIDTH], dtype=pl.BF16)
            q = prefill_project_inline_prefix128(norm, q_weight_full, q, active_count)
            k = prefill_project_inline_prefix128(norm, k_weight_full, k, active_count)
            v = prefill_project_inline_prefix128(norm, v_weight_full, v, active_count)
            gated = pl.create_tensor([128, FULL_Q_WIDTH], dtype=pl.BF16)
            prefill_attention_slots_inline_prefix128(q, k, v, q_gamma_full, k_gamma_full, cos, sin, mask, out_weight_full, key_pool, value_pool, block_tables, cache_offsets, token_counts, gated, full, active_count)
            attention = pl.create_tensor([128, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_inline_prefix128(gated, out_weight_full, attention, active_count)
            residual = prefill_residual_inline_prefix128(running, attention, residual)
        post_norm = pl.create_tensor([128, HIDDEN], dtype=pl.BF16)
        post_norm = prefill_norm_inline_prefix128(residual, post_gamma_layer, post_norm)
        mlp_gate = pl.create_tensor([128, INTERMEDIATE], dtype=pl.BF16)
        mlp_up = pl.create_tensor([128, INTERMEDIATE], dtype=pl.BF16)
        mlp_gate = prefill_project_inline_prefix128(post_norm, mlp_gate_layer, mlp_gate, active_count)
        mlp_up = prefill_project_inline_prefix128(post_norm, mlp_up_layer, mlp_up, active_count)
        middle = pl.create_tensor([128, INTERMEDIATE], dtype=pl.BF16)
        middle = prefill_swiglu_inline_prefix128(mlp_gate, mlp_up, middle)
        mlp_output = pl.create_tensor([128, HIDDEN], dtype=pl.BF16)
        mlp_output = prefill_project_inline_prefix128(middle, mlp_down_layer, mlp_output, active_count)
        with pl.spmd(128 // 2, name_hint='qwen_prefill_residual_store'):
            row = pl.tile.get_block_idx() * 2
            batch = row // ROWS
            token_row = row % ROWS
            io_a = pl.cast(pl.slice(residual, [2, HIDDEN], [row, 0]), pl.FP32)
            io_b = pl.cast(pl.slice(mlp_output, [2, HIDDEN], [row, 0]), pl.FP32)
            next_hidden = pl.cast(pl.add(io_a, io_b), pl.BF16, mode='rint')
            running[row:row + 2, 0:HIDDEN] = next_hidden
            if batch < active_count:
                traces[batch:batch + 1, layer + 1:layer + 2, token_row:token_row + 2, 0:HIDDEN] = pl.reshape(next_hidden, [1, 1, 2, HIDDEN])
    normalized = pl.create_tensor([128, HIDDEN], dtype=pl.BF16)
    normalized = prefill_norm_inline_prefix128(running, final_gamma, normalized)
    logits_rows = pl.reshape(logits, [PACKED, VOCAB])
    prefill_last_logits_inline_prefix128(normalized, lm_head, logits_rows, token_counts, active_count)
    with pl.spmd(128, name_hint='qwen_prefill_store_final_norm'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            traces[batch:batch + 1, 65:66, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(pl.slice(normalized, [1, HIDDEN], [row, 0]), [1, 1, 1, HIDDEN])
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit.inline
def prefill_batch_prefix256_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    running = pl.create_tensor([256, HIDDEN], dtype=pl.BF16)
    with pl.spmd(256, name_hint='qwen_prefill_embedding'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if token_row < count:
                token = pl.read(input_ids, [batch, token_row])
                embedded = pl.slice(embedding, [1, HIDDEN], [token, 0])
            else:
                embedded = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
            running[row:row + 1, 0:HIDDEN] = embedded
            traces[batch:batch + 1, 0:1, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(embedded, [1, 1, 1, HIDDEN])
        else:
            running[row:row + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
    for layer in pl.range(64):
        input_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        mlp_up_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        mlp_down_layer: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        norm = pl.create_tensor([256, HIDDEN], dtype=pl.BF16)
        norm = prefill_norm_inline_prefix256(running, input_gamma_layer, norm)
        residual = pl.create_tensor([256, HIDDEN], dtype=pl.BF16)
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[HIDDEN, QKV], pl.BF16, pl.NZ] = delta_qkv_weight[linear]
            z_weight_layer: pl.Tensor[[HIDDEN, VALUE_WIDTH], pl.BF16, pl.NZ] = delta_z_weight[linear]
            a_weight_layer: pl.Tensor[[HIDDEN, 128], pl.BF16, pl.NZ] = delta_a_weight[linear]
            b_weight_layer: pl.Tensor[[HIDDEN, 128], pl.BF16, pl.NZ] = delta_b_weight[linear]
            conv_weight_layer: pl.Tensor[[4, QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[VALUE_WIDTH, HIDDEN], pl.BF16, pl.NZ] = delta_out_weight[linear]
            qkv = pl.create_tensor([256, QKV], dtype=pl.BF16)
            z = pl.create_tensor([256, VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([256, 128], dtype=pl.BF16)
            b = pl.create_tensor([256, 128], dtype=pl.BF16)
            qkv = prefill_project_inline_prefix256(norm, qkv_weight_layer, qkv, active_count)
            z = prefill_project_inline_prefix256(norm, z_weight_layer, z, active_count)
            a = prefill_project_inline_prefix256(norm, a_weight_layer, a, active_count)
            b = prefill_project_inline_prefix256(norm, b_weight_layer, b, active_count)
            gated = pl.create_tensor([256, VALUE_WIDTH], dtype=pl.BF16)
            prefill_delta_slots_inline_prefix256(qkv, z, a, b, conv_weight_layer, a_log_layer, dt_bias_layer, delta_norm_layer, delta_out_layer, conv_states, recurrent_states, gated, linear, token_counts, active_count)
            attention = pl.create_tensor([256, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_inline_prefix256(gated, delta_out_layer, attention, active_count)
            residual = prefill_residual_inline_prefix256(running, attention, residual)
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[full]
            k_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[full]
            v_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[full]
            q_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ] = full_out_weight[full]
            q = pl.create_tensor([256, FULL_Q_PROJECTION], dtype=pl.BF16)
            k = pl.create_tensor([256, FULL_KV_WIDTH], dtype=pl.BF16)
            v = pl.create_tensor([256, FULL_KV_WIDTH], dtype=pl.BF16)
            q = prefill_project_inline_prefix256(norm, q_weight_full, q, active_count)
            k = prefill_project_inline_prefix256(norm, k_weight_full, k, active_count)
            v = prefill_project_inline_prefix256(norm, v_weight_full, v, active_count)
            gated = pl.create_tensor([256, FULL_Q_WIDTH], dtype=pl.BF16)
            prefill_attention_slots_inline_prefix256(q, k, v, q_gamma_full, k_gamma_full, cos, sin, mask, out_weight_full, key_pool, value_pool, block_tables, cache_offsets, token_counts, gated, full, active_count)
            attention = pl.create_tensor([256, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_inline_prefix256(gated, out_weight_full, attention, active_count)
            residual = prefill_residual_inline_prefix256(running, attention, residual)
        post_norm = pl.create_tensor([256, HIDDEN], dtype=pl.BF16)
        post_norm = prefill_norm_inline_prefix256(residual, post_gamma_layer, post_norm)
        mlp_gate = pl.create_tensor([256, INTERMEDIATE], dtype=pl.BF16)
        mlp_up = pl.create_tensor([256, INTERMEDIATE], dtype=pl.BF16)
        mlp_gate = prefill_project_inline_prefix256(post_norm, mlp_gate_layer, mlp_gate, active_count)
        mlp_up = prefill_project_inline_prefix256(post_norm, mlp_up_layer, mlp_up, active_count)
        middle = pl.create_tensor([256, INTERMEDIATE], dtype=pl.BF16)
        middle = prefill_swiglu_inline_prefix256(mlp_gate, mlp_up, middle)
        mlp_output = pl.create_tensor([256, HIDDEN], dtype=pl.BF16)
        mlp_output = prefill_project_inline_prefix256(middle, mlp_down_layer, mlp_output, active_count)
        with pl.spmd(256 // 2, name_hint='qwen_prefill_residual_store'):
            row = pl.tile.get_block_idx() * 2
            batch = row // ROWS
            token_row = row % ROWS
            io_a = pl.cast(pl.slice(residual, [2, HIDDEN], [row, 0]), pl.FP32)
            io_b = pl.cast(pl.slice(mlp_output, [2, HIDDEN], [row, 0]), pl.FP32)
            next_hidden = pl.cast(pl.add(io_a, io_b), pl.BF16, mode='rint')
            running[row:row + 2, 0:HIDDEN] = next_hidden
            if batch < active_count:
                traces[batch:batch + 1, layer + 1:layer + 2, token_row:token_row + 2, 0:HIDDEN] = pl.reshape(next_hidden, [1, 1, 2, HIDDEN])
    normalized = pl.create_tensor([256, HIDDEN], dtype=pl.BF16)
    normalized = prefill_norm_inline_prefix256(running, final_gamma, normalized)
    logits_rows = pl.reshape(logits, [PACKED, VOCAB])
    prefill_last_logits_inline_prefix256(normalized, lm_head, logits_rows, token_counts, active_count)
    with pl.spmd(256, name_hint='qwen_prefill_store_final_norm'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            traces[batch:batch + 1, 65:66, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(pl.slice(normalized, [1, HIDDEN], [row, 0]), [1, 1, 1, HIDDEN])
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit.inline
def prefill_batch_prefix384_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    running = pl.create_tensor([384, HIDDEN], dtype=pl.BF16)
    with pl.spmd(384, name_hint='qwen_prefill_embedding'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if token_row < count:
                token = pl.read(input_ids, [batch, token_row])
                embedded = pl.slice(embedding, [1, HIDDEN], [token, 0])
            else:
                embedded = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
            running[row:row + 1, 0:HIDDEN] = embedded
            traces[batch:batch + 1, 0:1, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(embedded, [1, 1, 1, HIDDEN])
        else:
            running[row:row + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
    for layer in pl.range(64):
        input_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        mlp_up_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        mlp_down_layer: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        norm = pl.create_tensor([384, HIDDEN], dtype=pl.BF16)
        norm = prefill_norm_inline_prefix384(running, input_gamma_layer, norm)
        residual = pl.create_tensor([384, HIDDEN], dtype=pl.BF16)
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[HIDDEN, QKV], pl.BF16, pl.NZ] = delta_qkv_weight[linear]
            z_weight_layer: pl.Tensor[[HIDDEN, VALUE_WIDTH], pl.BF16, pl.NZ] = delta_z_weight[linear]
            a_weight_layer: pl.Tensor[[HIDDEN, 128], pl.BF16, pl.NZ] = delta_a_weight[linear]
            b_weight_layer: pl.Tensor[[HIDDEN, 128], pl.BF16, pl.NZ] = delta_b_weight[linear]
            conv_weight_layer: pl.Tensor[[4, QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[VALUE_WIDTH, HIDDEN], pl.BF16, pl.NZ] = delta_out_weight[linear]
            qkv = pl.create_tensor([384, QKV], dtype=pl.BF16)
            z = pl.create_tensor([384, VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([384, 128], dtype=pl.BF16)
            b = pl.create_tensor([384, 128], dtype=pl.BF16)
            qkv = prefill_project_inline_prefix384(norm, qkv_weight_layer, qkv, active_count)
            z = prefill_project_inline_prefix384(norm, z_weight_layer, z, active_count)
            a = prefill_project_inline_prefix384(norm, a_weight_layer, a, active_count)
            b = prefill_project_inline_prefix384(norm, b_weight_layer, b, active_count)
            gated = pl.create_tensor([384, VALUE_WIDTH], dtype=pl.BF16)
            prefill_delta_slots_inline_prefix384(qkv, z, a, b, conv_weight_layer, a_log_layer, dt_bias_layer, delta_norm_layer, delta_out_layer, conv_states, recurrent_states, gated, linear, token_counts, active_count)
            attention = pl.create_tensor([384, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_inline_prefix384(gated, delta_out_layer, attention, active_count)
            residual = prefill_residual_inline_prefix384(running, attention, residual)
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[full]
            k_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[full]
            v_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[full]
            q_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ] = full_out_weight[full]
            q = pl.create_tensor([384, FULL_Q_PROJECTION], dtype=pl.BF16)
            k = pl.create_tensor([384, FULL_KV_WIDTH], dtype=pl.BF16)
            v = pl.create_tensor([384, FULL_KV_WIDTH], dtype=pl.BF16)
            q = prefill_project_inline_prefix384(norm, q_weight_full, q, active_count)
            k = prefill_project_inline_prefix384(norm, k_weight_full, k, active_count)
            v = prefill_project_inline_prefix384(norm, v_weight_full, v, active_count)
            gated = pl.create_tensor([384, FULL_Q_WIDTH], dtype=pl.BF16)
            prefill_attention_slots_inline_prefix384(q, k, v, q_gamma_full, k_gamma_full, cos, sin, mask, out_weight_full, key_pool, value_pool, block_tables, cache_offsets, token_counts, gated, full, active_count)
            attention = pl.create_tensor([384, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_inline_prefix384(gated, out_weight_full, attention, active_count)
            residual = prefill_residual_inline_prefix384(running, attention, residual)
        post_norm = pl.create_tensor([384, HIDDEN], dtype=pl.BF16)
        post_norm = prefill_norm_inline_prefix384(residual, post_gamma_layer, post_norm)
        mlp_gate = pl.create_tensor([384, INTERMEDIATE], dtype=pl.BF16)
        mlp_up = pl.create_tensor([384, INTERMEDIATE], dtype=pl.BF16)
        mlp_gate = prefill_project_inline_prefix384(post_norm, mlp_gate_layer, mlp_gate, active_count)
        mlp_up = prefill_project_inline_prefix384(post_norm, mlp_up_layer, mlp_up, active_count)
        middle = pl.create_tensor([384, INTERMEDIATE], dtype=pl.BF16)
        middle = prefill_swiglu_inline_prefix384(mlp_gate, mlp_up, middle)
        mlp_output = pl.create_tensor([384, HIDDEN], dtype=pl.BF16)
        mlp_output = prefill_project_inline_prefix384(middle, mlp_down_layer, mlp_output, active_count)
        with pl.spmd(384 // 2, name_hint='qwen_prefill_residual_store'):
            row = pl.tile.get_block_idx() * 2
            batch = row // ROWS
            token_row = row % ROWS
            io_a = pl.cast(pl.slice(residual, [2, HIDDEN], [row, 0]), pl.FP32)
            io_b = pl.cast(pl.slice(mlp_output, [2, HIDDEN], [row, 0]), pl.FP32)
            next_hidden = pl.cast(pl.add(io_a, io_b), pl.BF16, mode='rint')
            running[row:row + 2, 0:HIDDEN] = next_hidden
            if batch < active_count:
                traces[batch:batch + 1, layer + 1:layer + 2, token_row:token_row + 2, 0:HIDDEN] = pl.reshape(next_hidden, [1, 1, 2, HIDDEN])
    normalized = pl.create_tensor([384, HIDDEN], dtype=pl.BF16)
    normalized = prefill_norm_inline_prefix384(running, final_gamma, normalized)
    logits_rows = pl.reshape(logits, [PACKED, VOCAB])
    prefill_last_logits_inline_prefix384(normalized, lm_head, logits_rows, token_counts, active_count)
    with pl.spmd(384, name_hint='qwen_prefill_store_final_norm'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            traces[batch:batch + 1, 65:66, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(pl.slice(normalized, [1, HIDDEN], [row, 0]), [1, 1, 1, HIDDEN])
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit.inline
def prefill_batch_prefix512_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    running = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
    with pl.spmd(512, name_hint='qwen_prefill_embedding'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if token_row < count:
                token = pl.read(input_ids, [batch, token_row])
                embedded = pl.slice(embedding, [1, HIDDEN], [token, 0])
            else:
                embedded = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
            running[row:row + 1, 0:HIDDEN] = embedded
            traces[batch:batch + 1, 0:1, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(embedded, [1, 1, 1, HIDDEN])
        else:
            running[row:row + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
    for layer in pl.range(64):
        input_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_gate_weight[layer]
        mlp_up_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ] = mlp_up_weight[layer]
        mlp_down_layer: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ] = mlp_down_weight[layer]
        norm = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
        norm = prefill_norm_inline_prefix512(running, input_gamma_layer, norm)
        residual = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[HIDDEN, QKV], pl.BF16, pl.NZ] = delta_qkv_weight[linear]
            z_weight_layer: pl.Tensor[[HIDDEN, VALUE_WIDTH], pl.BF16, pl.NZ] = delta_z_weight[linear]
            a_weight_layer: pl.Tensor[[HIDDEN, 128], pl.BF16, pl.NZ] = delta_a_weight[linear]
            b_weight_layer: pl.Tensor[[HIDDEN, 128], pl.BF16, pl.NZ] = delta_b_weight[linear]
            conv_weight_layer: pl.Tensor[[4, QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[VALUE_WIDTH, HIDDEN], pl.BF16, pl.NZ] = delta_out_weight[linear]
            qkv = pl.create_tensor([512, QKV], dtype=pl.BF16)
            z = pl.create_tensor([512, VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([512, 128], dtype=pl.BF16)
            b = pl.create_tensor([512, 128], dtype=pl.BF16)
            qkv = prefill_project_inline_prefix512(norm, qkv_weight_layer, qkv, active_count)
            z = prefill_project_inline_prefix512(norm, z_weight_layer, z, active_count)
            a = prefill_project_inline_prefix512(norm, a_weight_layer, a, active_count)
            b = prefill_project_inline_prefix512(norm, b_weight_layer, b, active_count)
            gated = pl.create_tensor([512, VALUE_WIDTH], dtype=pl.BF16)
            prefill_delta_slots_inline_prefix512(qkv, z, a, b, conv_weight_layer, a_log_layer, dt_bias_layer, delta_norm_layer, delta_out_layer, conv_states, recurrent_states, gated, linear, token_counts, active_count)
            attention = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_inline_prefix512(gated, delta_out_layer, attention, active_count)
            residual = prefill_residual_inline_prefix512(running, attention, residual)
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ] = full_q_weight[full]
            k_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_k_weight[full]
            v_weight_full: pl.Tensor[[HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ] = full_v_weight[full]
            q_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ] = full_out_weight[full]
            q = pl.create_tensor([512, FULL_Q_PROJECTION], dtype=pl.BF16)
            k = pl.create_tensor([512, FULL_KV_WIDTH], dtype=pl.BF16)
            v = pl.create_tensor([512, FULL_KV_WIDTH], dtype=pl.BF16)
            q = prefill_project_inline_prefix512(norm, q_weight_full, q, active_count)
            k = prefill_project_inline_prefix512(norm, k_weight_full, k, active_count)
            v = prefill_project_inline_prefix512(norm, v_weight_full, v, active_count)
            gated = pl.create_tensor([512, FULL_Q_WIDTH], dtype=pl.BF16)
            prefill_attention_slots_inline_prefix512(q, k, v, q_gamma_full, k_gamma_full, cos, sin, mask, out_weight_full, key_pool, value_pool, block_tables, cache_offsets, token_counts, gated, full, active_count)
            attention = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
            attention = prefill_project_inline_prefix512(gated, out_weight_full, attention, active_count)
            residual = prefill_residual_inline_prefix512(running, attention, residual)
        post_norm = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
        post_norm = prefill_norm_inline_prefix512(residual, post_gamma_layer, post_norm)
        mlp_gate = pl.create_tensor([512, INTERMEDIATE], dtype=pl.BF16)
        mlp_up = pl.create_tensor([512, INTERMEDIATE], dtype=pl.BF16)
        mlp_gate = prefill_project_inline_prefix512(post_norm, mlp_gate_layer, mlp_gate, active_count)
        mlp_up = prefill_project_inline_prefix512(post_norm, mlp_up_layer, mlp_up, active_count)
        middle = pl.create_tensor([512, INTERMEDIATE], dtype=pl.BF16)
        middle = prefill_swiglu_inline_prefix512(mlp_gate, mlp_up, middle)
        mlp_output = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
        mlp_output = prefill_project_inline_prefix512(middle, mlp_down_layer, mlp_output, active_count)
        with pl.spmd(512 // 2, name_hint='qwen_prefill_residual_store'):
            row = pl.tile.get_block_idx() * 2
            batch = row // ROWS
            token_row = row % ROWS
            io_a = pl.cast(pl.slice(residual, [2, HIDDEN], [row, 0]), pl.FP32)
            io_b = pl.cast(pl.slice(mlp_output, [2, HIDDEN], [row, 0]), pl.FP32)
            next_hidden = pl.cast(pl.add(io_a, io_b), pl.BF16, mode='rint')
            running[row:row + 2, 0:HIDDEN] = next_hidden
            if batch < active_count:
                traces[batch:batch + 1, layer + 1:layer + 2, token_row:token_row + 2, 0:HIDDEN] = pl.reshape(next_hidden, [1, 1, 2, HIDDEN])
    normalized = pl.create_tensor([512, HIDDEN], dtype=pl.BF16)
    normalized = prefill_norm_inline_prefix512(running, final_gamma, normalized)
    logits_rows = pl.reshape(logits, [PACKED, VOCAB])
    prefill_last_logits_inline_prefix512(normalized, lm_head, logits_rows, token_counts, active_count)
    with pl.spmd(512, name_hint='qwen_prefill_store_final_norm'):
        row = pl.tile.get_block_idx()
        batch = row // ROWS
        token_row = row % ROWS
        if batch < active_count:
            traces[batch:batch + 1, 65:66, token_row:token_row + 1, 0:HIDDEN] = pl.reshape(pl.slice(normalized, [1, HIDDEN], [row, 0]), [1, 1, 1, HIDDEN])
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit.inline
def prefill_batch_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    prefix_group = (active_count + 1) // 2
    if prefix_group == 1:
        prefill_batch_prefix128_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    if prefix_group == 2:
        prefill_batch_prefix256_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    if prefix_group == 3:
        prefill_batch_prefix384_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    if prefix_group == 4:
        prefill_batch_prefix512_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def rank_prefill_shared(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * HIDDEN, INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * QKV, HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * VALUE_WIDTH, HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * HIDDEN, VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * FULL_Q_PROJECTION, HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * HIDDEN, FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[BATCH, ROWS, CONTEXT], pl.FP32], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[BATCH, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[BATCH, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.InOut[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    return prefill_batch_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)

@pl.jit.host
def prefill_fwd(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64, HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ], mlp_up_weight: pl.Tensor[[64, HIDDEN, INTERMEDIATE], pl.BF16, pl.NZ], mlp_down_weight: pl.Tensor[[64, INTERMEDIATE, HIDDEN], pl.BF16, pl.NZ], delta_qkv_weight: pl.Tensor[[48, HIDDEN, QKV], pl.BF16, pl.NZ], delta_z_weight: pl.Tensor[[48, HIDDEN, VALUE_WIDTH], pl.BF16, pl.NZ], delta_a_weight: pl.Tensor[[48, HIDDEN, 128], pl.BF16, pl.NZ], delta_b_weight: pl.Tensor[[48, HIDDEN, 128], pl.BF16, pl.NZ], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48, VALUE_WIDTH, HIDDEN], pl.BF16, pl.NZ], full_q_weight: pl.Tensor[[16, HIDDEN, FULL_Q_PROJECTION], pl.BF16, pl.NZ], full_k_weight: pl.Tensor[[16, HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ], full_v_weight: pl.Tensor[[16, HIDDEN, FULL_KV_WIDTH], pl.BF16, pl.NZ], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16, FULL_Q_WIDTH, HIDDEN], pl.BF16, pl.NZ], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[BATCH, ROWS, CONTEXT], pl.FP32], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[BATCH, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[BATCH, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.InOut[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    rank_prefill_shared(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, device=0)

@pl.jit.inline
def prefill_project_inline_nd(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    output_tiles = pl.tensor.dim(weight, 0) // 128
    input_columns = pl.tensor.dim(weight, 1)
    if active_count <= 128 // ROWS:
        with pl.spmd(output_tiles, name_hint='prefill_packed_projection_active'):
            task = pl.tile.get_block_idx()
            row = 0
            col = task * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [128, 256], [col, k_start])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=True, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
        with pl.spmd((PACKED // 128 - 1) * output_tiles, name_hint='prefill_zero_unused_projection'):
            zero_task = pl.tile.get_block_idx()
            zero_row = (zero_task // output_tiles + 1) * 128
            zero_column = zero_task % output_tiles * 128
            output[zero_row:zero_row + 128, zero_column:zero_column + 128] = pl.full([128, 128], dtype=pl.BF16, value=0.0)
    else:
        with pl.spmd(PACKED // 128 * output_tiles, name_hint='prefill_packed_projection'):
            task = pl.tile.get_block_idx()
            row = task % (PACKED // 128) * 128
            col = task // (PACKED // 128) * 128
            accumulator = pl.create_tensor([128, 128], dtype=pl.FP32)
            for chunk in pl.pipeline(0, input_columns // 256, stage=2):
                k_start = chunk * 256
                hidden_tile = pl.slice(hidden, [128, 256], [row, k_start])
                weight_tile = pl.slice(weight, [128, 256], [col, k_start])
                accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=True, init_cond=chunk == 0)
            output[row:row + 128, col:col + 128] = pl.cast(accumulator, pl.BF16, mode='rint')
    return output

@pl.jit.inline
def prefill_last_logits_inline(hidden: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, token_counts: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    selected_hidden = pl.create_tensor([16, HIDDEN], dtype=pl.BF16)
    with pl.spmd(16, name_hint='prefill_gather_last_hidden'):
        batch = pl.tile.get_block_idx()
        if batch < active_count:
            count = pl.read(token_counts, [batch])
            if count > 0:
                source_row = batch * ROWS + count - 1
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.slice(hidden, [1, HIDDEN], [source_row, 0])
            else:
                selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
        else:
            selected_hidden[batch:batch + 1, 0:HIDDEN] = pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0)
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
        for batch in pl.range(BATCH):
            if batch < active_count:
                count = pl.read(token_counts, [batch])
                if count > 0:
                    destination_row = batch * ROWS + count - 1
                    output[destination_row:destination_row + 1, column:column + 128] = pl.slice(projected, [1, 128], [batch, 0])
    return output
