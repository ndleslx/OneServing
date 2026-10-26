"""Standalone full-network Qwen3.8-27B PyPTO kernels and TP collective."""
import pypto.language as pl
import pypto.language.distributed as pld

linear_ROWS = 16

linear_HIDDEN = 5120

linear_OUTPUT = pl.dynamic('QWEN_LINEAR_OUTPUT')

linear_INPUT = pl.dynamic('QWEN_LINEAR_INPUT')

linear_OUTPUT_TILE = 128

linear_K_TILE = 512

@pl.jit.inline
def linear_hidden_projection_inline(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Tensor):
    output_columns = pl.tensor.dim(weight, 0)
    input_columns = pl.tensor.dim(weight, 1)
    with pl.spmd(output_columns // linear_OUTPUT_TILE, name_hint='qwen_hidden_projection'):
        start = pl.tile.get_block_idx() * linear_OUTPUT_TILE
        accumulator = pl.create_tensor([linear_ROWS, linear_OUTPUT_TILE], dtype=pl.FP32)
        for chunk in pl.pipeline(0, input_columns // linear_K_TILE, stage=2):
            k_start = chunk * linear_K_TILE
            hidden_tile = pl.slice(hidden, [linear_ROWS, linear_K_TILE], [0, k_start])
            weight_tile = pl.slice(weight, [linear_OUTPUT_TILE, linear_K_TILE], [start, k_start])
            accumulator = pl.matmul_acc(accumulator, hidden_tile, weight_tile, b_trans=True, init_cond=chunk == 0)
        result[0:linear_ROWS, start:start + linear_OUTPUT_TILE] = pl.cast(accumulator, pl.BF16, mode='rint')
    return result

@pl.jit
def linear_hidden_projection(hidden: pl.Tensor[[linear_ROWS, linear_INPUT], pl.BF16], weight: pl.Tensor[[linear_OUTPUT, linear_INPUT], pl.BF16], result: pl.Out[pl.Tensor[[linear_ROWS, linear_OUTPUT], pl.BF16]]):
    result = linear_hidden_projection_inline(hidden, weight, result)
    return result

rmsnorm_ROWS = 16

rmsnorm_HIDDEN = 5120

rmsnorm_CHUNK = 128

@pl.jit.inline
def rmsnorm_rmsnorm_inline(hidden: pl.Tensor[[rmsnorm_ROWS, rmsnorm_HIDDEN], pl.BF16], gamma: pl.Tensor[[1, rmsnorm_HIDDEN], pl.FP32], output: pl.Tensor[[rmsnorm_ROWS, rmsnorm_HIDDEN], pl.BF16]):
    with pl.spmd(1, name_hint='qwen_rmsnorm'):
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

@pl.jit
def rmsnorm_rmsnorm(hidden: pl.Tensor[[rmsnorm_ROWS, rmsnorm_HIDDEN], pl.BF16], gamma: pl.Tensor[[1, rmsnorm_HIDDEN], pl.FP32], output: pl.Out[pl.Tensor[[rmsnorm_ROWS, rmsnorm_HIDDEN], pl.BF16]]):
    output = rmsnorm_rmsnorm_inline(hidden, gamma, output)
    return output

gated_norm_ROWS = 240

gated_norm_DIM = 128

gated_norm_ROW_TILE = 16

@pl.jit.inline
def gated_norm_gated_rmsnorm_inline(core: pl.Tensor, gate: pl.Tensor, weight: pl.Tensor, output: pl.Tensor):
    with pl.spmd(gated_norm_ROWS // gated_norm_ROW_TILE, name_hint='qwen_gated_rmsnorm'):
        row = pl.tile.get_block_idx() * gated_norm_ROW_TILE
        source = pl.cast(pl.slice(core, [gated_norm_ROW_TILE, gated_norm_DIM], [row, 0]), pl.FP32)
        gate_tile = pl.cast(pl.slice(gate, [gated_norm_ROW_TILE, gated_norm_DIM], [row, 0]), pl.FP32)
        weight_tile = pl.cast(pl.slice(weight, [1, gated_norm_DIM], [0, 0]), pl.FP32)
        variance = pl.mul(pl.row_sum(pl.mul(source, source)), 1.0 / gated_norm_DIM)
        inverse = pl.rsqrt(pl.add(variance, 1e-06), high_precision=True)
        normalized = pl.cast(pl.row_expand_mul(source, inverse), pl.BF16, mode='rint')
        weighted = pl.cast(pl.col_expand_mul(pl.cast(normalized, pl.FP32), weight_tile), pl.BF16, mode='rint')
        sigmoid = pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0))
        result = pl.mul(pl.cast(weighted, pl.FP32), sigmoid)
        output[row:row + gated_norm_ROW_TILE, 0:gated_norm_DIM] = pl.cast(result, pl.BF16, mode='rint')
    return output

@pl.jit
def gated_norm_gated_rmsnorm(core: pl.Tensor[[gated_norm_ROWS, gated_norm_DIM], pl.BF16], gate: pl.Tensor[[gated_norm_ROWS, gated_norm_DIM], pl.BF16], weight: pl.Tensor[[1, gated_norm_DIM], pl.BF16], output: pl.Out[pl.Tensor[[gated_norm_ROWS, gated_norm_DIM], pl.BF16]]):
    output = gated_norm_gated_rmsnorm_inline(core, gate, weight, output)
    return output

causal_conv_TOKENS = pl.dynamic('QWEN_CONV_TOKENS')

causal_conv_CHANNELS = 10240

causal_conv_KERNEL = 4

causal_conv_CHANNEL_TILE = 128

@pl.jit.inline
def causal_conv_causal_conv_silu_inline(projected: pl.Tensor, weight_tap_channel: pl.Tensor, state_tap_channel: pl.Tensor, output: pl.Tensor, token_count: pl.Scalar[pl.INDEX]):
    with pl.spmd(causal_conv_CHANNELS // causal_conv_CHANNEL_TILE, name_hint='qwen_causal_conv'):
        column = pl.tile.get_block_idx() * causal_conv_CHANNEL_TILE
        previous_0 = pl.cast(pl.slice(state_tap_channel, [1, causal_conv_CHANNEL_TILE], [0, column]), pl.FP32)
        previous_1 = pl.cast(pl.slice(state_tap_channel, [1, causal_conv_CHANNEL_TILE], [1, column]), pl.FP32)
        previous_2 = pl.cast(pl.slice(state_tap_channel, [1, causal_conv_CHANNEL_TILE], [2, column]), pl.FP32)
        previous_3 = pl.cast(pl.slice(state_tap_channel, [1, causal_conv_CHANNEL_TILE], [3, column]), pl.FP32)
        tap_0 = pl.cast(pl.slice(weight_tap_channel, [1, causal_conv_CHANNEL_TILE], [0, column]), pl.FP32)
        tap_1 = pl.cast(pl.slice(weight_tap_channel, [1, causal_conv_CHANNEL_TILE], [1, column]), pl.FP32)
        tap_2 = pl.cast(pl.slice(weight_tap_channel, [1, causal_conv_CHANNEL_TILE], [2, column]), pl.FP32)
        tap_3 = pl.cast(pl.slice(weight_tap_channel, [1, causal_conv_CHANNEL_TILE], [3, column]), pl.FP32)
        for token in pl.range(token_count):
            current = pl.cast(pl.slice(projected, [1, causal_conv_CHANNEL_TILE], [token, column]), pl.FP32)
            convolved = pl.add(pl.add(pl.mul(previous_1, tap_0), pl.mul(previous_2, tap_1)), pl.add(pl.mul(previous_3, tap_2), pl.mul(current, tap_3)))
            convolved = pl.cast(pl.cast(convolved, pl.BF16, mode='rint'), pl.FP32)
            activated = pl.div(convolved, pl.add(pl.exp(pl.neg(convolved)), 1.0), high_precision=True)
            output[token:token + 1, column:column + causal_conv_CHANNEL_TILE] = pl.cast(activated, pl.BF16, mode='rint')
            previous_0 = previous_1
            previous_1 = previous_2
            previous_2 = previous_3
            previous_3 = current
        for token in pl.range(token_count, pl.tensor.dim(output, 0)):
            output[token:token + 1, column:column + causal_conv_CHANNEL_TILE] = pl.full([1, causal_conv_CHANNEL_TILE], dtype=pl.BF16, value=0.0)
        state_tap_channel[0:1, column:column + causal_conv_CHANNEL_TILE] = pl.cast(previous_0, pl.BF16, mode='rint')
        state_tap_channel[1:2, column:column + causal_conv_CHANNEL_TILE] = pl.cast(previous_1, pl.BF16, mode='rint')
        state_tap_channel[2:3, column:column + causal_conv_CHANNEL_TILE] = pl.cast(previous_2, pl.BF16, mode='rint')
        state_tap_channel[3:4, column:column + causal_conv_CHANNEL_TILE] = pl.cast(previous_3, pl.BF16, mode='rint')
    return (output, state_tap_channel)

@pl.jit
def causal_conv_causal_conv_silu(projected: pl.Tensor[[causal_conv_TOKENS, causal_conv_CHANNELS], pl.BF16], weight_tap_channel: pl.Tensor[[causal_conv_KERNEL, causal_conv_CHANNELS], pl.BF16], state_tap_channel: pl.InOut[pl.Tensor[[causal_conv_KERNEL, causal_conv_CHANNELS], pl.BF16]], output: pl.Out[pl.Tensor[[causal_conv_TOKENS, causal_conv_CHANNELS], pl.BF16]]):
    output, state_tap_channel = causal_conv_causal_conv_silu_inline(projected, weight_tap_channel, state_tap_channel, output, pl.tensor.dim(projected, 0))
    return (output, state_tap_channel)

delta_prepare_TOKENS = 5

delta_prepare_KEY_HEADS = 16

delta_prepare_VALUE_HEADS = 48

delta_prepare_DIM = 128

delta_prepare_CHANNELS = (delta_prepare_KEY_HEADS * 2 + delta_prepare_VALUE_HEADS) * delta_prepare_DIM

delta_prepare_PAD_ROWS = 16

delta_prepare_PAD_HEADS = 64

@pl.jit.inline
def delta_prepare_expand_qkv_inline(conv_output: pl.Tensor, query: pl.Tensor, key: pl.Tensor, value: pl.Tensor):
    query_rows = pl.reshape(query, [delta_prepare_TOKENS * delta_prepare_VALUE_HEADS, delta_prepare_DIM])
    key_rows = pl.reshape(key, [delta_prepare_TOKENS * delta_prepare_VALUE_HEADS, delta_prepare_DIM])
    value_rows = pl.reshape(value, [delta_prepare_TOKENS * delta_prepare_VALUE_HEADS, delta_prepare_DIM])
    with pl.spmd(delta_prepare_TOKENS * delta_prepare_VALUE_HEADS, name_hint='qwen_delta_expand_qkv'):
        row = pl.tile.get_block_idx()
        token = row // delta_prepare_VALUE_HEADS
        head = row % delta_prepare_VALUE_HEADS
        key_head = head // (delta_prepare_VALUE_HEADS // delta_prepare_KEY_HEADS)
        q = pl.slice(conv_output, [1, delta_prepare_DIM], [token, key_head * delta_prepare_DIM])
        k = pl.slice(conv_output, [1, delta_prepare_DIM], [token, delta_prepare_KEY_HEADS * delta_prepare_DIM + key_head * delta_prepare_DIM])
        v = pl.slice(conv_output, [1, delta_prepare_DIM], [token, 2 * delta_prepare_KEY_HEADS * delta_prepare_DIM + head * delta_prepare_DIM])
        query_rows[row:row + 1, 0:delta_prepare_DIM] = q
        key_rows[row:row + 1, 0:delta_prepare_DIM] = k
        value_rows[row:row + 1, 0:delta_prepare_DIM] = v
    return (query, key, value)

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

@pl.jit
def delta_prepare_expand_qkv(conv_output: pl.Tensor[[delta_prepare_TOKENS, delta_prepare_CHANNELS], pl.BF16], query: pl.Out[pl.Tensor[[delta_prepare_TOKENS, delta_prepare_VALUE_HEADS, delta_prepare_DIM], pl.BF16]], key: pl.Out[pl.Tensor[[delta_prepare_TOKENS, delta_prepare_VALUE_HEADS, delta_prepare_DIM], pl.BF16]], value: pl.Out[pl.Tensor[[delta_prepare_TOKENS, delta_prepare_VALUE_HEADS, delta_prepare_DIM], pl.BF16]]):
    query, key, value = delta_prepare_expand_qkv_inline(conv_output, query, key, value)
    return (query, key, value)

@pl.jit
def delta_prepare_decay_beta(a: pl.Tensor[[delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS], pl.BF16], b: pl.Tensor[[delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS], pl.BF16], a_log: pl.Tensor[[1, delta_prepare_PAD_HEADS], pl.BF16], dt_bias: pl.Tensor[[1, delta_prepare_PAD_HEADS], pl.BF16], log_decay: pl.Out[pl.Tensor[[delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS], pl.FP32]], beta: pl.Out[pl.Tensor[[delta_prepare_PAD_ROWS, delta_prepare_PAD_HEADS], pl.BF16]]):
    log_decay, beta = delta_prepare_decay_beta_inline(a, b, a_log, dt_bias, log_decay, beta)
    return (log_decay, beta)

delta_window16_ROWS = 16

delta_window16_KEY_HEADS = 16

delta_window16_HEADS = 48

delta_window16_DIM = 128

delta_window16_VALUE_TILE = 32

delta_window16_NORM_TILE = 16

delta_window16_COLUMN_REP = 16

delta_window16_VALUE_WIDTH = delta_window16_HEADS * delta_window16_DIM

@pl.jit.inline
def delta_window16_expand_qkv_window16_inline(conv_output: pl.Tensor, query: pl.Tensor, key: pl.Tensor, value: pl.Tensor):
    query_rows = pl.reshape(query, [delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM])
    key_rows = pl.reshape(key, [delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM])
    value_rows = pl.reshape(value, [delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM])
    with pl.spmd(delta_window16_ROWS * delta_window16_HEADS, name_hint='qwen_window_delta_expand_qkv'):
        row = pl.tile.get_block_idx()
        token = row // delta_window16_HEADS
        head = row % delta_window16_HEADS
        key_head = head // (delta_window16_HEADS // delta_window16_KEY_HEADS)
        query_rows[row:row + 1, 0:delta_window16_DIM] = pl.slice(conv_output, [1, delta_window16_DIM], [token, key_head * delta_window16_DIM])
        key_rows[row:row + 1, 0:delta_window16_DIM] = pl.slice(conv_output, [1, delta_window16_DIM], [token, delta_window16_KEY_HEADS * delta_window16_DIM + key_head * delta_window16_DIM])
        value_rows[row:row + 1, 0:delta_window16_DIM] = pl.slice(conv_output, [1, delta_window16_DIM], [token, 2 * delta_window16_KEY_HEADS * delta_window16_DIM + head * delta_window16_DIM])
    return (query, key, value)

@pl.jit.inline
def delta_window16_recurrence_window16_inline(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor, state_value_key: pl.Tensor, output: pl.Tensor, token_count: pl.Scalar[pl.INDEX]):
    q_rows = pl.reshape(query, [delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM])
    k_rows = pl.reshape(key, [delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM])
    q_norm = pl.create_tensor([delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM], dtype=pl.FP32)
    k_norm = pl.create_tensor([delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM], dtype=pl.FP32)
    with pl.spmd(delta_window16_ROWS * delta_window16_HEADS // delta_window16_NORM_TILE, name_hint='qwen_window_delta_norm') as norm_task:
        row = pl.tile.get_block_idx() * delta_window16_NORM_TILE
        q_tile = pl.cast(pl.slice(q_rows, [delta_window16_NORM_TILE, delta_window16_DIM], [row, 0]), pl.FP32)
        k_tile = pl.cast(pl.slice(k_rows, [delta_window16_NORM_TILE, delta_window16_DIM], [row, 0]), pl.FP32)
        q_norm[row:row + delta_window16_NORM_TILE, 0:delta_window16_DIM] = pl.mul(pl.row_expand_mul(q_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(q_tile, q_tile)), 1e-06), high_precision=True)), delta_window16_DIM ** (-0.5))
        k_norm[row:row + delta_window16_NORM_TILE, 0:delta_window16_DIM] = pl.row_expand_mul(k_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(k_tile, k_tile)), 1e-06), high_precision=True))
    state_rows = pl.reshape(state_value_key, [delta_window16_HEADS * delta_window16_DIM, delta_window16_DIM])
    output_rows = pl.reshape(output, [delta_window16_ROWS, delta_window16_HEADS * delta_window16_DIM])
    with pl.spmd(delta_window16_HEADS * (delta_window16_DIM // delta_window16_VALUE_TILE), name_hint='qwen_window_delta_recurrence', deps=[norm_task]):
        task = pl.tile.get_block_idx()
        head = task // (delta_window16_DIM // delta_window16_VALUE_TILE)
        value_start = task % (delta_window16_DIM // delta_window16_VALUE_TILE) * delta_window16_VALUE_TILE
        state_start = head * delta_window16_DIM + value_start
        state = pl.slice(state_rows, [delta_window16_VALUE_TILE, delta_window16_DIM], [state_start, 0])
        for token in pl.range(token_count):
            row = token * delta_window16_HEADS + head
            q = q_norm[row:row + 1, 0:delta_window16_DIM]
            k = k_norm[row:row + 1, 0:delta_window16_DIM]
            g = pl.read(log_decay, [token, head])
            decay = pl.exp(pl.mul(pl.full([1, delta_window16_DIM], dtype=pl.FP32, value=1.0), g))
            b = pl.read(beta, [token, head])
            decayed = pl.col_expand_mul(state, decay)
            remembered = pl.row_sum(pl.col_expand_mul(decayed, k))
            v_row = pl.cast(pl.reshape(pl.slice(value, [1, 1, delta_window16_VALUE_TILE], [token, head, value_start], drop_dims=[0]), [1, delta_window16_VALUE_TILE]), pl.FP32)
            v_col = pl.mul(pl.row_sum(pl.transpose(pl.col_expand_mul(pl.full([delta_window16_COLUMN_REP, delta_window16_VALUE_TILE], dtype=pl.FP32, value=1.0), v_row), 0, 1)), 1.0 / delta_window16_COLUMN_REP)
            delta = pl.mul(pl.sub(v_col, remembered), b)
            state = pl.add(decayed, pl.col_expand_mul(pl.row_expand_mul(pl.full([delta_window16_VALUE_TILE, delta_window16_DIM], dtype=pl.FP32, value=1.0), delta), k))
            output_rows[token:token + 1, head * delta_window16_DIM + value_start:head * delta_window16_DIM + value_start + delta_window16_VALUE_TILE] = pl.set_validshape(pl.cast(pl.matmul(q, state, pl.FP32, b_trans=True), pl.BF16, mode='rint'), 1, delta_window16_VALUE_TILE)
        for token in pl.range(token_count, delta_window16_ROWS):
            output_rows[token:token + 1, head * delta_window16_DIM + value_start:head * delta_window16_DIM + value_start + delta_window16_VALUE_TILE] = pl.full([1, delta_window16_VALUE_TILE], dtype=pl.BF16, value=0.0)
        state_rows[state_start:state_start + delta_window16_VALUE_TILE, 0:delta_window16_DIM] = state
    return (output, state_value_key)

@pl.jit.inline
def delta_window16_attention_tail_window16_inline(core: pl.Tensor, z_projection: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor, gated_output: pl.Tensor, attention_output: pl.Tensor):
    gate_rows = pl.create_tensor([delta_window16_ROWS * delta_window16_HEADS, delta_window16_DIM], dtype=pl.BF16)
    with pl.spmd(delta_window16_ROWS * delta_window16_HEADS, name_hint='qwen_window_z_gate'):
        row = pl.tile.get_block_idx()
        token = row // delta_window16_HEADS
        head = row % delta_window16_HEADS
        gate_rows[row:row + 1, 0:delta_window16_DIM] = pl.slice(z_projection, [1, delta_window16_DIM], [token, head * delta_window16_DIM])
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
    padded = pl.reshape(gated_output, [delta_window16_ROWS, delta_window16_VALUE_WIDTH])
    attention_output = linear_hidden_projection_inline(padded, out_weight, attention_output)
    return (gated_output, attention_output)

attention3_prepare_ROWS = 16

attention3_prepare_HIDDEN = 5120

attention3_prepare_HEAD_DIM = 256

attention3_prepare_Q_HEADS = 24

attention3_prepare_KV_HEADS = 4

attention3_prepare_Q_PROJECTION = attention3_prepare_Q_HEADS * attention3_prepare_HEAD_DIM * 2

attention3_prepare_KV_WIDTH = attention3_prepare_KV_HEADS * attention3_prepare_HEAD_DIM

attention3_prepare_Q_WIDTH = attention3_prepare_Q_HEADS * attention3_prepare_HEAD_DIM

attention3_prepare_ROTARY = 64

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
def attention3_prepare_attention3_prepare_inline(hidden: pl.Tensor, q_weight: pl.Tensor, k_weight: pl.Tensor, v_weight: pl.Tensor, q_gamma: pl.Tensor, k_gamma: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, q_norm: pl.Tensor, k_norm: pl.Tensor, q_rot: pl.Tensor, k_rot: pl.Tensor, value: pl.Tensor, gate: pl.Tensor):
    q_projected = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_Q_PROJECTION], dtype=pl.BF16)
    k_projected = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_KV_WIDTH], dtype=pl.BF16)
    q_projected = linear_hidden_projection_inline(hidden, q_weight, q_projected)
    k_projected = linear_hidden_projection_inline(hidden, k_weight, k_projected)
    value = linear_hidden_projection_inline(hidden, v_weight, value)
    with pl.spmd(attention3_prepare_Q_HEADS, name_hint='qwen_attention_query_norm'):
        head = pl.tile.get_block_idx()
        projection_start = head * attention3_prepare_HEAD_DIM * 2
        result_start = head * attention3_prepare_HEAD_DIM
        raw = pl.slice(q_projected, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], [0, projection_start])
        normalized = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        normalized = attention3_prepare_normalize_head_inline(raw, q_gamma, normalized)
        q_norm[0:attention3_prepare_ROWS, result_start:result_start + attention3_prepare_HEAD_DIM] = normalized
        gate[0:attention3_prepare_ROWS, result_start:result_start + attention3_prepare_HEAD_DIM] = pl.slice(q_projected, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], [0, projection_start + attention3_prepare_HEAD_DIM])
        rotated = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        rotated = attention3_prepare_rotary_head_inline(normalized, cos, sin, rotated)
        q_rot[0:attention3_prepare_ROWS, result_start:result_start + attention3_prepare_HEAD_DIM] = rotated
    with pl.spmd(attention3_prepare_KV_HEADS, name_hint='qwen_attention_key_norm'):
        head = pl.tile.get_block_idx()
        start = head * attention3_prepare_HEAD_DIM
        raw = pl.slice(k_projected, [attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], [0, start])
        normalized = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        normalized = attention3_prepare_normalize_head_inline(raw, k_gamma, normalized)
        k_norm[0:attention3_prepare_ROWS, start:start + attention3_prepare_HEAD_DIM] = normalized
        rotated = pl.create_tensor([attention3_prepare_ROWS, attention3_prepare_HEAD_DIM], dtype=pl.BF16)
        rotated = attention3_prepare_rotary_head_inline(normalized, cos, sin, rotated)
        k_rot[0:attention3_prepare_ROWS, start:start + attention3_prepare_HEAD_DIM] = rotated
    return (q_norm, k_norm, q_rot, k_rot, value, gate)

@pl.jit
def attention3_prepare_attention3_prepare(hidden: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_HIDDEN], pl.BF16], q_weight: pl.Tensor[[attention3_prepare_Q_PROJECTION, attention3_prepare_HIDDEN], pl.BF16], k_weight: pl.Tensor[[attention3_prepare_KV_WIDTH, attention3_prepare_HIDDEN], pl.BF16], v_weight: pl.Tensor[[attention3_prepare_KV_WIDTH, attention3_prepare_HIDDEN], pl.BF16], q_gamma: pl.Tensor[[1, attention3_prepare_HEAD_DIM], pl.FP32], k_gamma: pl.Tensor[[1, attention3_prepare_HEAD_DIM], pl.FP32], cos: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_ROTARY], pl.BF16], sin: pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_ROTARY], pl.BF16], q_norm: pl.Out[pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_Q_WIDTH], pl.BF16]], k_norm: pl.Out[pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_KV_WIDTH], pl.BF16]], q_rot: pl.Out[pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_Q_WIDTH], pl.BF16]], k_rot: pl.Out[pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_KV_WIDTH], pl.BF16]], value: pl.Out[pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_KV_WIDTH], pl.BF16]], gate: pl.Out[pl.Tensor[[attention3_prepare_ROWS, attention3_prepare_Q_WIDTH], pl.BF16]]):
    return attention3_prepare_attention3_prepare_inline(hidden, q_weight, k_weight, v_weight, q_gamma, k_gamma, cos, sin, q_norm, k_norm, q_rot, k_rot, value, gate)

attention512_core_ROWS = 16

attention512_core_CONTEXT = 512

attention512_core_KEY_BLOCK = 16

attention512_core_HEAD_DIM = 256

attention512_core_Q_HEADS = 24

attention512_core_KV_HEADS = 4

attention512_core_Q_WIDTH = attention512_core_Q_HEADS * attention512_core_HEAD_DIM

@pl.jit.inline
def attention512_core_attention512_core_inline(query: pl.Tensor, key_cache: pl.Tensor, value_cache: pl.Tensor, gate: pl.Tensor, mask: pl.Tensor, output_weight: pl.Tensor, logits_buffer: pl.Tensor, probability_buffer: pl.Tensor, weighted_output: pl.Tensor, gated_output: pl.Tensor, projected_output: pl.Tensor):
    with pl.spmd(attention512_core_Q_HEADS, name_hint='qwen_attention512_scores'):
        score_head = pl.tile.get_block_idx()
        score_kv_head = score_head // (attention512_core_Q_HEADS // attention512_core_KV_HEADS)
        score_start = score_head * attention512_core_HEAD_DIM
        score_q = pl.slice(query, [attention512_core_ROWS, attention512_core_HEAD_DIM], [0, score_start])
        for score_block in pl.range(attention512_core_CONTEXT // attention512_core_KEY_BLOCK):
            score_key_start = score_block * attention512_core_KEY_BLOCK
            score_k = pl.reshape(pl.slice(key_cache, [1, attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM], [score_kv_head, score_key_start, 0]), [attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM])
            score_raw = pl.matmul(score_q, score_k, pl.FP32, b_trans=True)
            score_rounded = pl.cast(score_raw, pl.BF16, mode='rint')
            score_scaled = pl.cast(pl.mul(pl.cast(score_rounded, pl.FP32), attention512_core_HEAD_DIM ** (-0.5)), pl.BF16, mode='rint')
            score_mask = pl.slice(mask, [attention512_core_ROWS, attention512_core_KEY_BLOCK], [0, score_key_start])
            logits_buffer[score_head * attention512_core_ROWS:(score_head + 1) * attention512_core_ROWS, score_key_start:score_key_start + attention512_core_KEY_BLOCK] = pl.add(pl.cast(score_scaled, pl.FP32), score_mask)
    with pl.spmd(attention512_core_Q_HEADS * (attention512_core_ROWS // 8), name_hint='qwen_attention512_softmax'):
        softmax_row = pl.tile.get_block_idx() * 8
        softmax_logits = pl.slice(logits_buffer, [8, attention512_core_CONTEXT], [softmax_row, 0])
        softmax_exp = pl.exp(pl.row_expand_sub(softmax_logits, pl.row_max(softmax_logits)))
        probability_buffer[softmax_row:softmax_row + 8, 0:attention512_core_CONTEXT] = pl.cast(pl.row_expand_div(softmax_exp, pl.row_sum(softmax_exp)), pl.BF16, mode='rint')
    with pl.spmd(attention512_core_Q_HEADS, name_hint='qwen_attention512_weighted'):
        weighted_head = pl.tile.get_block_idx()
        weighted_kv_head = weighted_head // (attention512_core_Q_HEADS // attention512_core_KV_HEADS)
        weighted_start = weighted_head * attention512_core_HEAD_DIM
        weighted_sum = pl.full([attention512_core_ROWS, attention512_core_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for weighted_block in pl.range(attention512_core_CONTEXT // attention512_core_KEY_BLOCK):
            weighted_key_start = weighted_block * attention512_core_KEY_BLOCK
            weighted_probs = pl.slice(probability_buffer, [attention512_core_ROWS, attention512_core_KEY_BLOCK], [weighted_head * attention512_core_ROWS, weighted_key_start])
            weighted_values = pl.reshape(pl.slice(value_cache, [1, attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM], [weighted_kv_head, weighted_key_start, 0]), [attention512_core_KEY_BLOCK, attention512_core_HEAD_DIM])
            weighted_sum = pl.add(weighted_sum, pl.matmul(weighted_probs, weighted_values, pl.FP32))
        weighted = pl.cast(weighted_sum, pl.BF16, mode='rint')
        weighted_output[0:attention512_core_ROWS, weighted_start:weighted_start + attention512_core_HEAD_DIM] = weighted
        gate_tile = pl.cast(pl.slice(gate, [attention512_core_ROWS, attention512_core_HEAD_DIM], [0, weighted_start]), pl.FP32)
        sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        gated_output[0:attention512_core_ROWS, weighted_start:weighted_start + attention512_core_HEAD_DIM] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint')
    projected_output = linear_hidden_projection_inline(gated_output, output_weight, projected_output)
    return (logits_buffer, probability_buffer, weighted_output, gated_output, projected_output)

@pl.jit
def attention512_core_attention512_core(query: pl.Tensor[[attention512_core_ROWS, attention512_core_Q_WIDTH], pl.BF16], key_cache: pl.Tensor[[attention512_core_KV_HEADS, attention512_core_CONTEXT, attention512_core_HEAD_DIM], pl.BF16], value_cache: pl.Tensor[[attention512_core_KV_HEADS, attention512_core_CONTEXT, attention512_core_HEAD_DIM], pl.BF16], gate: pl.Tensor[[attention512_core_ROWS, attention512_core_Q_WIDTH], pl.BF16], mask: pl.Tensor[[attention512_core_ROWS, attention512_core_CONTEXT], pl.FP32], output_weight: pl.Tensor[[5120, attention512_core_Q_WIDTH], pl.BF16], logits_buffer: pl.Out[pl.Tensor[[attention512_core_Q_HEADS * attention512_core_ROWS, attention512_core_CONTEXT], pl.FP32]], probability_buffer: pl.Out[pl.Tensor[[attention512_core_Q_HEADS * attention512_core_ROWS, attention512_core_CONTEXT], pl.BF16]], weighted_output: pl.Out[pl.Tensor[[attention512_core_ROWS, attention512_core_Q_WIDTH], pl.BF16]], gated_output: pl.Out[pl.Tensor[[attention512_core_ROWS, attention512_core_Q_WIDTH], pl.BF16]], projected_output: pl.Out[pl.Tensor[[attention512_core_ROWS, 5120], pl.BF16]]):
    return attention512_core_attention512_core_inline(query, key_cache, value_cache, gate, mask, output_weight, logits_buffer, probability_buffer, weighted_output, gated_output, projected_output)

residual_ROWS = 16

residual_HIDDEN = 5120

residual_TILE = 128

@pl.jit.inline
def residual_residual_add_inline(left: pl.Tensor, right: pl.Tensor, output: pl.Tensor):
    with pl.spmd(residual_ROWS * (residual_HIDDEN // residual_TILE), name_hint='qwen_residual_add'):
        task = pl.tile.get_block_idx()
        row = task // (residual_HIDDEN // residual_TILE)
        column = task % (residual_HIDDEN // residual_TILE) * residual_TILE
        a = pl.cast(pl.slice(left, [1, residual_TILE], [row, column]), pl.FP32)
        b = pl.cast(pl.slice(right, [1, residual_TILE], [row, column]), pl.FP32)
        output[row:row + 1, column:column + residual_TILE] = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
    return output

swiglu_ROWS = 16

swiglu_INTERMEDIATE = 17408

swiglu_CHANNEL_TILE = 128

@pl.jit.inline
def swiglu_swiglu_inline(gate: pl.Tensor, up: pl.Tensor, output: pl.Tensor):
    with pl.spmd(swiglu_INTERMEDIATE // swiglu_CHANNEL_TILE, name_hint='qwen_swiglu'):
        column = pl.tile.get_block_idx() * swiglu_CHANNEL_TILE
        gate_tile = pl.cast(pl.slice(gate, [swiglu_ROWS, swiglu_CHANNEL_TILE], [0, column]), pl.FP32)
        up_tile = pl.cast(pl.slice(up, [swiglu_ROWS, swiglu_CHANNEL_TILE], [0, column]), pl.FP32)
        activated = pl.cast(pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0)), pl.BF16, mode='rint')
        multiplied = pl.mul(pl.cast(activated, pl.FP32), up_tile)
        output[0:swiglu_ROWS, column:column + swiglu_CHANNEL_TILE] = pl.cast(multiplied, pl.BF16, mode='rint')
    return output

@pl.jit
def swiglu_swiglu(gate: pl.Tensor[[swiglu_ROWS, swiglu_INTERMEDIATE], pl.BF16], up: pl.Tensor[[swiglu_ROWS, swiglu_INTERMEDIATE], pl.BF16], output: pl.Out[pl.Tensor[[swiglu_ROWS, swiglu_INTERMEDIATE], pl.BF16]]):
    output = swiglu_swiglu_inline(gate, up, output)
    return output

layer0_stem_ROWS = 16

layer0_stem_HIDDEN = 5120

layer0_stem_QKV = 10240

layer0_stem_Z = 6144

layer0_stem_GATE_PAD = 128

@pl.jit.inline
def layer0_stem_layer0_stem_inline(hidden: pl.Tensor, gamma: pl.Tensor, qkv_weight: pl.Tensor, z_weight: pl.Tensor, a_weight: pl.Tensor, b_weight: pl.Tensor, norm_output: pl.Tensor, qkv_output: pl.Tensor, z_output: pl.Tensor, a_output: pl.Tensor, b_output: pl.Tensor):
    norm_output = rmsnorm_rmsnorm_inline(hidden, gamma, norm_output)
    qkv_output = linear_hidden_projection_inline(norm_output, qkv_weight, qkv_output)
    z_output = linear_hidden_projection_inline(norm_output, z_weight, z_output)
    a_output = linear_hidden_projection_inline(norm_output, a_weight, a_output)
    b_output = linear_hidden_projection_inline(norm_output, b_weight, b_output)
    return (norm_output, qkv_output, z_output, a_output, b_output)

@pl.jit
def layer0_stem_layer0_stem(hidden: pl.Tensor[[layer0_stem_ROWS, layer0_stem_HIDDEN], pl.BF16], gamma: pl.Tensor[[1, layer0_stem_HIDDEN], pl.FP32], qkv_weight: pl.Tensor[[layer0_stem_QKV, layer0_stem_HIDDEN], pl.BF16], z_weight: pl.Tensor[[layer0_stem_Z, layer0_stem_HIDDEN], pl.BF16], a_weight: pl.Tensor[[layer0_stem_GATE_PAD, layer0_stem_HIDDEN], pl.BF16], b_weight: pl.Tensor[[layer0_stem_GATE_PAD, layer0_stem_HIDDEN], pl.BF16], norm_output: pl.Out[pl.Tensor[[layer0_stem_ROWS, layer0_stem_HIDDEN], pl.BF16]], qkv_output: pl.Out[pl.Tensor[[layer0_stem_ROWS, layer0_stem_QKV], pl.BF16]], z_output: pl.Out[pl.Tensor[[layer0_stem_ROWS, layer0_stem_Z], pl.BF16]], a_output: pl.Out[pl.Tensor[[layer0_stem_ROWS, layer0_stem_GATE_PAD], pl.BF16]], b_output: pl.Out[pl.Tensor[[layer0_stem_ROWS, layer0_stem_GATE_PAD], pl.BF16]]):
    return layer0_stem_layer0_stem_inline(hidden, gamma, qkv_weight, z_weight, a_weight, b_weight, norm_output, qkv_output, z_output, a_output, b_output)

ROWS = 16

BATCH = 8

CONTEXT = 512

BLOCK_SIZE = 16

POOL_BLOCKS = 256

HIDDEN = 5120

INTERMEDIATE = 17408

QKV = 10240

VALUE_WIDTH = 6144

DELTA_HEADS = 48

DELTA_DIM = 128

FULL_HEAD_DIM = 256

FULL_Q_WIDTH = 6144

FULL_Q_PROJECTION = 12288

FULL_KV_WIDTH = 1024

ROTARY = 64

VOCAB = 248320

@pl.jit.inline
def full_step_inline(input_ids: pl.Tensor[[ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * HIDDEN, INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * QKV, HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * VALUE_WIDTH, HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * HIDDEN, VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * FULL_Q_PROJECTION, HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * HIDDEN, FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[ROWS, CONTEXT], pl.FP32], token_count: pl.Scalar[pl.INT32], cache_offset: pl.Scalar[pl.INT32], conv_states: pl.Tensor[[48, 4, QKV], pl.BF16], recurrent_states: pl.Tensor[[48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32], key_pool: pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16], value_pool: pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16], block_table: pl.Tensor[[CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Tensor[[66, ROWS, HIDDEN], pl.BF16], logits: pl.Tensor[[ROWS, VOCAB], pl.BF16]):
    running = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
    with pl.spmd(ROWS * (HIDDEN // 128), name_hint='qwen_full_token_embedding'):
        embed_task = pl.tile.get_block_idx()
        embed_token = embed_task // (HIDDEN // 128)
        embed_column = embed_task % (HIDDEN // 128) * 128
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
        input_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16] = pl.slice(mlp_gate_weight, [INTERMEDIATE, HIDDEN], [layer * INTERMEDIATE, 0])
        mlp_up_layer: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16] = pl.slice(mlp_up_weight, [INTERMEDIATE, HIDDEN], [layer * INTERMEDIATE, 0])
        mlp_down_layer: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16] = pl.slice(mlp_down_weight, [HIDDEN, INTERMEDIATE], [layer * HIDDEN, 0])
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[QKV, HIDDEN], pl.BF16] = pl.slice(delta_qkv_weight, [QKV, HIDDEN], [linear * QKV, 0])
            z_weight_layer: pl.Tensor[[VALUE_WIDTH, HIDDEN], pl.BF16] = pl.slice(delta_z_weight, [VALUE_WIDTH, HIDDEN], [linear * VALUE_WIDTH, 0])
            a_weight_layer: pl.Tensor[[128, HIDDEN], pl.BF16] = pl.slice(delta_a_weight, [128, HIDDEN], [linear * 128, 0])
            b_weight_layer: pl.Tensor[[128, HIDDEN], pl.BF16] = pl.slice(delta_b_weight, [128, HIDDEN], [linear * 128, 0])
            conv_weight_layer: pl.Tensor[[4, QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[HIDDEN, VALUE_WIDTH], pl.BF16] = pl.slice(delta_out_weight, [HIDDEN, VALUE_WIDTH], [linear * HIDDEN, 0])
            conv_source: pl.Tensor[[4, QKV], pl.BF16] = pl.reshape(pl.slice(conv_states, [1, 4, QKV], [linear, 0, 0]), [4, QKV])
            recurrent_source: pl.Tensor[[DELTA_HEADS * DELTA_DIM, DELTA_DIM], pl.FP32] = pl.reshape(pl.slice(recurrent_states, [1, DELTA_HEADS, DELTA_DIM, DELTA_DIM], [linear, 0, 0, 0]), [DELTA_HEADS * DELTA_DIM, DELTA_DIM])
            conv = pl.create_tensor([4, QKV], dtype=pl.BF16)
            recurrent = pl.create_tensor([DELTA_HEADS, DELTA_DIM, DELTA_DIM], dtype=pl.FP32)
            recurrent_rows = pl.reshape(recurrent, [DELTA_HEADS * DELTA_DIM, DELTA_DIM])
            with pl.spmd(4 * (QKV // 128), name_hint='qwen_full_load_conv'):
                task = pl.tile.get_block_idx()
                tap = task // (QKV // 128)
                col = task % (QKV // 128) * 128
                conv[tap:tap + 1, col:col + 128] = pl.slice(conv_source, [1, 128], [tap, col])
            with pl.spmd(DELTA_HEADS * DELTA_DIM // 16, name_hint='qwen_full_load_recurrent'):
                row = pl.tile.get_block_idx() * 16
                recurrent_rows[row:row + 16, 0:DELTA_DIM] = pl.slice(recurrent_source, [16, DELTA_DIM], [row, 0])
            norm_delta = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            qkv = pl.create_tensor([ROWS, QKV], dtype=pl.BF16)
            z = pl.create_tensor([ROWS, VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([ROWS, 128], dtype=pl.BF16)
            b = pl.create_tensor([ROWS, 128], dtype=pl.BF16)
            norm_delta, qkv, z, a, b = layer0_stem_layer0_stem_inline(running, input_gamma_layer, qkv_weight_layer, z_weight_layer, a_weight_layer, b_weight_layer, norm_delta, qkv, z, a, b)
            conv_output = pl.create_tensor([ROWS, QKV], dtype=pl.BF16)
            query = pl.create_tensor([ROWS, DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            key = pl.create_tensor([ROWS, DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            value = pl.create_tensor([ROWS, DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            log_decay = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            beta = pl.create_tensor([ROWS, DELTA_HEADS], dtype=pl.FP32)
            core = pl.create_tensor([ROWS, DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            conv_output, conv = causal_conv_causal_conv_silu_inline(qkv, conv_weight_layer, conv, conv_output, token_count)
            query, key, value = delta_window16_expand_qkv_window16_inline(conv_output, query, key, value)
            gate_values = pl.create_tensor([ROWS, 64], dtype=pl.FP32)
            beta_bf16 = pl.create_tensor([ROWS, 64], dtype=pl.BF16)
            gate_values, beta_bf16 = delta_prepare_decay_beta_inline(a, b, a_log_layer, dt_bias_layer, gate_values, beta_bf16)
            with pl.spmd(ROWS, name_hint='qwen_full_delta_gate_unpad'):
                token = pl.tile.get_block_idx()
                log_decay[token:token + 1, 0:DELTA_HEADS] = pl.slice(gate_values, [1, DELTA_HEADS], [token, 0])
                beta[token:token + 1, 0:DELTA_HEADS] = pl.cast(pl.slice(beta_bf16, [1, DELTA_HEADS], [token, 0]), pl.FP32)
            core, recurrent = delta_window16_recurrence_window16_inline(query, key, value, log_decay, beta, recurrent, core, token_count)
            gated = pl.create_tensor([ROWS * DELTA_HEADS, DELTA_DIM], dtype=pl.BF16)
            attention_delta = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            gated, attention_delta = delta_window16_attention_tail_window16_inline(core, z, delta_norm_layer, delta_out_layer, gated, attention_delta)
            residual_delta = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            residual_delta = residual_residual_add_inline(running, attention_delta, residual_delta)
            post_norm_delta = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            post_norm_delta = rmsnorm_rmsnorm_inline(residual_delta, post_gamma_layer, post_norm_delta)
            mlp_gate_delta = pl.create_tensor([ROWS, INTERMEDIATE], dtype=pl.BF16)
            mlp_up_delta = pl.create_tensor([ROWS, INTERMEDIATE], dtype=pl.BF16)
            mlp_gate_delta = linear_hidden_projection_inline(post_norm_delta, mlp_gate_layer, mlp_gate_delta)
            mlp_up_delta = linear_hidden_projection_inline(post_norm_delta, mlp_up_layer, mlp_up_delta)
            mlp_middle_delta = pl.create_tensor([ROWS, INTERMEDIATE], dtype=pl.BF16)
            mlp_middle_delta = swiglu_swiglu_inline(mlp_gate_delta, mlp_up_delta, mlp_middle_delta)
            mlp_output_delta = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            mlp_output_delta = linear_hidden_projection_inline(mlp_middle_delta, mlp_down_layer, mlp_output_delta)
            following_delta = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            following_delta = residual_residual_add_inline(residual_delta, mlp_output_delta, following_delta)
            conv_dest: pl.Tensor[[4, QKV], pl.BF16] = pl.reshape(pl.slice(conv_states, [1, 4, QKV], [linear, 0, 0]), [4, QKV])
            recurrent_dest: pl.Tensor[[DELTA_HEADS * DELTA_DIM, DELTA_DIM], pl.FP32] = pl.reshape(pl.slice(recurrent_states, [1, DELTA_HEADS, DELTA_DIM, DELTA_DIM], [linear, 0, 0, 0]), [DELTA_HEADS * DELTA_DIM, DELTA_DIM])
            recurrent_rows = pl.reshape(recurrent, [DELTA_HEADS * DELTA_DIM, DELTA_DIM])
            with pl.spmd(4 * (QKV // 128), name_hint='qwen_full_store_conv'):
                task = pl.tile.get_block_idx()
                tap = task // (QKV // 128)
                col = task % (QKV // 128) * 128
                conv_dest[tap:tap + 1, col:col + 128] = pl.slice(conv, [1, 128], [tap, col])
            with pl.spmd(DELTA_HEADS * DELTA_DIM // 16, name_hint='qwen_full_store_recurrent'):
                row = pl.tile.get_block_idx() * 16
                recurrent_dest[row:row + 16, 0:DELTA_DIM] = pl.slice(recurrent_rows, [16, DELTA_DIM], [row, 0])
            with pl.spmd(ROWS, name_hint='qwen_full_store_delta_hidden'):
                row = pl.tile.get_block_idx()
                next_hidden = pl.slice(following_delta, [1, HIDDEN], [row, 0])
                running[row:row + 1, 0:HIDDEN] = next_hidden
                traces[layer + 1:layer + 2, row:row + 1, 0:HIDDEN] = pl.reshape(next_hidden, [1, 1, HIDDEN])
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[FULL_Q_PROJECTION, HIDDEN], pl.BF16] = pl.slice(full_q_weight, [FULL_Q_PROJECTION, HIDDEN], [full * FULL_Q_PROJECTION, 0])
            k_weight_full: pl.Tensor[[FULL_KV_WIDTH, HIDDEN], pl.BF16] = pl.slice(full_k_weight, [FULL_KV_WIDTH, HIDDEN], [full * FULL_KV_WIDTH, 0])
            v_weight_full: pl.Tensor[[FULL_KV_WIDTH, HIDDEN], pl.BF16] = pl.slice(full_v_weight, [FULL_KV_WIDTH, HIDDEN], [full * FULL_KV_WIDTH, 0])
            q_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[HIDDEN, FULL_Q_WIDTH], pl.BF16] = pl.slice(full_out_weight, [HIDDEN, FULL_Q_WIDTH], [full * HIDDEN, 0])
            norm_full = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            norm_full = rmsnorm_rmsnorm_inline(running, input_gamma_layer, norm_full)
            q_norm_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm_full = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            value_full = pl.create_tensor([ROWS, FULL_KV_WIDTH], dtype=pl.BF16)
            gate_full = pl.create_tensor([ROWS, FULL_Q_WIDTH], dtype=pl.BF16)
            q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full = attention3_prepare_attention3_prepare_inline(norm_full, q_weight_full, k_weight_full, v_weight_full, q_gamma_full, k_gamma_full, cos, sin, q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full)
            key_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            value_local = pl.create_tensor([4, CONTEXT, FULL_HEAD_DIM], dtype=pl.BF16)
            with pl.spmd(4 * (CONTEXT // BLOCK_SIZE), name_hint='qwen_full_load_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // (CONTEXT // BLOCK_SIZE)
                logical_block = task % (CONTEXT // BLOCK_SIZE)
                physical = pl.read(block_table, [logical_block])
                row = logical_block * BLOCK_SIZE
                key_local[head:head + 1, row:row + BLOCK_SIZE, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(key_pool, [1, 1, 1, BLOCK_SIZE, FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, BLOCK_SIZE, FULL_HEAD_DIM])
                value_local[head:head + 1, row:row + BLOCK_SIZE, 0:FULL_HEAD_DIM] = pl.reshape(pl.slice(value_pool, [1, 1, 1, BLOCK_SIZE, FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, BLOCK_SIZE, FULL_HEAD_DIM])
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
            scores, probabilities, weighted, full_gated, attention_full = attention512_core_attention512_core_inline(q_rot, key_local, value_local, gate_full, mask, out_weight_full, scores, probabilities, weighted, full_gated, attention_full)
            residual_full = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            residual_full = residual_residual_add_inline(running, attention_full, residual_full)
            post_norm_full = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            post_norm_full = rmsnorm_rmsnorm_inline(residual_full, post_gamma_layer, post_norm_full)
            mlp_gate_full = pl.create_tensor([ROWS, INTERMEDIATE], dtype=pl.BF16)
            mlp_up_full = pl.create_tensor([ROWS, INTERMEDIATE], dtype=pl.BF16)
            mlp_gate_full = linear_hidden_projection_inline(post_norm_full, mlp_gate_layer, mlp_gate_full)
            mlp_up_full = linear_hidden_projection_inline(post_norm_full, mlp_up_layer, mlp_up_full)
            mlp_middle_full = pl.create_tensor([ROWS, INTERMEDIATE], dtype=pl.BF16)
            mlp_middle_full = swiglu_swiglu_inline(mlp_gate_full, mlp_up_full, mlp_middle_full)
            mlp_output_full = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            mlp_output_full = linear_hidden_projection_inline(mlp_middle_full, mlp_down_layer, mlp_output_full)
            following_full = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
            following_full = residual_residual_add_inline(residual_full, mlp_output_full, following_full)
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
            with pl.spmd(ROWS, name_hint='qwen_full_store_attention_hidden'):
                row = pl.tile.get_block_idx()
                next_hidden = pl.slice(following_full, [1, HIDDEN], [row, 0])
                running[row:row + 1, 0:HIDDEN] = next_hidden
                traces[layer + 1:layer + 2, row:row + 1, 0:HIDDEN] = pl.reshape(next_hidden, [1, 1, HIDDEN])
    normalized = pl.create_tensor([ROWS, HIDDEN], dtype=pl.BF16)
    normalized = rmsnorm_rmsnorm_inline(running, final_gamma, normalized)
    with pl.spmd(ROWS, name_hint='qwen_full_store_final_norm'):
        row = pl.tile.get_block_idx()
        traces[65:66, row:row + 1, 0:HIDDEN] = pl.reshape(pl.slice(normalized, [1, HIDDEN], [row, 0]), [1, 1, HIDDEN])
    logits = linear_hidden_projection_inline(normalized, lm_head, logits)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def full_short_fwd(input_ids: pl.Tensor[[ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * HIDDEN, INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * QKV, HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * VALUE_WIDTH, HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * HIDDEN, VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * FULL_Q_PROJECTION, HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * HIDDEN, FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[ROWS, CONTEXT], pl.FP32], token_count: pl.Scalar[pl.INT32], cache_offset: pl.Scalar[pl.INT32], conv_states: pl.InOut[pl.Tensor[[48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], block_table: pl.Tensor[[CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[66, ROWS, HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[ROWS, VOCAB], pl.BF16]]):
    return full_step_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_count, cache_offset, conv_states, recurrent_states, key_pool, value_pool, block_table, traces, logits)

@pl.jit.inline
def batch_step_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    for batch in pl.range(BATCH):
        if batch < active_count:
            ids_slice = pl.slice(input_ids, [1, ROWS], [batch, 0])
            ids = pl.reshape(ids_slice, [ROWS])
            cos_slice = pl.slice(cos, [1, ROWS, ROTARY], [batch, 0, 0])
            cos_slot = pl.reshape(cos_slice, [ROWS, ROTARY])
            sin_slice = pl.slice(sin, [1, ROWS, ROTARY], [batch, 0, 0])
            sin_slot = pl.reshape(sin_slice, [ROWS, ROTARY])
            mask_slice = pl.slice(mask, [1, ROWS, CONTEXT], [batch, 0, 0])
            mask_slot = pl.reshape(mask_slice, [ROWS, CONTEXT])
            conv_slice = pl.slice(conv_states, [1, 48, 4, QKV], [batch, 0, 0, 0])
            conv_slot = pl.reshape(conv_slice, [48, 4, QKV])
            recurrent_slice = pl.slice(recurrent_states, [1, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], [batch, 0, 0, 0, 0])
            recurrent_slot = pl.reshape(recurrent_slice, [48, DELTA_HEADS, DELTA_DIM, DELTA_DIM])
            table_slice = pl.slice(block_tables, [1, CONTEXT // BLOCK_SIZE], [batch, 0])
            table_slot = pl.reshape(table_slice, [CONTEXT // BLOCK_SIZE])
            trace_slice = pl.slice(traces, [1, 66, ROWS, HIDDEN], [batch, 0, 0, 0])
            trace_slot = pl.reshape(trace_slice, [66, ROWS, HIDDEN])
            logits_slice = pl.slice(logits, [1, ROWS, VOCAB], [batch, 0, 0])
            logits_slot = pl.reshape(logits_slice, [ROWS, VOCAB])
            conv_slot, recurrent_slot, key_pool, value_pool, trace_slot, logits_slot = full_step_inline(ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos_slot, sin_slot, mask_slot, pl.read(token_counts, [batch]), pl.read(cache_offsets, [batch]), conv_slot, recurrent_slot, key_pool, value_pool, table_slot, trace_slot, logits_slot)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def rank_prefill_fwd(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * HIDDEN, INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * QKV, HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * VALUE_WIDTH, HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * HIDDEN, VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * FULL_Q_PROJECTION, HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * HIDDEN, FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[BATCH, ROWS, CONTEXT], pl.FP32], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[BATCH, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[BATCH, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    for batch in pl.range(BATCH):
        if batch < active_count:
            ids_slice = pl.slice(input_ids, [1, ROWS], [batch, 0])
            ids = pl.reshape(ids_slice, [ROWS])
            cos_slice = pl.slice(cos, [1, ROWS, ROTARY], [batch, 0, 0])
            cos_slot = pl.reshape(cos_slice, [ROWS, ROTARY])
            sin_slice = pl.slice(sin, [1, ROWS, ROTARY], [batch, 0, 0])
            sin_slot = pl.reshape(sin_slice, [ROWS, ROTARY])
            mask_slice = pl.slice(mask, [1, ROWS, CONTEXT], [batch, 0, 0])
            mask_slot = pl.reshape(mask_slice, [ROWS, CONTEXT])
            conv_slice = pl.slice(conv_states, [1, 48, 4, QKV], [batch, 0, 0, 0])
            conv_slot = pl.reshape(conv_slice, [48, 4, QKV])
            recurrent_slice = pl.slice(recurrent_states, [1, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], [batch, 0, 0, 0, 0])
            recurrent_slot = pl.reshape(recurrent_slice, [48, DELTA_HEADS, DELTA_DIM, DELTA_DIM])
            table_slice = pl.slice(block_tables, [1, CONTEXT // BLOCK_SIZE], [batch, 0])
            table_slot = pl.reshape(table_slice, [CONTEXT // BLOCK_SIZE])
            trace_slice = pl.slice(traces, [1, 66, ROWS, HIDDEN], [batch, 0, 0, 0])
            trace_slot = pl.reshape(trace_slice, [66, ROWS, HIDDEN])
            logits_slice = pl.slice(logits, [1, ROWS, VOCAB], [batch, 0, 0])
            logits_slot = pl.reshape(logits_slice, [ROWS, VOCAB])
            conv_slot, recurrent_slot, key_pool, value_pool, trace_slot, logits_slot = full_step_inline(ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos_slot, sin_slot, mask_slot, pl.read(token_counts, [batch]), pl.read(cache_offsets, [batch]), conv_slot, recurrent_slot, key_pool, value_pool, table_slot, trace_slot, logits_slot)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def rank_decode_fwd(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * HIDDEN, INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * QKV, HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * VALUE_WIDTH, HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * HIDDEN, VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * FULL_Q_PROJECTION, HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * HIDDEN, FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[BATCH, ROWS, CONTEXT], pl.FP32], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[BATCH, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[BATCH, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    conv_states, recurrent_states, key_pool, value_pool, traces, logits = batch_step_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

tp_rank_ROWS = 16

tp_rank_BATCH = 8

tp_rank_CONTEXT = 512

tp_rank_BLOCK_SIZE = 16

tp_rank_POOL_BLOCKS = 256

tp_rank_HIDDEN = 5120

tp_rank_INTERMEDIATE = 17408

tp_rank_QKV = 10240

tp_rank_VALUE_WIDTH = 6144

tp_rank_DELTA_HEADS = 48

tp_rank_DELTA_DIM = 128

tp_rank_FULL_HEAD_DIM = 256

tp_rank_FULL_Q_WIDTH = 6144

tp_rank_FULL_Q_PROJECTION = 12288

tp_rank_FULL_KV_WIDTH = 1024

tp_rank_ROTARY = 64

tp_rank_VOCAB = 248320

tp_rank_TP_VOCAB = tp_rank_VOCAB // 2

@pl.jit.inline
def tp_rank_full_step_inline(input_ids: pl.Tensor[[tp_rank_ROWS], pl.INT32], embedding: pl.Tensor[[tp_rank_VOCAB, tp_rank_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, tp_rank_HIDDEN], pl.FP32], lm_head: pl.Tensor[[tp_rank_TP_VOCAB, tp_rank_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, tp_rank_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, tp_rank_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * tp_rank_INTERMEDIATE, tp_rank_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * tp_rank_INTERMEDIATE, tp_rank_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * tp_rank_HIDDEN, tp_rank_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * tp_rank_QKV, tp_rank_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * tp_rank_VALUE_WIDTH, tp_rank_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, tp_rank_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, tp_rank_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, tp_rank_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, tp_rank_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * tp_rank_HIDDEN, tp_rank_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * tp_rank_FULL_Q_PROJECTION, tp_rank_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, tp_rank_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, tp_rank_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * tp_rank_HIDDEN, tp_rank_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[tp_rank_ROWS, tp_rank_ROTARY], pl.BF16], sin: pl.Tensor[[tp_rank_ROWS, tp_rank_ROTARY], pl.BF16], mask: pl.Tensor[[tp_rank_ROWS, tp_rank_CONTEXT], pl.FP32], token_count: pl.Scalar[pl.INT32], cache_offset: pl.Scalar[pl.INT32], conv_states: pl.Tensor[[48, 4, tp_rank_QKV], pl.BF16], recurrent_states: pl.Tensor[[48, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], pl.FP32], key_pool: pl.Tensor[[16, 4, tp_rank_POOL_BLOCKS, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM], pl.BF16], value_pool: pl.Tensor[[16, 4, tp_rank_POOL_BLOCKS, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM], pl.BF16], block_table: pl.Tensor[[tp_rank_CONTEXT // tp_rank_BLOCK_SIZE], pl.INT32], traces: pl.Tensor[[66, tp_rank_ROWS, tp_rank_HIDDEN], pl.BF16], logits: pl.Tensor[[tp_rank_ROWS, tp_rank_TP_VOCAB], pl.BF16]):
    running = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
    with pl.spmd(tp_rank_ROWS * (tp_rank_HIDDEN // 128), name_hint='qwen_full_token_embedding'):
        embed_task = pl.tile.get_block_idx()
        embed_token = embed_task // (tp_rank_HIDDEN // 128)
        embed_column = embed_task % (tp_rank_HIDDEN // 128) * 128
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
        input_gamma_layer: pl.Tensor[[1, tp_rank_HIDDEN], pl.FP32] = pl.slice(input_gamma, [1, tp_rank_HIDDEN], [layer, 0])
        post_gamma_layer: pl.Tensor[[1, tp_rank_HIDDEN], pl.FP32] = pl.slice(post_gamma, [1, tp_rank_HIDDEN], [layer, 0])
        mlp_gate_layer: pl.Tensor[[tp_rank_INTERMEDIATE, tp_rank_HIDDEN], pl.BF16] = pl.slice(mlp_gate_weight, [tp_rank_INTERMEDIATE, tp_rank_HIDDEN], [layer * tp_rank_INTERMEDIATE, 0])
        mlp_up_layer: pl.Tensor[[tp_rank_INTERMEDIATE, tp_rank_HIDDEN], pl.BF16] = pl.slice(mlp_up_weight, [tp_rank_INTERMEDIATE, tp_rank_HIDDEN], [layer * tp_rank_INTERMEDIATE, 0])
        mlp_down_layer: pl.Tensor[[tp_rank_HIDDEN, tp_rank_INTERMEDIATE], pl.BF16] = pl.slice(mlp_down_weight, [tp_rank_HIDDEN, tp_rank_INTERMEDIATE], [layer * tp_rank_HIDDEN, 0])
        if layer % 4 < 3:
            linear = layer - layer // 4
            qkv_weight_layer: pl.Tensor[[tp_rank_QKV, tp_rank_HIDDEN], pl.BF16] = pl.slice(delta_qkv_weight, [tp_rank_QKV, tp_rank_HIDDEN], [linear * tp_rank_QKV, 0])
            z_weight_layer: pl.Tensor[[tp_rank_VALUE_WIDTH, tp_rank_HIDDEN], pl.BF16] = pl.slice(delta_z_weight, [tp_rank_VALUE_WIDTH, tp_rank_HIDDEN], [linear * tp_rank_VALUE_WIDTH, 0])
            a_weight_layer: pl.Tensor[[128, tp_rank_HIDDEN], pl.BF16] = pl.slice(delta_a_weight, [128, tp_rank_HIDDEN], [linear * 128, 0])
            b_weight_layer: pl.Tensor[[128, tp_rank_HIDDEN], pl.BF16] = pl.slice(delta_b_weight, [128, tp_rank_HIDDEN], [linear * 128, 0])
            conv_weight_layer: pl.Tensor[[4, tp_rank_QKV], pl.BF16] = pl.slice(delta_conv_weight, [4, tp_rank_QKV], [linear * 4, 0])
            a_log_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_a_log, [1, 64], [linear, 0])
            dt_bias_layer: pl.Tensor[[1, 64], pl.BF16] = pl.slice(delta_dt_bias, [1, 64], [linear, 0])
            delta_norm_layer: pl.Tensor[[1, tp_rank_DELTA_DIM], pl.BF16] = pl.slice(delta_norm_weight, [1, tp_rank_DELTA_DIM], [linear, 0])
            delta_out_layer: pl.Tensor[[tp_rank_HIDDEN, tp_rank_VALUE_WIDTH], pl.BF16] = pl.slice(delta_out_weight, [tp_rank_HIDDEN, tp_rank_VALUE_WIDTH], [linear * tp_rank_HIDDEN, 0])
            conv_source: pl.Tensor[[4, tp_rank_QKV], pl.BF16] = pl.reshape(pl.slice(conv_states, [1, 4, tp_rank_QKV], [linear, 0, 0]), [4, tp_rank_QKV])
            recurrent_source: pl.Tensor[[tp_rank_DELTA_HEADS * tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], pl.FP32] = pl.reshape(pl.slice(recurrent_states, [1, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], [linear, 0, 0, 0]), [tp_rank_DELTA_HEADS * tp_rank_DELTA_DIM, tp_rank_DELTA_DIM])
            conv = pl.create_tensor([4, tp_rank_QKV], dtype=pl.BF16)
            recurrent = pl.create_tensor([tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], dtype=pl.FP32)
            recurrent_rows = pl.reshape(recurrent, [tp_rank_DELTA_HEADS * tp_rank_DELTA_DIM, tp_rank_DELTA_DIM])
            with pl.spmd(4 * (tp_rank_QKV // 128), name_hint='qwen_full_load_conv'):
                task = pl.tile.get_block_idx()
                tap = task // (tp_rank_QKV // 128)
                col = task % (tp_rank_QKV // 128) * 128
                conv[tap:tap + 1, col:col + 128] = pl.slice(conv_source, [1, 128], [tap, col])
            with pl.spmd(tp_rank_DELTA_HEADS * tp_rank_DELTA_DIM // 16, name_hint='qwen_full_load_recurrent'):
                row = pl.tile.get_block_idx() * 16
                recurrent_rows[row:row + 16, 0:tp_rank_DELTA_DIM] = pl.slice(recurrent_source, [16, tp_rank_DELTA_DIM], [row, 0])
            norm_delta = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            qkv = pl.create_tensor([tp_rank_ROWS, tp_rank_QKV], dtype=pl.BF16)
            z = pl.create_tensor([tp_rank_ROWS, tp_rank_VALUE_WIDTH], dtype=pl.BF16)
            a = pl.create_tensor([tp_rank_ROWS, 128], dtype=pl.BF16)
            b = pl.create_tensor([tp_rank_ROWS, 128], dtype=pl.BF16)
            norm_delta, qkv, z, a, b = layer0_stem_layer0_stem_inline(running, input_gamma_layer, qkv_weight_layer, z_weight_layer, a_weight_layer, b_weight_layer, norm_delta, qkv, z, a, b)
            conv_output = pl.create_tensor([tp_rank_ROWS, tp_rank_QKV], dtype=pl.BF16)
            query = pl.create_tensor([tp_rank_ROWS, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM], dtype=pl.BF16)
            key = pl.create_tensor([tp_rank_ROWS, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM], dtype=pl.BF16)
            value = pl.create_tensor([tp_rank_ROWS, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM], dtype=pl.BF16)
            log_decay = pl.create_tensor([tp_rank_ROWS, tp_rank_DELTA_HEADS], dtype=pl.FP32)
            beta = pl.create_tensor([tp_rank_ROWS, tp_rank_DELTA_HEADS], dtype=pl.FP32)
            core = pl.create_tensor([tp_rank_ROWS, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM], dtype=pl.BF16)
            conv_output, conv = causal_conv_causal_conv_silu_inline(qkv, conv_weight_layer, conv, conv_output, token_count)
            query, key, value = delta_window16_expand_qkv_window16_inline(conv_output, query, key, value)
            gate_values = pl.create_tensor([tp_rank_ROWS, 64], dtype=pl.FP32)
            beta_bf16 = pl.create_tensor([tp_rank_ROWS, 64], dtype=pl.BF16)
            gate_values, beta_bf16 = delta_prepare_decay_beta_inline(a, b, a_log_layer, dt_bias_layer, gate_values, beta_bf16)
            with pl.spmd(tp_rank_ROWS, name_hint='qwen_full_delta_gate_unpad'):
                token = pl.tile.get_block_idx()
                log_decay[token:token + 1, 0:tp_rank_DELTA_HEADS] = pl.slice(gate_values, [1, tp_rank_DELTA_HEADS], [token, 0])
                beta[token:token + 1, 0:tp_rank_DELTA_HEADS] = pl.cast(pl.slice(beta_bf16, [1, tp_rank_DELTA_HEADS], [token, 0]), pl.FP32)
            core, recurrent = delta_window16_recurrence_window16_inline(query, key, value, log_decay, beta, recurrent, core, token_count)
            gated = pl.create_tensor([tp_rank_ROWS * tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM], dtype=pl.BF16)
            attention_delta = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            gated, attention_delta = delta_window16_attention_tail_window16_inline(core, z, delta_norm_layer, delta_out_layer, gated, attention_delta)
            residual_delta = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            residual_delta = residual_residual_add_inline(running, attention_delta, residual_delta)
            post_norm_delta = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            post_norm_delta = rmsnorm_rmsnorm_inline(residual_delta, post_gamma_layer, post_norm_delta)
            mlp_gate_delta = pl.create_tensor([tp_rank_ROWS, tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_up_delta = pl.create_tensor([tp_rank_ROWS, tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_gate_delta = linear_hidden_projection_inline(post_norm_delta, mlp_gate_layer, mlp_gate_delta)
            mlp_up_delta = linear_hidden_projection_inline(post_norm_delta, mlp_up_layer, mlp_up_delta)
            mlp_middle_delta = pl.create_tensor([tp_rank_ROWS, tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_middle_delta = swiglu_swiglu_inline(mlp_gate_delta, mlp_up_delta, mlp_middle_delta)
            mlp_output_delta = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            mlp_output_delta = linear_hidden_projection_inline(mlp_middle_delta, mlp_down_layer, mlp_output_delta)
            following_delta = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            following_delta = residual_residual_add_inline(residual_delta, mlp_output_delta, following_delta)
            conv_dest: pl.Tensor[[4, tp_rank_QKV], pl.BF16] = pl.reshape(pl.slice(conv_states, [1, 4, tp_rank_QKV], [linear, 0, 0]), [4, tp_rank_QKV])
            recurrent_dest: pl.Tensor[[tp_rank_DELTA_HEADS * tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], pl.FP32] = pl.reshape(pl.slice(recurrent_states, [1, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], [linear, 0, 0, 0]), [tp_rank_DELTA_HEADS * tp_rank_DELTA_DIM, tp_rank_DELTA_DIM])
            recurrent_rows = pl.reshape(recurrent, [tp_rank_DELTA_HEADS * tp_rank_DELTA_DIM, tp_rank_DELTA_DIM])
            with pl.spmd(4 * (tp_rank_QKV // 128), name_hint='qwen_full_store_conv'):
                task = pl.tile.get_block_idx()
                tap = task // (tp_rank_QKV // 128)
                col = task % (tp_rank_QKV // 128) * 128
                conv_dest[tap:tap + 1, col:col + 128] = pl.slice(conv, [1, 128], [tap, col])
            with pl.spmd(tp_rank_DELTA_HEADS * tp_rank_DELTA_DIM // 16, name_hint='qwen_full_store_recurrent'):
                row = pl.tile.get_block_idx() * 16
                recurrent_dest[row:row + 16, 0:tp_rank_DELTA_DIM] = pl.slice(recurrent_rows, [16, tp_rank_DELTA_DIM], [row, 0])
            with pl.spmd(tp_rank_ROWS, name_hint='qwen_full_store_delta_hidden'):
                row = pl.tile.get_block_idx()
                next_hidden = pl.slice(following_delta, [1, tp_rank_HIDDEN], [row, 0])
                running[row:row + 1, 0:tp_rank_HIDDEN] = next_hidden
                traces[layer + 1:layer + 2, row:row + 1, 0:tp_rank_HIDDEN] = pl.reshape(next_hidden, [1, 1, tp_rank_HIDDEN])
        else:
            full = layer // 4
            q_weight_full: pl.Tensor[[tp_rank_FULL_Q_PROJECTION, tp_rank_HIDDEN], pl.BF16] = pl.slice(full_q_weight, [tp_rank_FULL_Q_PROJECTION, tp_rank_HIDDEN], [full * tp_rank_FULL_Q_PROJECTION, 0])
            k_weight_full: pl.Tensor[[tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], pl.BF16] = pl.slice(full_k_weight, [tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], [full * tp_rank_FULL_KV_WIDTH, 0])
            v_weight_full: pl.Tensor[[tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], pl.BF16] = pl.slice(full_v_weight, [tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], [full * tp_rank_FULL_KV_WIDTH, 0])
            q_gamma_full: pl.Tensor[[1, tp_rank_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_q_gamma, [1, tp_rank_FULL_HEAD_DIM], [full, 0])
            k_gamma_full: pl.Tensor[[1, tp_rank_FULL_HEAD_DIM], pl.FP32] = pl.slice(full_k_gamma, [1, tp_rank_FULL_HEAD_DIM], [full, 0])
            out_weight_full: pl.Tensor[[tp_rank_HIDDEN, tp_rank_FULL_Q_WIDTH], pl.BF16] = pl.slice(full_out_weight, [tp_rank_HIDDEN, tp_rank_FULL_Q_WIDTH], [full * tp_rank_HIDDEN, 0])
            norm_full = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            norm_full = rmsnorm_rmsnorm_inline(running, input_gamma_layer, norm_full)
            q_norm_full = pl.create_tensor([tp_rank_ROWS, tp_rank_FULL_Q_WIDTH], dtype=pl.BF16)
            k_norm_full = pl.create_tensor([tp_rank_ROWS, tp_rank_FULL_KV_WIDTH], dtype=pl.BF16)
            q_rot = pl.create_tensor([tp_rank_ROWS, tp_rank_FULL_Q_WIDTH], dtype=pl.BF16)
            k_rot = pl.create_tensor([tp_rank_ROWS, tp_rank_FULL_KV_WIDTH], dtype=pl.BF16)
            value_full = pl.create_tensor([tp_rank_ROWS, tp_rank_FULL_KV_WIDTH], dtype=pl.BF16)
            gate_full = pl.create_tensor([tp_rank_ROWS, tp_rank_FULL_Q_WIDTH], dtype=pl.BF16)
            q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full = attention3_prepare_attention3_prepare_inline(norm_full, q_weight_full, k_weight_full, v_weight_full, q_gamma_full, k_gamma_full, cos, sin, q_norm_full, k_norm_full, q_rot, k_rot, value_full, gate_full)
            key_local = pl.create_tensor([4, tp_rank_CONTEXT, tp_rank_FULL_HEAD_DIM], dtype=pl.BF16)
            value_local = pl.create_tensor([4, tp_rank_CONTEXT, tp_rank_FULL_HEAD_DIM], dtype=pl.BF16)
            with pl.spmd(4 * (tp_rank_CONTEXT // tp_rank_BLOCK_SIZE), name_hint='qwen_full_load_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // (tp_rank_CONTEXT // tp_rank_BLOCK_SIZE)
                logical_block = task % (tp_rank_CONTEXT // tp_rank_BLOCK_SIZE)
                physical = pl.read(block_table, [logical_block])
                row = logical_block * tp_rank_BLOCK_SIZE
                key_local[head:head + 1, row:row + tp_rank_BLOCK_SIZE, 0:tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(key_pool, [1, 1, 1, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM])
                value_local[head:head + 1, row:row + tp_rank_BLOCK_SIZE, 0:tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_pool, [1, 1, 1, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM], [full, head, physical, 0, 0]), [1, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM])
            with pl.spmd(4, name_hint='qwen_full_update_kv'):
                head = pl.tile.get_block_idx()
                start = head * tp_rank_FULL_HEAD_DIM
                for token in pl.range(token_count):
                    key_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(k_rot, [1, tp_rank_FULL_HEAD_DIM], [token, start]), [1, 1, tp_rank_FULL_HEAD_DIM])
                    value_local[head:head + 1, cache_offset + token:cache_offset + token + 1, 0:tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_full, [1, tp_rank_FULL_HEAD_DIM], [token, start]), [1, 1, tp_rank_FULL_HEAD_DIM])
            scores = pl.create_tensor([24 * tp_rank_ROWS, tp_rank_CONTEXT], dtype=pl.FP32)
            probabilities = pl.create_tensor([24 * tp_rank_ROWS, tp_rank_CONTEXT], dtype=pl.BF16)
            weighted = pl.create_tensor([tp_rank_ROWS, tp_rank_FULL_Q_WIDTH], dtype=pl.BF16)
            full_gated = pl.create_tensor([tp_rank_ROWS, tp_rank_FULL_Q_WIDTH], dtype=pl.BF16)
            attention_full = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            scores, probabilities, weighted, full_gated, attention_full = attention512_core_attention512_core_inline(q_rot, key_local, value_local, gate_full, mask, out_weight_full, scores, probabilities, weighted, full_gated, attention_full)
            residual_full = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            residual_full = residual_residual_add_inline(running, attention_full, residual_full)
            post_norm_full = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            post_norm_full = rmsnorm_rmsnorm_inline(residual_full, post_gamma_layer, post_norm_full)
            mlp_gate_full = pl.create_tensor([tp_rank_ROWS, tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_up_full = pl.create_tensor([tp_rank_ROWS, tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_gate_full = linear_hidden_projection_inline(post_norm_full, mlp_gate_layer, mlp_gate_full)
            mlp_up_full = linear_hidden_projection_inline(post_norm_full, mlp_up_layer, mlp_up_full)
            mlp_middle_full = pl.create_tensor([tp_rank_ROWS, tp_rank_INTERMEDIATE], dtype=pl.BF16)
            mlp_middle_full = swiglu_swiglu_inline(mlp_gate_full, mlp_up_full, mlp_middle_full)
            mlp_output_full = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            mlp_output_full = linear_hidden_projection_inline(mlp_middle_full, mlp_down_layer, mlp_output_full)
            following_full = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
            following_full = residual_residual_add_inline(residual_full, mlp_output_full, following_full)
            with pl.spmd(4 * tp_rank_ROWS, name_hint='qwen_full_store_paged_kv'):
                task = pl.tile.get_block_idx()
                head = task // tp_rank_ROWS
                row = task % tp_rank_ROWS
                if row < token_count:
                    position = cache_offset + row
                    logical_block = position // tp_rank_BLOCK_SIZE
                    physical = pl.read(block_table, [logical_block])
                    slot = position % tp_rank_BLOCK_SIZE
                    key_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(key_local, [1, 1, tp_rank_FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, tp_rank_FULL_HEAD_DIM])
                    value_pool[full:full + 1, head:head + 1, physical:physical + 1, slot:slot + 1, 0:tp_rank_FULL_HEAD_DIM] = pl.reshape(pl.slice(value_local, [1, 1, tp_rank_FULL_HEAD_DIM], [head, position, 0]), [1, 1, 1, 1, tp_rank_FULL_HEAD_DIM])
            with pl.spmd(tp_rank_ROWS, name_hint='qwen_full_store_attention_hidden'):
                row = pl.tile.get_block_idx()
                next_hidden = pl.slice(following_full, [1, tp_rank_HIDDEN], [row, 0])
                running[row:row + 1, 0:tp_rank_HIDDEN] = next_hidden
                traces[layer + 1:layer + 2, row:row + 1, 0:tp_rank_HIDDEN] = pl.reshape(next_hidden, [1, 1, tp_rank_HIDDEN])
    normalized = pl.create_tensor([tp_rank_ROWS, tp_rank_HIDDEN], dtype=pl.BF16)
    normalized = rmsnorm_rmsnorm_inline(running, final_gamma, normalized)
    with pl.spmd(tp_rank_ROWS, name_hint='qwen_full_store_final_norm'):
        row = pl.tile.get_block_idx()
        traces[65:66, row:row + 1, 0:tp_rank_HIDDEN] = pl.reshape(pl.slice(normalized, [1, tp_rank_HIDDEN], [row, 0]), [1, 1, tp_rank_HIDDEN])
    logits = linear_hidden_projection_inline(normalized, lm_head, logits)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def tp_rank_full_short_fwd(input_ids: pl.Tensor[[tp_rank_ROWS], pl.INT32], embedding: pl.Tensor[[tp_rank_VOCAB, tp_rank_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, tp_rank_HIDDEN], pl.FP32], lm_head: pl.Tensor[[tp_rank_TP_VOCAB, tp_rank_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, tp_rank_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, tp_rank_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * tp_rank_INTERMEDIATE, tp_rank_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * tp_rank_INTERMEDIATE, tp_rank_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * tp_rank_HIDDEN, tp_rank_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * tp_rank_QKV, tp_rank_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * tp_rank_VALUE_WIDTH, tp_rank_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, tp_rank_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, tp_rank_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, tp_rank_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, tp_rank_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * tp_rank_HIDDEN, tp_rank_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * tp_rank_FULL_Q_PROJECTION, tp_rank_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, tp_rank_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, tp_rank_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * tp_rank_HIDDEN, tp_rank_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[tp_rank_ROWS, tp_rank_ROTARY], pl.BF16], sin: pl.Tensor[[tp_rank_ROWS, tp_rank_ROTARY], pl.BF16], mask: pl.Tensor[[tp_rank_ROWS, tp_rank_CONTEXT], pl.FP32], token_count: pl.Scalar[pl.INT32], cache_offset: pl.Scalar[pl.INT32], conv_states: pl.InOut[pl.Tensor[[48, 4, tp_rank_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[48, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, tp_rank_POOL_BLOCKS, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, tp_rank_POOL_BLOCKS, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM], pl.BF16]], block_table: pl.Tensor[[tp_rank_CONTEXT // tp_rank_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[66, tp_rank_ROWS, tp_rank_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[tp_rank_ROWS, tp_rank_TP_VOCAB], pl.BF16]]):
    return tp_rank_full_step_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_count, cache_offset, conv_states, recurrent_states, key_pool, value_pool, block_table, traces, logits)

@pl.jit.inline
def tp_rank_batch_step_inline(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.Tensor, recurrent_states: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor, block_tables: pl.Tensor, traces: pl.Tensor, logits: pl.Tensor, active_count: pl.Scalar[pl.INT32]):
    for batch in pl.range(tp_rank_BATCH):
        if batch < active_count:
            ids_slice = pl.slice(input_ids, [1, tp_rank_ROWS], [batch, 0])
            ids = pl.reshape(ids_slice, [tp_rank_ROWS])
            cos_slice = pl.slice(cos, [1, tp_rank_ROWS, tp_rank_ROTARY], [batch, 0, 0])
            cos_slot = pl.reshape(cos_slice, [tp_rank_ROWS, tp_rank_ROTARY])
            sin_slice = pl.slice(sin, [1, tp_rank_ROWS, tp_rank_ROTARY], [batch, 0, 0])
            sin_slot = pl.reshape(sin_slice, [tp_rank_ROWS, tp_rank_ROTARY])
            mask_slice = pl.slice(mask, [1, tp_rank_ROWS, tp_rank_CONTEXT], [batch, 0, 0])
            mask_slot = pl.reshape(mask_slice, [tp_rank_ROWS, tp_rank_CONTEXT])
            conv_slice = pl.slice(conv_states, [1, 48, 4, tp_rank_QKV], [batch, 0, 0, 0])
            conv_slot = pl.reshape(conv_slice, [48, 4, tp_rank_QKV])
            recurrent_slice = pl.slice(recurrent_states, [1, 48, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], [batch, 0, 0, 0, 0])
            recurrent_slot = pl.reshape(recurrent_slice, [48, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM])
            table_slice = pl.slice(block_tables, [1, tp_rank_CONTEXT // tp_rank_BLOCK_SIZE], [batch, 0])
            table_slot = pl.reshape(table_slice, [tp_rank_CONTEXT // tp_rank_BLOCK_SIZE])
            trace_slice = pl.slice(traces, [1, 66, tp_rank_ROWS, tp_rank_HIDDEN], [batch, 0, 0, 0])
            trace_slot = pl.reshape(trace_slice, [66, tp_rank_ROWS, tp_rank_HIDDEN])
            logits_slice = pl.slice(logits, [1, tp_rank_ROWS, tp_rank_TP_VOCAB], [batch, 0, 0])
            logits_slot = pl.reshape(logits_slice, [tp_rank_ROWS, tp_rank_TP_VOCAB])
            conv_slot, recurrent_slot, key_pool, value_pool, trace_slot, logits_slot = tp_rank_full_step_inline(ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos_slot, sin_slot, mask_slot, pl.read(token_counts, [batch]), pl.read(cache_offsets, [batch]), conv_slot, recurrent_slot, key_pool, value_pool, table_slot, trace_slot, logits_slot)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def tp_rank_prefill_fwd(input_ids: pl.Tensor[[tp_rank_BATCH, tp_rank_ROWS], pl.INT32], embedding: pl.Tensor[[tp_rank_VOCAB, tp_rank_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, tp_rank_HIDDEN], pl.FP32], lm_head: pl.Tensor[[tp_rank_TP_VOCAB, tp_rank_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, tp_rank_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, tp_rank_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * tp_rank_INTERMEDIATE, tp_rank_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * tp_rank_INTERMEDIATE, tp_rank_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * tp_rank_HIDDEN, tp_rank_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * tp_rank_QKV, tp_rank_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * tp_rank_VALUE_WIDTH, tp_rank_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, tp_rank_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, tp_rank_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, tp_rank_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, tp_rank_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * tp_rank_HIDDEN, tp_rank_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * tp_rank_FULL_Q_PROJECTION, tp_rank_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, tp_rank_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, tp_rank_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * tp_rank_HIDDEN, tp_rank_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[tp_rank_BATCH, tp_rank_ROWS, tp_rank_ROTARY], pl.BF16], sin: pl.Tensor[[tp_rank_BATCH, tp_rank_ROWS, tp_rank_ROTARY], pl.BF16], mask: pl.Tensor[[tp_rank_BATCH, tp_rank_ROWS, tp_rank_CONTEXT], pl.FP32], token_counts: pl.Tensor[[tp_rank_BATCH], pl.INT32], cache_offsets: pl.Tensor[[tp_rank_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[tp_rank_BATCH, 48, 4, tp_rank_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[tp_rank_BATCH, 48, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, tp_rank_POOL_BLOCKS, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, tp_rank_POOL_BLOCKS, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[tp_rank_BATCH, tp_rank_CONTEXT // tp_rank_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[tp_rank_BATCH, 66, tp_rank_ROWS, tp_rank_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[tp_rank_BATCH, tp_rank_ROWS, tp_rank_TP_VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    for batch in pl.range(tp_rank_BATCH):
        if batch < active_count:
            ids_slice = pl.slice(input_ids, [1, tp_rank_ROWS], [batch, 0])
            ids = pl.reshape(ids_slice, [tp_rank_ROWS])
            cos_slice = pl.slice(cos, [1, tp_rank_ROWS, tp_rank_ROTARY], [batch, 0, 0])
            cos_slot = pl.reshape(cos_slice, [tp_rank_ROWS, tp_rank_ROTARY])
            sin_slice = pl.slice(sin, [1, tp_rank_ROWS, tp_rank_ROTARY], [batch, 0, 0])
            sin_slot = pl.reshape(sin_slice, [tp_rank_ROWS, tp_rank_ROTARY])
            mask_slice = pl.slice(mask, [1, tp_rank_ROWS, tp_rank_CONTEXT], [batch, 0, 0])
            mask_slot = pl.reshape(mask_slice, [tp_rank_ROWS, tp_rank_CONTEXT])
            conv_slice = pl.slice(conv_states, [1, 48, 4, tp_rank_QKV], [batch, 0, 0, 0])
            conv_slot = pl.reshape(conv_slice, [48, 4, tp_rank_QKV])
            recurrent_slice = pl.slice(recurrent_states, [1, 48, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], [batch, 0, 0, 0, 0])
            recurrent_slot = pl.reshape(recurrent_slice, [48, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM])
            table_slice = pl.slice(block_tables, [1, tp_rank_CONTEXT // tp_rank_BLOCK_SIZE], [batch, 0])
            table_slot = pl.reshape(table_slice, [tp_rank_CONTEXT // tp_rank_BLOCK_SIZE])
            trace_slice = pl.slice(traces, [1, 66, tp_rank_ROWS, tp_rank_HIDDEN], [batch, 0, 0, 0])
            trace_slot = pl.reshape(trace_slice, [66, tp_rank_ROWS, tp_rank_HIDDEN])
            logits_slice = pl.slice(logits, [1, tp_rank_ROWS, tp_rank_TP_VOCAB], [batch, 0, 0])
            logits_slot = pl.reshape(logits_slice, [tp_rank_ROWS, tp_rank_TP_VOCAB])
            conv_slot, recurrent_slot, key_pool, value_pool, trace_slot, logits_slot = tp_rank_full_step_inline(ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos_slot, sin_slot, mask_slot, pl.read(token_counts, [batch]), pl.read(cache_offsets, [batch]), conv_slot, recurrent_slot, key_pool, value_pool, table_slot, trace_slot, logits_slot)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit
def tp_rank_decode_fwd(input_ids: pl.Tensor[[tp_rank_BATCH, tp_rank_ROWS], pl.INT32], embedding: pl.Tensor[[tp_rank_VOCAB, tp_rank_HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, tp_rank_HIDDEN], pl.FP32], lm_head: pl.Tensor[[tp_rank_TP_VOCAB, tp_rank_HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, tp_rank_HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, tp_rank_HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * tp_rank_INTERMEDIATE, tp_rank_HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * tp_rank_INTERMEDIATE, tp_rank_HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * tp_rank_HIDDEN, tp_rank_INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * tp_rank_QKV, tp_rank_HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * tp_rank_VALUE_WIDTH, tp_rank_HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, tp_rank_HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, tp_rank_HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, tp_rank_QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, tp_rank_DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * tp_rank_HIDDEN, tp_rank_VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * tp_rank_FULL_Q_PROJECTION, tp_rank_HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * tp_rank_FULL_KV_WIDTH, tp_rank_HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, tp_rank_FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, tp_rank_FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * tp_rank_HIDDEN, tp_rank_FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[tp_rank_BATCH, tp_rank_ROWS, tp_rank_ROTARY], pl.BF16], sin: pl.Tensor[[tp_rank_BATCH, tp_rank_ROWS, tp_rank_ROTARY], pl.BF16], mask: pl.Tensor[[tp_rank_BATCH, tp_rank_ROWS, tp_rank_CONTEXT], pl.FP32], token_counts: pl.Tensor[[tp_rank_BATCH], pl.INT32], cache_offsets: pl.Tensor[[tp_rank_BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[tp_rank_BATCH, 48, 4, tp_rank_QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[tp_rank_BATCH, 48, tp_rank_DELTA_HEADS, tp_rank_DELTA_DIM, tp_rank_DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, tp_rank_POOL_BLOCKS, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, tp_rank_POOL_BLOCKS, tp_rank_BLOCK_SIZE, tp_rank_FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[tp_rank_BATCH, tp_rank_CONTEXT // tp_rank_BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[tp_rank_BATCH, 66, tp_rank_ROWS, tp_rank_HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[tp_rank_BATCH, tp_rank_ROWS, tp_rank_TP_VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    conv_states, recurrent_states, key_pool, value_pool, traces, logits = tp_rank_batch_step_inline(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count)
    return (conv_states, recurrent_states, key_pool, value_pool, traces, logits)

@pl.jit.host
def tp_prefill_fwd(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.InOut[pl.Tensor], recurrent_states: pl.InOut[pl.Tensor], key_pool: pl.InOut[pl.Tensor], value_pool: pl.InOut[pl.Tensor], block_tables: pl.Tensor, traces: pl.Out[pl.Tensor], logits: pl.Out[pl.Tensor], active_count: pl.Scalar[pl.INT32]):
    for rank in pl.range(pld.world_size()):
        tp_rank_prefill_fwd(input_ids[rank], embedding[rank], final_gamma[rank], lm_head[rank], input_gamma[rank], post_gamma[rank], mlp_gate_weight[rank], mlp_up_weight[rank], mlp_down_weight[rank], delta_qkv_weight[rank], delta_z_weight[rank], delta_a_weight[rank], delta_b_weight[rank], delta_conv_weight[rank], delta_a_log[rank], delta_dt_bias[rank], delta_norm_weight[rank], delta_out_weight[rank], full_q_weight[rank], full_k_weight[rank], full_v_weight[rank], full_q_gamma[rank], full_k_gamma[rank], full_out_weight[rank], cos[rank], sin[rank], mask[rank], token_counts[rank], cache_offsets[rank], conv_states[rank], recurrent_states[rank], key_pool[rank], value_pool[rank], block_tables[rank], traces[rank], logits[rank], active_count, device=rank)

@pl.jit.host
def tp_decode_fwd(input_ids: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor, mlp_gate_weight: pl.Tensor, mlp_up_weight: pl.Tensor, mlp_down_weight: pl.Tensor, delta_qkv_weight: pl.Tensor, delta_z_weight: pl.Tensor, delta_a_weight: pl.Tensor, delta_b_weight: pl.Tensor, delta_conv_weight: pl.Tensor, delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm_weight: pl.Tensor, delta_out_weight: pl.Tensor, full_q_weight: pl.Tensor, full_k_weight: pl.Tensor, full_v_weight: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out_weight: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, mask: pl.Tensor, token_counts: pl.Tensor, cache_offsets: pl.Tensor, conv_states: pl.InOut[pl.Tensor], recurrent_states: pl.InOut[pl.Tensor], key_pool: pl.InOut[pl.Tensor], value_pool: pl.InOut[pl.Tensor], block_tables: pl.Tensor, traces: pl.Out[pl.Tensor], logits: pl.Out[pl.Tensor], active_count: pl.Scalar[pl.INT32]):
    for rank in pl.range(pld.world_size()):
        tp_rank_decode_fwd(input_ids[rank], embedding[rank], final_gamma[rank], lm_head[rank], input_gamma[rank], post_gamma[rank], mlp_gate_weight[rank], mlp_up_weight[rank], mlp_down_weight[rank], delta_qkv_weight[rank], delta_z_weight[rank], delta_a_weight[rank], delta_b_weight[rank], delta_conv_weight[rank], delta_a_log[rank], delta_dt_bias[rank], delta_norm_weight[rank], delta_out_weight[rank], full_q_weight[rank], full_k_weight[rank], full_v_weight[rank], full_q_gamma[rank], full_k_gamma[rank], full_out_weight[rank], cos[rank], sin[rank], mask[rank], token_counts[rank], cache_offsets[rank], conv_states[rank], recurrent_states[rank], key_pool[rank], value_pool[rank], block_tables[rank], traces[rank], logits[rank], active_count, device=rank)

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
def prefill_fwd(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * HIDDEN, INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * QKV, HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * VALUE_WIDTH, HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * HIDDEN, VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * FULL_Q_PROJECTION, HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * HIDDEN, FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[BATCH, ROWS, CONTEXT], pl.FP32], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[BATCH, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[BATCH, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    rank_prefill_fwd(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, device=0)

@pl.jit.host
def decode_fwd(input_ids: pl.Tensor[[BATCH, ROWS], pl.INT32], embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], final_gamma: pl.Tensor[[1, HIDDEN], pl.FP32], lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16], input_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], post_gamma: pl.Tensor[[64, HIDDEN], pl.FP32], mlp_gate_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_up_weight: pl.Tensor[[64 * INTERMEDIATE, HIDDEN], pl.BF16], mlp_down_weight: pl.Tensor[[64 * HIDDEN, INTERMEDIATE], pl.BF16], delta_qkv_weight: pl.Tensor[[48 * QKV, HIDDEN], pl.BF16], delta_z_weight: pl.Tensor[[48 * VALUE_WIDTH, HIDDEN], pl.BF16], delta_a_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_b_weight: pl.Tensor[[48 * 128, HIDDEN], pl.BF16], delta_conv_weight: pl.Tensor[[48 * 4, QKV], pl.BF16], delta_a_log: pl.Tensor[[48, 64], pl.BF16], delta_dt_bias: pl.Tensor[[48, 64], pl.BF16], delta_norm_weight: pl.Tensor[[48, DELTA_DIM], pl.BF16], delta_out_weight: pl.Tensor[[48 * HIDDEN, VALUE_WIDTH], pl.BF16], full_q_weight: pl.Tensor[[16 * FULL_Q_PROJECTION, HIDDEN], pl.BF16], full_k_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_v_weight: pl.Tensor[[16 * FULL_KV_WIDTH, HIDDEN], pl.BF16], full_q_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_k_gamma: pl.Tensor[[16, FULL_HEAD_DIM], pl.FP32], full_out_weight: pl.Tensor[[16 * HIDDEN, FULL_Q_WIDTH], pl.BF16], cos: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], sin: pl.Tensor[[BATCH, ROWS, ROTARY], pl.BF16], mask: pl.Tensor[[BATCH, ROWS, CONTEXT], pl.FP32], token_counts: pl.Tensor[[BATCH], pl.INT32], cache_offsets: pl.Tensor[[BATCH], pl.INT32], conv_states: pl.InOut[pl.Tensor[[BATCH, 48, 4, QKV], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[BATCH, 48, DELTA_HEADS, DELTA_DIM, DELTA_DIM], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[16, 4, POOL_BLOCKS, BLOCK_SIZE, FULL_HEAD_DIM], pl.BF16]], block_tables: pl.Tensor[[BATCH, CONTEXT // BLOCK_SIZE], pl.INT32], traces: pl.Out[pl.Tensor[[BATCH, 66, ROWS, HIDDEN], pl.BF16]], logits: pl.Out[pl.Tensor[[BATCH, ROWS, VOCAB], pl.BF16]], active_count: pl.Scalar[pl.INT32]):
    rank_decode_fwd(input_ids, embedding, final_gamma, lm_head, input_gamma, post_gamma, mlp_gate_weight, mlp_up_weight, mlp_down_weight, delta_qkv_weight, delta_z_weight, delta_a_weight, delta_b_weight, delta_conv_weight, delta_a_log, delta_dt_bias, delta_norm_weight, delta_out_weight, full_q_weight, full_k_weight, full_v_weight, full_q_gamma, full_k_gamma, full_out_weight, cos, sin, mask, token_counts, cache_offsets, conv_states, recurrent_states, key_pool, value_pool, block_tables, traces, logits, active_count, device=0)
