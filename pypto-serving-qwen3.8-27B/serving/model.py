"""Qwen3.5 TP2 unified batched model and MTP production kernels.

Packed R token rows are independent of request batch B; persistent state uses request slots.
No runtime imports of reference repositories or host tensor implementations.
"""
import pypto.language as pl
import pypto.language.distributed as pld
HIDDEN = 5120
VOCAB = 248320
TP_VOCAB = 124160
CHUNK = 16
BLOCK_SIZE = 128
SLOTS = 32
MAX_LAUNCH_BLOCKS = 32767
MAX_MIXED_LAUNCH_BLOCKS = 64
B = pl.dynamic('QWEN_BATCH')
R = pl.dynamic('QWEN_ROWS')
L = pl.dynamic('QWEN_LOGIT_ROWS')
TEACHER_RETURN_R = pl.dynamic('QWEN_TEACHER_RETURN_ROWS_SOURCE2')
POOL = pl.dynamic('QWEN_POOL_BLOCKS')
TABLE = pl.dynamic('QWEN_TABLE_BLOCKS')
N_COPY = pl.dynamic('QWEN_COPY_PAGES')
SIG_ROWS = pl.dynamic('QWEN_SIG_ROWS')
SIG_COLS = pl.dynamic('QWEN_SIG_COLS')

@pl.jit.inline
def linear_partial_m16(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(w, 0)
    inner = pl.tensor.dim(w, 1)
    logical_tasks = rows // 16 * (width // 128)
    launch_blocks = pl.min(logical_tasks, MAX_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_batched_linear_m16'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = task // (width // 128) * 16
            col = task % (width // 128) * 128
            acc = pl.create_tensor([16, 128], dtype=pl.FP32)
            for k in pl.range(inner // 512):
                a = pl.slice(x, [16, 512], [row, k * 512])
                b = pl.slice(w, [128, 512], [col, k * 512])
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=k == 0)
            out[row:row + 16, col:col + 128] = acc
    return out


@pl.jit.inline
def linear_partial_m16_ffn_tail(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(w, 0)
    inner = pl.tensor.dim(w, 1)
    logical_tasks = (rows % 128) // 16 * (width // 128)
    launch_blocks = pl.min(logical_tasks, MAX_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_ffn_down_m16_tail'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = rows // 128 * 128 + task // (width // 128) * 16
            col = task % (width // 128) * 128
            acc = pl.create_tensor([16, 128], dtype=pl.FP32)
            for k in pl.range(inner // 512):
                a = pl.slice(x, [16, 512], [row, k * 512])
                b = pl.slice(w, [128, 512], [col, k * 512])
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=k == 0)
            out[row:row + 16, col:col + 128] = acc
    return out

@pl.jit.inline
def linear_partial_m32(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(w, 0)
    inner = pl.tensor.dim(w, 1)
    logical_tasks = rows // 32 * (width // 128)
    launch_blocks = pl.min(logical_tasks, MAX_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_batched_linear_m32'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = task // (width // 128) * 32
            col = task % (width // 128) * 128
            acc = pl.create_tensor([32, 128], dtype=pl.FP32)
            for k in pl.range(inner // 512):
                a = pl.slice(x, [32, 512], [row, k * 512])
                b = pl.slice(w, [128, 512], [col, k * 512])
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=k == 0)
            out[row:row + 32, col:col + 128] = acc
    return out

@pl.jit.inline
def linear_partial_m64(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(w, 0)
    inner = pl.tensor.dim(w, 1)
    logical_tasks = rows // 64 * (width // 128)
    launch_blocks = pl.min(logical_tasks, MAX_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_batched_linear_m64'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = task // (width // 128) * 64
            col = task % (width // 128) * 128
            acc = pl.create_tensor([64, 128], dtype=pl.FP32)
            for k in pl.range(inner // 512):
                a = pl.slice(x, [64, 512], [row, k * 512])
                b = pl.slice(w, [128, 512], [col, k * 512])
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=k == 0)
            out[row:row + 64, col:col + 128] = acc
    return out

@pl.jit.inline
def linear_partial_m128_ffn(x: pl.Tensor[[R, 8704], pl.BF16], w: pl.Tensor[[5120, 8704], pl.BF16], out: pl.Tensor[[R, 5120], pl.FP32]):
    rows = pl.tensor.dim(x, 0)
    logical_tasks = rows // 128 * 40
    launch_blocks = pl.min(logical_tasks, MAX_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_ffn_down_m128_fp32'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = task // 40 * 128
            col = task % 40 * 128
            acc = pl.create_tensor([128, 128], dtype=pl.FP32)
            for k in pl.pipeline(0, 17, stage=2):
                a = pl.slice(x, [128, 512], [row, k * 512])
                b = pl.slice(w, [128, 512], [col, k * 512])
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=k == 0)
            out[row:row + 128, col:col + 128] = acc
    return out

@pl.jit.inline
def linear_partial(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    if rows % 64 == 0:
        out = linear_partial_m64(x, w, out)
    elif rows % 32 == 0:
        out = linear_partial_m32(x, w, out)
    else:
        out = linear_partial_m16(x, w, out)
    return out

@pl.jit.inline
def linear_bf16_m16(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(w, 0)
    inner = pl.tensor.dim(w, 1)
    logical_tasks = rows // 16 * (width // 128)
    launch_blocks = pl.min(logical_tasks, MAX_MIXED_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_batched_linear_bf16_m16'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = task // (width // 128) * 16
            col = task % (width // 128) * 128
            acc = pl.create_tensor([16, 128], dtype=pl.FP32)
            for k in pl.range(inner // 512):
                a = pl.slice(x, [16, 512], [row, k * 512])
                b = pl.slice(w, [128, 512], [col, k * 512])
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=k == 0)
            out[row:row + 16, col:col + 128] = pl.cast(acc, pl.BF16, mode='rint')
    return out

@pl.jit.inline
def linear_bf16_m32(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(w, 0)
    inner = pl.tensor.dim(w, 1)
    logical_tasks = rows // 32 * (width // 128)
    launch_blocks = pl.min(logical_tasks, MAX_MIXED_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_batched_linear_bf16_m32'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = task // (width // 128) * 32
            col = task % (width // 128) * 128
            acc = pl.create_tensor([32, 128], dtype=pl.FP32)
            for k in pl.range(inner // 512):
                a = pl.slice(x, [32, 512], [row, k * 512])
                b = pl.slice(w, [128, 512], [col, k * 512])
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=k == 0)
            out[row:row + 32, col:col + 128] = pl.cast(acc, pl.BF16, mode='rint')
    return out

@pl.jit.inline
def linear_bf16_m64(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(w, 0)
    inner = pl.tensor.dim(w, 1)
    logical_tasks = rows // 64 * (width // 128)
    launch_blocks = pl.min(logical_tasks, MAX_MIXED_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_batched_linear_bf16_m64'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = task // (width // 128) * 64
            col = task % (width // 128) * 128
            acc = pl.create_tensor([64, 128], dtype=pl.FP32)
            for k in pl.range(inner // 512):
                a = pl.slice(x, [64, 512], [row, k * 512])
                b = pl.slice(w, [128, 512], [col, k * 512])
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=k == 0)
            out[row:row + 64, col:col + 128] = pl.cast(acc, pl.BF16, mode='rint')
    return out

@pl.jit.inline
def linear_bf16_m128_prefix(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(w, 0)
    inner = pl.tensor.dim(w, 1)
    logical_tasks = rows // 128 * (width // 128)
    launch_blocks = pl.min(logical_tasks, MAX_MIXED_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_linear_bf16_m128_prefix'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = task // (width // 128) * 128
            col = task % (width // 128) * 128
            acc = pl.create_tensor([128, 128], dtype=pl.FP32)
            for k in pl.range(inner // 512):
                a = pl.slice(x, [128, 512], [row, k * 512])
                b = pl.slice(w, [128, 512], [col, k * 512])
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=k == 0)
            for epilogue in pl.range(8):
                fragment = pl.slice(acc, [16, 128], [epilogue * 16, 0])
                out[row + epilogue * 16:row + epilogue * 16 + 16, col:col + 128] = pl.cast(fragment, pl.BF16, mode='rint')
    return out


@pl.jit.inline
def linear_bf16_m16_tail(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(w, 0)
    inner = pl.tensor.dim(w, 1)
    logical_tasks = (rows % 128) // 16 * (width // 128)
    launch_blocks = pl.min(logical_tasks, MAX_MIXED_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_linear_bf16_m16_tail'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = rows // 128 * 128 + task // (width // 128) * 16
            col = task % (width // 128) * 128
            acc = pl.create_tensor([16, 128], dtype=pl.FP32)
            for k in pl.range(inner // 512):
                a = pl.slice(x, [16, 512], [row, k * 512])
                b = pl.slice(w, [128, 512], [col, k * 512])
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=k == 0)
            out[row:row + 16, col:col + 128] = pl.cast(acc, pl.BF16, mode='rint')
    return out


@pl.jit.inline
def linear_bf16(x: pl.Tensor, w: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    if rows >= 2048:
        out = linear_bf16_m128_prefix(x, w, out)
        if rows % 128 != 0:
            out = linear_bf16_m16_tail(x, w, out)
    elif rows % 64 == 0:
        out = linear_bf16_m64(x, w, out)
    elif rows % 32 == 0:
        out = linear_bf16_m32(x, w, out)
    else:
        out = linear_bf16_m16(x, w, out)
    return out





@pl.jit.inline
def rms_norm(x: pl.Tensor, gamma: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    with pl.spmd(rows // 16, name_hint='qwen_rms128'):
        row = pl.tile.get_block_idx() * 16
        total = pl.full([1, 16], dtype=pl.FP32, value=0.0)
        for k in pl.range(width // 128):
            a = pl.cast(pl.slice(x, [16, 128], [row, k * 128]), pl.FP32)
            total = pl.add(total, pl.reshape(pl.row_sum(pl.mul(a, a)), [1, 16]))
        inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(total, 1.0 / 5120), 1e-06), high_precision=True), [16, 1])
        for k in pl.range(width // 128):
            a = pl.cast(pl.slice(x, [16, 128], [row, k * 128]), pl.FP32)
            g = pl.slice(gamma, [1, 128], [0, k * 128])
            out[row:row + 16, k * 128:k * 128 + 128] = pl.cast(pl.col_expand_mul(pl.row_expand_mul(a, inv), g), pl.BF16, mode='rint')
    return out

@pl.jit.inline
def residual_add(x: pl.Tensor, y: pl.Tensor, out: pl.InOut[pl.Tensor]):
    rows = pl.tensor.dim(x, 0)
    logical_tasks = rows // 16 * 10
    launch_blocks = pl.min(logical_tasks, 64)
    with pl.spmd(launch_blocks, name_hint='qwen_residual'):
        for t in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = t // 10 * 16
            col = t % 10 * 512
            a = pl.cast(pl.slice(x, [16, 512], [row, col]), pl.FP32)
            b = pl.cast(pl.slice(y, [16, 512], [row, col]), pl.FP32)
            out[row:row + 16, col:col + 512] = pl.cast(pl.add(a, b), pl.BF16, mode='rint')
    return out

@pl.jit.inline
def swiglu(gate: pl.Tensor, up: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(gate, 0)
    logical_tasks = rows // 16 * 68
    launch_blocks = pl.min(logical_tasks, MAX_LAUNCH_BLOCKS)
    with pl.spmd(launch_blocks, name_hint='qwen_swiglu'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, launch_blocks):
            row = task // 68 * 16
            col = task % 68 * 128
            g = pl.cast(pl.slice(gate, [16, 128], [row, col]), pl.FP32)
            u = pl.cast(pl.slice(up, [16, 128], [row, col]), pl.FP32)
            sigmoid_arg_1 = g
            sigmoid_value_1_exp = pl.exp(pl.neg(pl.abs(sigmoid_arg_1)))
            sigmoid_value_1_pos = pl.cmp(sigmoid_arg_1, 0.0, cmp_type=5)
            sigmoid_value_1 = pl.div(pl.sub(sigmoid_value_1_exp, pl.mul(sigmoid_value_1_pos, pl.sub(sigmoid_value_1_exp, 1.0))), pl.add(sigmoid_value_1_exp, 1.0), high_precision=True)
            act = pl.cast(pl.mul(g, sigmoid_value_1), pl.BF16, mode='rint')
            out[row:row + 16, col:col + 128] = pl.cast(pl.mul(pl.cast(act, pl.FP32), u), pl.BF16, mode='rint')
    return out

@pl.jit.inline(auto_scope=False)
def tp_sum(partial: pl.Tensor, window: pld.DistributedTensor[[16384, 5120], pl.FP32], signal: pld.DistributedTensor[[2, 1], pl.INT32], out: pl.Tensor, rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32]):
    rows = pl.tensor.dim(partial, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='tp_reuse', allow_early_resolve=False) as reuse:
        for peer in pl.range(2):
            pld.system.wait(signal, offsets=[peer, 0], expected=(epoch - 1) * 2, cmp=pld.WaitCmp.Ge)
    with pl.spmd(32, name_hint='tp_publish', deps=[reuse]) as publish:
        for tile in pl.range(pl.tile.get_block_idx(), rows * 10, 32):
            row = tile // 10
            col = tile % 10 * 512
            value = pl.load(partial, [row, col], [1, 512])
            pld.tile.remote_store(value, window, peer=rank, offsets=[row, col])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='tp_ready', deps=[publish]) as ready:
        for peer in pl.range(2):
            pld.system.notify(signal, peer=peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='tp_wait', deps=[ready], allow_early_resolve=False) as wait:
        for peer in pl.range(2):
            pld.system.wait(signal, offsets=[peer, 0], expected=epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(32, name_hint='tp_reduce', deps=[wait]) as reduce:
        for tile in pl.range(pl.tile.get_block_idx(), rows * 10, 32):
            row = tile // 10
            col = tile % 10 * 512
            accumulator = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
            for peer in pl.range(2):
                remote = pld.tile.remote_load(window, peer=peer, offsets=[row, col], shape=[1, 512])
                accumulator = pl.add(accumulator, remote)
            out = pl.store(pl.cast(accumulator, pl.BF16, mode='rint'), [row, col], out)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='tp_release', deps=[reduce]) as release:
        for peer in pl.range(2):
            pld.system.notify(signal, peer=peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    return out

@pl.jit.inline(auto_scope=False)
def tp_finish(signal: pld.DistributedTensor[[2, 1], pl.INT32], rank: pl.Scalar[pl.INT32], epochs: pl.Scalar[pl.INT32]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='tp_clear_credits', allow_early_resolve=False):
        for peer in pl.range(2):
            pld.system.wait(signal, offsets=[peer, 0], expected=epochs * 2, cmp=pld.WaitCmp.Ge)
        for peer in pl.range(2):
            pld.system.notify(signal, peer=rank, offsets=[peer, 0], value=0 - epochs * 2, op=pld.NotifyOp.AtomicAdd)

@pl.jit.inline
def zero_tail(out: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor):
    rows = pl.tensor.dim(out, 0)
    width = pl.tensor.dim(out, 1)
    batch = pl.tensor.dim(counts, 0)
    with pl.spmd(32, name_hint='qwen_zero_packed_tail'):
        total = pl.read(row_offsets, [batch - 1]) + pl.read(counts, [batch - 1])
        for task in pl.range(pl.tile.get_block_idx(), (rows - total) * (width // 128), 32):
            row = total + task // (width // 128)
            col = task % (width // 128) * 128
            out[row:row + 1, col:col + 128] = pl.full([1, 128], dtype=pl.BF16, value=0.0)
    return out

@pl.jit.inline
def embed(ids: pl.Tensor, embedding: pl.Tensor, row_sequences: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(ids, 0)
    with pl.spmd(rows, name_hint='qwen_embed'):
        row = pl.tile.get_block_idx()
        seq = pl.read(row_sequences, [row])
        if seq >= 0:
            token = pl.read(ids, [row])
            out[row:row + 1, 0:5120] = pl.slice(embedding, [1, 5120], [token, 0])
        else:
            out[row:row + 1, 0:5120] = pl.full([1, 5120], dtype=pl.BF16, value=0.0)
    return out

@pl.jit.inline
def conv_update(projected: pl.Tensor, weight: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor, slots: pl.Tensor, states: pl.Tensor, snapshots: pl.Tensor, out: pl.Tensor, layer: pl.Scalar[pl.INT32], snapshot_mode: pl.Scalar[pl.INT32], max_count: pl.Scalar[pl.INT32]):
    batch = pl.tensor.dim(counts, 0)
    sequence_groups = pl.min(batch, 3)
    with pl.spmd(sequence_groups * 20, name_hint='qwen_conv_prefix4') as prefix_task:
        worker = pl.tile.get_block_idx()
        col = worker // sequence_groups * 256
        sequence_lane = worker % sequence_groups
        w0 = pl.cast(pl.slice(weight, [1, 256], [0, col]), pl.FP32)
        w1 = pl.cast(pl.slice(weight, [1, 256], [1, col]), pl.FP32)
        w2 = pl.cast(pl.slice(weight, [1, 256], [2, col]), pl.FP32)
        w3 = pl.cast(pl.slice(weight, [1, 256], [3, col]), pl.FP32)
        for seq in pl.range(sequence_lane, batch, sequence_groups):
            count = pl.min(pl.read(counts, [seq]), 4)
            offset_row = pl.read(row_offsets, [seq])
            if count > 0:
                slot = pl.read(slots, [seq])
                s0 = pl.cast(pl.reshape(pl.slice(states, [1, 1, 1, 256], [slot, layer, 0, col]), [1, 256]), pl.FP32)
                s1 = pl.cast(pl.reshape(pl.slice(states, [1, 1, 1, 256], [slot, layer, 1, col]), [1, 256]), pl.FP32)
                s2 = pl.cast(pl.reshape(pl.slice(states, [1, 1, 1, 256], [slot, layer, 2, col]), [1, 256]), pl.FP32)
                s3 = pl.cast(pl.reshape(pl.slice(states, [1, 1, 1, 256], [slot, layer, 3, col]), [1, 256]), pl.FP32)
                for token in pl.range(count):
                    x = pl.cast(pl.slice(projected, [1, 256], [offset_row + token, col]), pl.FP32)
                    c = pl.add(pl.add(pl.mul(s1, w0), pl.mul(s2, w1)), pl.add(pl.mul(s3, w2), pl.mul(x, w3)))
                    c = pl.cast(pl.cast(c, pl.BF16, mode='rint'), pl.FP32)
                    sigmoid_arg_4 = c
                    sigmoid_value_4_exp = pl.exp(pl.neg(pl.abs(sigmoid_arg_4)))
                    sigmoid_value_4_pos = pl.cmp(sigmoid_arg_4, 0.0, cmp_type=5)
                    sigmoid_value_4 = pl.div(pl.sub(sigmoid_value_4_exp, pl.mul(sigmoid_value_4_pos, pl.sub(sigmoid_value_4_exp, 1.0))), pl.add(sigmoid_value_4_exp, 1.0), high_precision=True)
                    out[offset_row + token:offset_row + token + 1, col:col + 256] = pl.cast(pl.mul(c, sigmoid_value_4), pl.BF16, mode='rint')
                    s0 = s1
                    s1 = s2
                    s2 = s3
                    s3 = x
                    if snapshot_mode > 0 and token < 4:
                        snapshots[token:token + 1, slot:slot + 1, layer:layer + 1, 0:1, col:col + 256] = pl.reshape(pl.cast(s0, pl.BF16, mode='rint'), [1, 1, 1, 1, 256])
                        snapshots[token:token + 1, slot:slot + 1, layer:layer + 1, 1:2, col:col + 256] = pl.reshape(pl.cast(s1, pl.BF16, mode='rint'), [1, 1, 1, 1, 256])
                        snapshots[token:token + 1, slot:slot + 1, layer:layer + 1, 2:3, col:col + 256] = pl.reshape(pl.cast(s2, pl.BF16, mode='rint'), [1, 1, 1, 1, 256])
                        snapshots[token:token + 1, slot:slot + 1, layer:layer + 1, 3:4, col:col + 256] = pl.reshape(pl.cast(s3, pl.BF16, mode='rint'), [1, 1, 1, 1, 256])
                states[slot:slot + 1, layer:layer + 1, 0:1, col:col + 256] = pl.reshape(pl.cast(s0, pl.BF16, mode='rint'), [1, 1, 1, 256])
                states[slot:slot + 1, layer:layer + 1, 1:2, col:col + 256] = pl.reshape(pl.cast(s1, pl.BF16, mode='rint'), [1, 1, 1, 256])
                states[slot:slot + 1, layer:layer + 1, 2:3, col:col + 256] = pl.reshape(pl.cast(s2, pl.BF16, mode='rint'), [1, 1, 1, 256])
                states[slot:slot + 1, layer:layer + 1, 3:4, col:col + 256] = pl.reshape(pl.cast(s3, pl.BF16, mode='rint'), [1, 1, 1, 256])
    if max_count > 4:
        # Output t>=4 depends only on projected[t-3:t+1], not recurrent state.
        # Three disjoint time lanes per column partition; the first four rows
        # retain the original BF16/snapshot path above.
        with pl.spmd(60, name_hint='qwen_conv_prefill_time16', deps=[prefix_task]) as tail_task:
            time16_worker = pl.tile.get_block_idx()
            time16_col = time16_worker // 3 * 256
            time16_time_lane = time16_worker % 3
            prefill_w0 = pl.cast(pl.load(weight, [0, time16_col], [1, 256], target_memory=pl.MemorySpace.Vec), pl.FP32)
            prefill_w1 = pl.cast(pl.load(weight, [1, time16_col], [1, 256], target_memory=pl.MemorySpace.Vec), pl.FP32)
            prefill_w2 = pl.cast(pl.load(weight, [2, time16_col], [1, 256], target_memory=pl.MemorySpace.Vec), pl.FP32)
            prefill_w3 = pl.cast(pl.load(weight, [3, time16_col], [1, 256], target_memory=pl.MemorySpace.Vec), pl.FP32)
            for time16_seq in pl.range(batch):
                time16_count = pl.read(counts, [time16_seq])
                time16_offset_row = pl.read(row_offsets, [time16_seq])
                if time16_count > 4:
                    for time16_time_tile in pl.range(time16_time_lane, (time16_count - 4 + 15) // 16, 3):
                        time16_token = 4 + time16_time_tile * 16
                        time16_actual = pl.min(16, time16_count - time16_token)
                        x0 = pl.tile.fillpad_expand(pl.cast(pl.load(projected, [time16_offset_row + time16_token - 3, time16_col], [16, 256], valid_shape=[time16_actual, 256], target_memory=pl.MemorySpace.Vec), pl.FP32), [16, 256], pad_value=0)
                        x1 = pl.tile.fillpad_expand(pl.cast(pl.load(projected, [time16_offset_row + time16_token - 2, time16_col], [16, 256], valid_shape=[time16_actual, 256], target_memory=pl.MemorySpace.Vec), pl.FP32), [16, 256], pad_value=0)
                        x2 = pl.tile.fillpad_expand(pl.cast(pl.load(projected, [time16_offset_row + time16_token - 1, time16_col], [16, 256], valid_shape=[time16_actual, 256], target_memory=pl.MemorySpace.Vec), pl.FP32), [16, 256], pad_value=0)
                        x3 = pl.tile.fillpad_expand(pl.cast(pl.load(projected, [time16_offset_row + time16_token, time16_col], [16, 256], valid_shape=[time16_actual, 256], target_memory=pl.MemorySpace.Vec), pl.FP32), [16, 256], pad_value=0)
                        prefill_c = pl.add(pl.add(pl.tile.col_expand_mul(x0, prefill_w0), pl.tile.col_expand_mul(x1, prefill_w1)), pl.add(pl.tile.col_expand_mul(x2, prefill_w2), pl.tile.col_expand_mul(x3, prefill_w3)))
                        prefill_c = pl.cast(pl.cast(prefill_c, pl.BF16, mode='rint'), pl.FP32)
                        prefill_e = pl.exp(pl.neg(pl.abs(prefill_c)))
                        prefill_positive = pl.tile.select(pl.tile.cmps(prefill_c, 0.0, cmp_type=5), pl.tile.full([16, 256], dtype=pl.FP32, value=1.0), 0.0)
                        prefill_sigmoid = pl.div(pl.sub(prefill_e, pl.mul(prefill_positive, pl.sub(prefill_e, 1.0))), pl.add(prefill_e, 1.0), high_precision=True)
                        prefill_result = pl.set_validshape(pl.cast(pl.mul(prefill_c, prefill_sigmoid), pl.BF16, mode='rint'), time16_actual, 256)
                        pl.store(prefill_result, [time16_offset_row + time16_token, time16_col], out)
                    # Unique time-lane owner; final history is part of the same
                    # output-writing task, so all projected readers complete
                    # before the convolved consumer and next-layer reuse.
                    if time16_time_lane == 0:
                        time16_slot = pl.read(slots, [time16_seq])
                        time16_final_history = pl.slice(projected, [4, 256], [time16_offset_row + time16_count - 4, time16_col])
                        states[time16_slot:time16_slot + 1, layer:layer + 1, 0:4, time16_col:time16_col + 256] = pl.reshape(time16_final_history, [1, 1, 4, 256])
    out = zero_tail(out, counts, row_offsets)
    return (out, states, snapshots)

@pl.jit.inline
def delta_parameters(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt: pl.Tensor, g: pl.Tensor, beta: pl.Tensor):
    rows = pl.tensor.dim(a, 0)
    with pl.spmd(rows // 16, name_hint='qwen_gdn_parameters'):
        row = pl.tile.get_block_idx() * 16
        av = pl.cast(pl.slice(a, [16, 32], [row, 0]), pl.FP32)
        bv = pl.cast(pl.slice(b, [16, 32], [row, 0]), pl.FP32)
        al = pl.cast(pl.slice(a_log, [1, 32], [0, 0]), pl.FP32)
        bias = pl.cast(pl.slice(dt, [1, 32], [0, 0]), pl.FP32)
        x = pl.col_expand_add(av, bias)
        softplus = pl.add(pl.maximum(x, 0.0), pl.log(pl.add(pl.exp(pl.neg(pl.abs(x))), 1.0)))
        g[row:row + 16, 0:32] = pl.neg(pl.col_expand_mul(softplus, pl.exp(al)))
        e = pl.exp(pl.neg(pl.abs(bv)))
        pos = pl.cmp(bv, 0.0, cmp_type=5)
        value = pl.div(pl.sub(e, pl.mul(pos, pl.sub(e, 1.0))), pl.add(e, 1.0), high_precision=True)
        beta[row:row + 16, 0:32] = pl.cast(pl.cast(value, pl.BF16, mode='rint'), pl.FP32)
    return (g, beta)

@pl.jit.inline
def delta_normalize(conv: pl.Tensor, qn: pl.Tensor, kn: pl.Tensor):
    rows = pl.tensor.dim(conv, 0)
    logical_tasks = rows // 16 * 24
    physical_blocks = pl.min(logical_tasks, MAX_MIXED_LAUNCH_BLOCKS)
    with pl.spmd(physical_blocks, name_hint='qwen_delta_qk_norm'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, physical_blocks):
            row = task // 24 * 16
            head = task % 24
            q0 = pl.cast(pl.slice(conv, [16, 128], [row, head // 3 * 128]), pl.FP32)
            k0 = pl.cast(pl.slice(conv, [16, 128], [row, 1024 + head // 3 * 128]), pl.FP32)
            qs = pl.cast(pl.cast(pl.mul(q0, q0), pl.BF16, mode='rint'), pl.FP32)
            ks = pl.cast(pl.cast(pl.mul(k0, k0), pl.BF16, mode='rint'), pl.FP32)
            qsum = pl.cast(pl.cast(pl.reshape(pl.row_sum(qs), [1, 16]), pl.BF16, mode='rint'), pl.FP32)
            ksum = pl.cast(pl.cast(pl.reshape(pl.row_sum(ks), [1, 16]), pl.BF16, mode='rint'), pl.FP32)
            qeps = pl.cast(pl.cast(pl.add(qsum, 1e-06), pl.BF16, mode='rint'), pl.FP32)
            keps = pl.cast(pl.cast(pl.add(ksum, 1e-06), pl.BF16, mode='rint'), pl.FP32)
            qi = pl.reshape(pl.cast(pl.cast(pl.rsqrt(qeps, high_precision=True), pl.BF16, mode='rint'), pl.FP32), [16, 1])
            ki = pl.reshape(pl.cast(pl.cast(pl.rsqrt(keps, high_precision=True), pl.BF16, mode='rint'), pl.FP32), [16, 1])
            qn[row:row + 16, head * 128:head * 128 + 128] = pl.mul(pl.cast(pl.cast(pl.row_expand_mul(q0, qi), pl.BF16, mode='rint'), pl.FP32), 128 ** (-0.5))
            kn[row:row + 16, head * 128:head * 128 + 128] = pl.cast(pl.cast(pl.row_expand_mul(k0, ki), pl.BF16, mode='rint'), pl.FP32)
    return (qn, kn)

@pl.jit.inline
def delta_gate(core: pl.Tensor, z: pl.Tensor, norm: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(core, 0)
    logical_tasks = rows // 16 * 24
    physical_blocks = pl.min(logical_tasks, MAX_MIXED_LAUNCH_BLOCKS)
    with pl.spmd(physical_blocks, name_hint='qwen_delta_gate'):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, physical_blocks):
            row = task // 24 * 16
            col = task % 24 * 128
            x = pl.cast(pl.slice(core, [16, 128], [row, col]), pl.FP32)
            gate = pl.cast(pl.slice(z, [16, 128], [row, col]), pl.FP32)
            inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(pl.reshape(pl.row_sum(pl.mul(x, x)), [1, 16]), 1.0 / 128), 1e-06), high_precision=True), [16, 1])
            normalized = pl.cast(pl.row_expand_mul(x, inv), pl.BF16, mode='rint')
            weighted = pl.cast(pl.col_expand_mul(pl.cast(normalized, pl.FP32), pl.cast(pl.slice(norm, [1, 128], [0, 0]), pl.FP32)), pl.BF16, mode='rint')
            sigmoid_arg_2 = gate
            sigmoid_value_2_exp = pl.exp(pl.neg(pl.abs(sigmoid_arg_2)))
            sigmoid_value_2_pos = pl.cmp(sigmoid_arg_2, 0.0, cmp_type=5)
            sigmoid_value_2 = pl.div(pl.sub(sigmoid_value_2_exp, pl.mul(sigmoid_value_2_pos, pl.sub(sigmoid_value_2_exp, 1.0))), pl.add(sigmoid_value_2_exp, 1.0), high_precision=True)
            out[row:row + 16, col:col + 128] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), pl.mul(gate, sigmoid_value_2)), pl.BF16, mode='rint')
    return out

@pl.jit.inline
def norm_rope(raw: pl.Tensor[[16, 256], pl.BF16], gamma: pl.Tensor[[1, 256], pl.FP32], cos: pl.Tensor[[16, 64], pl.BF16], sin: pl.Tensor[[16, 64], pl.BF16]) -> pl.Tensor[[16, 256], pl.BF16]:
    a = pl.cast(raw, pl.FP32)
    first = pl.slice(a, [16, 128], [0, 0])
    second = pl.slice(a, [16, 128], [0, 128])
    total = pl.add(pl.reshape(pl.row_sum(pl.mul(first, first)), [1, 16]), pl.reshape(pl.row_sum(pl.mul(second, second)), [1, 16]))
    inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(total, 1.0 / 256), 1e-06), high_precision=True), [16, 1])
    normalized = pl.cast(pl.col_expand_mul(pl.row_expand_mul(a, inv), gamma), pl.BF16, mode='rint')
    x = pl.cast(pl.slice(normalized, [16, 32], [0, 0]), pl.FP32)
    y = pl.cast(pl.slice(normalized, [16, 32], [0, 32]), pl.FP32)
    c0 = pl.cast(pl.slice(cos, [16, 32], [0, 0]), pl.FP32)
    c1 = pl.cast(pl.slice(cos, [16, 32], [0, 32]), pl.FP32)
    s0 = pl.cast(pl.slice(sin, [16, 32], [0, 0]), pl.FP32)
    s1 = pl.cast(pl.slice(sin, [16, 32], [0, 32]), pl.FP32)
    xc = pl.cast(pl.cast(pl.mul(x, c0), pl.BF16, mode='rint'), pl.FP32)
    ys = pl.cast(pl.cast(pl.mul(y, s0), pl.BF16, mode='rint'), pl.FP32)
    yc = pl.cast(pl.cast(pl.mul(y, c1), pl.BF16, mode='rint'), pl.FP32)
    xs = pl.cast(pl.cast(pl.mul(x, s1), pl.BF16, mode='rint'), pl.FP32)
    r0 = pl.cast(pl.sub(xc, ys), pl.BF16, mode='rint')
    r1 = pl.cast(pl.add(yc, xs), pl.BF16, mode='rint')
    return pl.concat(pl.concat(r0, r1), pl.slice(normalized, [16, 192], [0, 64]))

@pl.jit.inline
def prepare_heads(qp: pl.Tensor, kp: pl.Tensor, qgamma: pl.Tensor, kgamma: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, q: pl.Tensor, k: pl.Tensor, gate: pl.Tensor):
    rows = pl.tensor.dim(qp, 0)
    with pl.spmd(rows // 16 * 12, name_hint='qwen_q_norm_rope'):
        task = pl.tile.get_block_idx()
        row = task // 12 * 16
        head = task % 12
        raw = pl.slice(qp, [16, 256], [row, head * 512])
        ct = pl.slice(cos, [16, 64], [row, 0])
        st = pl.slice(sin, [16, 64], [row, 0])
        qr = norm_rope(raw, qgamma, ct, st)
        q[row:row + 16, head * 256:head * 256 + 256] = qr
        gate[row:row + 16, head * 256:head * 256 + 256] = pl.slice(qp, [16, 256], [row, head * 512 + 256])
    with pl.spmd(rows // 16 * 2, name_hint='qwen_k_norm_rope'):
        task = pl.tile.get_block_idx()
        row = task // 2 * 16
        head = task % 2
        raw = pl.slice(kp, [16, 256], [row, head * 256])
        ct = pl.slice(cos, [16, 64], [row, 0])
        st = pl.slice(sin, [16, 64], [row, 0])
        kr = norm_rope(raw, kgamma, ct, st)
        k[row:row + 16, head * 256:head * 256 + 256] = kr
    return (q, k, gate)

@pl.jit.inline
def write_kv(k: pl.Tensor, v: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor, positions: pl.Tensor, slots: pl.Tensor, tables: pl.Tensor, kpool: pl.Tensor, vpool: pl.Tensor, layer: pl.Scalar[pl.INT32]):
    batch = pl.tensor.dim(counts, 0)
    with pl.spmd(batch * 2, name_hint='qwen_append_paged_kv'):
        task = pl.tile.get_block_idx()
        seq = task // 2
        head = task % 2
        count = pl.read(counts, [seq])
        offset_row = pl.read(row_offsets, [seq])
        if count > 0:
            slot = pl.read(slots, [seq])
            position = pl.read(positions, [seq])
            if layer == 16 and position == 1:
                first_page = pl.read(tables, [slot, 0])
                kpool[16:17, head:head + 1, first_page:first_page + 1, 0:1, 0:256] = pl.full([1, 1, 1, 1, 256], dtype=pl.BF16, value=0.0)
                vpool[16:17, head:head + 1, first_page:first_page + 1, 0:1, 0:256] = pl.full([1, 1, 1, 1, 256], dtype=pl.BF16, value=0.0)
            for token in pl.range(count):
                page = pl.read(tables, [slot, (position + token) // 128])
                offset = (position + token) % 128
                kt = pl.slice(k, [1, 256], [offset_row + token, head * 256])
                vt = pl.slice(v, [1, 256], [offset_row + token, head * 256])
                kpool[layer:layer + 1, head:head + 1, page:page + 1, offset:offset + 1, 0:256] = pl.reshape(kt, [1, 1, 1, 1, 256])
                vpool[layer:layer + 1, head:head + 1, page:page + 1, offset:offset + 1, 0:256] = pl.reshape(vt, [1, 1, 1, 1, 256])
    return (kpool, vpool)

@pl.jit.inline
def gqa_decode_pack(q: pl.Tensor, gate: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor, positions: pl.Tensor, slots: pl.Tensor, tables: pl.Tensor, kpool: pl.Tensor, vpool: pl.Tensor, qpacked: pl.Tensor, gpacked: pl.Tensor, ktail: pl.Tensor, vtail: pl.Tensor, layer: pl.Scalar[pl.INT32]):
    batch = pl.tensor.dim(counts, 0)
    tasks = batch * 2
    blocks = pl.min(tasks, MAX_MIXED_LAUNCH_BLOCKS)
    pool_blocks = pl.tensor.dim(kpool, 2)
    cache_rows = pl.tensor.dim(kpool, 0) * pl.tensor.dim(kpool, 1) * pool_blocks * 128
    kflat = pl.reshape(kpool, [cache_rows, 256])
    vflat = pl.reshape(vpool, [cache_rows, 256])
    with pl.spmd(blocks, name_hint='qwen_gqa_decode_pack'):
        for pack_task in pl.range(pl.tile.get_block_idx(), tasks, blocks):
            seq = pack_task // 2
            kvhead = pack_task % 2
            count = pl.read(counts, [seq])
            offset_row = pl.read(row_offsets, [seq])
            packed_row = pack_task * 32
            if count > 0:
                if count <= 4:
                    for token in pl.range(4):
                        qrow = pl.tile.full([1, 1536], dtype=pl.BF16, value=0.0)
                        grow = pl.tile.full([1, 1536], dtype=pl.BF16, value=0.0)
                        if token < count:
                            qrow = pl.load(q, [offset_row + token, kvhead * 1536], [1, 1536], target_memory=pl.MemorySpace.Vec)
                            grow = pl.load(gate, [offset_row + token, kvhead * 1536], [1, 1536], target_memory=pl.MemorySpace.Vec)
                        pl.store(pl.tile.reshape(qrow, [6, 256]), [packed_row + token * 6, 0], qpacked)
                        pl.store(pl.tile.reshape(grow, [6, 256]), [packed_row + token * 6, 0], gpacked)
                    padding = pl.tile.full([8, 256], dtype=pl.BF16, value=0.0)
                    pl.store(padding, [packed_row + 24, 0], qpacked)
                    pl.store(padding, [packed_row + 24, 0], gpacked)
                position = pl.read(positions, [seq])
                slot = pl.read(slots, [seq])
                last_block = (position + count - 1) // 128
                last_count = (position + count - 1) % 128 + 1
                if last_count < 128:
                    page = pl.read(tables, [slot, last_block])
                    cache_row = ((layer * 2 + kvhead) * pool_blocks + page) * 128
                    for slab in pl.range(8):
                        slab_valid = pl.max(0, pl.min(16, last_count - slab * 16))
                        if slab_valid > 0:
                            kl = pl.load(kflat, [cache_row + slab * 16, 0], [16, 256], valid_shape=[slab_valid, 256], target_memory=pl.MemorySpace.Vec)
                            vl = pl.load(vflat, [cache_row + slab * 16, 0], [16, 256], valid_shape=[slab_valid, 256], target_memory=pl.MemorySpace.Vec)
                            kr = pl.tile.fillpad_expand(kl, [16, 256], pad_value=0)
                            vr = pl.tile.fillpad_expand(vl, [16, 256], pad_value=0)
                            pl.store(kr, [pack_task * 128 + slab * 16, 0], ktail)
                            pl.store(vr, [pack_task * 128 + slab * 16, 0], vtail)
                        else:
                            kz = pl.tile.full([16, 256], dtype=pl.BF16, value=0.0)
                            vz = pl.tile.full([16, 256], dtype=pl.BF16, value=0.0)
                            pl.store(kz, [pack_task * 128 + slab * 16, 0], ktail)
                            pl.store(vz, [pack_task * 128 + slab * 16, 0], vtail)
    return qpacked, gpacked, ktail, vtail

@pl.jit.inline
def gqa_decode_m32(qpacked: pl.Tensor, gpacked: pl.Tensor, ktail: pl.Tensor, vtail: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor, positions: pl.Tensor, slots: pl.Tensor, tables: pl.Tensor, kpool: pl.Tensor, vpool: pl.Tensor, out: pl.Tensor, layer: pl.Scalar[pl.INT32], minimum_position: pl.Scalar[pl.INT32]):
    batch = pl.tensor.dim(counts, 0)
    tasks = batch * 2
    blocks = pl.min(tasks, MAX_MIXED_LAUNCH_BLOCKS)
    pool_blocks = pl.tensor.dim(kpool, 2)
    cache_rows = pl.tensor.dim(kpool, 0) * pl.tensor.dim(kpool, 1) * pool_blocks * 128
    kflat = pl.reshape(kpool, [cache_rows, 256])
    vflat = pl.reshape(vpool, [cache_rows, 256])
    with pl.spmd(blocks, name_hint='qwen_gqa_decode_m32', optimizations=[pl.cross_core_slot(slot_num=2)]):
        for scan_task in pl.range(pl.tile.get_block_idx(), tasks, blocks):
            seq = scan_task // 2
            kvhead = scan_task % 2
            count = pl.read(counts, [seq])
            if count > 0:
                if count <= 4:
                    slot = pl.read(slots, [seq])
                    position = pl.read(positions, [seq])
                    offset_row = pl.read(row_offsets, [seq])
                    packed_row = scan_task * 32
                    pages = (position + count + 127) // 128
                    qt = pl.load(qpacked, [packed_row, 0], [32, 256], target_memory=pl.MemorySpace.Mat)
                    maximum = pl.tile.full([1, 32], dtype=pl.FP32, value=-1e+30)
                    denominator = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
                    total = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                    lane_index = pl.tile.cast(pl.tile.arange(0, [1, 32], dtype=pl.INT32), pl.FP32)
                    token1 = pl.tile.select(pl.tile.cmps(lane_index, 6.0, cmp_type=5), pl.tile.full([1, 32], dtype=pl.FP32, value=1.0), 0.0)
                    token2 = pl.tile.select(pl.tile.cmps(lane_index, 12.0, cmp_type=5), pl.tile.full([1, 32], dtype=pl.FP32, value=1.0), 0.0)
                    token3 = pl.tile.select(pl.tile.cmps(lane_index, 18.0, cmp_type=5), pl.tile.full([1, 32], dtype=pl.FP32, value=1.0), 0.0)
                    lane_token = pl.tile.add(pl.tile.add(token1, token2), token3)
                    query_position_row = pl.tile.adds(lane_token, pl.cast(pl.cast(position, pl.INT32), pl.FP32))
                    query_positions = pl.tile.reshape(query_position_row, [32, 1])
                    query_valid = pl.tile.select(pl.tile.cmps(lane_index, pl.cast(pl.cast(count * 6, pl.INT32), pl.FP32), cmp_type=2), pl.tile.full([1, 32], dtype=pl.FP32, value=1.0), 0.0)
                    reduce_tmp = pl.create_tile([32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    for block in pl.range(pages):
                        page = pl.read(tables, [slot, block])
                        key_count = pl.min(128, position + count - block * 128)
                        cache_row = ((layer * 2 + kvhead) * pool_blocks + page) * 128
                        if key_count < 128:
                            kt = pl.load(ktail, [scan_task * 128, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                        else:
                            kt = pl.load(kflat, [cache_row, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                        raw_acc = pl.matmul(pl.tile.move(qt, target_memory=pl.MemorySpace.Left), pl.tile.move(pl.tile.transpose_view(kt), target_memory=pl.MemorySpace.Right), out_dtype=pl.FP32)
                        raw = pl.tile.cast(raw_acc, pl.BF16, mode='rint')
                        raw = pl.tile.fillpad_expand(raw, [32, 128], pad_value=0)
                        scaled = pl.tile.cast(pl.tile.mul(pl.tile.cast(raw, pl.FP32), 1.0 / 16), pl.BF16, mode='rint')
                        key_positions = pl.tile.cast(pl.tile.arange(pl.cast(block * 128, pl.INT32), [1, 128], dtype=pl.INT32), pl.FP32)
                        key_matrix = pl.tile.col_expand_mul(pl.tile.full([32, 128], dtype=pl.FP32, value=1.0), key_positions)
                        causal = pl.tile.select(pl.tile.cmps(pl.tile.row_expand_sub(key_matrix, query_positions), 0.0, cmp_type=3), pl.tile.full([32, 128], dtype=pl.FP32, value=1.0), 0.0)
                        minimum = pl.tile.select(pl.tile.cmps(key_positions, pl.cast(pl.cast(minimum_position, pl.INT32), pl.FP32), cmp_type=5), pl.tile.full([1, 128], dtype=pl.FP32, value=1.0), 0.0)
                        key_valid = pl.tile.select(pl.tile.cmps(key_positions, pl.cast(pl.cast(position + count - 1, pl.INT32), pl.FP32), cmp_type=3), pl.tile.full([1, 128], dtype=pl.FP32, value=1.0), 0.0)
                        valid = pl.tile.row_expand_mul(pl.tile.col_expand_mul(causal, pl.tile.mul(minimum, key_valid)), pl.tile.reshape(query_valid, [32, 1]))
                        scores = pl.tile.add(pl.tile.mul(pl.tile.cast(scaled, pl.FP32), valid), pl.tile.mul(pl.tile.sub(valid, 1.0), 1e+30))
                        next_maximum = pl.tile.maximum(maximum, pl.tile.reshape(pl.tile.row_max(scores, reduce_tmp), [1, 32]))
                        correction = pl.tile.exp(pl.tile.sub(maximum, next_maximum))
                        mass = pl.tile.mul(pl.tile.exp(pl.tile.row_expand_sub(scores, pl.tile.reshape(next_maximum, [32, 1]))), valid)
                        denominator = pl.tile.add(pl.tile.mul(denominator, correction), pl.tile.reshape(pl.tile.row_sum(mass, reduce_tmp), [1, 32]))
                        probability = pl.tile.cast(mass, pl.BF16, mode='rint')
                        if key_count < 128:
                            vt = pl.load(vtail, [scan_task * 128, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                        else:
                            vt = pl.load(vflat, [cache_row, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                        pv_acc = pl.matmul(pl.tile.move(probability, target_memory=pl.MemorySpace.Left), pl.tile.move(vt, target_memory=pl.MemorySpace.Right), out_dtype=pl.FP32)
                        total = pl.tile.add(pl.tile.row_expand_mul(total, pl.tile.reshape(correction, [32, 1])), pv_acc)
                        maximum = next_maximum
                    weighted = pl.tile.cast(pl.tile.row_expand_div(total, pl.tile.reshape(pl.tile.maximums(denominator, 1.0), [32, 1])), pl.BF16, mode='rint')
                    gpad = pl.load(gpacked, [packed_row, 0], [32, 256], target_memory=pl.MemorySpace.Vec)
                    g = pl.tile.cast(pl.tile.fillpad_expand(gpad, [32, 256], pad_value=0), pl.FP32)
                    sigmoid_value_exp = pl.tile.exp(pl.tile.neg(pl.tile.abs(g)))
                    sigmoid_value_pos = pl.tile.select(pl.tile.cmps(g, 0.0, cmp_type=5), pl.tile.full([32, 256], dtype=pl.FP32, value=1.0), 0.0)
                    sigmoid_value = pl.tile.div(pl.tile.sub(sigmoid_value_exp, pl.tile.mul(sigmoid_value_pos, pl.tile.sub(sigmoid_value_exp, 1.0))), pl.tile.add(sigmoid_value_exp, 1.0), high_precision=True)
                    sigmoid = pl.tile.cast(sigmoid_value, pl.BF16, mode='rint')
                    result = pl.tile.cast(pl.tile.mul(pl.tile.cast(weighted, pl.FP32), pl.tile.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint')
                    for token in pl.range(count):
                        for local_head in pl.range(6):
                            result_row = pl.tile.slice(result, [1, 256], [token * 6 + local_head, 0])
                            pl.store(result_row, [offset_row + token, (kvhead * 6 + local_head) * 256], out)
    return out

@pl.jit.inline
def paged_attention_k128_prefill(q: pl.Tensor, gate: pl.Tensor, ktail: pl.Tensor, vtail: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor, positions: pl.Tensor, slots: pl.Tensor, tables: pl.Tensor, kpool: pl.Tensor, vpool: pl.Tensor, out: pl.Tensor, layer: pl.Scalar[pl.INT32], minimum_position: pl.Scalar[pl.INT32]):
    batch = pl.tensor.dim(counts, 0)
    rows = pl.tensor.dim(q, 0)
    query_tiles_bound = (rows + 31) // 32 + batch - 1
    logical_tasks = query_tiles_bound * 12
    pool_blocks = pl.tensor.dim(kpool, 2)
    cache_rows = pl.tensor.dim(kpool, 0) * pl.tensor.dim(kpool, 1) * pool_blocks * 128
    kflat = pl.reshape(kpool, [cache_rows, 256])
    vflat = pl.reshape(vpool, [cache_rows, 256])
    physical_blocks = pl.min(logical_tasks, MAX_MIXED_LAUNCH_BLOCKS)
    with pl.spmd(physical_blocks, name_hint='qwen_paged_attention', optimizations=[pl.cross_core_slot(slot_num=2)]):
        for task in pl.range(pl.tile.get_block_idx(), logical_tasks, physical_blocks):
            tile_index = task // 12
            head = task % 12
            kvhead = head // 6
            tile_prefix = 0
            selected_seq = 0
            query_start = 0
            selected = 0
            for candidate_seq in pl.range(batch):
                candidate_count = pl.read(counts, [candidate_seq])
                next_prefix = tile_prefix + (candidate_count + 31) // 32
                if tile_index >= tile_prefix:
                    if tile_index < next_prefix:
                        selected_seq = candidate_seq
                        query_start = (tile_index - tile_prefix) * 32
                        selected = 1
                tile_prefix = next_prefix
            if selected > 0:
                if pl.read(counts, [selected_seq]) > 4:
                    seq = selected_seq
                    count = pl.read(counts, [seq])
                    slot = pl.read(slots, [seq])
                    position = pl.read(positions, [seq])
                    offset_row = pl.read(row_offsets, [seq])
                    query_count = pl.min(32, count - query_start)
                    pages = (position + query_start + query_count + 127) // 128
                    qt = pl.load(q, [offset_row + query_start, head * 256], [32, 256], valid_shape=[query_count, 256], target_memory=pl.MemorySpace.Mat)
                    maximum = pl.tile.full([1, 32], dtype=pl.FP32, value=-1e+30)
                    denominator = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
                    total = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
                    query_position_row = pl.tile.cast(pl.tile.arange(pl.cast(position + query_start, pl.INT32), [1, 32], dtype=pl.INT32), pl.FP32)
                    query_positions = pl.tile.reshape(query_position_row, [32, 1])
                    query_valid = pl.tile.select(pl.tile.cmps(query_position_row, pl.cast(pl.cast(position + query_start + query_count - 1, pl.INT32), pl.FP32), cmp_type=3), pl.tile.full([1, 32], dtype=pl.FP32, value=1.0), 0.0)
                    reduce_tmp = pl.create_tile([32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    for block in pl.range(pages):
                        page = pl.read(tables, [slot, block])
                        key_count = pl.min(128, position + query_start + query_count - block * 128)
                        cache_row = ((layer * 2 + kvhead) * pool_blocks + page) * 128
                        if block * 128 + 128 > position + count:
                            kt = pl.load(ktail, [(seq * 2 + kvhead) * 128, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                        else:
                            kt = pl.load(kflat, [cache_row, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                        raw_acc = pl.matmul(pl.tile.move(qt, target_memory=pl.MemorySpace.Left), pl.tile.move(pl.tile.transpose_view(kt), target_memory=pl.MemorySpace.Right), out_dtype=pl.FP32)
                        raw = pl.tile.cast(raw_acc, pl.BF16, mode='rint')
                        raw = pl.tile.fillpad_expand(raw, [32, 128], pad_value=0)
                        scaled = pl.tile.cast(pl.tile.mul(pl.tile.cast(raw, pl.FP32), 1.0 / 16), pl.BF16, mode='rint')
                        key_positions = pl.tile.cast(pl.tile.arange(pl.cast(block * 128, pl.INT32), [1, 128], dtype=pl.INT32), pl.FP32)
                        key_matrix = pl.tile.col_expand_mul(pl.tile.full([32, 128], dtype=pl.FP32, value=1.0), key_positions)
                        causal = pl.tile.select(pl.tile.cmps(pl.tile.row_expand_sub(key_matrix, query_positions), 0.0, cmp_type=3), pl.tile.full([32, 128], dtype=pl.FP32, value=1.0), 0.0)
                        minimum = pl.tile.select(pl.tile.cmps(key_positions, pl.cast(pl.cast(minimum_position, pl.INT32), pl.FP32), cmp_type=5), pl.tile.full([1, 128], dtype=pl.FP32, value=1.0), 0.0)
                        key_valid = pl.tile.select(pl.tile.cmps(key_positions, pl.cast(pl.cast(position + query_start + query_count - 1, pl.INT32), pl.FP32), cmp_type=3), pl.tile.full([1, 128], dtype=pl.FP32, value=1.0), 0.0)
                        valid = pl.tile.row_expand_mul(pl.tile.col_expand_mul(causal, pl.tile.mul(minimum, key_valid)), pl.tile.reshape(query_valid, [32, 1]))
                        scores = pl.tile.add(pl.tile.mul(pl.tile.cast(scaled, pl.FP32), valid), pl.tile.mul(pl.tile.sub(valid, 1.0), 1e+30))
                        next_maximum = pl.tile.maximum(maximum, pl.tile.reshape(pl.tile.row_max(scores, reduce_tmp), [1, 32]))
                        correction = pl.tile.exp(pl.tile.sub(maximum, next_maximum))
                        mass = pl.tile.mul(pl.tile.exp(pl.tile.row_expand_sub(scores, pl.tile.reshape(next_maximum, [32, 1]))), valid)
                        denominator = pl.tile.add(pl.tile.mul(denominator, correction), pl.tile.reshape(pl.tile.row_sum(mass, reduce_tmp), [1, 32]))
                        probability = pl.tile.cast(mass, pl.BF16, mode='rint')
                        if block * 128 + 128 > position + count:
                            vt = pl.load(vtail, [(seq * 2 + kvhead) * 128, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                        else:
                            vt = pl.load(vflat, [cache_row, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                        pv_acc = pl.matmul(pl.tile.move(probability, target_memory=pl.MemorySpace.Left), pl.tile.move(vt, target_memory=pl.MemorySpace.Right), out_dtype=pl.FP32)
                        total = pl.tile.add(pl.tile.row_expand_mul(total, pl.tile.reshape(correction, [32, 1])), pv_acc)
                        maximum = next_maximum
                    weighted = pl.tile.cast(pl.tile.row_expand_div(total, pl.tile.reshape(pl.tile.maximums(denominator, 1.0), [32, 1])), pl.BF16, mode='rint')
                    gpad = pl.load(gate, [offset_row + query_start, head * 256], [32, 256], valid_shape=[query_count, 256], target_memory=pl.MemorySpace.Vec)
                    g = pl.tile.cast(pl.tile.fillpad_expand(gpad, [32, 256], pad_value=0), pl.FP32)
                    sigmoid_value_exp = pl.tile.exp(pl.tile.neg(pl.tile.abs(g)))
                    sigmoid_value_pos = pl.tile.select(pl.tile.cmps(g, 0.0, cmp_type=5), pl.tile.full([32, 256], dtype=pl.FP32, value=1.0), 0.0)
                    sigmoid_value = pl.tile.div(pl.tile.sub(sigmoid_value_exp, pl.tile.mul(sigmoid_value_pos, pl.tile.sub(sigmoid_value_exp, 1.0))), pl.tile.add(sigmoid_value_exp, 1.0), high_precision=True)
                    sigmoid = pl.tile.cast(sigmoid_value, pl.BF16, mode='rint')
                    result = pl.tile.cast(pl.tile.mul(pl.tile.cast(weighted, pl.FP32), pl.tile.cast(sigmoid, pl.FP32)), pl.BF16, mode='rint')
                    result = pl.set_validshape(result, query_count, 256)
                    pl.store(result, [offset_row + query_start, head * 256], out)
    out = zero_tail(out, counts, row_offsets)
    return out

@pl.jit.inline
def paged_attention(q: pl.Tensor, gate: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor, positions: pl.Tensor, slots: pl.Tensor, tables: pl.Tensor, kpool: pl.Tensor, vpool: pl.Tensor, out: pl.Tensor, layer: pl.Scalar[pl.INT32], minimum_position: pl.Scalar[pl.INT32]):
    batch = pl.tensor.dim(counts, 0)
    qpacked = pl.create_tensor([batch * 64, 256], dtype=pl.BF16)
    gpacked = pl.create_tensor([batch * 64, 256], dtype=pl.BF16)
    ktail = pl.create_tensor([batch * 256, 256], dtype=pl.BF16)
    vtail = pl.create_tensor([batch * 256, 256], dtype=pl.BF16)
    gqa_decode_pack(q, gate, counts, row_offsets, positions, slots, tables, kpool, vpool, qpacked, gpacked, ktail, vtail, layer)
    out = gqa_decode_m32(qpacked, gpacked, ktail, vtail, counts, row_offsets, positions, slots, tables, kpool, vpool, out, layer, minimum_position)
    out = paged_attention_k128_prefill(q, gate, ktail, vtail, counts, row_offsets, positions, slots, tables, kpool, vpool, out, layer, minimum_position)
    return out

@pl.jit.inline
def full_attention(x: pl.Tensor, qw: pl.Tensor, kw: pl.Tensor, vw: pl.Tensor, qgamma: pl.Tensor, kgamma: pl.Tensor, ow: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor, positions: pl.Tensor, slots: pl.Tensor, tables: pl.Tensor, kpool: pl.Tensor, vpool: pl.Tensor, partial: pl.Tensor, layer: pl.Scalar[pl.INT32], minimum_position: pl.Scalar[pl.INT32]):
    rows = pl.tensor.dim(x, 0)
    qp = pl.create_tensor([rows, 6144], dtype=pl.BF16)
    kp = pl.create_tensor([rows, 512], dtype=pl.BF16)
    vp = pl.create_tensor([rows, 512], dtype=pl.BF16)
    qp = linear_bf16(x, qw, qp)
    kp = linear_bf16(x, kw, kp)
    vp = linear_bf16(x, vw, vp)
    q = pl.create_tensor([rows, 3072], dtype=pl.BF16)
    k = pl.create_tensor([rows, 512], dtype=pl.BF16)
    draft_attn_gate_ws = pl.create_tensor([rows, 3072], dtype=pl.BF16)
    q, k, draft_attn_gate_ws = prepare_heads(qp, kp, qgamma, kgamma, cos, sin, q, k, draft_attn_gate_ws)
    kpool, vpool = write_kv(k, vp, counts, row_offsets, positions, slots, tables, kpool, vpool, layer)
    attended = pl.create_tensor([rows, 3072], dtype=pl.BF16)
    attended = paged_attention(q, draft_attn_gate_ws, counts, row_offsets, positions, slots, tables, kpool, vpool, attended, layer, minimum_position)
    partial = linear_partial(attended, ow, partial)
    return (partial, kpool, vpool)

@pl.jit.inline(auto_scope=False)
def mlp_partial(x: pl.Tensor, gate_weight: pl.Tensor, up_weight: pl.Tensor, down_weight: pl.Tensor, partial: pl.Tensor):
    with pl.scope():
        rows = pl.tensor.dim(x, 0)
        draft_ffn_gate_ws = pl.create_tensor([rows, 8704], dtype=pl.BF16)
        draft_ffn_up_ws = pl.create_tensor([rows, 8704], dtype=pl.BF16)
        draft_ffn_middle_ws = pl.create_tensor([rows, 8704], dtype=pl.BF16)
        draft_ffn_gate_ws = linear_bf16(x, gate_weight, draft_ffn_gate_ws)
        draft_ffn_up_ws = linear_bf16(x, up_weight, draft_ffn_up_ws)
        draft_ffn_middle_ws = swiglu(draft_ffn_gate_ws, draft_ffn_up_ws, draft_ffn_middle_ws)
        linear_partial(draft_ffn_middle_ws, down_weight, partial)
    return partial

@pl.jit.inline
def compact_hidden(hidden: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor, out: pl.Tensor):
    rows = pl.tensor.dim(out, 0)
    batch = pl.tensor.dim(counts, 0)
    with pl.spmd(rows, name_hint='qwen_compact_logit_input'):
        row = pl.tile.get_block_idx()
        seq = row // 4
        token = row % 4
        if seq < batch:
            count = pl.read(counts, [seq])
            keep = pl.min(count, 4)
            offset = pl.read(row_offsets, [seq])
            if token < keep:
                source = offset + count - keep + token
                out[row:row + 1, 0:5120] = pl.slice(hidden, [1, 5120], [source, 0])
            else:
                out[row:row + 1, 0:5120] = pl.full([1, 5120], dtype=pl.BF16, value=0.0)
        else:
            out[row:row + 1, 0:5120] = pl.full([1, 5120], dtype=pl.BF16, value=0.0)
    return out

@pl.jit.inline(auto_scope=False)
def full_attention_scratch(x: pl.Tensor, qw: pl.Tensor, kw: pl.Tensor, vw: pl.Tensor, qgamma: pl.Tensor, kgamma: pl.Tensor, ow: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor, positions: pl.Tensor, slots: pl.Tensor, tables: pl.Tensor, kpool: pl.Tensor, vpool: pl.Tensor, fa_partial_out: pl.Tensor[[R, 5120], pl.FP32], layer: pl.Scalar[pl.INT32], minimum_position: pl.Scalar[pl.INT32], fa_projected_q_ws: pl.Tensor[[R, 6144], pl.BF16], fa_projected_k_ws: pl.Tensor[[R, 512], pl.BF16], fa_projected_v_ws: pl.Tensor[[R, 512], pl.BF16], fa_q_ws: pl.Tensor[[R, 3072], pl.BF16], fa_k_ws: pl.Tensor[[R, 512], pl.BF16], fa_gate_ws: pl.Tensor[[R, 3072], pl.BF16], fa_attended_ws: pl.Tensor[[R, 3072], pl.BF16]):
    rows = pl.tensor.dim(x, 0)
    linear_bf16(x, qw, fa_projected_q_ws)
    linear_bf16(x, kw, fa_projected_k_ws)
    linear_bf16(x, vw, fa_projected_v_ws)
    prepare_heads(fa_projected_q_ws, fa_projected_k_ws, qgamma, kgamma, cos, sin, fa_q_ws, fa_k_ws, fa_gate_ws)
    write_kv(fa_k_ws, fa_projected_v_ws, counts, row_offsets, positions, slots, tables, kpool, vpool, layer)
    paged_attention(fa_q_ws, fa_gate_ws, counts, row_offsets, positions, slots, tables, kpool, vpool, fa_attended_ws, layer, minimum_position)
    linear_partial(fa_attended_ws, ow, fa_partial_out)
    return (fa_partial_out, kpool, vpool)

@pl.jit.inline(auto_scope=False)
def ffn_gate_up_nz_m16(x: pl.Tensor, gate_w: pl.Tensor[[1, 8704, 5120], pl.BF16, pl.NZ], up_w: pl.Tensor[[1, 8704, 5120], pl.BF16, pl.NZ], middle: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    logical_tasks = rows // 16 * 68
    for worker in pl.spmd(64, name_hint='qwen_ffn_gate_up_nz_m16'):
        for task in pl.range(worker, logical_tasks, 64):
            row = task // 68 * 16
            col = task % 68 * 128
            gate_acc = pl.create_tensor([1, 16, 128], dtype=pl.FP32)
            up_acc = pl.create_tensor([1, 16, 128], dtype=pl.FP32)
            for kb in pl.pipeline(0, 20, stage=2):
                a = pl.slice(x, [16, 256], [row, kb * 256])
                gw = pl.slice(gate_w, [1, 128, 256], [0, col, kb * 256])
                uw = pl.slice(up_w, [1, 128, 256], [0, col, kb * 256])
                gate_acc = pl.matmul_acc(gate_acc, a, gw, b_trans=True, init_cond=kb == 0)
                up_acc = pl.matmul_acc(up_acc, a, uw, b_trans=True, init_cond=kb == 0)
            gate_rounded = pl.cast(pl.reshape(gate_acc, [16, 128]), pl.BF16, mode='rint')
            up_rounded = pl.cast(pl.reshape(up_acc, [16, 128]), pl.BF16, mode='rint')
            for epilogue in pl.range(16 // 16):
                g = pl.cast(pl.slice(gate_rounded, [16, 128], [epilogue * 16, 0]), pl.FP32)
                u = pl.cast(pl.slice(up_rounded, [16, 128], [epilogue * 16, 0]), pl.FP32)
                e = pl.exp(pl.neg(pl.abs(g)))
                positive = pl.cmp(g, 0.0, cmp_type=5)
                sigmoid = pl.div(pl.sub(e, pl.mul(positive, pl.sub(e, 1.0))), pl.add(e, 1.0), high_precision=True)
                act = pl.cast(pl.mul(g, sigmoid), pl.BF16, mode='rint')
                middle[row + epilogue * 16:row + epilogue * 16 + 16, col:col + 128] = pl.cast(pl.mul(pl.cast(act, pl.FP32), u), pl.BF16, mode='rint')
    return middle


@pl.jit.inline(auto_scope=False)
def ffn_gate_up_nz_m16_tail(x: pl.Tensor, gate_w: pl.Tensor[[1, 8704, 5120], pl.BF16, pl.NZ], up_w: pl.Tensor[[1, 8704, 5120], pl.BF16, pl.NZ], middle: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    logical_tasks = (rows % 128) // 16 * 68
    for worker in pl.spmd(64, name_hint='qwen_ffn_gate_up_nz_m16_tail'):
        for task in pl.range(worker, logical_tasks, 64):
            row = rows // 128 * 128 + task // 68 * 16
            col = task % 68 * 128
            gate_acc = pl.create_tensor([1, 16, 128], dtype=pl.FP32)
            up_acc = pl.create_tensor([1, 16, 128], dtype=pl.FP32)
            for kb in pl.pipeline(0, 20, stage=2):
                a = pl.slice(x, [16, 256], [row, kb * 256])
                gw = pl.slice(gate_w, [1, 128, 256], [0, col, kb * 256])
                uw = pl.slice(up_w, [1, 128, 256], [0, col, kb * 256])
                gate_acc = pl.matmul_acc(gate_acc, a, gw, b_trans=True, init_cond=kb == 0)
                up_acc = pl.matmul_acc(up_acc, a, uw, b_trans=True, init_cond=kb == 0)
            gate_rounded = pl.cast(pl.reshape(gate_acc, [16, 128]), pl.BF16, mode='rint')
            up_rounded = pl.cast(pl.reshape(up_acc, [16, 128]), pl.BF16, mode='rint')
            for epilogue in pl.range(16 // 16):
                g = pl.cast(pl.slice(gate_rounded, [16, 128], [epilogue * 16, 0]), pl.FP32)
                u = pl.cast(pl.slice(up_rounded, [16, 128], [epilogue * 16, 0]), pl.FP32)
                e = pl.exp(pl.neg(pl.abs(g)))
                positive = pl.cmp(g, 0.0, cmp_type=5)
                sigmoid = pl.div(pl.sub(e, pl.mul(positive, pl.sub(e, 1.0))), pl.add(e, 1.0), high_precision=True)
                act = pl.cast(pl.mul(g, sigmoid), pl.BF16, mode='rint')
                middle[row + epilogue * 16:row + epilogue * 16 + 16, col:col + 128] = pl.cast(pl.mul(pl.cast(act, pl.FP32), u), pl.BF16, mode='rint')
    return middle


@pl.jit.inline(auto_scope=False)
def ffn_gate_up_nz_m32(x: pl.Tensor, gate_w: pl.Tensor[[1, 8704, 5120], pl.BF16, pl.NZ], up_w: pl.Tensor[[1, 8704, 5120], pl.BF16, pl.NZ], middle: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    logical_tasks = rows // 32 * 68
    for worker in pl.spmd(64, name_hint='qwen_ffn_gate_up_nz_m32'):
        for task in pl.range(worker, logical_tasks, 64):
            row = task // 68 * 32
            col = task % 68 * 128
            gate_acc = pl.create_tensor([1, 32, 128], dtype=pl.FP32)
            up_acc = pl.create_tensor([1, 32, 128], dtype=pl.FP32)
            for kb in pl.pipeline(0, 20, stage=2):
                a = pl.slice(x, [32, 256], [row, kb * 256])
                gw = pl.slice(gate_w, [1, 128, 256], [0, col, kb * 256])
                uw = pl.slice(up_w, [1, 128, 256], [0, col, kb * 256])
                gate_acc = pl.matmul_acc(gate_acc, a, gw, b_trans=True, init_cond=kb == 0)
                up_acc = pl.matmul_acc(up_acc, a, uw, b_trans=True, init_cond=kb == 0)
            gate_rounded = pl.cast(pl.reshape(gate_acc, [32, 128]), pl.BF16, mode='rint')
            up_rounded = pl.cast(pl.reshape(up_acc, [32, 128]), pl.BF16, mode='rint')
            for epilogue in pl.range(32 // 16):
                g = pl.cast(pl.slice(gate_rounded, [16, 128], [epilogue * 16, 0]), pl.FP32)
                u = pl.cast(pl.slice(up_rounded, [16, 128], [epilogue * 16, 0]), pl.FP32)
                e = pl.exp(pl.neg(pl.abs(g)))
                positive = pl.cmp(g, 0.0, cmp_type=5)
                sigmoid = pl.div(pl.sub(e, pl.mul(positive, pl.sub(e, 1.0))), pl.add(e, 1.0), high_precision=True)
                act = pl.cast(pl.mul(g, sigmoid), pl.BF16, mode='rint')
                middle[row + epilogue * 16:row + epilogue * 16 + 16, col:col + 128] = pl.cast(pl.mul(pl.cast(act, pl.FP32), u), pl.BF16, mode='rint')
    return middle


@pl.jit.inline(auto_scope=False)
def ffn_gate_up_nz_m64(x: pl.Tensor, gate_w: pl.Tensor[[1, 8704, 5120], pl.BF16, pl.NZ], up_w: pl.Tensor[[1, 8704, 5120], pl.BF16, pl.NZ], middle: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    logical_tasks = rows // 64 * 68
    for worker in pl.spmd(64, name_hint='qwen_ffn_gate_up_nz_m64'):
        for task in pl.range(worker, logical_tasks, 64):
            row = task // 68 * 64
            col = task % 68 * 128
            gate_acc = pl.create_tensor([1, 64, 128], dtype=pl.FP32)
            up_acc = pl.create_tensor([1, 64, 128], dtype=pl.FP32)
            for kb in pl.pipeline(0, 20, stage=2):
                a = pl.slice(x, [64, 256], [row, kb * 256])
                gw = pl.slice(gate_w, [1, 128, 256], [0, col, kb * 256])
                uw = pl.slice(up_w, [1, 128, 256], [0, col, kb * 256])
                gate_acc = pl.matmul_acc(gate_acc, a, gw, b_trans=True, init_cond=kb == 0)
                up_acc = pl.matmul_acc(up_acc, a, uw, b_trans=True, init_cond=kb == 0)
            gate_rounded = pl.cast(pl.reshape(gate_acc, [64, 128]), pl.BF16, mode='rint')
            up_rounded = pl.cast(pl.reshape(up_acc, [64, 128]), pl.BF16, mode='rint')
            for epilogue in pl.range(64 // 16):
                g = pl.cast(pl.slice(gate_rounded, [16, 128], [epilogue * 16, 0]), pl.FP32)
                u = pl.cast(pl.slice(up_rounded, [16, 128], [epilogue * 16, 0]), pl.FP32)
                e = pl.exp(pl.neg(pl.abs(g)))
                positive = pl.cmp(g, 0.0, cmp_type=5)
                sigmoid = pl.div(pl.sub(e, pl.mul(positive, pl.sub(e, 1.0))), pl.add(e, 1.0), high_precision=True)
                act = pl.cast(pl.mul(g, sigmoid), pl.BF16, mode='rint')
                middle[row + epilogue * 16:row + epilogue * 16 + 16, col:col + 128] = pl.cast(pl.mul(pl.cast(act, pl.FP32), u), pl.BF16, mode='rint')
    return middle


@pl.jit.inline(auto_scope=False)
def ffn_gate_up_nz_m128(x: pl.Tensor, gate_w: pl.Tensor[[1, 8704, 5120], pl.BF16, pl.NZ], up_w: pl.Tensor[[1, 8704, 5120], pl.BF16, pl.NZ], middle: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    logical_tasks = rows // 128 * 68
    for worker in pl.spmd(64, name_hint='qwen_ffn_gate_up_nz_m128'):
        for task in pl.range(worker, logical_tasks, 64):
            row = task // 68 * 128
            col = task % 68 * 128
            gate_acc = pl.create_tensor([1, 128, 128], dtype=pl.FP32)
            up_acc = pl.create_tensor([1, 128, 128], dtype=pl.FP32)
            for kb in pl.pipeline(0, 20, stage=2):
                a = pl.slice(x, [128, 256], [row, kb * 256])
                gw = pl.slice(gate_w, [1, 128, 256], [0, col, kb * 256])
                uw = pl.slice(up_w, [1, 128, 256], [0, col, kb * 256])
                gate_acc = pl.matmul_acc(gate_acc, a, gw, b_trans=True, init_cond=kb == 0)
                up_acc = pl.matmul_acc(up_acc, a, uw, b_trans=True, init_cond=kb == 0)
            # Preserve each original FP32 -> BF16 -> FP32 rounding per element.
            # Slice the 128-row accumulator before conversion; keep Vec epilogues M16.
            for epilogue in pl.range(128 // 16):
                gate_fragment = pl.slice(pl.reshape(gate_acc, [128, 128]), [16, 128], [epilogue * 16, 0])
                up_fragment = pl.slice(pl.reshape(up_acc, [128, 128]), [16, 128], [epilogue * 16, 0])
                gate_rounded = pl.cast(gate_fragment, pl.BF16, mode='rint')
                up_rounded = pl.cast(up_fragment, pl.BF16, mode='rint')
                g = pl.cast(gate_rounded, pl.FP32)
                u = pl.cast(up_rounded, pl.FP32)
                e = pl.exp(pl.neg(pl.abs(g)))
                positive = pl.cmp(g, 0.0, cmp_type=5)
                sigmoid = pl.div(pl.sub(e, pl.mul(positive, pl.sub(e, 1.0))), pl.add(e, 1.0), high_precision=True)
                act = pl.cast(pl.mul(g, sigmoid), pl.BF16, mode='rint')
                middle[row + epilogue * 16:row + epilogue * 16 + 16, col:col + 128] = pl.cast(pl.mul(pl.cast(act, pl.FP32), u), pl.BF16, mode='rint')
    return middle


@pl.jit.inline(auto_scope=False)
def ffn_gate_up_nz(x: pl.Tensor, gate_w: pl.Tensor, up_w: pl.Tensor, middle: pl.Tensor):
    rows = pl.tensor.dim(x, 0)
    if rows >= 2048:
        middle = ffn_gate_up_nz_m128(x, gate_w, up_w, middle)
        if rows % 128 != 0:
            middle = ffn_gate_up_nz_m16_tail(x, gate_w, up_w, middle)
    elif rows % 128 == 0:
        middle = ffn_gate_up_nz_m128(x, gate_w, up_w, middle)
    elif rows % 64 == 0:
        middle = ffn_gate_up_nz_m64(x, gate_w, up_w, middle)
    elif rows % 32 == 0:
        middle = ffn_gate_up_nz_m32(x, gate_w, up_w, middle)
    else:
        middle = ffn_gate_up_nz_m16(x, gate_w, up_w, middle)
    return middle


@pl.jit.inline(auto_scope=False)
def mlp_partial_scratch(x: pl.Tensor, gate_weight: pl.Tensor, up_weight: pl.Tensor, down_weight: pl.Tensor, ffn_partial_out: pl.Tensor[[R, 5120], pl.FP32], ffn_middle_ws: pl.Tensor[[R, 8704], pl.BF16]):
    with pl.scope():
        ffn_gate_up_nz(x, gate_weight, up_weight, ffn_middle_ws)
        rows = pl.tensor.dim(ffn_middle_ws, 0)
        if rows >= 2048:
            linear_partial_m128_ffn(ffn_middle_ws, down_weight, ffn_partial_out)
            if rows % 128 != 0:
                linear_partial_m16_ffn_tail(ffn_middle_ws, down_weight, ffn_partial_out)
        else:
            linear_partial(ffn_middle_ws, down_weight, ffn_partial_out)
    return ffn_partial_out

@pl.jit.inline
def make_c16_constants(max_count: pl.Scalar[pl.INT32]):
    inclusive = pl.create_tensor([16, 16], dtype=pl.FP32)
    strict = pl.create_tensor([16, 16], dtype=pl.FP32)
    identity = pl.create_tensor([16, 128], dtype=pl.FP32)
    onehot = pl.create_tensor([24, 32], dtype=pl.FP32)
    if max_count > 4:
        with pl.spmd(32, name_hint='gdn_c16_constants'):
            c16_kernel_1_row = pl.tile.get_block_idx()
            if c16_kernel_1_row < 16:
                for c16_kernel_1_col in pl.range(128):
                    c16_kernel_1_diagonal = 0.0
                    if c16_kernel_1_col == c16_kernel_1_row:
                        c16_kernel_1_diagonal = 1.0
                    pl.write(identity, [c16_kernel_1_row, c16_kernel_1_col], c16_kernel_1_diagonal)
                    if c16_kernel_1_col < 16:
                        c16_kernel_1_lower = 0.0
                        c16_kernel_1_below = 0.0
                        if c16_kernel_1_col <= c16_kernel_1_row:
                            c16_kernel_1_lower = 1.0
                        if c16_kernel_1_col < c16_kernel_1_row:
                            c16_kernel_1_below = 1.0
                        pl.write(inclusive, [c16_kernel_1_row, c16_kernel_1_col], c16_kernel_1_lower)
                        pl.write(strict, [c16_kernel_1_row, c16_kernel_1_col], c16_kernel_1_below)
            if c16_kernel_1_row < 24:
                for c16_kernel_1_col in pl.range(32):
                    c16_kernel_1_selected = 0.0
                    if c16_kernel_1_col == c16_kernel_1_row:
                        c16_kernel_1_selected = 1.0
                    pl.write(onehot, [c16_kernel_1_row, c16_kernel_1_col], c16_kernel_1_selected)
    return (inclusive, strict, identity, onehot)

@pl.jit.inline
def delta_core_c16(conv: pl.Tensor, qn: pl.Tensor, kn: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor, counts: pl.Tensor, row_offsets: pl.Tensor, slots: pl.Tensor, states: pl.Tensor, snapshots: pl.Tensor, out: pl.Tensor, layer: pl.Scalar[pl.INT32], snapshot_mode: pl.Scalar[pl.INT32], max_count: pl.Scalar[pl.INT32], inclusive: pl.Tensor, strict: pl.Tensor, identity: pl.Tensor, onehot: pl.Tensor, offsets: pl.Tensor, query_gm: pl.Tensor, key_gm: pl.Tensor, value_gm: pl.Tensor, gate_gm: pl.Tensor, beta_gm: pl.Tensor, prefix_gm: pl.Tensor, gram_gm: pl.Tensor, qk_gm: pl.Tensor, pair_gm: pl.Tensor, negative_gm: pl.Tensor, weighted_key_gm: pl.Tensor, weighted_value_gm: pl.Tensor, factor_gm: pl.Tensor, product_gm: pl.Tensor, query_ready: pl.Tensor, key_ready: pl.Tensor, transform_key_ready: pl.Tensor, transform_value_ready: pl.Tensor, causal_ready: pl.Tensor, decay_ready: pl.Tensor, carry_cube: pl.Tensor, carry_vector: pl.Tensor):
    batch = pl.tensor.dim(counts, 0)
    rows = pl.tensor.dim(out, 0)
    snapshot_rows = pl.reshape(snapshots, [4 * 32 * 48 * 24 * 128, 128])
    c16_state_rows = pl.reshape(states, [32 * 48 * 24 * 128, 128])
    logical = batch * 48
    physical = pl.min(logical, 64)
    with pl.spmd(physical, name_hint='gdn_c16_serial_prefix') as serial_task:
        for c16_kernel_1_task in pl.range(pl.tile.get_block_idx(), logical, physical):
            c16_kernel_1_sequence = c16_kernel_1_task // 48
            c16_kernel_1_head = c16_kernel_1_task % 48 // 2
            c16_kernel_1_value_start = c16_kernel_1_task % 2 * 64
            c16_kernel_1_count = pl.cast(pl.read(counts, [c16_kernel_1_sequence]), pl.INDEX)
            c16_kernel_1_limit = c16_kernel_1_count
            if c16_kernel_1_count > 4:
                c16_kernel_1_limit = 0
                if snapshot_mode > 0:
                    c16_kernel_1_limit = 4
            if c16_kernel_1_limit > 0:
                c16_kernel_1_slot = pl.cast(pl.read(slots, [c16_kernel_1_sequence]), pl.INDEX)
                c16_kernel_1_token_start = pl.cast(pl.read(row_offsets, [c16_kernel_1_sequence]), pl.INDEX)
                c16_kernel_1_carried = pl.reshape(pl.slice(states, [1, 1, 1, 64, 128], [c16_kernel_1_slot, layer, c16_kernel_1_head, c16_kernel_1_value_start, 0]), [64, 128])
                for c16_kernel_1_token in pl.range(c16_kernel_1_limit):
                    c16_kernel_1_row = c16_kernel_1_token_start + c16_kernel_1_token
                    c16_kernel_1_query = pl.slice(qn, [1, 128], [c16_kernel_1_row, c16_kernel_1_head * 128])
                    c16_kernel_1_key = pl.slice(kn, [1, 128], [c16_kernel_1_row, c16_kernel_1_head * 128])
                    c16_kernel_1_gate = pl.read(log_decay, [c16_kernel_1_row, c16_kernel_1_head])
                    c16_kernel_1_beta_value = pl.read(beta, [c16_kernel_1_row, c16_kernel_1_head])
                    c16_kernel_1_decay = pl.exp(pl.mul(pl.full([1, 128], dtype=pl.FP32, value=1.0), c16_kernel_1_gate))
                    c16_kernel_1_decayed = pl.col_expand_mul(c16_kernel_1_carried, c16_kernel_1_decay)
                    c16_kernel_1_remembered = pl.row_sum(pl.col_expand_mul(c16_kernel_1_decayed, c16_kernel_1_key))
                    c16_kernel_1_value = pl.cast(pl.slice(conv, [1, 64], [c16_kernel_1_row, 2048 + c16_kernel_1_head * 128 + c16_kernel_1_value_start]), pl.FP32)
                    c16_kernel_1_delta = pl.mul(pl.sub(pl.reshape(c16_kernel_1_value, [64, 1]), c16_kernel_1_remembered), c16_kernel_1_beta_value)
                    c16_kernel_1_carried = pl.add(c16_kernel_1_decayed, pl.col_expand_mul(pl.row_expand_mul(pl.full([64, 128], dtype=pl.FP32, value=1.0), c16_kernel_1_delta), c16_kernel_1_key))
                    c16_kernel_1_result = pl.reshape(pl.row_sum(pl.col_expand_mul(c16_kernel_1_carried, c16_kernel_1_query)), [1, 64])
                    out[c16_kernel_1_row:c16_kernel_1_row + 1, c16_kernel_1_head * 128 + c16_kernel_1_value_start:c16_kernel_1_head * 128 + c16_kernel_1_value_start + 64] = pl.cast(c16_kernel_1_result, pl.BF16, mode='rint')
                    if snapshot_mode > 0 and c16_kernel_1_token < 4:
                        c16_kernel_1_snap_start = ((c16_kernel_1_token * 32 + c16_kernel_1_slot) * 48 + layer) * 3072 + c16_kernel_1_head * 128 + c16_kernel_1_value_start
                        snapshot_rows[c16_kernel_1_snap_start:c16_kernel_1_snap_start + 64, 0:128] = c16_kernel_1_carried
                states[c16_kernel_1_slot:c16_kernel_1_slot + 1, layer:layer + 1, c16_kernel_1_head:c16_kernel_1_head + 1, c16_kernel_1_value_start:c16_kernel_1_value_start + 64, 0:128] = pl.reshape(c16_kernel_1_carried, [1, 1, 1, 64, 128])
    if max_count > 4:
        capacity = rows + batch * 16
        chunk_capacity = capacity // 16
        with pl.spmd(1, name_hint='gdn_c16_scratch_offsets', deps=[serial_task]) as offsets_task:
            c16_kernel_2_block = pl.tile.get_block_idx()
            c16_kernel_2_total = c16_kernel_2_block * 0
            for c16_kernel_2_sequence in pl.range(batch):
                pl.write(offsets, [c16_kernel_2_sequence], pl.cast(c16_kernel_2_total, pl.INT32))
                c16_kernel_2_count = pl.cast(pl.read(counts, [c16_kernel_2_sequence]), pl.INDEX)
                c16_kernel_2_skip = 0
                if snapshot_mode > 0:
                    c16_kernel_2_skip = pl.min(c16_kernel_2_count, 4)
                if c16_kernel_2_count > 4:
                    c16_kernel_2_total = c16_kernel_2_total + (c16_kernel_2_count - c16_kernel_2_skip + 15) // 16 * 16
            pl.write(offsets, [batch], pl.cast(c16_kernel_2_total, pl.INT32))
        dependency = offsets_task
        for head_base in pl.range(0, 24, 2):
            for local_head in pl.range(2):
                head = head_base + local_head
                with pl.spmd(64, name_hint='gdn_c16_pack_all', deps=[dependency]) as pack_task:
                    c16_kernel_3_block = pl.tile.get_block_idx()
                    for c16_kernel_3_sequence in pl.range(batch):
                        c16_kernel_3_count = pl.cast(pl.read(counts, [c16_kernel_3_sequence]), pl.INDEX)
                        c16_kernel_3_skip = 0
                        if snapshot_mode > 0:
                            c16_kernel_3_skip = pl.min(c16_kernel_3_count, 4)
                        if c16_kernel_3_count > 4:
                            c16_kernel_3_origin = pl.cast(pl.read(offsets, [c16_kernel_3_sequence]), pl.INDEX)
                            c16_kernel_3_token_origin = pl.cast(pl.read(row_offsets, [c16_kernel_3_sequence]), pl.INDEX) + c16_kernel_3_skip
                            c16_kernel_3_chunks = (c16_kernel_3_count - c16_kernel_3_skip + 15) // 16
                            for c16_kernel_3_chunk in pl.range(c16_kernel_3_block, c16_kernel_3_chunks, 64):
                                c16_kernel_3_row = c16_kernel_3_origin + c16_kernel_3_chunk * 16
                                c16_kernel_3_token = c16_kernel_3_token_origin + c16_kernel_3_chunk * 16
                                c16_kernel_3_actual = pl.min(16, c16_kernel_3_count - c16_kernel_3_skip - c16_kernel_3_chunk * 16)
                                query_gm[c16_kernel_3_row:c16_kernel_3_row + 16, 0:128] = pl.set_validshape(pl.fillpad(pl.slice(qn, [16, 128], [c16_kernel_3_token, head * 128], valid_shape=[c16_kernel_3_actual, 128]), pad_value=pl.PadValue.zero), 16, 128)
                                key_gm[c16_kernel_3_row:c16_kernel_3_row + 16, 0:128] = pl.set_validshape(pl.fillpad(pl.slice(kn, [16, 128], [c16_kernel_3_token, head * 128], valid_shape=[c16_kernel_3_actual, 128]), pad_value=pl.PadValue.zero), 16, 128)
                                value_gm[c16_kernel_3_row:c16_kernel_3_row + 16, 0:128] = pl.cast(pl.set_validshape(pl.fillpad(pl.slice(conv, [16, 128], [c16_kernel_3_token, 2048 + head * 128], valid_shape=[c16_kernel_3_actual, 128]), pad_value=pl.PadValue.zero), 16, 128), pl.FP32)
                                c16_kernel_3_gates = pl.set_validshape(pl.fillpad(pl.slice(log_decay, [16, 32], [c16_kernel_3_token, 0], valid_shape=[c16_kernel_3_actual, 32]), pad_value=pl.PadValue.zero), 16, 32)
                                c16_kernel_3_betas = pl.set_validshape(pl.fillpad(pl.slice(beta, [16, 32], [c16_kernel_3_token, 0], valid_shape=[c16_kernel_3_actual, 32]), pad_value=pl.PadValue.zero), 16, 32)
                                c16_kernel_3_gate_column = pl.row_sum(pl.col_expand_mul(c16_kernel_3_gates, onehot[head:head + 1, 0:32]))
                                c16_kernel_3_beta_column = pl.row_sum(pl.col_expand_mul(c16_kernel_3_betas, onehot[head:head + 1, 0:32]))
                                gate_gm[c16_kernel_3_row:c16_kernel_3_row + 16, 0:128] = pl.row_expand_mul(pl.full([16, 128], dtype=pl.FP32, value=1.0), c16_kernel_3_gate_column)
                                beta_gm[c16_kernel_3_row:c16_kernel_3_row + 16, 0:128] = pl.row_expand_mul(pl.full([16, 128], dtype=pl.FP32, value=1.0), c16_kernel_3_beta_column)
                with pl.spmd(64, name_hint='gdn_c16_gram_prefix', deps=[pack_task]) as gram_task:
                    c16_kernel_4_block = pl.tile.get_block_idx()
                    c16_kernel_4_used_chunks = pl.cast(pl.read(offsets, [batch]), pl.INDEX) // 16
                    for c16_kernel_4_chunk in pl.range(c16_kernel_4_block, c16_kernel_4_used_chunks, 64):
                        c16_kernel_4_row = c16_kernel_4_chunk * 16
                        prefix_gm[c16_kernel_4_row:c16_kernel_4_row + 16, 0:128] = pl.matmul(inclusive[0:16, 0:16], gate_gm[c16_kernel_4_row:c16_kernel_4_row + 16, 0:128], pl.FP32)
                        gram_gm[c16_kernel_4_row:c16_kernel_4_row + 16, 0:16] = pl.matmul(key_gm[c16_kernel_4_row:c16_kernel_4_row + 16, 0:128], key_gm[c16_kernel_4_row:c16_kernel_4_row + 16, 0:128], pl.FP32, b_trans=True)
                        qk_gm[c16_kernel_4_row:c16_kernel_4_row + 16, 0:16] = pl.matmul(query_gm[c16_kernel_4_row:c16_kernel_4_row + 16, 0:128], key_gm[c16_kernel_4_row:c16_kernel_4_row + 16, 0:128], pl.FP32, b_trans=True)
                with pl.spmd(64, name_hint='gdn_c16_masks_weights', deps=[gram_task]) as mask_task:
                    c16_kernel_5_block = pl.tile.get_block_idx()
                    c16_kernel_5_used_chunks = pl.cast(pl.read(offsets, [batch]), pl.INDEX) // 16
                    for c16_kernel_5_chunk in pl.range(c16_kernel_5_block, c16_kernel_5_used_chunks, 64):
                        c16_kernel_5_row = c16_kernel_5_chunk * 16
                        c16_kernel_5_prepared = local_head * capacity + c16_kernel_5_row
                        c16_kernel_5_cumulative = prefix_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:128]
                        c16_kernel_5_square = prefix_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:16]
                        c16_kernel_5_difference = pl.sub(c16_kernel_5_square, pl.transpose(c16_kernel_5_square, 0, 1))
                        c16_kernel_5_pair = pl.mul(pl.exp(pl.minimum(c16_kernel_5_difference, 0.0)), inclusive[0:16, 0:16])
                        pair_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:16] = c16_kernel_5_pair
                        c16_kernel_5_beta_column = pl.mul(pl.row_sum(beta_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:128]), 1.0 / 128)
                        c16_kernel_5_negative = pl.neg(pl.mul(pl.row_expand_mul(pl.mul(c16_kernel_5_pair, gram_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:16]), c16_kernel_5_beta_column), strict[0:16, 0:16]))
                        negative_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:128] = pl.concat(c16_kernel_5_negative, pl.full([16, 112], dtype=pl.FP32, value=0.0))
                        causal_ready[c16_kernel_5_prepared:c16_kernel_5_prepared + 16, 0:16] = pl.mul(c16_kernel_5_pair, qk_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:16])
                        c16_kernel_5_direct = pl.exp(c16_kernel_5_cumulative)
                        query_ready[c16_kernel_5_prepared:c16_kernel_5_prepared + 16, 0:128] = pl.mul(query_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:128], c16_kernel_5_direct)
                        weighted_key_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:128] = pl.row_expand_mul(pl.mul(key_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:128], c16_kernel_5_direct), c16_kernel_5_beta_column)
                        weighted_value_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:128] = pl.row_expand_mul(value_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:128], c16_kernel_5_beta_column)
                        c16_kernel_5_last = prefix_gm[c16_kernel_5_row + 15:c16_kernel_5_row + 16, 0:128]
                        c16_kernel_5_repeated_last = pl.col_expand_mul(pl.full([16, 128], dtype=pl.FP32, value=1.0), c16_kernel_5_last)
                        key_ready[c16_kernel_5_prepared:c16_kernel_5_prepared + 16, 0:128] = pl.mul(key_gm[c16_kernel_5_row:c16_kernel_5_row + 16, 0:128], pl.exp(pl.minimum(pl.sub(c16_kernel_5_repeated_last, c16_kernel_5_cumulative), 0.0)))
                        decay_ready[local_head * chunk_capacity + c16_kernel_5_chunk:local_head * chunk_capacity + c16_kernel_5_chunk + 1, 0:128] = pl.exp(c16_kernel_5_last)
                with pl.spmd(64, name_hint='gdn_c16_seed_all', deps=[mask_task]) as seed_task:
                    c16_kernel_6_block = pl.tile.get_block_idx()
                    c16_kernel_6_used = pl.cast(pl.read(offsets, [batch]), pl.INDEX)
                    for c16_kernel_6_chunk in pl.range(c16_kernel_6_block, chunk_capacity, 64):
                        c16_kernel_6_row = c16_kernel_6_chunk * 16
                        if c16_kernel_6_row < c16_kernel_6_used:
                            factor_gm[c16_kernel_6_row:c16_kernel_6_row + 16, 0:128] = pl.add(identity[0:16, 0:128], negative_gm[c16_kernel_6_row:c16_kernel_6_row + 16, 0:128])
                        else:
                            factor_gm[c16_kernel_6_row:c16_kernel_6_row + 16, 0:128] = identity[0:16, 0:128]
                        product_gm[c16_kernel_6_row:c16_kernel_6_row + 16, 0:128] = identity[0:16, 0:128]
                with pl.spmd(64, name_hint='gdn_c16_inverse_all0', deps=[seed_task]) as inverse0:
                    for c16_kernel_7_chunk in pl.range(pl.tile.get_block_idx(), chunk_capacity, 64):
                        c16_kernel_7_row = c16_kernel_7_chunk * 16
                        c16_kernel_7_narrow = factor_gm[c16_kernel_7_row:c16_kernel_7_row + 16, 0:16]
                        c16_kernel_7_wide = factor_gm[c16_kernel_7_row:c16_kernel_7_row + 16, 0:128]
                        c16_kernel_7_twice_eye = pl.mul(identity[0:16, 0:128], 2.0)
                        product_gm[c16_kernel_7_row:c16_kernel_7_row + 16, 128:256] = pl.matmul(c16_kernel_7_narrow, product_gm[c16_kernel_7_row:c16_kernel_7_row + 16, 0:128], pl.FP32)
                        factor_gm[c16_kernel_7_row:c16_kernel_7_row + 16, 128:256] = pl.add(pl.matmul(c16_kernel_7_narrow, pl.sub(c16_kernel_7_wide, c16_kernel_7_twice_eye), pl.FP32), c16_kernel_7_twice_eye)
                with pl.spmd(64, name_hint='gdn_c16_inverse_all1', deps=[inverse0]) as inverse1:
                    for c16_kernel_8_chunk in pl.range(pl.tile.get_block_idx(), chunk_capacity, 64):
                        c16_kernel_8_row = c16_kernel_8_chunk * 16
                        c16_kernel_8_narrow = factor_gm[c16_kernel_8_row:c16_kernel_8_row + 16, 128:144]
                        c16_kernel_8_wide = factor_gm[c16_kernel_8_row:c16_kernel_8_row + 16, 128:256]
                        c16_kernel_8_twice_eye = pl.mul(identity[0:16, 0:128], 2.0)
                        product_gm[c16_kernel_8_row:c16_kernel_8_row + 16, 256:384] = pl.matmul(c16_kernel_8_narrow, product_gm[c16_kernel_8_row:c16_kernel_8_row + 16, 128:256], pl.FP32)
                        factor_gm[c16_kernel_8_row:c16_kernel_8_row + 16, 256:384] = pl.add(pl.matmul(c16_kernel_8_narrow, pl.sub(c16_kernel_8_wide, c16_kernel_8_twice_eye), pl.FP32), c16_kernel_8_twice_eye)
                with pl.spmd(64, name_hint='gdn_c16_inverse_all2', deps=[inverse1]) as inverse2:
                    for c16_kernel_9_chunk in pl.range(pl.tile.get_block_idx(), chunk_capacity, 64):
                        c16_kernel_9_row = c16_kernel_9_chunk * 16
                        c16_kernel_9_narrow = factor_gm[c16_kernel_9_row:c16_kernel_9_row + 16, 256:272]
                        c16_kernel_9_wide = factor_gm[c16_kernel_9_row:c16_kernel_9_row + 16, 256:384]
                        c16_kernel_9_twice_eye = pl.mul(identity[0:16, 0:128], 2.0)
                        product_gm[c16_kernel_9_row:c16_kernel_9_row + 16, 384:512] = pl.matmul(c16_kernel_9_narrow, product_gm[c16_kernel_9_row:c16_kernel_9_row + 16, 256:384], pl.FP32)
                        factor_gm[c16_kernel_9_row:c16_kernel_9_row + 16, 384:512] = pl.add(pl.matmul(c16_kernel_9_narrow, pl.sub(c16_kernel_9_wide, c16_kernel_9_twice_eye), pl.FP32), c16_kernel_9_twice_eye)
                with pl.spmd(64, name_hint='gdn_c16_inverse_all3', deps=[inverse2]) as inverse3:
                    for c16_kernel_10_chunk in pl.range(pl.tile.get_block_idx(), chunk_capacity, 64):
                        c16_kernel_10_row = c16_kernel_10_chunk * 16
                        c16_kernel_10_narrow = factor_gm[c16_kernel_10_row:c16_kernel_10_row + 16, 384:400]
                        c16_kernel_10_wide = factor_gm[c16_kernel_10_row:c16_kernel_10_row + 16, 384:512]
                        c16_kernel_10_twice_eye = pl.mul(identity[0:16, 0:128], 2.0)
                        product_gm[c16_kernel_10_row:c16_kernel_10_row + 16, 512:640] = pl.matmul(c16_kernel_10_narrow, product_gm[c16_kernel_10_row:c16_kernel_10_row + 16, 384:512], pl.FP32)
                        factor_gm[c16_kernel_10_row:c16_kernel_10_row + 16, 512:640] = pl.add(pl.matmul(c16_kernel_10_narrow, pl.sub(c16_kernel_10_wide, c16_kernel_10_twice_eye), pl.FP32), c16_kernel_10_twice_eye)
                with pl.spmd(64, name_hint='gdn_c16_transform_all', deps=[inverse3]) as transform_task:
                    c16_kernel_11_used_chunks = pl.cast(pl.read(offsets, [batch]), pl.INDEX) // 16
                    for c16_kernel_11_chunk in pl.range(pl.tile.get_block_idx(), c16_kernel_11_used_chunks, 64):
                        c16_kernel_11_row = c16_kernel_11_chunk * 16
                        c16_kernel_11_prepared = local_head * capacity + c16_kernel_11_row
                        c16_kernel_11_transform = product_gm[c16_kernel_11_row:c16_kernel_11_row + 16, 512:528]
                        transform_key_ready[c16_kernel_11_prepared:c16_kernel_11_prepared + 16, 0:128] = pl.matmul(c16_kernel_11_transform, weighted_key_gm[c16_kernel_11_row:c16_kernel_11_row + 16, 0:128], pl.FP32)
                        transform_value_ready[c16_kernel_11_prepared:c16_kernel_11_prepared + 16, 0:128] = pl.matmul(c16_kernel_11_transform, weighted_value_gm[c16_kernel_11_row:c16_kernel_11_row + 16, 0:128], pl.FP32)
                dependency = transform_task
            scan_logical = batch * 2 * 4
            scan_physical = pl.min(scan_logical, 64)
            with pl.spmd(scan_physical, name_hint='gdn_c16_seed_carry', deps=[dependency]) as carry_seed_task:
                for c16_seed_c16_kernel_12_task in pl.range(pl.tile.get_block_idx(), scan_logical, scan_physical):
                    c16_seed_c16_kernel_12_sequence = c16_seed_c16_kernel_12_task // 8
                    c16_seed_c16_kernel_12_local_head = c16_seed_c16_kernel_12_task % 8 // 4
                    c16_seed_c16_kernel_12_value_start = c16_seed_c16_kernel_12_task % 4 * 32
                    c16_seed_c16_kernel_12_head = head_base + c16_seed_c16_kernel_12_local_head
                    c16_seed_c16_kernel_12_count = pl.cast(pl.read(counts, [c16_seed_c16_kernel_12_sequence]), pl.INDEX)
                    if c16_seed_c16_kernel_12_count > 4:
                        c16_seed_c16_kernel_12_skip = 0
                        if snapshot_mode > 0:
                            c16_seed_c16_kernel_12_skip = 4
                        c16_seed_c16_kernel_12_slot = pl.cast(pl.read(slots, [c16_seed_c16_kernel_12_sequence]), pl.INDEX)
                        c16_seed_c16_kernel_12_origin = pl.cast(pl.read(offsets, [c16_seed_c16_kernel_12_sequence]), pl.INDEX)
                        c16_seed_c16_kernel_12_token_origin = pl.cast(pl.read(row_offsets, [c16_seed_c16_kernel_12_sequence]), pl.INDEX) + c16_seed_c16_kernel_12_skip
                        c16_seed_c16_kernel_12_chunks = (c16_seed_c16_kernel_12_count - c16_seed_c16_kernel_12_skip + 15) // 16
                        c16_seed_c16_kernel_12_carry_start = c16_seed_c16_kernel_12_task * 2 * 32
                        c16_seed_c16_kernel_12_seed = pl.reshape(pl.slice(states, [1, 1, 1, 32, 128], [c16_seed_c16_kernel_12_slot, layer, c16_seed_c16_kernel_12_head, c16_seed_c16_kernel_12_value_start, 0]), [32, 128])
                        carry_cube[c16_seed_c16_kernel_12_carry_start:c16_seed_c16_kernel_12_carry_start + 32, 0:128] = c16_seed_c16_kernel_12_seed
                        carry_vector[c16_seed_c16_kernel_12_carry_start:c16_seed_c16_kernel_12_carry_start + 32, 0:128] = c16_seed_c16_kernel_12_seed
            with pl.spmd(scan_physical, name_hint='gdn_c16_chunk_scan', deps=[carry_seed_task]) as scan_task:
                for c16_kernel_12_task in pl.range(pl.tile.get_block_idx(), scan_logical, scan_physical):
                    c16_kernel_12_sequence = c16_kernel_12_task // 8
                    c16_kernel_12_local_head = c16_kernel_12_task % 8 // 4
                    c16_kernel_12_value_start = c16_kernel_12_task % 4 * 32
                    c16_kernel_12_head = head_base + c16_kernel_12_local_head
                    c16_kernel_12_count = pl.cast(pl.read(counts, [c16_kernel_12_sequence]), pl.INDEX)
                    if c16_kernel_12_count > 4:
                        c16_kernel_12_skip = 0
                        if snapshot_mode > 0:
                            c16_kernel_12_skip = 4
                        c16_kernel_12_slot = pl.cast(pl.read(slots, [c16_kernel_12_sequence]), pl.INDEX)
                        c16_kernel_12_origin = pl.cast(pl.read(offsets, [c16_kernel_12_sequence]), pl.INDEX)
                        c16_kernel_12_token_origin = pl.cast(pl.read(row_offsets, [c16_kernel_12_sequence]), pl.INDEX) + c16_kernel_12_skip
                        c16_kernel_12_chunks = (c16_kernel_12_count - c16_kernel_12_skip + 15) // 16
                        c16_kernel_12_carried = pl.tile.load(carry_vector, [c16_kernel_12_task * 2 * 32, 0], [32, 128], target_memory=pl.MemorySpace.Vec)
                        for c16_kernel_12_chunk in pl.range(c16_kernel_12_chunks):
                            c16_kernel_12_row = c16_kernel_12_origin + c16_kernel_12_chunk * 16
                            c16_kernel_12_prepared = c16_kernel_12_local_head * capacity + c16_kernel_12_row
                            c16_kernel_12_cube_state = pl.tile.move(c16_kernel_12_carried, pl.MemorySpace.Mat, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)
                            c16_kernel_12_w = pl.tile.load(transform_key_ready, [c16_kernel_12_prepared, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                            c16_kernel_12_explained_acc = pl.tile.matmul(c16_kernel_12_w, pl.tile.transpose_view(c16_kernel_12_cube_state))
                            c16_kernel_12_explained = pl.tile.move(c16_kernel_12_explained_acc, pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                            c16_kernel_12_z = pl.tile.load(transform_value_ready, [c16_kernel_12_prepared, c16_kernel_12_value_start], [16, 32], target_memory=pl.MemorySpace.Vec)
                            c16_kernel_12_residual = pl.sub(c16_kernel_12_z, c16_kernel_12_explained)
                            c16_kernel_12_q = pl.tile.load(query_ready, [c16_kernel_12_prepared, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                            c16_kernel_12_direct_acc = pl.tile.matmul(c16_kernel_12_q, pl.tile.transpose_view(c16_kernel_12_cube_state))
                            c16_kernel_12_direct_output = pl.tile.move(c16_kernel_12_direct_acc, pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                            c16_kernel_12_a = pl.tile.load(causal_ready, [c16_kernel_12_prepared, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
                            c16_kernel_12_residual_mat = pl.tile.move(c16_kernel_12_residual, pl.MemorySpace.Mat, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)
                            c16_kernel_12_local_acc = pl.tile.matmul(c16_kernel_12_a, c16_kernel_12_residual_mat)
                            c16_kernel_12_local_output = pl.tile.move(c16_kernel_12_local_acc, pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                            c16_kernel_12_token = c16_kernel_12_token_origin + c16_kernel_12_chunk * 16
                            c16_kernel_12_actual = pl.min(16, c16_kernel_12_count - c16_kernel_12_skip - c16_kernel_12_chunk * 16)
                            c16_kernel_12_result = pl.set_validshape(pl.cast(pl.add(c16_kernel_12_direct_output, c16_kernel_12_local_output), pl.BF16, mode='rint'), c16_kernel_12_actual, 32)
                            pl.tile.store(c16_kernel_12_result, [c16_kernel_12_token, c16_kernel_12_head * 128 + c16_kernel_12_value_start], out)
                            c16_kernel_12_decay_row = c16_kernel_12_local_head * chunk_capacity + c16_kernel_12_row // 16
                            c16_kernel_12_decay_tile = pl.tile.load(decay_ready, [c16_kernel_12_decay_row, 0], [1, 128], target_memory=pl.MemorySpace.Vec)
                            c16_kernel_12_retained = pl.col_expand_mul(c16_kernel_12_carried, c16_kernel_12_decay_tile)
                            c16_kernel_12_k = pl.tile.load(key_ready, [c16_kernel_12_prepared, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                            c16_kernel_12_update_acc = pl.tile.matmul(pl.tile.transpose_view(c16_kernel_12_residual_mat), c16_kernel_12_k)
                            c16_kernel_12_update = pl.tile.move(c16_kernel_12_update_acc, pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                            c16_kernel_12_carried = pl.add(c16_kernel_12_retained, c16_kernel_12_update)
                        c16_kernel_12_pool_row = ((c16_kernel_12_slot * 48 + layer) * 24 + c16_kernel_12_head) * 128 + c16_kernel_12_value_start
                        pl.tile.store(c16_kernel_12_carried, [c16_kernel_12_pool_row, 0], c16_state_rows)
            dependency = scan_task
    with pl.spmd(32, name_hint='gdn_c16_zero_tail'):
        c16_kernel_13_total = pl.cast(pl.read(row_offsets, [batch - 1]), pl.INDEX) + pl.cast(pl.read(counts, [batch - 1]), pl.INDEX)
        for c16_kernel_13_task in pl.range(pl.tile.get_block_idx(), (rows - c16_kernel_13_total) * 24, 32):
            c16_kernel_13_row = c16_kernel_13_total + c16_kernel_13_task // 24
            c16_kernel_13_col = c16_kernel_13_task % 24 * 128
            out[c16_kernel_13_row:c16_kernel_13_row + 1, c16_kernel_13_col:c16_kernel_13_col + 128] = pl.full([1, 128], dtype=pl.BF16, value=0.0)
    return (out, states, snapshots)

@pl.jit(auto_scope=False)
def model_fwd(input_ids: pl.Tensor[[R], pl.INT32], embedding: pl.Tensor[[248320, 5120], pl.BF16], final_gamma: pl.Tensor[[1, 5120], pl.FP32], lm_head: pl.Tensor[[124160, 5120], pl.BF16], input_gamma: pl.Tensor[[64, 5120], pl.FP32], post_gamma: pl.Tensor[[64, 5120], pl.FP32], mlp_gate: pl.Tensor[[64, 8704, 5120], pl.BF16, pl.NZ], mlp_up: pl.Tensor[[64, 8704, 5120], pl.BF16, pl.NZ], mlp_down: pl.Tensor[[327680, 8704], pl.BF16], delta_qkv: pl.Tensor[[245760, 5120], pl.BF16], delta_z: pl.Tensor[[147456, 5120], pl.BF16], delta_out: pl.Tensor[[245760, 3072], pl.BF16], delta_a: pl.Tensor[[6144, 5120], pl.BF16], delta_b: pl.Tensor[[6144, 5120], pl.BF16], delta_conv: pl.Tensor[[192, 5120], pl.BF16], delta_a_log: pl.Tensor[[48, 32], pl.BF16], delta_dt_bias: pl.Tensor[[48, 32], pl.BF16], delta_norm: pl.Tensor[[48, 128], pl.BF16], full_q: pl.Tensor[[98304, 5120], pl.BF16], full_k: pl.Tensor[[8192, 5120], pl.BF16], full_v: pl.Tensor[[8192, 5120], pl.BF16], full_q_gamma: pl.Tensor[[16, 256], pl.FP32], full_k_gamma: pl.Tensor[[16, 256], pl.FP32], full_out: pl.Tensor[[81920, 3072], pl.BF16], mtp_fc: pl.Tensor[[5120, 5120], pl.BF16], mtp_embedding_gamma: pl.Tensor[[1, 5120], pl.FP32], mtp_hidden_gamma: pl.Tensor[[1, 5120], pl.FP32], mtp_final_gamma: pl.Tensor[[1, 5120], pl.FP32], mtp_input_gamma: pl.Tensor[[1, 5120], pl.FP32], mtp_post_gamma: pl.Tensor[[1, 5120], pl.FP32], mtp_q_gamma: pl.Tensor[[1, 256], pl.FP32], mtp_k_gamma: pl.Tensor[[1, 256], pl.FP32], mtp_q: pl.Tensor[[6144, 5120], pl.BF16], mtp_k: pl.Tensor[[512, 5120], pl.BF16], mtp_v: pl.Tensor[[512, 5120], pl.BF16], mtp_out: pl.Tensor[[5120, 3072], pl.BF16], mtp_mlp_gate: pl.Tensor[[8704, 5120], pl.BF16], mtp_mlp_up: pl.Tensor[[8704, 5120], pl.BF16], mtp_mlp_down: pl.Tensor[[5120, 8704], pl.BF16], counts: pl.Tensor[[B], pl.INT32], row_offsets: pl.Tensor[[B], pl.INT32], row_sequences: pl.Tensor[[R], pl.INT32], positions: pl.Tensor[[B], pl.INT32], slot_ids: pl.Tensor[[B], pl.INT32], cos: pl.Tensor[[R, 64], pl.BF16], sin: pl.Tensor[[R, 64], pl.BF16], block_tables: pl.Tensor[[32, TABLE], pl.INT32], conv_states: pl.InOut[pl.Tensor[[32, 48, 4, 5120], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[32, 48, 24, 128, 128], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[17, 2, POOL, 128, 256], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[17, 2, POOL, 128, 256], pl.BF16]], conv_snapshots: pl.InOut[pl.Tensor[[4, 32, 48, 4, 5120], pl.BF16]], recurrent_snapshots: pl.InOut[pl.Tensor[[4, 32, 48, 3072, 128], pl.FP32]], hidden_out: pl.Out[pl.Tensor[[R, 5120], pl.BF16]], logits_out: pl.Out[pl.Tensor[[L, 124160], pl.BF16]], snapshot_mode: pl.Scalar[pl.INT32], comm_data: pld.DistributedTensor[[16384, 5120], pl.FP32], comm_signals: pld.DistributedTensor[[2, 1], pl.INT32], rank: pl.Scalar[pl.INT32], max_count: pl.Scalar[pl.INT32]):
    rows = pl.tensor.dim(input_ids, 0)
    with pl.scope():
        c16_batch = pl.tensor.dim(counts, 0)
        c16_capacity = rows + c16_batch * 16
        c16_chunk_capacity = c16_capacity // 16
        c16_inclusive, c16_strict, c16_identity, c16_onehot = make_c16_constants(max_count)
        c16_workspace_offsets = pl.create_tensor([c16_batch + 1], dtype=pl.INT32)
        c16_workspace_query_gm = pl.create_tensor([c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_key_gm = pl.create_tensor([c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_value_gm = pl.create_tensor([c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_gate_gm = pl.create_tensor([c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_beta_gm = pl.create_tensor([c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_prefix_gm = pl.create_tensor([c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_gram_gm = pl.create_tensor([c16_capacity, 16], dtype=pl.FP32)
        c16_workspace_qk_gm = pl.create_tensor([c16_capacity, 16], dtype=pl.FP32)
        c16_workspace_pair_gm = pl.create_tensor([c16_capacity, 16], dtype=pl.FP32)
        c16_workspace_negative_gm = pl.create_tensor([c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_weighted_key_gm = pl.create_tensor([c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_weighted_value_gm = pl.create_tensor([c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_factor_gm = pl.create_tensor([c16_capacity, 640], dtype=pl.FP32)
        c16_workspace_product_gm = pl.create_tensor([c16_capacity, 640], dtype=pl.FP32)
        c16_workspace_query_ready = pl.create_tensor([2 * c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_key_ready = pl.create_tensor([2 * c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_transform_key_ready = pl.create_tensor([2 * c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_transform_value_ready = pl.create_tensor([2 * c16_capacity, 128], dtype=pl.FP32)
        c16_workspace_causal_ready = pl.create_tensor([2 * c16_capacity, 16], dtype=pl.FP32)
        c16_workspace_decay_ready = pl.create_tensor([2 * c16_chunk_capacity, 128], dtype=pl.FP32)
        c16_workspace_carry_cube = pl.create_tensor([c16_batch * 2 * 4 * 2 * 32, 128], dtype=pl.FP32)
        c16_workspace_carry_vector = pl.create_tensor([c16_batch * 2 * 4 * 2 * 32, 128], dtype=pl.FP32)
        normalized = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        attention_partial = pl.create_tensor([rows, 5120], dtype=pl.FP32)
        attention_output = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        residual = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        post_norm = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        mlp_part = pl.create_tensor([rows, 5120], dtype=pl.FP32)
        mlp_output = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        projected = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        z = pl.create_tensor([rows, 3072], dtype=pl.BF16)
        a = pl.create_tensor([rows, 128], dtype=pl.BF16)
        b = pl.create_tensor([rows, 128], dtype=pl.BF16)
        convolved = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        core = pl.create_tensor([rows, 3072], dtype=pl.BF16)
        decay_values = pl.create_tensor([rows, 32], dtype=pl.FP32)
        beta_values = pl.create_tensor([rows, 32], dtype=pl.FP32)
        qn = pl.create_tensor([rows, 3072], dtype=pl.FP32)
        kn = pl.create_tensor([rows, 3072], dtype=pl.FP32)
        gated = pl.create_tensor([rows, 3072], dtype=pl.BF16)
        attn_qp = pl.create_tensor([rows, 6144], dtype=pl.BF16)
        attn_kp = pl.create_tensor([rows, 512], dtype=pl.BF16)
        attn_vp = pl.create_tensor([rows, 512], dtype=pl.BF16)
        attn_q = pl.create_tensor([rows, 3072], dtype=pl.BF16)
        attn_k = pl.create_tensor([rows, 512], dtype=pl.BF16)
        attn_gate = pl.create_tensor([rows, 3072], dtype=pl.BF16)
        attn_attended = pl.create_tensor([rows, 3072], dtype=pl.BF16)
        ff_middle = pl.create_tensor([rows, 8704], dtype=pl.BF16)
        for layer in pl.range(64):
            if layer == 0:
                embed(input_ids, embedding, row_sequences, hidden_out)
            gamma = pl.slice(input_gamma, [1, 5120], [layer, 0])
            post = pl.slice(post_gamma, [1, 5120], [layer, 0])
            with pl.scope():
                rms_norm(hidden_out, gamma, normalized)
                if layer % 4 != 3:
                    linear = layer - layer // 4
                    qw = pl.slice(delta_qkv, [5120, 5120], [linear * 5120, 0])
                    zw = pl.slice(delta_z, [3072, 5120], [linear * 3072, 0])
                    aw = pl.slice(delta_a, [128, 5120], [linear * 128, 0])
                    bw = pl.slice(delta_b, [128, 5120], [linear * 128, 0])
                    cw = pl.slice(delta_conv, [4, 5120], [linear * 4, 0])
                    al = pl.slice(delta_a_log, [1, 32], [linear, 0])
                    dt = pl.slice(delta_dt_bias, [1, 32], [linear, 0])
                    nw = pl.slice(delta_norm, [1, 128], [linear, 0])
                    ow = pl.slice(delta_out, [5120, 3072], [linear * 5120, 0])
                    linear_bf16(normalized, qw, projected)
                    linear_bf16(normalized, zw, z)
                    linear_bf16(normalized, aw, a)
                    linear_bf16(normalized, bw, b)
                    conv_update(projected, cw, counts, row_offsets, slot_ids, conv_states, conv_snapshots, convolved, linear, snapshot_mode, max_count)
                    delta_parameters(a, b, al, dt, decay_values, beta_values)
                    delta_normalize(convolved, qn, kn)
                    delta_core_c16(convolved, qn, kn, decay_values, beta_values, counts, row_offsets, slot_ids, recurrent_states, recurrent_snapshots, core, linear, snapshot_mode, max_count, c16_inclusive, c16_strict, c16_identity, c16_onehot, c16_workspace_offsets, c16_workspace_query_gm, c16_workspace_key_gm, c16_workspace_value_gm, c16_workspace_gate_gm, c16_workspace_beta_gm, c16_workspace_prefix_gm, c16_workspace_gram_gm, c16_workspace_qk_gm, c16_workspace_pair_gm, c16_workspace_negative_gm, c16_workspace_weighted_key_gm, c16_workspace_weighted_value_gm, c16_workspace_factor_gm, c16_workspace_product_gm, c16_workspace_query_ready, c16_workspace_key_ready, c16_workspace_transform_key_ready, c16_workspace_transform_value_ready, c16_workspace_causal_ready, c16_workspace_decay_ready, c16_workspace_carry_cube, c16_workspace_carry_vector)
                    delta_gate(core, z, nw, gated)
                    linear_partial(gated, ow, attention_partial)
                else:
                    full = layer // 4
                    full_qw = pl.slice(full_q, [6144, 5120], [full * 6144, 0])
                    full_kw = pl.slice(full_k, [512, 5120], [full * 512, 0])
                    full_vw = pl.slice(full_v, [512, 5120], [full * 512, 0])
                    full_qg = pl.slice(full_q_gamma, [1, 256], [full, 0])
                    full_kg = pl.slice(full_k_gamma, [1, 256], [full, 0])
                    full_ow = pl.slice(full_out, [5120, 3072], [full * 5120, 0])
                    full_attention_scratch(normalized, full_qw, full_kw, full_vw, full_qg, full_kg, full_ow, cos, sin, counts, row_offsets, positions, slot_ids, block_tables, key_pool, value_pool, attention_partial, full, 0, attn_qp, attn_kp, attn_vp, attn_q, attn_k, attn_gate, attn_attended)
            tp_sum(attention_partial, comm_data, comm_signals, attention_output, rank, layer * 2 + 1)
            residual_add(hidden_out, attention_output, residual)
            rms_norm(residual, post, post_norm)
            gate_weight = pl.slice(mlp_gate, [1, 8704, 5120], [layer, 0, 0])
            up_weight = pl.slice(mlp_up, [1, 8704, 5120], [layer, 0, 0])
            down_weight = pl.slice(mlp_down, [5120, 8704], [layer * 5120, 0])
            mlp_partial_scratch(post_norm, gate_weight, up_weight, down_weight, mlp_part, ff_middle)
            tp_sum(mlp_part, comm_data, comm_signals, mlp_output, rank, layer * 2 + 2)
            residual_add(residual, mlp_output, hidden_out)
        rms_norm(hidden_out, final_gamma, hidden_out)
        zero_tail(hidden_out, counts, row_offsets)
        logit_rows = pl.tensor.dim(logits_out, 0)
        selected_hidden = pl.create_tensor([logit_rows, 5120], dtype=pl.BF16)
        compact_hidden(hidden_out, counts, row_offsets, selected_hidden)
        linear_bf16(selected_hidden, lm_head, logits_out)
        tp_finish(comm_signals, rank, 128)
    return (conv_states, recurrent_states, key_pool, value_pool, conv_snapshots, recurrent_snapshots, hidden_out, logits_out)

@pl.jit(auto_scope=False)
def drafter_fwd(input_ids: pl.Tensor[[R], pl.INT32], previous_hidden: pl.Tensor[[R, 5120], pl.BF16], embedding: pl.Tensor[[248320, 5120], pl.BF16], final_gamma: pl.Tensor[[1, 5120], pl.FP32], lm_head: pl.Tensor[[124160, 5120], pl.BF16], input_gamma: pl.Tensor[[64, 5120], pl.FP32], post_gamma: pl.Tensor[[64, 5120], pl.FP32], mlp_gate: pl.Tensor[[64, 8704, 5120], pl.BF16, pl.NZ], mlp_up: pl.Tensor[[64, 8704, 5120], pl.BF16, pl.NZ], mlp_down: pl.Tensor[[327680, 8704], pl.BF16], delta_qkv: pl.Tensor[[245760, 5120], pl.BF16], delta_z: pl.Tensor[[147456, 5120], pl.BF16], delta_out: pl.Tensor[[245760, 3072], pl.BF16], delta_a: pl.Tensor[[6144, 5120], pl.BF16], delta_b: pl.Tensor[[6144, 5120], pl.BF16], delta_conv: pl.Tensor[[192, 5120], pl.BF16], delta_a_log: pl.Tensor[[48, 32], pl.BF16], delta_dt_bias: pl.Tensor[[48, 32], pl.BF16], delta_norm: pl.Tensor[[48, 128], pl.BF16], full_q: pl.Tensor[[98304, 5120], pl.BF16], full_k: pl.Tensor[[8192, 5120], pl.BF16], full_v: pl.Tensor[[8192, 5120], pl.BF16], full_q_gamma: pl.Tensor[[16, 256], pl.FP32], full_k_gamma: pl.Tensor[[16, 256], pl.FP32], full_out: pl.Tensor[[81920, 3072], pl.BF16], mtp_fc: pl.Tensor[[5120, 5120], pl.BF16], mtp_embedding_gamma: pl.Tensor[[1, 5120], pl.FP32], mtp_hidden_gamma: pl.Tensor[[1, 5120], pl.FP32], mtp_final_gamma: pl.Tensor[[1, 5120], pl.FP32], mtp_input_gamma: pl.Tensor[[1, 5120], pl.FP32], mtp_post_gamma: pl.Tensor[[1, 5120], pl.FP32], mtp_q_gamma: pl.Tensor[[1, 256], pl.FP32], mtp_k_gamma: pl.Tensor[[1, 256], pl.FP32], mtp_q: pl.Tensor[[6144, 5120], pl.BF16], mtp_k: pl.Tensor[[512, 5120], pl.BF16], mtp_v: pl.Tensor[[512, 5120], pl.BF16], mtp_out: pl.Tensor[[5120, 3072], pl.BF16], mtp_mlp_gate: pl.Tensor[[8704, 5120], pl.BF16], mtp_mlp_up: pl.Tensor[[8704, 5120], pl.BF16], mtp_mlp_down: pl.Tensor[[5120, 8704], pl.BF16], counts: pl.Tensor[[B], pl.INT32], row_offsets: pl.Tensor[[B], pl.INT32], row_sequences: pl.Tensor[[R], pl.INT32], positions: pl.Tensor[[B], pl.INT32], slot_ids: pl.Tensor[[B], pl.INT32], cos: pl.Tensor[[R, 64], pl.BF16], sin: pl.Tensor[[R, 64], pl.BF16], block_tables: pl.Tensor[[32, TABLE], pl.INT32], conv_states: pl.InOut[pl.Tensor[[32, 48, 4, 5120], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[32, 48, 24, 128, 128], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[17, 2, POOL, 128, 256], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[17, 2, POOL, 128, 256], pl.BF16]], conv_snapshots: pl.InOut[pl.Tensor[[4, 32, 48, 4, 5120], pl.BF16]], recurrent_snapshots: pl.InOut[pl.Tensor[[4, 32, 48, 3072, 128], pl.FP32]], hidden_out: pl.Out[pl.Tensor[[R, 5120], pl.BF16]], logits_out: pl.Out[pl.Tensor[[L, 124160], pl.BF16]], snapshot_mode: pl.Scalar[pl.INT32], comm_data: pld.DistributedTensor[[16384, 5120], pl.FP32], comm_signals: pld.DistributedTensor[[2, 1], pl.INT32], rank: pl.Scalar[pl.INT32]):
    rows = pl.tensor.dim(input_ids, 0)
    with pl.scope():
        embedded = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        embed(input_ids, embedding, row_sequences, embedded)
        normalized = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        if rank == 0:
            rms_norm(embedded, mtp_embedding_gamma, normalized)
        else:
            rms_norm(previous_hidden, mtp_hidden_gamma, normalized)
        fc_partial = pl.create_tensor([rows, 5120], dtype=pl.FP32)
        linear_partial(normalized, mtp_fc, fc_partial)
        running = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        tp_sum(fc_partial, comm_data, comm_signals, running, rank, 1)
        rms_norm(running, mtp_input_gamma, normalized)
        attention_partial = pl.create_tensor([rows, 5120], dtype=pl.FP32)
        with pl.scope():
            full_attention(normalized, mtp_q, mtp_k, mtp_v, mtp_q_gamma, mtp_k_gamma, mtp_out, cos, sin, counts, row_offsets, positions, slot_ids, block_tables, key_pool, value_pool, attention_partial, 16, 1)
        attention_output = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        tp_sum(attention_partial, comm_data, comm_signals, attention_output, rank, 2)
        residual = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        residual_add(running, attention_output, residual)
        rms_norm(residual, mtp_post_gamma, normalized)
        partial = pl.create_tensor([rows, 5120], dtype=pl.FP32)
        mlp_partial(normalized, mtp_mlp_gate, mtp_mlp_up, mtp_mlp_down, partial)
        mlp_output = pl.create_tensor([rows, 5120], dtype=pl.BF16)
        tp_sum(partial, comm_data, comm_signals, mlp_output, rank, 3)
        residual_add(residual, mlp_output, running)
        rms_norm(running, mtp_final_gamma, hidden_out)
        zero_tail(hidden_out, counts, row_offsets)
        logit_rows = pl.tensor.dim(logits_out, 0)
        selected_hidden = pl.create_tensor([logit_rows, 5120], dtype=pl.BF16)
        compact_hidden(hidden_out, counts, row_offsets, selected_hidden)
        linear_bf16(selected_hidden, lm_head, logits_out)
        tp_finish(comm_signals, rank, 3)
    return (key_pool, value_pool, hidden_out, logits_out)

@pl.jit.host
def model_host(input_ids: pl.Tensor[[2, R], pl.INT32], embedding: pl.Tensor[[2, 248320, 5120], pl.BF16], final_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], lm_head: pl.Tensor[[2, 124160, 5120], pl.BF16], input_gamma: pl.Tensor[[2, 64, 5120], pl.FP32], post_gamma: pl.Tensor[[2, 64, 5120], pl.FP32], mlp_gate: pl.Tensor[[2, 64, 8704, 5120], pl.BF16], mlp_up: pl.Tensor[[2, 64, 8704, 5120], pl.BF16], mlp_down: pl.Tensor[[2, 327680, 8704], pl.BF16], delta_qkv: pl.Tensor[[2, 245760, 5120], pl.BF16], delta_z: pl.Tensor[[2, 147456, 5120], pl.BF16], delta_out: pl.Tensor[[2, 245760, 3072], pl.BF16], delta_a: pl.Tensor[[2, 6144, 5120], pl.BF16], delta_b: pl.Tensor[[2, 6144, 5120], pl.BF16], delta_conv: pl.Tensor[[2, 192, 5120], pl.BF16], delta_a_log: pl.Tensor[[2, 48, 32], pl.BF16], delta_dt_bias: pl.Tensor[[2, 48, 32], pl.BF16], delta_norm: pl.Tensor[[2, 48, 128], pl.BF16], full_q: pl.Tensor[[2, 98304, 5120], pl.BF16], full_k: pl.Tensor[[2, 8192, 5120], pl.BF16], full_v: pl.Tensor[[2, 8192, 5120], pl.BF16], full_q_gamma: pl.Tensor[[2, 16, 256], pl.FP32], full_k_gamma: pl.Tensor[[2, 16, 256], pl.FP32], full_out: pl.Tensor[[2, 81920, 3072], pl.BF16], mtp_fc: pl.Tensor[[2, 5120, 5120], pl.BF16], mtp_embedding_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], mtp_hidden_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], mtp_final_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], mtp_input_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], mtp_post_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], mtp_q_gamma: pl.Tensor[[2, 1, 256], pl.FP32], mtp_k_gamma: pl.Tensor[[2, 1, 256], pl.FP32], mtp_q: pl.Tensor[[2, 6144, 5120], pl.BF16], mtp_k: pl.Tensor[[2, 512, 5120], pl.BF16], mtp_v: pl.Tensor[[2, 512, 5120], pl.BF16], mtp_out: pl.Tensor[[2, 5120, 3072], pl.BF16], mtp_mlp_gate: pl.Tensor[[2, 8704, 5120], pl.BF16], mtp_mlp_up: pl.Tensor[[2, 8704, 5120], pl.BF16], mtp_mlp_down: pl.Tensor[[2, 5120, 8704], pl.BF16], counts: pl.Tensor[[2, B], pl.INT32], row_offsets: pl.Tensor[[2, B], pl.INT32], row_sequences: pl.Tensor[[2, R], pl.INT32], positions: pl.Tensor[[2, B], pl.INT32], slot_ids: pl.Tensor[[2, B], pl.INT32], cos: pl.Tensor[[2, R, 64], pl.BF16], sin: pl.Tensor[[2, R, 64], pl.BF16], block_tables: pl.Tensor[[2, 32, TABLE], pl.INT32], conv_states: pl.InOut[pl.Tensor[[2, 32, 48, 4, 5120], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[2, 32, 48, 24, 128, 128], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[2, 17, 2, POOL, 128, 256], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[2, 17, 2, POOL, 128, 256], pl.BF16]], conv_snapshots: pl.InOut[pl.Tensor[[2, 4, 32, 48, 4, 5120], pl.BF16]], recurrent_snapshots: pl.InOut[pl.Tensor[[2, 4, 32, 48, 3072, 128], pl.FP32]], hidden_out: pl.Out[pl.Tensor[[2, R, 5120], pl.BF16]], logits_out: pl.Out[pl.Tensor[[2, L, 124160], pl.BF16]], snapshot_mode: pl.Scalar[pl.INT32], max_count: pl.Scalar[pl.INT32], decision_valid: pl.Tensor[[2, L], pl.INT32], decision_out: pl.Out[pl.Tensor[[2, L, 16], pl.FP32]]):
    data_buffer = pld.alloc_window_buffer([16384, 5120], dtype=pl.FP32)
    signal_buffer = pld.alloc_window_buffer([2, 1], dtype=pl.INT32)
    for rank in pl.range(2):
        data = pld.window(data_buffer, [16384, 5120], dtype=pl.FP32)
        signal = pld.window(signal_buffer, [2, 1], dtype=pl.INT32)
        model_fwd(input_ids[rank], embedding[rank], final_gamma[rank], lm_head[rank], input_gamma[rank], post_gamma[rank], mlp_gate[rank], mlp_up[rank], mlp_down[rank], delta_qkv[rank], delta_z[rank], delta_out[rank], delta_a[rank], delta_b[rank], delta_conv[rank], delta_a_log[rank], delta_dt_bias[rank], delta_norm[rank], full_q[rank], full_k[rank], full_v[rank], full_q_gamma[rank], full_k_gamma[rank], full_out[rank], mtp_fc[rank], mtp_embedding_gamma[rank], mtp_hidden_gamma[rank], mtp_final_gamma[rank], mtp_input_gamma[rank], mtp_post_gamma[rank], mtp_q_gamma[rank], mtp_k_gamma[rank], mtp_q[rank], mtp_k[rank], mtp_v[rank], mtp_out[rank], mtp_mlp_gate[rank], mtp_mlp_up[rank], mtp_mlp_down[rank], counts[rank], row_offsets[rank], row_sequences[rank], positions[rank], slot_ids[rank], cos[rank], sin[rank], block_tables[rank], conv_states[rank], recurrent_states[rank], key_pool[rank], value_pool[rank], conv_snapshots[rank], recurrent_snapshots[rank], hidden_out[rank], logits_out[rank], snapshot_mode, data, signal, rank, max_count, device=rank)
        packed_greedy_fwd(logits_out[rank], decision_valid[rank], decision_out[rank], rank, device=rank)

@pl.jit.host
def drafter_host(input_ids: pl.Tensor[[2, R], pl.INT32], previous_hidden: pl.Tensor[[2, R, 5120], pl.BF16], embedding: pl.Tensor[[2, 248320, 5120], pl.BF16], final_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], lm_head: pl.Tensor[[2, 124160, 5120], pl.BF16], input_gamma: pl.Tensor[[2, 64, 5120], pl.FP32], post_gamma: pl.Tensor[[2, 64, 5120], pl.FP32], mlp_gate: pl.Tensor[[2, 64, 8704, 5120], pl.BF16], mlp_up: pl.Tensor[[2, 64, 8704, 5120], pl.BF16], mlp_down: pl.Tensor[[2, 327680, 8704], pl.BF16], delta_qkv: pl.Tensor[[2, 245760, 5120], pl.BF16], delta_z: pl.Tensor[[2, 147456, 5120], pl.BF16], delta_out: pl.Tensor[[2, 245760, 3072], pl.BF16], delta_a: pl.Tensor[[2, 6144, 5120], pl.BF16], delta_b: pl.Tensor[[2, 6144, 5120], pl.BF16], delta_conv: pl.Tensor[[2, 192, 5120], pl.BF16], delta_a_log: pl.Tensor[[2, 48, 32], pl.BF16], delta_dt_bias: pl.Tensor[[2, 48, 32], pl.BF16], delta_norm: pl.Tensor[[2, 48, 128], pl.BF16], full_q: pl.Tensor[[2, 98304, 5120], pl.BF16], full_k: pl.Tensor[[2, 8192, 5120], pl.BF16], full_v: pl.Tensor[[2, 8192, 5120], pl.BF16], full_q_gamma: pl.Tensor[[2, 16, 256], pl.FP32], full_k_gamma: pl.Tensor[[2, 16, 256], pl.FP32], full_out: pl.Tensor[[2, 81920, 3072], pl.BF16], mtp_fc: pl.Tensor[[2, 5120, 5120], pl.BF16], mtp_embedding_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], mtp_hidden_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], mtp_final_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], mtp_input_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], mtp_post_gamma: pl.Tensor[[2, 1, 5120], pl.FP32], mtp_q_gamma: pl.Tensor[[2, 1, 256], pl.FP32], mtp_k_gamma: pl.Tensor[[2, 1, 256], pl.FP32], mtp_q: pl.Tensor[[2, 6144, 5120], pl.BF16], mtp_k: pl.Tensor[[2, 512, 5120], pl.BF16], mtp_v: pl.Tensor[[2, 512, 5120], pl.BF16], mtp_out: pl.Tensor[[2, 5120, 3072], pl.BF16], mtp_mlp_gate: pl.Tensor[[2, 8704, 5120], pl.BF16], mtp_mlp_up: pl.Tensor[[2, 8704, 5120], pl.BF16], mtp_mlp_down: pl.Tensor[[2, 5120, 8704], pl.BF16], counts: pl.Tensor[[2, B], pl.INT32], row_offsets: pl.Tensor[[2, B], pl.INT32], row_sequences: pl.Tensor[[2, R], pl.INT32], positions: pl.Tensor[[2, B], pl.INT32], slot_ids: pl.Tensor[[2, B], pl.INT32], cos: pl.Tensor[[2, R, 64], pl.BF16], sin: pl.Tensor[[2, R, 64], pl.BF16], block_tables: pl.Tensor[[2, 32, TABLE], pl.INT32], conv_states: pl.InOut[pl.Tensor[[2, 32, 48, 4, 5120], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[2, 32, 48, 24, 128, 128], pl.FP32]], key_pool: pl.InOut[pl.Tensor[[2, 17, 2, POOL, 128, 256], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[2, 17, 2, POOL, 128, 256], pl.BF16]], conv_snapshots: pl.InOut[pl.Tensor[[2, 4, 32, 48, 4, 5120], pl.BF16]], recurrent_snapshots: pl.InOut[pl.Tensor[[2, 4, 32, 48, 3072, 128], pl.FP32]], hidden_out: pl.Out[pl.Tensor[[2, R, 5120], pl.BF16]], logits_out: pl.Out[pl.Tensor[[2, L, 124160], pl.BF16]], snapshot_mode: pl.Scalar[pl.INT32], decision_valid: pl.Tensor[[2, L], pl.INT32], decision_out: pl.Out[pl.Tensor[[2, L, 16], pl.FP32]], tail_rows: pl.Tensor[[2, 32], pl.INT32], tails: pl.InOut[pl.Tensor[[2, 32, 5120], pl.BF16]], tail_status: pl.Out[pl.Tensor[[2, 32, 16], pl.INT32]], teacher_return: pl.Out[pl.Tensor[[2, TEACHER_RETURN_R, 16], pl.FP32]]):
    data_buffer = pld.alloc_window_buffer([16384, 5120], dtype=pl.FP32)
    signal_buffer = pld.alloc_window_buffer([2, 1], dtype=pl.INT32)
    for rank in pl.range(2):
        data = pld.window(data_buffer, [16384, 5120], dtype=pl.FP32)
        signal = pld.window(signal_buffer, [2, 1], dtype=pl.INT32)
        drafter_fwd(input_ids[rank], previous_hidden[rank], embedding[rank], final_gamma[rank], lm_head[rank], input_gamma[rank], post_gamma[rank], mlp_gate[rank], mlp_up[rank], mlp_down[rank], delta_qkv[rank], delta_z[rank], delta_out[rank], delta_a[rank], delta_b[rank], delta_conv[rank], delta_a_log[rank], delta_dt_bias[rank], delta_norm[rank], full_q[rank], full_k[rank], full_v[rank], full_q_gamma[rank], full_k_gamma[rank], full_out[rank], mtp_fc[rank], mtp_embedding_gamma[rank], mtp_hidden_gamma[rank], mtp_final_gamma[rank], mtp_input_gamma[rank], mtp_post_gamma[rank], mtp_q_gamma[rank], mtp_k_gamma[rank], mtp_q[rank], mtp_k[rank], mtp_v[rank], mtp_out[rank], mtp_mlp_gate[rank], mtp_mlp_up[rank], mtp_mlp_down[rank], counts[rank], row_offsets[rank], row_sequences[rank], positions[rank], slot_ids[rank], cos[rank], sin[rank], block_tables[rank], conv_states[rank], recurrent_states[rank], key_pool[rank], value_pool[rank], conv_snapshots[rank], recurrent_snapshots[rank], hidden_out[rank], logits_out[rank], snapshot_mode, data, signal, rank, device=rank)
        resident_tail_update_fwd(hidden_out[rank], tail_rows[rank], tails[rank], tail_status[rank], device=rank)
        packed_greedy_fwd(logits_out[rank], decision_valid[rank], decision_out[rank], rank, device=rank)
        teacher_return_fwd(decision_out[rank], tail_status[rank], teacher_return[rank], device=rank)

@pl.jit
def zero_state(slot_ids: pl.Tensor[[32], pl.INT32], conv_states: pl.InOut[pl.Tensor[[32, 48, 4, 5120], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[32, 48, 24, 128, 128], pl.FP32]]):
    with pl.spmd(32 * 48, name_hint='qwen_zero_slot_state'):
        task = pl.tile.get_block_idx()
        slot = pl.read(slot_ids, [task // 48])
        layer = task % 48
        if slot >= 0:
            for tap in pl.range(4):
                for col in pl.range(40):
                    conv_states[slot:slot + 1, layer:layer + 1, tap:tap + 1, col * 128:col * 128 + 128] = pl.full([1, 1, 1, 128], dtype=pl.BF16, value=0.0)
            for head in pl.range(24):
                for row in pl.range(8):
                    recurrent_states[slot:slot + 1, layer:layer + 1, head:head + 1, row * 16:row * 16 + 16, 0:128] = pl.full([1, 1, 1, 16, 128], dtype=pl.FP32, value=0.0)
    return (conv_states, recurrent_states)

@pl.jit.host
def zero_state_host(slot_ids: pl.Tensor, conv_states: pl.InOut[pl.Tensor], recurrent_states: pl.InOut[pl.Tensor]):
    for rank in pl.range(2):
        zero_state(slot_ids[rank], conv_states[rank], recurrent_states[rank], device=rank)

@pl.jit
def restore_state(slot_ids: pl.Tensor[[32], pl.INT32], accepted_counts: pl.Tensor[[32], pl.INT32], conv_snapshots: pl.Tensor[[4, 32, 48, 4, 5120], pl.BF16], recurrent_snapshots: pl.Tensor[[4, 32, 48, 3072, 128], pl.FP32], conv_states: pl.InOut[pl.Tensor[[32, 48, 4, 5120], pl.BF16]], recurrent_states: pl.InOut[pl.Tensor[[32, 48, 24, 128, 128], pl.FP32]]):
    snapshot_rows = pl.reshape(recurrent_snapshots, [4 * 32 * 48 * 24 * 128, 128])
    with pl.spmd(32 * 48, name_hint='qwen_restore_verified_prefix'):
        task = pl.tile.get_block_idx()
        entry = task // 48
        layer = task % 48
        slot = pl.read(slot_ids, [entry])
        if slot >= 0:
            step = pl.read(accepted_counts, [entry]) - 1
            for tap in pl.range(4):
                for col in pl.range(40):
                    source = pl.slice(conv_snapshots, [1, 1, 1, 1, 128], [step, slot, layer, tap, col * 128])
                    conv_states[slot:slot + 1, layer:layer + 1, tap:tap + 1, col * 128:col * 128 + 128] = pl.reshape(source, [1, 1, 1, 128])
            for head in pl.range(24):
                for row in pl.range(8):
                    recurrent_source = pl.slice(snapshot_rows, [16, 128], [((step * 32 + slot) * 48 + layer) * 3072 + head * 128 + row * 16, 0])
                    recurrent_states[slot:slot + 1, layer:layer + 1, head:head + 1, row * 16:row * 16 + 16, 0:128] = pl.reshape(recurrent_source, [1, 1, 1, 16, 128])
    return (conv_states, recurrent_states)

@pl.jit.host
def restore_host(slot_ids: pl.Tensor, accepted_counts: pl.Tensor, conv_snapshots: pl.Tensor, recurrent_snapshots: pl.Tensor, conv_states: pl.InOut[pl.Tensor], recurrent_states: pl.InOut[pl.Tensor]):
    for rank in pl.range(2):
        restore_state(slot_ids[rank], accepted_counts[rank], conv_snapshots[rank], recurrent_snapshots[rank], conv_states[rank], recurrent_states[rank], device=rank)

@pl.jit
def cache_copy_pages(src_pages: pl.Tensor[[32], pl.INT32], dst_pages: pl.Tensor[[32], pl.INT32], key_pool: pl.InOut[pl.Tensor[[17, 2, POOL, 128, 256], pl.BF16]], value_pool: pl.InOut[pl.Tensor[[17, 2, POOL, 128, 256], pl.BF16]]):
    with pl.spmd(32 * 17 * 2, name_hint='qwen_prefix_page_cow'):
        task = pl.tile.get_block_idx()
        entry = task // 34
        layer = task % 34 // 2
        head = task % 2
        source = pl.read(src_pages, [entry])
        destination = pl.read(dst_pages, [entry])
        if source >= 0 and destination >= 0:
            k = pl.slice(key_pool, [1, 1, 1, 128, 256], [layer, head, source, 0, 0])
            v = pl.slice(value_pool, [1, 1, 1, 128, 256], [layer, head, source, 0, 0])
            key_pool[layer:layer + 1, head:head + 1, destination:destination + 1, 0:128, 0:256] = k
            value_pool[layer:layer + 1, head:head + 1, destination:destination + 1, 0:128, 0:256] = v
    return (key_pool, value_pool)

@pl.jit.host
def cache_copy_pages_host(src_pages: pl.Tensor, dst_pages: pl.Tensor, key_pool: pl.InOut[pl.Tensor], value_pool: pl.InOut[pl.Tensor]):
    for rank in pl.range(2):
        cache_copy_pages(src_pages[rank], dst_pages[rank], key_pool[rank], value_pool[rank], device=rank)

# Device greedy v71: independent dynamic logits extent, no hidden transfer/TP window reset.
# Sampler shares producer L; no independent HOST shape specialization.

@pl.jit.inline(auto_scope=False)
def greedy_local_inline(logits: pl.Tensor[[L, 124160], pl.BF16], row_valid: pl.Tensor[[L], pl.INT32], token_ids: pl.Tensor[[L, 16], pl.INT32], scores_out: pl.Tensor[[L, 16], pl.FP32], status_out: pl.Tensor[[L, 16], pl.INT32], rank: pl.Scalar[pl.INT32]):
    rows = pl.tensor.dim(logits, 0)
    partial_scores = pl.create_tensor([rows * 128, 16], dtype=pl.FP32)
    partial_ids = pl.create_tensor([rows * 128, 16], dtype=pl.INT32)
    partial_status = pl.create_tensor([rows * 128, 16], dtype=pl.INT32)
    for worker in pl.spmd(64, name_hint='qwen_greedy_local_chunks'):
        for task in pl.range(worker, rows * 128, 64):
            row = task // 128
            chunk = task % 128
            flag = pl.cast(0, pl.INT32)
            token = pl.cast(-1, pl.INT32)
            value = pl.cast(0.0, pl.FP32)
            if pl.read(row_valid, [row]) != 0:
                if chunk < 122:
                    if chunk < 121:
                        block_flat = pl.cast(pl.slice(logits, [1, 1024], [row, chunk * 1024]), pl.FP32)
                        block_x = pl.reshape(block_flat, [8, 128])
                        block_finite = pl.row_min(pl.cmp(pl.abs(block_x), 3.3895313892515355e38, cmp_type=3))
                        block_max = pl.row_max(block_x)
                        block_broadcast = pl.row_expand_add(pl.full([8, 128], dtype=pl.FP32, value=0.0), block_max)
                        block_hit = pl.cast(pl.cmp(block_x, block_broadcast, cmp_type=0), pl.INT32)
                        block_ramp = pl.col_expand(pl.full([8, 128], dtype=pl.INT32, value=0), pl.arange(0, [1, 128], dtype=pl.INT32))
                        block_candidates = pl.add(pl.mul(block_hit, pl.sub(block_ramp, 2048)), 2048)
                        block_winners = pl.row_min(block_candidates)
                        finite = pl.cast(1.0, pl.FP32)
                        best = pl.cast(-3.4028234663852886e38, pl.FP32)
                        winner = pl.cast(2147483647, pl.INT32)
                        for lane in pl.range(8):
                            lane_finite_block = pl.read(block_finite, [lane, 0])
                            if lane_finite_block < finite:
                                finite = lane_finite_block
                            lane_value = pl.read(block_max, [lane, 0])
                            lane_token = pl.cast(lane * 128 + pl.read(block_winners, [lane, 0]), pl.INT32)
                            if lane_value > best:
                                best = lane_value
                                winner = lane_token
                            else:
                                if lane_value == best:
                                    winner = pl.min(winner, lane_token)
                    else:
                        tail_flat = pl.cast(pl.slice(logits, [1, 256], [row, 123904]), pl.FP32)
                        tail_x = pl.reshape(tail_flat, [8, 32])
                        tail_finite = pl.row_min(pl.cmp(pl.abs(tail_x), 3.3895313892515355e38, cmp_type=3))
                        tail_max = pl.row_max(tail_x)
                        tail_broadcast = pl.row_expand_add(pl.full([8, 32], dtype=pl.FP32, value=0.0), tail_max)
                        tail_hit = pl.cast(pl.cmp(tail_x, tail_broadcast, cmp_type=0), pl.INT32)
                        tail_ramp = pl.col_expand(pl.full([8, 32], dtype=pl.INT32, value=0), pl.arange(0, [1, 32], dtype=pl.INT32))
                        tail_candidates = pl.add(pl.mul(tail_hit, pl.sub(tail_ramp, 2048)), 2048)
                        tail_winners = pl.row_min(tail_candidates)
                        finite = pl.cast(1.0, pl.FP32)
                        best = pl.cast(-3.4028234663852886e38, pl.FP32)
                        winner = pl.cast(2147483647, pl.INT32)
                        for lane in pl.range(8):
                            lane_finite_tail = pl.read(tail_finite, [lane, 0])
                            if lane_finite_tail < finite:
                                finite = lane_finite_tail
                            lane_value = pl.read(tail_max, [lane, 0])
                            lane_token = pl.cast(lane * 32 + pl.read(tail_winners, [lane, 0]), pl.INT32)
                            if lane_value > best:
                                best = lane_value
                                winner = lane_token
                            else:
                                if lane_value == best:
                                    winner = pl.min(winner, lane_token)
                    flag = pl.cast(2, pl.INT32)
                    if finite == 1.0:
                        flag = pl.cast(1, pl.INT32)
                        token = pl.cast(rank * 124160 + chunk * 1024 + winner, pl.INT32)
                        value = best
            score_pad = pl.full([1, 16], dtype=pl.FP32, value=0.0)
            id_pad = pl.full([1, 16], dtype=pl.INT32, value=-1)
            flag_pad = pl.full([1, 16], dtype=pl.INT32, value=0)
            pl.write(score_pad, [0, 0], value)
            pl.write(id_pad, [0, 0], token)
            pl.write(flag_pad, [0, 0], flag)
            partial_scores[task:task + 1, :] = score_pad
            partial_ids[task:task + 1, :] = id_pad
            partial_status[task:task + 1, :] = flag_pad
    for worker in pl.spmd(32, name_hint='qwen_greedy_local_merge'):
        for row in pl.range(worker, rows, 32):
            best_score = pl.cast(-3.4028234663852886e38, pl.FP32)
            best_id = pl.cast(2147483647, pl.INT32)
            bad = pl.cast(0, pl.INT32)
            for chunk in pl.range(128):
                f = pl.read(partial_status, [row * 128 + chunk, 0])
                if f == 2:
                    bad = pl.cast(1, pl.INT32)
                if f == 1:
                    v = pl.read(partial_scores, [row * 128 + chunk, 0])
                    t = pl.read(partial_ids, [row * 128 + chunk, 0])
                    if v > best_score:
                        best_score = v
                        best_id = t
                    else:
                        if v == best_score:
                            best_id = pl.min(best_id, t)
            ids = pl.full([1, 16], dtype=pl.INT32, value=-1)
            vals = pl.full([1, 16], dtype=pl.FP32, value=0.0)
            flags = pl.full([1, 16], dtype=pl.INT32, value=0)
            if pl.read(row_valid, [row]) != 0:
                pl.write(flags, [0, 0], pl.cast(2, pl.INT32))
                if bad == 0:
                    if best_id < 248320:
                        pl.write(ids, [0, 0], best_id)
                        pl.write(vals, [0, 0], best_score)
                        pl.write(flags, [0, 0], pl.cast(1, pl.INT32))
            token_ids[row:row + 1, :] = ids
            scores_out[row:row + 1, :] = vals
            status_out[row:row + 1, :] = flags
    return token_ids, scores_out, status_out

@pl.jit(auto_scope=False)
def packed_greedy_fwd(logits: pl.Tensor[[L, 124160], pl.BF16], row_valid: pl.Tensor[[L], pl.INT32], decision_out: pl.Out[pl.Tensor[[L, 16], pl.FP32]], rank: pl.Scalar[pl.INT32]):
    rows = pl.tensor.dim(logits, 0)
    token_ids = pl.create_tensor([rows, 16], dtype=pl.INT32)
    scores = pl.create_tensor([rows, 16], dtype=pl.FP32)
    status = pl.create_tensor([rows, 16], dtype=pl.INT32)
    # Same finite/vocab-tail/local-tie reduction as the validated standalone path.
    greedy_local_inline(logits, row_valid, token_ids, scores, status, rank)
    for worker in pl.spmd(32, name_hint='qwen_pack_local_decision'):
        for row in pl.range(worker, rows, 32):
            record = pl.full([1, 16], dtype=pl.FP32, value=0.0)
            # Token <248320 and -1, status0/1/2 are exactly representable FP32.
            pl.write(record, [0, 0], pl.cast(pl.read(token_ids, [row, 0]), pl.FP32))
            pl.write(record, [0, 1], pl.read(scores, [row, 0]))
            pl.write(record, [0, 2], pl.cast(pl.read(status, [row, 0]), pl.FP32))
            decision_out[row:row + 1, :] = record
    return decision_out

"""Unadopted Device copy proposal for accepted-prefix teacher inputs.

No model math or state commit; caller owns both buffers until completion.
Source and destination extents are independent. Never imported by production.
"""

V72_GATHER_SOURCE_ROWS = pl.dynamic('QWEN_RESIDENT_SOURCE_ROWS')
V72_PREVIOUS_ROWS = pl.dynamic('QWEN_RESIDENT_GATHER_ROWS')


@pl.jit.inline(auto_scope=False)
def resident_hidden_gather_inline(
    source: pl.Tensor[[V72_GATHER_SOURCE_ROWS, 5120], pl.BF16],
    source_rows: pl.Tensor[[V72_PREVIOUS_ROWS], pl.INT32],
    previous: pl.Tensor[[V72_PREVIOUS_ROWS, 5120], pl.BF16],
    status: pl.Tensor[[V72_PREVIOUS_ROWS, 16], pl.INT32],
):
    count = pl.tensor.dim(previous, 0)
    source_count = pl.tensor.dim(source, 0)
    for worker in pl.spmd(64, name_hint='resident_teacher_gather'):
        for row in pl.range(worker, count, 64):
            index = pl.read(source_rows, [row])
            value = pl.full([1, 5120], dtype=pl.BF16, value=0.0)
            flags = pl.full([1, 16], dtype=pl.INT32, value=0)
            if index >= 0:
                pl.write(flags, [0, 0], pl.cast(2, pl.INT32))
                if index < source_count:
                    value = pl.slice(source, [1, 5120], [index, 0])
                    pl.write(flags, [0, 0], pl.cast(1, pl.INT32))
            else:
                if index < -1:
                    pl.write(flags, [0, 0], pl.cast(2, pl.INT32))
            previous[row:row + 1, :] = value
            status[row:row + 1, :] = flags
    return previous, status


@pl.jit(auto_scope=False)
def resident_hidden_gather_fwd(
    source: pl.Tensor[[V72_GATHER_SOURCE_ROWS, 5120], pl.BF16],
    source_rows: pl.Tensor[[V72_PREVIOUS_ROWS], pl.INT32],
    previous: pl.Out[pl.Tensor[[V72_PREVIOUS_ROWS, 5120], pl.BF16]],
    status: pl.Out[pl.Tensor[[V72_PREVIOUS_ROWS, 16], pl.INT32]],
):
    resident_hidden_gather_inline(source, source_rows, previous, status)
    return previous, status


@pl.jit.host
def resident_hidden_gather_host(
    source: pl.Tensor[[2, V72_GATHER_SOURCE_ROWS, 5120], pl.BF16],
    source_rows: pl.Tensor[[2, V72_PREVIOUS_ROWS], pl.INT32],
    previous: pl.Out[pl.Tensor[[2, V72_PREVIOUS_ROWS, 5120], pl.BF16]],
    status: pl.Out[pl.Tensor[[2, V72_PREVIOUS_ROWS, 16], pl.INT32]],
):
    for rank in pl.range(2):
        resident_hidden_gather_fwd(source[rank], source_rows[rank], previous[rank], status[rank], device=rank)

"""Unadopted bounded per-slot tail update; inactive slots retain their data.

The Host validates a unique request generation for each fixed destination slot.
Source rowmap is immutable for this blocking call; -1 means preserve the slot.
No allocation, model execution, or production import occurs in this module.
"""

V72_TAIL_SOURCE_ROWS = pl.dynamic('QWEN_RESIDENT_TAIL_SOURCE_ROWS')


@pl.jit.inline(auto_scope=False)
def resident_tail_update_inline(
    source: pl.Tensor[[V72_TAIL_SOURCE_ROWS, 5120], pl.BF16],
    source_rows: pl.Tensor[[32], pl.INT32],
    tails: pl.InOut[pl.Tensor[[32, 5120], pl.BF16]],
    status: pl.Tensor[[32, 16], pl.INT32],
):
    source_count = pl.tensor.dim(source, 0)
    for slot in pl.spmd(32, name_hint='resident_tail_update'):
        index = pl.read(source_rows, [slot])
        flags = pl.full([1, 16], dtype=pl.INT32, value=0)
        if index >= 0:
            pl.write(flags, [0, 0], pl.cast(2, pl.INT32))
            if index < source_count:
                tails[slot:slot + 1, :] = pl.slice(source, [1, 5120], [index, 0])
                pl.write(flags, [0, 0], pl.cast(1, pl.INT32))
        else:
            if index < -1:
                pl.write(flags, [0, 0], pl.cast(2, pl.INT32))
        status[slot:slot + 1, :] = flags
    return tails, status


@pl.jit(auto_scope=False)
def resident_tail_update_fwd(
    source: pl.Tensor[[V72_TAIL_SOURCE_ROWS, 5120], pl.BF16],
    source_rows: pl.Tensor[[32], pl.INT32],
    tails: pl.InOut[pl.Tensor[[32, 5120], pl.BF16]],
    status: pl.Out[pl.Tensor[[32, 16], pl.INT32]],
):
    resident_tail_update_inline(source, source_rows, tails, status)
    return tails, status


@pl.jit.host
def resident_tail_update_host(
    source: pl.Tensor[[2, V72_TAIL_SOURCE_ROWS, 5120], pl.BF16],
    source_rows: pl.Tensor[[2, 32], pl.INT32],
    tails: pl.InOut[pl.Tensor[[2, 32, 5120], pl.BF16]],
    status: pl.Out[pl.Tensor[[2, 32, 16], pl.INT32]],
):
    for rank in pl.range(2):
        resident_tail_update_fwd(source[rank], source_rows[rank], tails[rank], status[rank], device=rank)


@pl.jit(auto_scope=False)
def prefix_restore_state(
    slot_ids: pl.Tensor[[32], pl.INT32],
    conv_source: pl.Tensor[[48, 4, 5120], pl.BF16],
    recurrent_source: pl.Tensor[[48, 24, 128, 128], pl.FP32],
    conv_states: pl.InOut[pl.Tensor[[32, 48, 4, 5120], pl.BF16]],
    recurrent_states: pl.InOut[pl.Tensor[[32, 48, 24, 128, 128], pl.FP32]],
):
    with pl.spmd(48, name_hint='prefix_device_state_restore'):
        layer = pl.tile.get_block_idx()
        slot = pl.read(slot_ids, [0])
        if slot >= 0:
            if slot < 32:
                for tap in pl.range(4):
                    for col in pl.range(40):
                        conv_source_tile = pl.slice(conv_source, [1, 1, 128], [layer, tap, col * 128])
                        conv_states[slot:slot + 1, layer:layer + 1, tap:tap + 1, col * 128:col * 128 + 128] = pl.reshape(conv_source_tile, [1, 1, 1, 128])
                for head in pl.range(24):
                    for row in pl.range(8):
                        rec_source_tile = pl.slice(recurrent_source, [1, 1, 16, 128], [layer, head, row * 16, 0])
                        recurrent_states[slot:slot + 1, layer:layer + 1, head:head + 1, row * 16:row * 16 + 16, 0:128] = pl.reshape(rec_source_tile, [1, 1, 1, 16, 128])
    return conv_states, recurrent_states


@pl.jit.host
def prefix_restore_host(
    slot_ids: pl.Tensor[[2, 32], pl.INT32],
    conv_source: pl.Tensor[[2, 48, 4, 5120], pl.BF16],
    recurrent_source: pl.Tensor[[2, 48, 24, 128, 128], pl.FP32],
    conv_states: pl.InOut[pl.Tensor[[2, 32, 48, 4, 5120], pl.BF16]],
    recurrent_states: pl.InOut[pl.Tensor[[2, 32, 48, 24, 128, 128], pl.FP32]],
):
    for rank in pl.range(2):
        prefix_restore_state(slot_ids[rank], conv_source[rank], recurrent_source[rank], conv_states[rank], recurrent_states[rank], device=rank)


@pl.jit(auto_scope=False)
def teacher_return_fwd(
    decision: pl.Tensor[[L, 16], pl.FP32],
    tail_status: pl.Tensor[[32, 16], pl.INT32],
    packet: pl.Out[pl.Tensor[[TEACHER_RETURN_R, 16], pl.FP32]],
):
    logits = pl.tensor.dim(decision, 0)
    for worker in pl.spmd(64, name_hint='teacher_return_packet_source2'):
        for row in pl.range(worker, pl.tensor.dim(packet, 0), 64):
            value = pl.full([1, 16], dtype=pl.FP32, value=0.0)
            if row == 0:
                pl.write(value, [0, 0], pl.cast(pl.cast(100, pl.INT32), pl.FP32))
                pl.write(value, [0, 1], pl.cast(pl.cast(logits, pl.INT32), pl.FP32))
                pl.write(value, [0, 2], pl.cast(pl.cast(32, pl.INT32), pl.FP32))
            else:
                if row < 1 + logits:
                    value = pl.slice(decision, [1, 16], [row - 1, 0])
                else:
                    if row < 1 + logits + 32:
                        value = pl.cast(pl.slice(tail_status, [1, 16], [row - 1 - logits, 0]), pl.FP32)
            packet[row:row + 1, :] = value
    return packet
