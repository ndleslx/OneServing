"""Read the real Qwen3.8-27B text checkpoint into PyPTO weight banks."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open


HIDDEN = 5120
INTERMEDIATE = 17408
QKV = 10240
VALUE_WIDTH = 6144
FULL_HEAD_DIM = 256
EXPECTED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_DIGESTS = {
    "config.json": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    "model.safetensors.index.json": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
}


def pack_nz_matrix(logical: torch.Tensor) -> torch.Tensor:
    """Pack final TP-local BF16 rows/cols once; each layer stays a separate matrix."""
    if logical.dtype != torch.bfloat16 or logical.ndim != 2:
        raise ValueError('NZ gate requires final 2D BF16 TP-local matrix')
    rows, cols = logical.shape
    c0 = 32 // logical.element_size()
    if rows % 16 or cols % c0:
        raise ValueError('NZ gate requires padded 16-row/C0-column shape')
    return logical.reshape(rows//16,16,cols//c0,c0).permute(2,0,1,3).contiguous().reshape(rows,cols)


def verify_model_identity(root: Path) -> None:
    metadata = root / ".cache/huggingface/download/config.json.metadata"
    if metadata.read_text().splitlines()[0] != EXPECTED_REVISION:
        raise ValueError("Mounted model revision differs from deployment contract")
    for filename, expected in EXPECTED_DIGESTS.items():
        digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Mounted model identity differs: {filename}")


class TextWeights:
    def __init__(self, root: Path) -> None:
        self.root = root
        verify_model_identity(root)
        config = json.loads((root / "config.json").read_text())
        text = config["text_config"]
        expected = {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": FULL_HEAD_DIM,
            "vocab_size": 248320,
            "dtype": "bfloat16",
        }
        for key, value in expected.items():
            if text.get(key) != value:
                raise ValueError(f"Unexpected {key}: {text.get(key)!r}")
        expected_types = ["linear_attention"] * 3 + ["full_attention"]
        if text["layer_types"] != expected_types * 16:
            raise ValueError("Unexpected Qwen3.5 layer order")
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.all_layers = range(64)
        self.linear_layers = [index for index in self.all_layers if index % 4 != 3]
        self.full_layers = [index for index in self.all_layers if index % 4 == 3]

    def read(self, name: str) -> torch.Tensor:
        filename = self.index.get(name)
        if filename is None:
            raise KeyError(f"Checkpoint is missing {name}")
        with safe_open(self.root / filename, framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(name)
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 checkpoint tensor: {name}")
        return tensor.contiguous()

    def layer(self, index: int, tail: str) -> torch.Tensor:
        return self.read(f"model.language_model.layers.{index}.{tail}")

    def stack(self, indices: range | list[int], tail: str, transform=None) -> torch.Tensor:
        tensors = [self.layer(index, tail) for index in indices]
        if transform is not None:
            tensors = [transform(tensor) for tensor in tensors]
        return torch.cat(tensors, dim=0).contiguous()

    def banks(self) -> Iterator[tuple[str, torch.Tensor]]:
        gamma = lambda value: (1.0 + value.float()).reshape(1, -1)
        pad_gate = lambda value: torch.nn.functional.pad(value, (0, 0, 0, 128 - value.shape[0]))

        def pad_parameter(value: torch.Tensor) -> torch.Tensor:
            result = torch.zeros((1, 64), dtype=torch.bfloat16)
            result[0, :value.numel()] = value
            return result

        yield "embedding", self.read("model.language_model.embed_tokens.weight")
        yield "final_gamma", gamma(self.read("model.language_model.norm.weight"))
        yield "lm_head", self.read("lm_head.weight")
        yield "input_gamma", self.stack(self.all_layers, "input_layernorm.weight", gamma)
        yield "post_gamma", self.stack(self.all_layers, "post_attention_layernorm.weight", gamma)
        for bank, tail in (
            ("mlp_gate", "mlp.gate_proj.weight"),
            ("mlp_up", "mlp.up_proj.weight"),
            ("mlp_down", "mlp.down_proj.weight"),
        ):
            yield bank, self.stack(self.all_layers, tail)
        for bank, tail, transform in (
            ("delta_qkv", "linear_attn.in_proj_qkv.weight", None),
            ("delta_z", "linear_attn.in_proj_z.weight", None),
            ("delta_a", "linear_attn.in_proj_a.weight", pad_gate),
            ("delta_b", "linear_attn.in_proj_b.weight", pad_gate),
            ("delta_conv", "linear_attn.conv1d.weight",
             lambda value: value.squeeze(1).transpose(0, 1).contiguous()),
            ("delta_a_log", "linear_attn.A_log", pad_parameter),
            ("delta_dt_bias", "linear_attn.dt_bias", pad_parameter),
            ("delta_norm", "linear_attn.norm.weight", lambda value: value.reshape(1, 128)),
            ("delta_out", "linear_attn.out_proj.weight", None),
        ):
            yield bank, self.stack(self.linear_layers, tail, transform)
        for bank, tail, transform in (
            ("full_q", "self_attn.q_proj.weight", None),
            ("full_k", "self_attn.k_proj.weight", None),
            ("full_v", "self_attn.v_proj.weight", None),
            ("full_q_gamma", "self_attn.q_norm.weight", gamma),
            ("full_k_gamma", "self_attn.k_norm.weight", gamma),
            ("full_out", "self_attn.o_proj.weight", None),
        ):
            yield bank, self.stack(self.full_layers, tail, transform)

    @staticmethod
    def tp_slice(value: torch.Tensor, kind: str, rank: int) -> torch.Tensor:
        """Own one of two contiguous head/intermediate shards, before padding.

        Qwen GDN Q/K/V occupy separate contiguous segments. Full-attention
        Q/gate are interleaved within each head, so splitting whole heads
        preserves their checkpoint layout.
        """
        if rank not in (0, 1):
            raise ValueError("This model contract requires TP=2")
        if kind == "replicated":
            return value.contiguous()
        if kind == "qkv":
            segments = value.split((2048, 2048, 6144), dim=0)
            return torch.cat([x.chunk(2, dim=0)[rank] for x in segments], dim=0).contiguous()
        if kind == "column":
            return value.chunk(2, dim=0)[rank].contiguous()
        if kind == "row":
            return value.chunk(2, dim=1)[rank].contiguous()
        raise ValueError(f"Unknown TP layout: {kind}")

    def sharded_banks(self, rank: int) -> Iterator[tuple[str, torch.Tensor]]:
        """Stream only this rank's banks, including the checkpoint MTP layer."""
        if rank not in (0, 1):
            raise ValueError("This model contract requires TP=2")
        gamma = lambda value: (1.0 + value.float()).reshape(1, -1)

        def bank(indices, tail, kind="replicated", transform=None):
            pieces = []
            for index in indices:
                value = self.tp_slice(self.layer(index, tail), kind, rank)
                pieces.append(transform(value) if transform else value)
            return torch.cat(pieces, dim=0).contiguous()

        def pad_projection(value):
            return torch.nn.functional.pad(value, (0, 0, 0, 128 - value.shape[0]))

        def parameter(value):
            return torch.nn.functional.pad(value.reshape(1, -1), (0, 32 - value.numel()))

        yield "embedding", self.read("model.language_model.embed_tokens.weight")
        yield "final_gamma", gamma(self.read("model.language_model.norm.weight"))
        yield "lm_head", self.tp_slice(self.read("lm_head.weight"), "column", rank)
        yield "input_gamma", bank(self.all_layers, "input_layernorm.weight", transform=gamma)
        yield "post_gamma", bank(self.all_layers, "post_attention_layernorm.weight", transform=gamma)
        for name, tail, kind in (
            ("mlp_gate", "mlp.gate_proj.weight", "column"),
            ("mlp_up", "mlp.up_proj.weight", "column"),
            ("mlp_down", "mlp.down_proj.weight", "row"),
            ("delta_qkv", "linear_attn.in_proj_qkv.weight", "qkv"),
            ("delta_z", "linear_attn.in_proj_z.weight", "column"),
            ("delta_out", "linear_attn.out_proj.weight", "row"),
        ):
            indices = self.all_layers if name.startswith("mlp") else self.linear_layers
            if name in ("mlp_gate", "mlp_up"):
                # The leading layer axis is semantic: never flatten before NZ packing.
                packed_layers = torch.empty((len(indices), 8704, 5120), dtype=torch.bfloat16)
                for local_layer, index in enumerate(indices):
                    local = self.tp_slice(self.layer(index, tail), kind, rank)
                    if tuple(local.shape) != (8704, 5120):
                        raise ValueError('NZ gate TP-local shape mismatch')
                    packed_layers[local_layer].copy_(pack_nz_matrix(local))
                yield name, packed_layers
            else:
                yield name, bank(indices, tail, kind)
        for name, tail in (("delta_a", "in_proj_a.weight"), ("delta_b", "in_proj_b.weight")):
            yield name, bank(self.linear_layers, "linear_attn." + tail, "column", pad_projection)
        yield "delta_conv", bank(
            self.linear_layers, "linear_attn.conv1d.weight", "qkv",
            lambda value: value.squeeze(1).transpose(0, 1).contiguous())
        for name, tail in (("delta_a_log", "A_log"), ("delta_dt_bias", "dt_bias")):
            yield name, bank(self.linear_layers, "linear_attn." + tail, "column", parameter)
        yield "delta_norm", bank(self.linear_layers, "linear_attn.norm.weight",
                                  transform=lambda value: value.reshape(1, 128))
        for name, tail, kind, transform in (
            ("full_q", "q_proj.weight", "column", None),
            ("full_k", "k_proj.weight", "column", None),
            ("full_v", "v_proj.weight", "column", None),
            ("full_q_gamma", "q_norm.weight", "replicated", gamma),
            ("full_k_gamma", "k_norm.weight", "replicated", gamma),
            ("full_out", "o_proj.weight", "row", None),
        ):
            yield name, bank(self.full_layers, "self_attn." + tail, kind, transform)
        # FC consumes [normalized_embedding, normalized_hidden]. Its input-axis
        # shard is reduced in FP32, just like attention/MLP row projections.
        yield "mtp_fc", self.tp_slice(self.read("mtp.fc.weight"), "row", rank)
        for name, path in (
            ("mtp_embedding_gamma", "mtp.pre_fc_norm_embedding.weight"),
            ("mtp_hidden_gamma", "mtp.pre_fc_norm_hidden.weight"),
            ("mtp_final_gamma", "mtp.norm.weight"),
            ("mtp_input_gamma", "mtp.layers.0.input_layernorm.weight"),
            ("mtp_post_gamma", "mtp.layers.0.post_attention_layernorm.weight"),
            ("mtp_q_gamma", "mtp.layers.0.self_attn.q_norm.weight"),
            ("mtp_k_gamma", "mtp.layers.0.self_attn.k_norm.weight"),
        ):
            yield name, gamma(self.read(path))
        for name, tail, kind in (
            ("mtp_q", "self_attn.q_proj.weight", "column"),
            ("mtp_k", "self_attn.k_proj.weight", "column"),
            ("mtp_v", "self_attn.v_proj.weight", "column"),
            ("mtp_out", "self_attn.o_proj.weight", "row"),
            ("mtp_mlp_gate", "mlp.gate_proj.weight", "column"),
            ("mtp_mlp_up", "mlp.up_proj.weight", "column"),
            ("mtp_mlp_down", "mlp.down_proj.weight", "row"),
        ):
            yield name, self.tp_slice(self.read("mtp.layers.0." + tail), kind, rank)


# Fixed rank-local ABI, independently checked against the checkpoint.
TP_BANK_SPECS = (('embedding', (248320, 5120), 'bfloat16'),
 ('final_gamma', (1, 5120), 'float32'),
 ('lm_head', (124160, 5120), 'bfloat16'),
 ('input_gamma', (64, 5120), 'float32'),
 ('post_gamma', (64, 5120), 'float32'),
 ('mlp_gate', (64, 8704, 5120), 'bfloat16'),
 ('mlp_up', (64, 8704, 5120), 'bfloat16'),
 ('mlp_down', (327680, 8704), 'bfloat16'),
 ('delta_qkv', (245760, 5120), 'bfloat16'),
 ('delta_z', (147456, 5120), 'bfloat16'),
 ('delta_out', (245760, 3072), 'bfloat16'),
 ('delta_a', (6144, 5120), 'bfloat16'),
 ('delta_b', (6144, 5120), 'bfloat16'),
 ('delta_conv', (192, 5120), 'bfloat16'),
 ('delta_a_log', (48, 32), 'bfloat16'),
 ('delta_dt_bias', (48, 32), 'bfloat16'),
 ('delta_norm', (48, 128), 'bfloat16'),
 ('full_q', (98304, 5120), 'bfloat16'),
 ('full_k', (8192, 5120), 'bfloat16'),
 ('full_v', (8192, 5120), 'bfloat16'),
 ('full_q_gamma', (16, 256), 'float32'),
 ('full_k_gamma', (16, 256), 'float32'),
 ('full_out', (81920, 3072), 'bfloat16'),
 ('mtp_fc', (5120, 5120), 'bfloat16'),
 ('mtp_embedding_gamma', (1, 5120), 'float32'),
 ('mtp_hidden_gamma', (1, 5120), 'float32'),
 ('mtp_final_gamma', (1, 5120), 'float32'),
 ('mtp_input_gamma', (1, 5120), 'float32'),
 ('mtp_post_gamma', (1, 5120), 'float32'),
 ('mtp_q_gamma', (1, 256), 'float32'),
 ('mtp_k_gamma', (1, 256), 'float32'),
 ('mtp_q', (6144, 5120), 'bfloat16'),
 ('mtp_k', (512, 5120), 'bfloat16'),
 ('mtp_v', (512, 5120), 'bfloat16'),
 ('mtp_out', (5120, 3072), 'bfloat16'),
 ('mtp_mlp_gate', (8704, 5120), 'bfloat16'),
 ('mtp_mlp_up', (8704, 5120), 'bfloat16'),
 ('mtp_mlp_down', (5120, 8704), 'bfloat16'))
