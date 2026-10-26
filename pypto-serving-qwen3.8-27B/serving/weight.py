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
