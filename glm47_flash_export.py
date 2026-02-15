#!/usr/bin/env python3
"""Export GLM-4.7-Flash GGUF into a runtime .bin package.

Current scope:
- Exports real tensors for a baseline path:
  - token_embd.weight (runtime Q8 path)
  - output_norm.weight (fp32)
  - output.weight (runtime Q8 path)
  - layer-0 MLA attention (attn_norm/q_a/q_a_norm/q_b/kv_a_mqa/kv_a_norm/k_b/v_b/attn_output)
  - layer-0 dense FFN (ffn_norm/gate/up/down)
- Appends per-layer MLA attention blocks for all layers (for future full-layer runtime loop).
- Exports tokenizer/template sidecars.
- Keeps strict metadata/tensor checks.

This removes the previous metadata-only stub and enables a real weight-driven
baseline forward pass in C.
"""

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "llama.cpp" / "gguf-py"))
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

MAGIC = 0x67343763  # "g47c"
VERSION = 4
# Smaller quantization groups reduce re-quantization error vs GGUF at the cost
# of larger runtime checkpoint size.
GROUP_SIZE = 16
HEADER_SIZE = 256


@dataclass
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    rope_dim: int
    q_lora_rank: int
    kv_lora_rank: int
    kv_mqa_width: int
    n_routed_experts: int
    n_experts_used: int
    n_shared_experts: int
    bos: int
    eos: int


def require_field(reader: GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"missing required metadata field: {key}")
    return field.contents()


def first_present(reader: GGUFReader, keys: list[str], default=None):
    for key in keys:
        field = reader.get_field(key)
        if field is not None:
            return field.contents()
    return default


def tensor_map(reader: GGUFReader):
    return {t.name: t for t in reader.tensors}


def get_cfg(reader: GGUFReader) -> Config:
    arch = require_field(reader, "general.architecture")
    if arch != "deepseek2":
        raise ValueError(f"unexpected architecture {arch!r}, expected 'deepseek2'")

    vocab_field = first_present(
        reader, ["deepseek2.vocab_size", "tokenizer.ggml.tokens"], 0
    )
    vocab_size = len(vocab_field) if isinstance(vocab_field, list) else int(vocab_field)

    return Config(
        dim=int(require_field(reader, "deepseek2.embedding_length")),
        hidden_dim=int(require_field(reader, "deepseek2.feed_forward_length")),
        n_layers=int(require_field(reader, "deepseek2.block_count")),
        n_heads=int(require_field(reader, "deepseek2.attention.head_count")),
        n_kv_heads=int(require_field(reader, "deepseek2.attention.head_count_kv")),
        vocab_size=vocab_size,
        seq_len=int(require_field(reader, "deepseek2.context_length")),
        rope_dim=int(require_field(reader, "deepseek2.rope.dimension_count")),
        q_lora_rank=int(require_field(reader, "deepseek2.attention.q_lora_rank")),
        kv_lora_rank=int(require_field(reader, "deepseek2.attention.kv_lora_rank")),
        kv_mqa_width=int(require_field(reader, "deepseek2.attention.key_length")),
        n_routed_experts=int(require_field(reader, "deepseek2.expert_count")),
        n_experts_used=int(require_field(reader, "deepseek2.expert_used_count")),
        n_shared_experts=int(require_field(reader, "deepseek2.expert_shared_count")),
        bos=int(first_present(reader, ["tokenizer.ggml.bos_token_id"], 0)),
        eos=int(first_present(reader, ["tokenizer.ggml.eos_token_id"], 0)),
    )


def validate_tensors(reader: GGUFReader, cfg: Config):
    names = {t.name for t in reader.tensors}

    for key in ["token_embd.weight", "output_norm.weight", "output.weight"]:
        if key not in names:
            raise ValueError(f"missing required tensor: {key}")

    for key in [
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ]:
        if key not in names:
            raise ValueError(f"missing required tensor: {key}")

    # Keep strict per-layer checks for architecture sanity.
    for i in range(cfg.n_layers):
        layer_keys = [
            f"blk.{i}.attn_norm.weight",
            f"blk.{i}.attn_q_a.weight",
            f"blk.{i}.attn_q_a_norm.weight",
            f"blk.{i}.attn_q_b.weight",
            f"blk.{i}.attn_kv_a_mqa.weight",
            f"blk.{i}.attn_kv_a_norm.weight",
            f"blk.{i}.attn_k_b.weight",
            f"blk.{i}.attn_v_b.weight",
            f"blk.{i}.attn_output.weight",
            f"blk.{i}.ffn_norm.weight",
        ]
        if i > 0:
            layer_keys.extend(
                [
                    f"blk.{i}.ffn_gate_inp.weight",
                    f"blk.{i}.exp_probs_b.bias",
                    f"blk.{i}.ffn_gate_exps.weight",
                    f"blk.{i}.ffn_up_exps.weight",
                    f"blk.{i}.ffn_down_exps.weight",
                    f"blk.{i}.ffn_gate_shexp.weight",
                    f"blk.{i}.ffn_up_shexp.weight",
                    f"blk.{i}.ffn_down_shexp.weight",
                ]
            )
        for key in layer_keys:
            if key not in names:
                raise ValueError(f"missing required tensor: {key}")


def quantize_q80_rows(rows: np.ndarray, group_size: int = GROUP_SIZE):
    if rows.dtype != np.float32:
        rows = rows.astype(np.float32, copy=False)
    if rows.shape[1] % group_size != 0:
        raise ValueError(
            f"row width {rows.shape[1]} is not divisible by group size {group_size}"
        )

    r = rows.reshape(rows.shape[0], -1, group_size)
    wmax = np.max(np.abs(r), axis=2)
    scale = np.where(wmax == 0.0, 1.0, wmax / 127.0).astype(np.float32)
    q = np.round(r / scale[:, :, None]).clip(-127, 127).astype(np.int8)
    return q.reshape(rows.shape[0], rows.shape[1]), scale


def pack_runtime_rows(tensor, rows: np.ndarray, group_size: int = GROUP_SIZE):
    del tensor
    rows = rows.astype(np.float32, copy=False)
    if rows.ndim != 2:
        raise ValueError(f"expected 2D rows, got shape {rows.shape}")
    return quantize_q80_rows(rows, group_size=group_size)


def prepare_layer_attention_quantized(
    tm: dict,
    cfg: Config,
    layer_idx: int,
    expected_dims: Optional[Tuple[int, int, int]] = None,
):
    pfx = f"blk.{layer_idx}"
    attn_norm_t = tm[f"{pfx}.attn_norm.weight"]
    q_a_norm_t = tm[f"{pfx}.attn_q_a_norm.weight"]
    kv_a_norm_t = tm[f"{pfx}.attn_kv_a_norm.weight"]
    q_a_t = tm[f"{pfx}.attn_q_a.weight"]
    q_b_t = tm[f"{pfx}.attn_q_b.weight"]
    kv_a_t = tm[f"{pfx}.attn_kv_a_mqa.weight"]
    k_b_t = tm[f"{pfx}.attn_k_b.weight"]
    v_b_t = tm[f"{pfx}.attn_v_b.weight"]
    attn_out_t = tm[f"{pfx}.attn_output.weight"]

    attn_norm = dequantize(attn_norm_t.data, attn_norm_t.tensor_type).astype(np.float32)
    q_a_norm = dequantize(q_a_norm_t.data, q_a_norm_t.tensor_type).astype(np.float32)
    kv_a_norm = dequantize(kv_a_norm_t.data, kv_a_norm_t.tensor_type).astype(np.float32)

    q_a = dequantize(q_a_t.data, q_a_t.tensor_type).astype(np.float32)
    q_b = dequantize(q_b_t.data, q_b_t.tensor_type).astype(np.float32)
    kv_a_mqa = dequantize(kv_a_t.data, kv_a_t.tensor_type).astype(np.float32)
    k_b = dequantize(k_b_t.data, k_b_t.tensor_type).astype(np.float32)
    v_b = dequantize(v_b_t.data, v_b_t.tensor_type).astype(np.float32)
    attn_out = dequantize(attn_out_t.data, attn_out_t.tensor_type).astype(np.float32)

    if attn_norm.shape != (cfg.dim,):
        raise ValueError(
            f"{pfx}.attn_norm.weight shape mismatch: got {attn_norm.shape}, expected ({cfg.dim},)"
        )
    if q_a_norm.shape != (cfg.q_lora_rank,):
        raise ValueError(
            f"{pfx}.attn_q_a_norm.weight shape mismatch: got {q_a_norm.shape}, expected ({cfg.q_lora_rank},)"
        )
    if kv_a_norm.shape != (cfg.kv_lora_rank,):
        raise ValueError(
            f"{pfx}.attn_kv_a_norm.weight shape mismatch: got {kv_a_norm.shape}, expected ({cfg.kv_lora_rank},)"
        )
    if q_a.shape != (cfg.q_lora_rank, cfg.dim):
        raise ValueError(
            f"{pfx}.attn_q_a.weight shape mismatch: got {q_a.shape}, expected ({cfg.q_lora_rank}, {cfg.dim})"
        )
    if kv_a_mqa.shape != (cfg.kv_lora_rank + cfg.rope_dim, cfg.dim):
        raise ValueError(
            f"{pfx}.attn_kv_a_mqa.weight shape mismatch: got {kv_a_mqa.shape}, expected ({cfg.kv_lora_rank + cfg.rope_dim}, {cfg.dim})"
        )
    if q_b.shape[1] != cfg.q_lora_rank:
        raise ValueError(f"{pfx}.attn_q_b.weight bad rank dim: got {q_b.shape}")
    if q_b.shape[0] % cfg.n_heads != 0:
        raise ValueError(
            f"{pfx}.attn_q_b.weight rows not divisible by n_heads: {q_b.shape[0]} vs {cfg.n_heads}"
        )

    head_k_dim = int(q_b.shape[0] // cfg.n_heads)
    qk_nope_dim = head_k_dim - cfg.rope_dim
    if qk_nope_dim <= 0:
        raise ValueError(
            f"{pfx}.attn_q_b.weight invalid derived dims: head_k={head_k_dim} rope={cfg.rope_dim}"
        )

    if k_b.shape != (cfg.n_heads, cfg.kv_lora_rank, qk_nope_dim):
        raise ValueError(
            f"{pfx}.attn_k_b.weight shape mismatch: got {k_b.shape}, expected ({cfg.n_heads}, {cfg.kv_lora_rank}, {qk_nope_dim})"
        )
    if v_b.shape[0] != cfg.n_heads or v_b.shape[2] != cfg.kv_lora_rank:
        raise ValueError(
            f"{pfx}.attn_v_b.weight shape mismatch: got {v_b.shape}, expected head-major with kv rank {cfg.kv_lora_rank}"
        )
    v_head_dim = int(v_b.shape[1])

    if attn_out.shape != (cfg.dim, cfg.n_heads * v_head_dim):
        raise ValueError(
            f"{pfx}.attn_output.weight shape mismatch: got {attn_out.shape}, expected ({cfg.dim}, {cfg.n_heads * v_head_dim})"
        )

    if expected_dims is not None:
        exp_qk_nope, exp_head_k, exp_v_head = expected_dims
        got_dims = (qk_nope_dim, head_k_dim, v_head_dim)
        if got_dims != expected_dims:
            raise ValueError(
                f"{pfx} attention dims mismatch: got {got_dims}, expected {expected_dims}"
            )

    k_b_rows = k_b.reshape(cfg.n_heads * cfg.kv_lora_rank, qk_nope_dim)
    v_b_rows = v_b.reshape(cfg.n_heads * v_head_dim, cfg.kv_lora_rank)

    q_a_q, q_a_s = pack_runtime_rows(q_a_t, q_a)
    q_b_q, q_b_s = pack_runtime_rows(q_b_t, q_b)
    kv_a_q, kv_a_s = pack_runtime_rows(kv_a_t, kv_a_mqa)
    k_b_q, k_b_s = pack_runtime_rows(k_b_t, k_b_rows)
    v_b_q, v_b_s = pack_runtime_rows(v_b_t, v_b_rows)
    attn_out_q, attn_out_s = pack_runtime_rows(attn_out_t, attn_out)

    return {
        "qk_nope_dim": qk_nope_dim,
        "head_k_dim": head_k_dim,
        "v_head_dim": v_head_dim,
        "attn_norm": attn_norm,
        "q_a_norm": q_a_norm,
        "kv_a_norm": kv_a_norm,
        "q_a_q": q_a_q,
        "q_a_s": q_a_s,
        "q_b_q": q_b_q,
        "q_b_s": q_b_s,
        "kv_a_q": kv_a_q,
        "kv_a_s": kv_a_s,
        "k_b_q": k_b_q,
        "k_b_s": k_b_s,
        "v_b_q": v_b_q,
        "v_b_s": v_b_s,
        "attn_out_q": attn_out_q,
        "attn_out_s": attn_out_s,
    }


def write_attention_layer_block(out_f, layer_pack: dict):
    out_f.write(layer_pack["attn_norm"].tobytes(order="C"))
    out_f.write(layer_pack["q_a_norm"].tobytes(order="C"))
    out_f.write(layer_pack["kv_a_norm"].tobytes(order="C"))
    out_f.write(layer_pack["q_a_q"].tobytes(order="C"))
    out_f.write(layer_pack["q_a_s"].tobytes(order="C"))
    out_f.write(layer_pack["q_b_q"].tobytes(order="C"))
    out_f.write(layer_pack["q_b_s"].tobytes(order="C"))
    out_f.write(layer_pack["kv_a_q"].tobytes(order="C"))
    out_f.write(layer_pack["kv_a_s"].tobytes(order="C"))
    out_f.write(layer_pack["k_b_q"].tobytes(order="C"))
    out_f.write(layer_pack["k_b_s"].tobytes(order="C"))
    out_f.write(layer_pack["v_b_q"].tobytes(order="C"))
    out_f.write(layer_pack["v_b_s"].tobytes(order="C"))
    out_f.write(layer_pack["attn_out_q"].tobytes(order="C"))
    out_f.write(layer_pack["attn_out_s"].tobytes(order="C"))


def prepare_layer_moe_shared_quantized(
    tm: dict,
    cfg: Config,
    layer_idx: int,
    expected_moe_dim: Optional[int] = None,
):
    if layer_idx <= 0:
        raise ValueError("moe shared path is defined for layers >= 1")

    pfx = f"blk.{layer_idx}"
    ffn_norm_t = tm[f"{pfx}.ffn_norm.weight"]
    gate_inp_t = tm[f"{pfx}.ffn_gate_inp.weight"]
    exp_probs_b_t = tm[f"{pfx}.exp_probs_b.bias"]
    gate_shexp_t = tm[f"{pfx}.ffn_gate_shexp.weight"]
    up_shexp_t = tm[f"{pfx}.ffn_up_shexp.weight"]
    down_shexp_t = tm[f"{pfx}.ffn_down_shexp.weight"]

    ffn_norm = dequantize(ffn_norm_t.data, ffn_norm_t.tensor_type).astype(np.float32)
    gate_inp = dequantize(gate_inp_t.data, gate_inp_t.tensor_type).astype(np.float32)
    exp_probs_b = dequantize(exp_probs_b_t.data, exp_probs_b_t.tensor_type).astype(
        np.float32
    )

    gate_shexp = dequantize(gate_shexp_t.data, gate_shexp_t.tensor_type).astype(
        np.float32
    )
    up_shexp = dequantize(up_shexp_t.data, up_shexp_t.tensor_type).astype(np.float32)
    down_shexp = dequantize(down_shexp_t.data, down_shexp_t.tensor_type).astype(
        np.float32
    )

    if ffn_norm.shape != (cfg.dim,):
        raise ValueError(
            f"{pfx}.ffn_norm.weight shape mismatch: got {ffn_norm.shape}, expected ({cfg.dim},)"
        )
    if gate_inp.shape != (cfg.n_routed_experts, cfg.dim):
        raise ValueError(
            f"{pfx}.ffn_gate_inp.weight shape mismatch: got {gate_inp.shape}, expected ({cfg.n_routed_experts}, {cfg.dim})"
        )
    if exp_probs_b.shape != (cfg.n_routed_experts,):
        raise ValueError(
            f"{pfx}.exp_probs_b.bias shape mismatch: got {exp_probs_b.shape}, expected ({cfg.n_routed_experts},)"
        )

    if gate_shexp.shape[1] != cfg.dim:
        raise ValueError(
            f"{pfx}.ffn_gate_shexp.weight shape mismatch: got {gate_shexp.shape}, expected second dim {cfg.dim}"
        )
    if up_shexp.shape != gate_shexp.shape:
        raise ValueError(
            f"{pfx}.ffn_up_shexp.weight shape mismatch: got {up_shexp.shape}, expected {gate_shexp.shape}"
        )
    moe_dim = int(gate_shexp.shape[0])
    if down_shexp.shape != (cfg.dim, moe_dim):
        raise ValueError(
            f"{pfx}.ffn_down_shexp.weight shape mismatch: got {down_shexp.shape}, expected ({cfg.dim}, {moe_dim})"
        )
    if expected_moe_dim is not None and moe_dim != expected_moe_dim:
        raise ValueError(
            f"{pfx} shared expert dim mismatch: got {moe_dim}, expected {expected_moe_dim}"
        )
    if moe_dim % GROUP_SIZE != 0:
        raise ValueError(
            f"{pfx} shared expert dim must be divisible by {GROUP_SIZE}: got {moe_dim}"
        )

    gate_sh_q, gate_sh_s = pack_runtime_rows(gate_shexp_t, gate_shexp)
    up_sh_q, up_sh_s = pack_runtime_rows(up_shexp_t, up_shexp)
    down_sh_q, down_sh_s = pack_runtime_rows(down_shexp_t, down_shexp)

    return {
        "moe_dim": moe_dim,
        "ffn_norm": ffn_norm,
        "gate_inp": gate_inp,
        "exp_probs_b": exp_probs_b,
        "gate_sh_q": gate_sh_q,
        "gate_sh_s": gate_sh_s,
        "up_sh_q": up_sh_q,
        "up_sh_s": up_sh_s,
        "down_sh_q": down_sh_q,
        "down_sh_s": down_sh_s,
    }


def write_moe_shared_layer_block(out_f, layer_pack: dict):
    out_f.write(layer_pack["ffn_norm"].tobytes(order="C"))
    out_f.write(layer_pack["gate_inp"].tobytes(order="C"))
    out_f.write(layer_pack["exp_probs_b"].tobytes(order="C"))
    out_f.write(layer_pack["gate_sh_q"].tobytes(order="C"))
    out_f.write(layer_pack["gate_sh_s"].tobytes(order="C"))
    out_f.write(layer_pack["up_sh_q"].tobytes(order="C"))
    out_f.write(layer_pack["up_sh_s"].tobytes(order="C"))
    out_f.write(layer_pack["down_sh_q"].tobytes(order="C"))
    out_f.write(layer_pack["down_sh_s"].tobytes(order="C"))


def prepare_layer_moe_routed_quantized(
    tm: dict,
    cfg: Config,
    layer_idx: int,
    moe_dim: int,
):
    if layer_idx <= 0:
        raise ValueError("moe routed path is defined for layers >= 1")

    pfx = f"blk.{layer_idx}"
    gate_exps_t = tm[f"{pfx}.ffn_gate_exps.weight"]
    up_exps_t = tm[f"{pfx}.ffn_up_exps.weight"]
    down_exps_t = tm[f"{pfx}.ffn_down_exps.weight"]

    gate_exps = dequantize(gate_exps_t.data, gate_exps_t.tensor_type).astype(np.float32)
    up_exps = dequantize(up_exps_t.data, up_exps_t.tensor_type).astype(np.float32)
    down_exps = dequantize(down_exps_t.data, down_exps_t.tensor_type).astype(np.float32)

    expected_gate_up = (cfg.n_routed_experts, moe_dim, cfg.dim)
    expected_down = (cfg.n_routed_experts, cfg.dim, moe_dim)
    if gate_exps.shape != expected_gate_up:
        raise ValueError(
            f"{pfx}.ffn_gate_exps.weight shape mismatch: got {gate_exps.shape}, expected {expected_gate_up}"
        )
    if up_exps.shape != expected_gate_up:
        raise ValueError(
            f"{pfx}.ffn_up_exps.weight shape mismatch: got {up_exps.shape}, expected {expected_gate_up}"
        )
    if down_exps.shape != expected_down:
        raise ValueError(
            f"{pfx}.ffn_down_exps.weight shape mismatch: got {down_exps.shape}, expected {expected_down}"
        )

    gate_rows = gate_exps.reshape(cfg.n_routed_experts * moe_dim, cfg.dim)
    up_rows = up_exps.reshape(cfg.n_routed_experts * moe_dim, cfg.dim)
    down_rows = down_exps.reshape(cfg.n_routed_experts * cfg.dim, moe_dim)

    gate_q, gate_s = pack_runtime_rows(gate_exps_t, gate_rows)
    up_q, up_s = pack_runtime_rows(up_exps_t, up_rows)
    down_q, down_s = pack_runtime_rows(down_exps_t, down_rows)

    return {
        "gate_q": gate_q,
        "gate_s": gate_s,
        "up_q": up_q,
        "up_s": up_s,
        "down_q": down_q,
        "down_s": down_s,
    }


def write_moe_routed_layer_block(out_f, layer_pack: dict):
    out_f.write(layer_pack["gate_q"].tobytes(order="C"))
    out_f.write(layer_pack["gate_s"].tobytes(order="C"))
    out_f.write(layer_pack["up_q"].tobytes(order="C"))
    out_f.write(layer_pack["up_s"].tobytes(order="C"))
    out_f.write(layer_pack["down_q"].tobytes(order="C"))
    out_f.write(layer_pack["down_s"].tobytes(order="C"))


def write_tokenizer_sidecar(reader: GGUFReader, out_prefix: Path):
    tok = require_field(reader, "tokenizer.ggml.tokens")
    if not isinstance(tok, list) or not tok:
        raise ValueError("tokenizer.ggml.tokens is empty or invalid")

    scores = first_present(reader, ["tokenizer.ggml.scores"], None)
    if not isinstance(scores, list) or len(scores) != len(tok):
        scores = [-1e6] * len(tok)

    bos = int(first_present(reader, ["tokenizer.ggml.bos_token_id"], 0))
    eos = int(first_present(reader, ["tokenizer.ggml.eos_token_id"], 0))
    max_len = max(len(t.encode("utf-8", errors="replace")) for t in tok)

    tokenizer_path = out_prefix.with_suffix(out_prefix.suffix + ".tokenizer")
    with tokenizer_path.open("wb") as out_f:
        out_f.write(struct.pack("<I", max_len))
        out_f.write(struct.pack("<I", bos))
        out_f.write(struct.pack("<I", eos))
        for i, token in enumerate(tok):
            token_bytes = token.encode("utf-8", errors="replace")
            out_f.write(struct.pack("<f", float(scores[i])))
            out_f.write(struct.pack("<I", len(token_bytes)))
            out_f.write(token_bytes)
    print(f"Written tokenizer sidecar: {tokenizer_path}")


def write_template_sidecars(reader: GGUFReader, out_prefix: Path):
    chat_template = first_present(
        reader,
        ["tokenizer.chat_template", "tokenizer.chat_template.default"],
        "<|user|>\n%s\n<|assistant|>\n",
    )

    if isinstance(chat_template, list):
        chat_template = (
            chat_template[0] if chat_template else "<|user|>\n%s\n<|assistant|>\n"
        )

    base = str(chat_template)
    for suffix in [
        ".template",
        ".template.with-thinking",
        ".template.with-system",
        ".template.with-system-and-thinking",
    ]:
        path = out_prefix.with_suffix(out_prefix.suffix + suffix)
        path.write_text(base, encoding="utf-8")
        print(f"Written template sidecar: {path}")


def write_runtime_bin(reader: GGUFReader, cfg: Config, out_path: Path):
    tm = tensor_map(reader)

    out_norm = dequantize(
        tm["output_norm.weight"].data, tm["output_norm.weight"].tensor_type
    ).astype(np.float32)
    if out_norm.shape != (cfg.dim,):
        raise ValueError(
            f"output_norm.weight shape mismatch: got {out_norm.shape}, expected ({cfg.dim},)"
        )

    tok_emb_t = tm["token_embd.weight"]
    out_w_t = tm["output.weight"]
    tok_emb = dequantize(tok_emb_t.data, tok_emb_t.tensor_type).astype(np.float32)
    out_w = dequantize(out_w_t.data, out_w_t.tensor_type).astype(np.float32)

    expected = (cfg.vocab_size, cfg.dim)
    if tok_emb.shape != expected:
        raise ValueError(
            f"token_embd.weight shape mismatch: got {tok_emb.shape}, expected {expected}"
        )
    if out_w.shape != expected:
        raise ValueError(
            f"output.weight shape mismatch: got {out_w.shape}, expected {expected}"
        )

    tok_q, tok_s = pack_runtime_rows(tok_emb_t, tok_emb)
    out_q, out_s = pack_runtime_rows(out_w_t, out_w)

    l0_ffn_norm = dequantize(
        tm["blk.0.ffn_norm.weight"].data, tm["blk.0.ffn_norm.weight"].tensor_type
    ).astype(np.float32)
    if l0_ffn_norm.shape != (cfg.dim,):
        raise ValueError(
            f"blk.0.ffn_norm.weight shape mismatch: got {l0_ffn_norm.shape}, expected ({cfg.dim},)"
        )

    l0_gate_t = tm["blk.0.ffn_gate.weight"]
    l0_up_t = tm["blk.0.ffn_up.weight"]
    l0_down_t = tm["blk.0.ffn_down.weight"]
    l0_gate = dequantize(l0_gate_t.data, l0_gate_t.tensor_type).astype(np.float32)
    l0_up = dequantize(l0_up_t.data, l0_up_t.tensor_type).astype(np.float32)
    l0_down = dequantize(l0_down_t.data, l0_down_t.tensor_type).astype(np.float32)

    if l0_gate.shape != (cfg.hidden_dim, cfg.dim):
        raise ValueError(
            f"blk.0.ffn_gate.weight shape mismatch: got {l0_gate.shape}, expected ({cfg.hidden_dim}, {cfg.dim})"
        )
    if l0_up.shape != (cfg.hidden_dim, cfg.dim):
        raise ValueError(
            f"blk.0.ffn_up.weight shape mismatch: got {l0_up.shape}, expected ({cfg.hidden_dim}, {cfg.dim})"
        )
    if l0_down.shape != (cfg.dim, cfg.hidden_dim):
        raise ValueError(
            f"blk.0.ffn_down.weight shape mismatch: got {l0_down.shape}, expected ({cfg.dim}, {cfg.hidden_dim})"
        )

    l0_gate_q, l0_gate_s = pack_runtime_rows(l0_gate_t, l0_gate)
    l0_up_q, l0_up_s = pack_runtime_rows(l0_up_t, l0_up)
    l0_down_q, l0_down_s = pack_runtime_rows(l0_down_t, l0_down)

    l0_attn = prepare_layer_attention_quantized(tm, cfg, 0)
    l0_qk_nope_dim = int(l0_attn["qk_nope_dim"])
    l0_head_k_dim = int(l0_attn["head_k_dim"])
    l0_v_head_dim = int(l0_attn["v_head_dim"])
    l0_attn_norm = l0_attn["attn_norm"]
    l0_q_a_norm = l0_attn["q_a_norm"]
    l0_kv_a_norm = l0_attn["kv_a_norm"]
    l0_q_a_q = l0_attn["q_a_q"]
    l0_q_a_s = l0_attn["q_a_s"]
    l0_q_b_q = l0_attn["q_b_q"]
    l0_q_b_s = l0_attn["q_b_s"]
    l0_kv_a_q = l0_attn["kv_a_q"]
    l0_kv_a_s = l0_attn["kv_a_s"]
    l0_k_b_q = l0_attn["k_b_q"]
    l0_k_b_s = l0_attn["k_b_s"]
    l0_v_b_q = l0_attn["v_b_q"]
    l0_v_b_s = l0_attn["v_b_s"]
    l0_attn_out_q = l0_attn["attn_out_q"]
    l0_attn_out_s = l0_attn["attn_out_s"]

    norm_bytes = out_norm.nbytes
    tok_q_bytes = tok_q.nbytes
    tok_s_bytes = tok_s.nbytes
    out_q_bytes = out_q.nbytes
    out_s_bytes = out_s.nbytes
    l0_ffn_norm_bytes = l0_ffn_norm.nbytes
    l0_gate_q_bytes = l0_gate_q.nbytes
    l0_gate_s_bytes = l0_gate_s.nbytes
    l0_up_q_bytes = l0_up_q.nbytes
    l0_up_s_bytes = l0_up_s.nbytes
    l0_down_q_bytes = l0_down_q.nbytes
    l0_down_s_bytes = l0_down_s.nbytes
    l0_attn_norm_bytes = l0_attn_norm.nbytes
    l0_q_a_norm_bytes = l0_q_a_norm.nbytes
    l0_kv_a_norm_bytes = l0_kv_a_norm.nbytes
    l0_q_a_q_bytes = l0_q_a_q.nbytes
    l0_q_a_s_bytes = l0_q_a_s.nbytes
    l0_q_b_q_bytes = l0_q_b_q.nbytes
    l0_q_b_s_bytes = l0_q_b_s.nbytes
    l0_kv_a_q_bytes = l0_kv_a_q.nbytes
    l0_kv_a_s_bytes = l0_kv_a_s.nbytes
    l0_k_b_q_bytes = l0_k_b_q.nbytes
    l0_k_b_s_bytes = l0_k_b_s.nbytes
    l0_v_b_q_bytes = l0_v_b_q.nbytes
    l0_v_b_s_bytes = l0_v_b_s.nbytes
    l0_attn_out_q_bytes = l0_attn_out_q.nbytes
    l0_attn_out_s_bytes = l0_attn_out_s.nbytes
    attn_layer_bytes = (
        l0_attn_norm_bytes
        + l0_q_a_norm_bytes
        + l0_kv_a_norm_bytes
        + l0_q_a_q_bytes
        + l0_q_a_s_bytes
        + l0_q_b_q_bytes
        + l0_q_b_s_bytes
        + l0_kv_a_q_bytes
        + l0_kv_a_s_bytes
        + l0_k_b_q_bytes
        + l0_k_b_s_bytes
        + l0_v_b_q_bytes
        + l0_v_b_s_bytes
        + l0_attn_out_q_bytes
        + l0_attn_out_s_bytes
    )
    all_attn_bytes = cfg.n_layers * attn_layer_bytes

    moe_ffn_dim = 0
    moe_layer_bytes = 0
    all_moe_bytes = 0
    moe_routed_layer_bytes = 0
    all_moe_routed_bytes = 0
    l1_moe = None
    l1_moe_routed = None
    if cfg.n_layers > 1:
        l1_moe = prepare_layer_moe_shared_quantized(tm, cfg, 1)
        moe_ffn_dim = int(l1_moe["moe_dim"])
        moe_layer_bytes = (
            l1_moe["ffn_norm"].nbytes
            + l1_moe["gate_inp"].nbytes
            + l1_moe["exp_probs_b"].nbytes
            + l1_moe["gate_sh_q"].nbytes
            + l1_moe["gate_sh_s"].nbytes
            + l1_moe["up_sh_q"].nbytes
            + l1_moe["up_sh_s"].nbytes
            + l1_moe["down_sh_q"].nbytes
            + l1_moe["down_sh_s"].nbytes
        )
        all_moe_bytes = (cfg.n_layers - 1) * moe_layer_bytes
        l1_moe_routed = prepare_layer_moe_routed_quantized(tm, cfg, 1, moe_ffn_dim)
        moe_routed_layer_bytes = (
            l1_moe_routed["gate_q"].nbytes
            + l1_moe_routed["gate_s"].nbytes
            + l1_moe_routed["up_q"].nbytes
            + l1_moe_routed["up_s"].nbytes
            + l1_moe_routed["down_q"].nbytes
            + l1_moe_routed["down_s"].nbytes
        )
        all_moe_routed_bytes = (cfg.n_layers - 1) * moe_routed_layer_bytes

    off_norm = HEADER_SIZE
    off_tok_q = off_norm + norm_bytes
    off_tok_s = off_tok_q + tok_q_bytes
    off_out_q = off_tok_s + tok_s_bytes
    off_out_s = off_out_q + out_q_bytes
    off_l0_ffn_norm = off_out_s + out_s_bytes
    off_l0_gate_q = off_l0_ffn_norm + l0_ffn_norm_bytes
    off_l0_gate_s = off_l0_gate_q + l0_gate_q_bytes
    off_l0_up_q = off_l0_gate_s + l0_gate_s_bytes
    off_l0_up_s = off_l0_up_q + l0_up_q_bytes
    off_l0_down_q = off_l0_up_s + l0_up_s_bytes
    off_l0_down_s = off_l0_down_q + l0_down_q_bytes
    off_l0_attn_norm = off_l0_down_s + l0_down_s_bytes
    off_l0_q_a_norm = off_l0_attn_norm + l0_attn_norm_bytes
    off_l0_kv_a_norm = off_l0_q_a_norm + l0_q_a_norm_bytes
    off_l0_q_a_q = off_l0_kv_a_norm + l0_kv_a_norm_bytes
    off_l0_q_a_s = off_l0_q_a_q + l0_q_a_q_bytes
    off_l0_q_b_q = off_l0_q_a_s + l0_q_a_s_bytes
    off_l0_q_b_s = off_l0_q_b_q + l0_q_b_q_bytes
    off_l0_kv_a_q = off_l0_q_b_s + l0_q_b_s_bytes
    off_l0_kv_a_s = off_l0_kv_a_q + l0_kv_a_q_bytes
    off_l0_k_b_q = off_l0_kv_a_s + l0_kv_a_s_bytes
    off_l0_k_b_s = off_l0_k_b_q + l0_k_b_q_bytes
    off_l0_v_b_q = off_l0_k_b_s + l0_k_b_s_bytes
    off_l0_v_b_s = off_l0_v_b_q + l0_v_b_q_bytes
    off_l0_attn_out_q = off_l0_v_b_s + l0_v_b_s_bytes
    off_l0_attn_out_s = off_l0_attn_out_q + l0_attn_out_q_bytes
    off_all_attn_layers = off_l0_attn_out_s + l0_attn_out_s_bytes
    off_all_moe_layers = off_all_attn_layers + all_attn_bytes
    off_all_moe_routed_layers = off_all_moe_layers + all_moe_bytes
    total_bytes = off_all_moe_routed_layers + all_moe_routed_bytes

    with out_path.open("wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(
            struct.pack(
                "<16i",
                cfg.dim,
                cfg.hidden_dim,
                cfg.n_layers,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.vocab_size,
                cfg.seq_len,
                cfg.rope_dim,
                cfg.q_lora_rank,
                cfg.kv_lora_rank,
                cfg.kv_mqa_width,
                cfg.n_routed_experts,
                cfg.n_experts_used,
                cfg.n_shared_experts,
                cfg.bos,
                cfg.eos,
            )
        )
        f.write(
            struct.pack(
                "<14Q",
                GROUP_SIZE,
                off_norm,
                off_tok_q,
                off_tok_s,
                off_out_q,
                off_out_s,
                total_bytes,
                off_l0_ffn_norm,
                off_l0_gate_q,
                off_l0_gate_s,
                off_l0_up_q,
                off_l0_up_s,
                off_l0_down_q,
                off_l0_down_s,
            )
        )
        # v4 extension block (fits in header padding and keeps base layout backward compatible)
        f.write(
            struct.pack(
                "<5i", l0_qk_nope_dim, l0_head_k_dim, l0_v_head_dim, 1, moe_ffn_dim
            )
        )
        pad = HEADER_SIZE - f.tell()
        if pad < 0:
            raise ValueError("header overflow")
        f.write(b"\0" * pad)

        f.write(out_norm.tobytes(order="C"))
        f.write(tok_q.tobytes(order="C"))
        f.write(tok_s.tobytes(order="C"))
        f.write(out_q.tobytes(order="C"))
        f.write(out_s.tobytes(order="C"))
        f.write(l0_ffn_norm.tobytes(order="C"))
        f.write(l0_gate_q.tobytes(order="C"))
        f.write(l0_gate_s.tobytes(order="C"))
        f.write(l0_up_q.tobytes(order="C"))
        f.write(l0_up_s.tobytes(order="C"))
        f.write(l0_down_q.tobytes(order="C"))
        f.write(l0_down_s.tobytes(order="C"))
        f.write(l0_attn_norm.tobytes(order="C"))
        f.write(l0_q_a_norm.tobytes(order="C"))
        f.write(l0_kv_a_norm.tobytes(order="C"))
        f.write(l0_q_a_q.tobytes(order="C"))
        f.write(l0_q_a_s.tobytes(order="C"))
        f.write(l0_q_b_q.tobytes(order="C"))
        f.write(l0_q_b_s.tobytes(order="C"))
        f.write(l0_kv_a_q.tobytes(order="C"))
        f.write(l0_kv_a_s.tobytes(order="C"))
        f.write(l0_k_b_q.tobytes(order="C"))
        f.write(l0_k_b_s.tobytes(order="C"))
        f.write(l0_v_b_q.tobytes(order="C"))
        f.write(l0_v_b_s.tobytes(order="C"))
        f.write(l0_attn_out_q.tobytes(order="C"))
        f.write(l0_attn_out_s.tobytes(order="C"))

        # Append full per-layer attention blocks for future multi-layer runtime integration.
        write_attention_layer_block(f, l0_attn)
        expected_dims = (l0_qk_nope_dim, l0_head_k_dim, l0_v_head_dim)
        for layer_idx in range(1, cfg.n_layers):
            layer_pack = prepare_layer_attention_quantized(
                tm, cfg, layer_idx, expected_dims=expected_dims
            )
            write_attention_layer_block(f, layer_pack)

        # Append shared-expert FFN tensors for MoE layers (1..n_layers-1).
        if cfg.n_layers > 1 and l1_moe is not None:
            write_moe_shared_layer_block(f, l1_moe)
            for layer_idx in range(2, cfg.n_layers):
                moe_pack = prepare_layer_moe_shared_quantized(
                    tm, cfg, layer_idx, expected_moe_dim=moe_ffn_dim
                )
                write_moe_shared_layer_block(f, moe_pack)

        # Append routed expert tensors for MoE layers (1..n_layers-1).
        if cfg.n_layers > 1 and l1_moe_routed is not None:
            write_moe_routed_layer_block(f, l1_moe_routed)
            for layer_idx in range(2, cfg.n_layers):
                routed_pack = prepare_layer_moe_routed_quantized(
                    tm, cfg, layer_idx, moe_ffn_dim
                )
                write_moe_routed_layer_block(f, routed_pack)

    print("Export summary:")
    print(f"  output: {out_path}")
    print(f"  arch: deepseek2")
    print(f"  layers: {cfg.n_layers}")
    print(f"  dim: {cfg.dim}")
    print(f"  vocab: {cfg.vocab_size} bos={cfg.bos} eos={cfg.eos}")
    print(
        "  bytes:"
        f" total={total_bytes:,}"
        f" norm={norm_bytes:,}"
        f" tok_q={tok_q_bytes:,}"
        f" tok_s={tok_s_bytes:,}"
        f" out_q={out_q_bytes:,}"
        f" out_s={out_s_bytes:,}"
        f" l0_norm={l0_ffn_norm_bytes:,}"
        f" l0_gate_q={l0_gate_q_bytes:,}"
        f" l0_gate_s={l0_gate_s_bytes:,}"
        f" l0_up_q={l0_up_q_bytes:,}"
        f" l0_up_s={l0_up_s_bytes:,}"
        f" l0_down_q={l0_down_q_bytes:,}"
        f" l0_down_s={l0_down_s_bytes:,}"
        f" l0_attn_norm={l0_attn_norm_bytes:,}"
        f" l0_q_a_norm={l0_q_a_norm_bytes:,}"
        f" l0_kv_a_norm={l0_kv_a_norm_bytes:,}"
        f" l0_q_a_q={l0_q_a_q_bytes:,}"
        f" l0_q_a_s={l0_q_a_s_bytes:,}"
        f" l0_q_b_q={l0_q_b_q_bytes:,}"
        f" l0_q_b_s={l0_q_b_s_bytes:,}"
        f" l0_kv_a_q={l0_kv_a_q_bytes:,}"
        f" l0_kv_a_s={l0_kv_a_s_bytes:,}"
        f" l0_k_b_q={l0_k_b_q_bytes:,}"
        f" l0_k_b_s={l0_k_b_s_bytes:,}"
        f" l0_v_b_q={l0_v_b_q_bytes:,}"
        f" l0_v_b_s={l0_v_b_s_bytes:,}"
        f" l0_attn_out_q={l0_attn_out_q_bytes:,}"
        f" l0_attn_out_s={l0_attn_out_s_bytes:,}"
        f" all_attn_layer={attn_layer_bytes:,}"
        f" all_attn_total={all_attn_bytes:,}"
        f" moe_layer={moe_layer_bytes:,}"
        f" moe_total={all_moe_bytes:,}"
        f" moe_routed_layer={moe_routed_layer_bytes:,}"
        f" moe_routed_total={all_moe_routed_bytes:,}"
    )
    print(
        f"  l0_mla_dims: qk_nope={l0_qk_nope_dim} head_k={l0_head_k_dim} v_head={l0_v_head_dim}"
    )
    if moe_ffn_dim > 0:
        print(f"  moe_shared_dim: {moe_ffn_dim}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input GGUF path")
    parser.add_argument("--output", required=True, help="Output .bin path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    reader = GGUFReader(str(in_path))
    cfg = get_cfg(reader)
    validate_tensors(reader, cfg)
    write_runtime_bin(reader, cfg, out_path)
    write_tokenizer_sidecar(reader, out_path)
    write_template_sidecars(reader, out_path)


if __name__ == "__main__":
    main()
