#!/usr/bin/env python3
"""Env-gated online-FP8 allow-list for BF16 dense layers on v0.30.0 modelopt.py (P4-1).

The MIXED_PRECISION checkpoint leaves GDN, QSA, PLE-kv and MTP attention in
BF16 (UnquantizedLinearMethod). With VLLM_QWEN38_FP8_DENSE=per_block|ptpc,
ModelOptMixedPrecisionConfig instead hands the Linear layers whose prefix
matches the allow-list to vLLM's stock Fp8PerBlockOnlineLinearMethod or
Fp8PtpcOnlineLinearMethod (online/fp8.py), which quantize at load time.
Unset, empty, "0" or "off" returns stock UnquantizedLinearMethod unchanged.

Output: docker/v030/modelopt.py with an R11 header. Mount it at IN_IMAGE in
the v0.30.0 container on BOTH ranks (weights load on each rank).

--savings prints per-rank weight bytes (BF16 vs FP8) for the allow-list,
computed from the local checkpoint's safetensors headers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

BASE_IMAGE = (
    "vllm/vllm-openai:v0.30.0-aarch64@"
    "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
)
BASE_IMAGE_DIGEST = (
    "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
)
UPSTREAM_FILE = "vllm/model_executor/layers/quantization/modelopt.py"
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/" + UPSTREAM_FILE
UPSTREAM_FILE_SHA256 = (
    "41e00493d2770af19db03f585e9edb0004093c98b851d0f6726c2870cba11e22"
)
UPSTREAM_PR = "n/a (recipe-local; stock --quantization-config does not reach the MTP drafter)"
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots"
    / "fab0aecb760cec45227f6656abcaafa11abca87a"
)

HEADER = f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {UPSTREAM_FILE}
# upstream_file_sha256: {UPSTREAM_FILE_SHA256}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_fp8_dense_overlay.py (do not hand-edit)
"""

# Hunk 1: the allow-list helper, placed just above the mixed config.
OLD_CLASS = """\
class ModelOptMixedPrecisionConfig(ModelOptQuantConfigBase):
"""
NEW_CLASS = '''\
# qwen38-fp8-dense (recipe overlay, plan P4-1). Default OFF: with
# VLLM_QWEN38_FP8_DENSE unset/""/"0"/"off" the helper returns None and the
# caller returns stock UnquantizedLinearMethod. Tensors stay split (C7).
_QWEN38_FP8_DENSE_DEFAULT_ALLOW = (
    r"(?:^|\\.)linear_attn\\.in_proj_qkvz$",
    r"(?:^|\\.)linear_attn\\.out_proj$",
    r"(?:^|\\.)self_attn\\.qkv_proj$",
    r"(?:^|\\.)self_attn\\.o_proj$",
    r"(?:^|\\.)ple\\.kv_proj$",
)


def _qwen38_fp8_dense_method(
    layer: torch.nn.Module, prefix: str
) -> "QuantizeMethodBase | None":
    import os
    import re

    mode = os.environ.get("VLLM_QWEN38_FP8_DENSE", "").strip().lower()
    if mode in ("", "0", "off"):
        return None
    if mode not in ("per_block", "ptpc"):
        raise ValueError(
            f"VLLM_QWEN38_FP8_DENSE={mode!r}: expected per_block, ptpc or off"
        )
    if not isinstance(layer, LinearBase):
        return None
    raw = os.environ.get("VLLM_QWEN38_FP8_DENSE_ALLOW", "").strip()
    # A tuple: logger.info_once lru-caches its args, and a list is unhashable.
    patterns = (
        tuple(p.strip() for p in raw.split(",") if p.strip())
        if raw
        else _QWEN38_FP8_DENSE_DEFAULT_ALLOW
    )
    if not any(re.search(p, prefix) for p in patterns):
        return None
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerBlockOnlineLinearMethod,
        Fp8PtpcOnlineLinearMethod,
    )

    logger.info_once("qwen38-fp8-dense: mode=%s allow=%s", mode, patterns)
    logger.debug("qwen38-fp8-dense: %s -> %s", prefix, mode)
    if mode == "per_block":
        return Fp8PerBlockOnlineLinearMethod()
    return Fp8PtpcOnlineLinearMethod()


class ModelOptMixedPrecisionConfig(ModelOptQuantConfigBase):
'''
# Hunk 2: layers listed in exclude_modules (GDN, QSA, MTP attention).
OLD_EXCLUDED = """\
        # Excluded layers
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            return None
"""
NEW_EXCLUDED = """\
        # Excluded layers
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                fp8_dense = _qwen38_fp8_dense_method(layer, prefix)
                if fp8_dense is not None:
                    return fp8_dense
                return UnquantizedLinearMethod()
            return None
"""
# Hunk 3: layers absent from quantized_layers (PLE kv_proj).
OLD_UNLISTED = """\
            if quant_algo is None or quant_algo not in LINEAR_ALGOS:
                # Layer not in quantized_layers — leave unquantized
                return UnquantizedLinearMethod()
"""
NEW_UNLISTED = """\
            if quant_algo is None or quant_algo not in LINEAR_ALGOS:
                # Layer not in quantized_layers — leave unquantized
                fp8_dense = _qwen38_fp8_dense_method(layer, prefix)
                if fp8_dense is not None:
                    return fp8_dense
                return UnquantizedLinearMethod()
"""
HUNKS = (
    (OLD_CLASS, NEW_CLASS),
    (OLD_EXCLUDED, NEW_EXCLUDED),
    (OLD_UNLISTED, NEW_UNLISTED),
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def overlay(stock: str) -> str:
    got = sha256(stock)
    if got != UPSTREAM_FILE_SHA256:
        raise SystemExit(
            f"apply_fp8_dense_overlay: stock {UPSTREAM_FILE} sha256 {got} "
            f"!= {UPSTREAM_FILE_SHA256}; rebase the P4-1 hunks"
        )
    text = stock
    for old, new in HUNKS:
        if text.count(old) != 1:
            raise SystemExit("apply_fp8_dense_overlay: anchor not unique/found")
        text = text.replace(old, new, 1)
    return HEADER + text


def unoverlay(text: str) -> str:
    """Invert overlay(); used by tests to prove the diff is only P4-1."""
    if not text.startswith(HEADER):
        raise ValueError("R11 header missing or stale")
    text = text[len(HEADER) :]
    for old, new in reversed(HUNKS):
        if text.count(new) != 1:
            raise ValueError("overlay hunk missing")
        text = text.replace(new, old, 1)
    return text


def extract(image: str, dest: Path) -> None:
    cid = subprocess.check_output(["docker", "create", image], text=True).strip()
    try:
        subprocess.check_call(["docker", "cp", f"{cid}:{IN_IMAGE}", str(dest)])
    finally:
        subprocess.check_call(["docker", "rm", cid], stdout=subprocess.DEVNULL)


# --- checkpoint -> runtime Linear prefixes (for tests and --savings) -------

# Checkpoint shard name -> vLLM packed module (Qwen4Exp packed_modules_mapping).
_PACKED = {
    "in_proj_qkv": "in_proj_qkvz",
    "in_proj_z": "in_proj_qkvz",
    "in_proj_b": "in_proj_ba",
    "in_proj_a": "in_proj_ba",
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "key_proj": "kv_proj",
    "value_proj": "kv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
}


def read_shapes(snapshot: Path) -> dict[str, tuple[str, list[int]]]:
    """Weight name -> (dtype, shape) for every 2-D *.weight in the checkpoint."""
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    headers: dict[str, dict] = {}
    out: dict[str, tuple[str, list[int]]] = {}
    for name, fname in weight_map.items():
        if not name.endswith(".weight"):
            continue
        if fname not in headers:
            with open(snapshot / fname, "rb") as fp:
                (n,) = struct.unpack("<Q", fp.read(8))
                headers[fname] = json.loads(fp.read(n))
        entry = headers[fname][name]
        if len(entry["shape"]) == 2:
            out[name] = (entry["dtype"], entry["shape"])
    return out


def runtime_prefix(ckpt_weight: str, num_hidden_layers: int = 48) -> str:
    """Checkpoint weight name -> vLLM module prefix it loads into."""
    name = ckpt_weight.removesuffix(".weight")
    if name.startswith("model.language_model."):
        name = "language_model.model." + name[len("model.language_model.") :]
    # Runtime MTP layer is mtp.layers.{num_hidden_layers} (mtp.py remap).
    name = re.sub(r"^mtp\.layers\.(\d+)\.",
                  lambda m: f"mtp.layers.{num_hidden_layers + int(m[1])}.", name)
    head, _, leaf = name.rpartition(".")
    return f"{head}.{_PACKED.get(leaf, leaf)}" if head else leaf


def allow_match(prefix: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(re.search(p, prefix) for p in patterns)


def per_rank_shape(prefix: str, n: int, k: int, tp: int) -> tuple[int, int]:
    leaf = prefix.rsplit(".", 1)[-1]
    if leaf == "kv_proj":  # PLE kv_proj is disable_tp=True (replicated)
        return n, k
    if leaf in ("out_proj", "o_proj"):  # RowParallelLinear
        return n, k // tp
    return n // tp, k  # Column/Merged/QKV ParallelLinear


def savings(
    snapshot: Path, patterns: tuple[str, ...] | list[str], tp: int = 2
) -> dict[str, dict[str, int]]:
    """Per-rank bytes by module leaf: bf16, per_block (fp8 + fp32 128x128
    scales), ptpc (fp8 + fp32 per-row scales), and layer count."""
    fused: dict[str, list[int]] = {}
    for name, (dtype, (n, k)) in read_shapes(snapshot).items():
        prefix = runtime_prefix(name)
        if not allow_match(prefix, patterns):
            continue
        if dtype != "BF16":
            raise ValueError(f"{name} is {dtype}, expected BF16")
        acc = fused.setdefault(prefix, [0, k])
        if acc[1] != k:
            raise ValueError(f"{prefix}: shard K mismatch")
        acc[0] += n
    rows: dict[str, dict[str, int]] = {}
    for prefix, (n, k) in fused.items():
        rn, rk = per_rank_shape(prefix, n, k, tp)
        leaf = prefix.rsplit(".", 1)[-1]
        if prefix.startswith("mtp."):
            leaf = "mtp." + leaf
        r = rows.setdefault(leaf, {"layers": 0, "bf16": 0, "per_block": 0, "ptpc": 0})
        r["layers"] += 1
        r["bf16"] += rn * rk * 2
        r["per_block"] += rn * rk + (-(-rn // 128)) * (-(-rk // 128)) * 4
        r["ptpc"] += rn * rk + rn * 4
    return rows


def _default_patterns(text: str) -> list[str]:
    """Read _QWEN38_FP8_DENSE_DEFAULT_ALLOW out of the rendered overlay."""
    import ast

    text = text.split("\nclass ModelOptMixedPrecisionConfig(", 1)[0]
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "_QWEN38_FP8_DENSE_DEFAULT_ALLOW"
            for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise ValueError("default allow-list not found")


def print_savings(snapshot: Path, patterns: list[str], tp: int) -> None:
    rows = savings(snapshot, patterns, tp)
    gib = 1024**3
    print(f"{'module':<16}{'layers':>7}{'bf16 MB':>10}{'blk MB':>9}{'ptpc MB':>9}")
    tot = {"bf16": 0, "per_block": 0, "ptpc": 0}
    for leaf, r in sorted(rows.items()):
        print(f"{leaf:<16}{r['layers']:>7}{r['bf16'] / 1e6:>10.1f}"
              f"{r['per_block'] / 1e6:>9.1f}{r['ptpc'] / 1e6:>9.1f}")
        for key in tot:
            tot[key] += r[key]
    print(f"{'total':<16}{sum(r['layers'] for r in rows.values()):>7}"
          f"{tot['bf16'] / 1e6:>10.1f}{tot['per_block'] / 1e6:>9.1f}"
          f"{tot['ptpc'] / 1e6:>9.1f}")
    for key in ("per_block", "ptpc"):
        d = tot["bf16"] - tot[key]
        print(f"saving per rank ({key}): {d / 1e9:.3f} GB = {d / gib:.3f} GiB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default=BASE_IMAGE)
    ap.add_argument(
        "--src",
        type=Path,
        help="stock modelopt.py already extracted from the image (skips docker)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "modelopt.py",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if --out differs from a fresh render",
    )
    ap.add_argument(
        "--savings",
        action="store_true",
        help="print per-rank BF16 vs FP8 bytes for the default allow-list and exit",
    )
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()
    if args.savings:
        print_savings(args.snapshot, _default_patterns(NEW_CLASS), args.tp)
        return 0
    if args.src is not None:
        stock = args.src.read_text()
    else:
        tmp = args.out.with_suffix(".stock.py")
        extract(args.image, tmp)
        stock = tmp.read_text()
        tmp.unlink(missing_ok=True)
    text = overlay(stock)
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            print(f"apply_fp8_dense_overlay: {args.out} is stale", file=sys.stderr)
            return 1
        print(f"apply_fp8_dense_overlay: {args.out} up to date")
        return 0
    args.out.write_text(text)
    print(f"apply_fp8_dense_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
