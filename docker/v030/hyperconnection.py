# R11-OVERLAY
# base_image_digest: sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
# upstream_file: vllm/models/qwen4_exp/nvidia/hyperconnection.py
# upstream_file_sha256: 2a15d6b22fbe1def4d2bcd85568269d1157c5e1c298f0c6af43c1fb9797489ba
# upstream_PR: none (local K5; fusion target of vLLM issue #54688)
# generator: docker/v030/apply_hc_overlay.py (do not hand-edit)
# embedded: docker/v030/hc_fused.py sha256 809924ead98fa0734a7576e5235caeb61b3e7a4ec63ec3706db7c4c1fb08c4fc
# features: K5 fused HC (H1 combine_stats, H2 down_inject_silu, H3 up_gatemix)
#   for M <= 32, opt-in via VLLM_QWEN38_HC_FUSED=1, default off, self-tested at load
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — NVIDIA model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This NVIDIA variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output, it fuses its
    residual combine with the RMSNorm. A missing injection selects unit-weight
    combine. Final mixers use ``use_combine=False`` and do not produce a new
    injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        _k5 = hcf_dispatch(self, hidden_states, None, None)
        if _k5 is not None:
            return _k5
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm. A missing injection applies the block output
        to every stream with unit weight.
        """
        if prev_block_output is not None:
            _k5 = hcf_dispatch(self, hidden_states, prev_block_output, prev_injection)
            if _k5 is not None:
                return _k5
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]


# ---- K5 embedded from docker/v030/hc_fused.py (do not hand-edit) ----
# K5 fused hyper-connection (plan section 4 K5, L4b; findings F14, F15, F09).
#
# Source of truth for the kernels. docker/v030/apply_hc_overlay.py embeds this
# file verbatim at the end of the v0.30.0 qwen4_exp/nvidia/hyperconnection.py
# overlay, so it must stay a plain module: no __future__ import, every
# top-level name prefixed _hcf / hcf / HCF.
#
# Stock GatedResidual.combine_and_mix is 5 kernels per module:
#   hc_combine_norm -> F.linear(xn, W_down[336,10240]) -> hc_silu
#   -> F.linear(lora, W_up[10240,320]) -> hc_gate_mix
# K5 is 3 kernels, decode only (M <= 32):
#   H1 hcf_combine_stats   new residual (bf16, rounded exactly as ops/hc.py
#                          _hc_combine_norm_kernel) + rrms [M, HC] fp32.
#                          mix() (no pending combine) runs the stats-only
#                          variant mirroring _grouped_gemma_rmsnorm_kernel.
#   H2 hcf_down_inject_silu xn recomputed per K tile with the reference bf16
#                          rounding (y = r*rrms; y += y*w; bf16), split-K GEMM
#                          over the 324 useful rows (320 without inject), the
#                          last CTA per N block sums the split partials in fixed
#                          order 0..SPLIT_K-1, rounds to bf16 (the cuBLAS output
#                          rounding), then SiLU(x/HC) -> lora, raw -> injection.
#   H3 hcf_up_gatemix      per output column j: gate[s*H+j] for the HC streams
#                          (never materialised), rounded to bf16 as the up GEMM
#                          output is, then sum_s sigmoid(g)*xn in stream order,
#                          / HC, stored bf16 (same op order as _hc_gate_mix_kernel).
#
# Numerics: new residual is class A0 (same expression, same reduction shape as
# stock). lora / injection / block_input are class A: the only difference is
# the GEMM summation order (tl.dot + fixed split-K order vs cuBLAS split-K).
# No fp32 atomics touch outputs; the only atomic is an int32 arrival counter.
#
# Default OFF. Active only with VLLM_QWEN38_HC_FUSED=1, CUDA, M <= 32, bf16
# unquantized weights of the expected shapes, and after a per-variant
# load-time self-test (real module weights, M in HCF_SELF_TEST_MS x 3 input
# distributions, CUDA-graph replay at M=4 and 32) passed. Anything else runs
# the stock path.
import logging
import os

import torch

try:
    import triton
    import triton.language as tl

    _HCF_HAS_TRITON = True
except Exception:  # noqa: BLE001 - no triton means stock path
    _HCF_HAS_TRITON = False

HCF_ENV = "VLLM_QWEN38_HC_FUSED"
HCF_MAX_M = 32
_hcf_logger = logging.getLogger("vllm.qwen38.hc_fused")
_HCF_STATE = {
    "enabled": os.environ.get(HCF_ENV, "0") == "1",
    "failed": False,  # any self-test failure disables K5 for the process
    "busy": False,  # set while the self-test runs the stock path
    "tested": set(),  # variant keys that passed
    "counters": {},  # device -> int32 arrival counters (self-resetting)
}
# Tile config. SPLIT_K=8 x 21 N-blocks of 16 rows = 168 CTAs for the (336, 10240)
# down+inject GEMM; 2560/16 = 160 CTAs for the up GEMM + gate mix.
HCF_CFG = {"split_k": 8, "block_n": 16, "block_k": 128, "block_j": 16, "block_k_up": 64}
_HCF_COUNTERS = 1024
# Self-test token counts. Must reach every Triton specialisation the dispatch
# can launch: BLOCK_M 16 and 32, and M == 1 / M % 16 == 0 / other (Triton
# specialises integer args on ==1 and divisibility by 16).
HCF_SELF_TEST_MS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 32)


def _hcf_interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


if _HCF_HAS_TRITON:

    @triton.jit
    def _hcf_gemm_out_bf16(x, INTERP: tl.constexpr):
        """fp32 -> bf16 -> fp32 with round-to-nearest-even, as cuBLAS writes
        its bf16 output. On GPU .to(bf16) is RNE; the Triton CPU interpreter
        truncates, so there the rounding is done on the bits."""
        if INTERP:
            b = x.to(tl.uint32, bitcast=True)
            b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
            return b.to(tl.float32, bitcast=True)
        else:
            return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _hcf_combine_stats_kernel(
        block_ptr,
        res_ptr,
        inj_ptr,
        out_ptr,
        rrms_ptr,
        stride_block,
        stride_res,
        stride_inj,
        stride_out,
        HC_DIM: tl.constexpr,
        HC: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        # Mirrors ops/hc.py _hc_combine_norm_kernel up to rrms (same tile
        # shape, same expression order) so the new residual is bit-identical.
        HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
        NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
        NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)

        pid = tl.program_id(0)
        row = pid // HC
        stream = pid % HC
        offs_hc = tl.arange(0, HC_PAD)
        mask_hc = offs_hc < HC
        tile_ids = tl.arange(0, NUM_TILES_PAD)
        offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        mask_inner = offs_inner < HC_DIM
        offs = stream * HC_DIM + offs_inner

        res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
        if inj_ptr is not None:
            inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
        block = tl.load(
            block_ptr + row * stride_block + offs_inner,
            mask_inner,
            other=0.0,
        )
        if inj_ptr is not None:
            inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
            block = block.to(tl.float32) * tl.sum(tl.where(offs_hc == stream, inj, 0.0))
        out = (res.to(tl.float32) + block.to(tl.float32)).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)

        out = out.to(tl.float32)
        sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
        rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)
        tl.store(rrms_ptr + pid, rrms)

    @triton.jit
    def _hcf_stats_kernel(
        x_ptr,
        rrms_ptr,
        stride_x,
        HC_DIM: tl.constexpr,
        HC: tl.constexpr,
        EPS: tl.constexpr,
    ) -> None:
        # Mirrors ops/hc.py _grouped_gemma_rmsnorm_kernel's rrms (mix() path).
        BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(HC_DIM)
        pid = tl.program_id(0)
        group_id = pid % HC
        row = pid // HC
        offs_g = tl.arange(0, BLOCK_SIZE)
        mask = offs_g < HC_DIM
        x = tl.load(
            x_ptr + row * stride_x + group_id * HC_DIM + offs_g, mask, other=0.0
        ).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(x * x) / HC_DIM + EPS)
        tl.store(rrms_ptr + pid, rrms)

    @triton.jit
    def _hcf_down_kernel(
        res_ptr,
        rrms_ptr,
        nw_ptr,
        w_ptr,
        ws_ptr,
        cnt_ptr,
        lora_ptr,
        inj_ptr,
        M,
        stride_res,
        stride_w,
        stride_lora,
        stride_inj,
        HC_DIM: tl.constexpr,
        HC: tl.constexpr,
        W_SHARED: tl.constexpr,
        R: tl.constexpr,
        NR: tl.constexpr,
        K_SPLIT: tl.constexpr,
        SPLIT_K: tl.constexpr,
        WS_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        DOT_F32: tl.constexpr,  # True under TRITON_INTERPRET (CPU validation)
    ) -> None:
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < NR  # skips the 12 pad rows of the merged weight

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for kk in range(0, K_SPLIT, BLOCK_K):
            k0 = pid_k * K_SPLIT + kk
            stream = k0 // HC_DIM
            offs_k = k0 + tl.arange(0, BLOCK_K)
            wt = tl.load(
                w_ptr + offs_n[:, None] * stride_w + offs_k[None, :],
                mask_n[:, None],
                other=0.0,
            )
            r = tl.load(
                res_ptr + offs_m[:, None] * stride_res + offs_k[None, :],
                mask_m[:, None],
                other=0.0,
            )
            rr = tl.load(rrms_ptr + offs_m * HC + stream, mask_m, other=0.0)
            if W_SHARED:
                nw = tl.load(nw_ptr + offs_k - stream * HC_DIM)
            else:
                nw = tl.load(nw_ptr + offs_k)
            # Reference xn: y = x*rrms; y += y*w; stored bf16 (ops/hc.py:343-345).
            y = r.to(tl.float32) * rr[:, None]
            y += y * nw.to(tl.float32)[None, :]
            xn = y.to(tl.bfloat16)
            if DOT_F32:
                # bf16 x bf16 products are exact in fp32; the Triton CPU
                # interpreter mis-evaluates bf16 tl.dot, so it takes this path.
                acc += tl.dot(
                    xn.to(tl.float32),
                    tl.trans(wt.to(tl.float32)),
                    input_precision="ieee",
                )
            else:
                acc = tl.dot(xn, tl.trans(wt), acc)

        ws_offs = offs_m[:, None] * WS_N + offs_n[None, :]
        tl.store(ws_ptr + pid_k * BLOCK_M * WS_N + ws_offs, acc)
        tl.debug_barrier()
        # Arrival counter (acq_rel, gpu scope). The last split to arrive for
        # this N block reduces all partials in fixed split order and resets it.
        prev = tl.atomic_add(cnt_ptr + pid_n, 1)
        if prev == SPLIT_K - 1:
            tot = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(
                    ws_ptr + s * BLOCK_M * WS_N + ws_offs, cache_modifier=".cg"
                )
            tl.atomic_xchg(cnt_ptr + pid_n, 0)
            # cuBLAS writes the GEMM output in bf16; hc_silu reads it back.
            v = _hcf_gemm_out_bf16(tot, DOT_F32)
            x = v / HC
            act = x * tl.sigmoid(x)
            is_lora = offs_n < R
            tl.store(
                lora_ptr + offs_m[:, None] * stride_lora + offs_n[None, :],
                act,
                mask=mask_m[:, None] & is_lora[None, :],
            )
            if inj_ptr is not None:
                is_inj = (offs_n >= R) & mask_n
                tl.store(
                    inj_ptr + offs_m[:, None] * stride_inj + (offs_n - R)[None, :],
                    v,
                    mask=mask_m[:, None] & is_inj[None, :],
                )

    @triton.jit
    def _hcf_up_kernel(
        res_ptr,
        rrms_ptr,
        nw_ptr,
        lora_ptr,
        w_ptr,
        out_ptr,
        M,
        stride_res,
        stride_lora,
        stride_w,
        stride_out,
        HC_DIM: tl.constexpr,
        HC: tl.constexpr,
        W_SHARED: tl.constexpr,
        R: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_K: tl.constexpr,
        DOT_F32: tl.constexpr,  # True under TRITON_INTERPRET (CPU validation)
    ) -> None:
        pid = tl.program_id(0)
        offs_m = tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        offs_j = pid * BLOCK_J + tl.arange(0, BLOCK_J)
        mask_j = offs_j < HC_DIM
        mask_mj = mask_m[:, None] & mask_j[None, :]

        mix = tl.zeros([BLOCK_M, BLOCK_J], dtype=tl.float32)
        for stream in tl.static_range(HC):
            col = stream * HC_DIM + offs_j
            g = tl.zeros([BLOCK_M, BLOCK_J], dtype=tl.float32)
            for k0 in tl.static_range(0, R, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < R
                a = tl.load(
                    lora_ptr + offs_m[:, None] * stride_lora + offs_k[None, :],
                    mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                wt = tl.load(
                    w_ptr + col[:, None] * stride_w + offs_k[None, :],
                    mask_j[:, None] & mask_k[None, :],
                    other=0.0,
                )
                if DOT_F32:
                    g += tl.dot(
                        a.to(tl.float32),
                        tl.trans(wt.to(tl.float32)),
                        input_precision="ieee",
                    )
                else:
                    g = tl.dot(a, tl.trans(wt), g)
            # The up GEMM output (gate) is bf16 in the stock path.
            g = _hcf_gemm_out_bf16(g, DOT_F32)
            r = tl.load(
                res_ptr + offs_m[:, None] * stride_res + col[None, :],
                mask_mj,
                other=0.0,
            )
            rr = tl.load(rrms_ptr + offs_m * HC + stream, mask_m, other=0.0)
            if W_SHARED:
                nw = tl.load(nw_ptr + offs_j, mask_j, other=0.0)
            else:
                nw = tl.load(nw_ptr + col, mask_j, other=0.0)
            y = r.to(tl.float32) * rr[:, None]
            y += y * nw.to(tl.float32)[None, :]
            xn = y.to(tl.bfloat16)
            mix += tl.sigmoid(g) * xn.to(tl.float32)
        mix /= HC
        tl.store(
            out_ptr + offs_m[:, None] * stride_out + offs_j[None, :], mix, mask=mask_mj
        )


def _hcf_counters(device: torch.device) -> torch.Tensor:
    key = str(device)
    buf = _HCF_STATE["counters"].get(key)
    if buf is None:
        buf = torch.zeros(_HCF_COUNTERS, dtype=torch.int32, device=device)
        _HCF_STATE["counters"][key] = buf
    return buf


def hcf_combine_stats(residual, block_output, injection, norm_weight, eps, hc_count):
    """H1. Returns (new_residual bf16 [M, D], rrms fp32 [M, HC]).

    block_output None = mix() path: residual is returned unchanged.
    """
    n, d = residual.shape
    hc_dim = d // hc_count
    assert residual.stride(1) == 1
    rrms = torch.empty(n, hc_count, dtype=torch.float32, device=residual.device)
    if block_output is None:
        _hcf_stats_kernel[(n * hc_count,)](
            residual, rrms, residual.stride(0), HC_DIM=hc_dim, HC=hc_count, EPS=eps
        )
        return residual, rrms
    assert block_output.shape == (n, hc_dim) and block_output.stride(1) == 1
    if injection is not None:
        assert injection.shape == (n, hc_count) and injection.stride(1) == 1
    out = residual.new_empty(residual.shape)
    _hcf_combine_stats_kernel[(n * hc_count,)](
        block_output,
        residual,
        injection,
        out,
        rrms,
        block_output.stride(0),
        residual.stride(0),
        injection.stride(0) if injection is not None else 0,
        out.stride(0),
        HC_DIM=hc_dim,
        HC=hc_count,
        EPS=eps,
        BLOCK_SIZE=512,
    )
    return out, rrms


def _hcf_block_m(m: int) -> int:
    return max(16, triton.next_power_of_2(m))


def hcf_down_inject_silu(residual, rrms, norm_weight, w_down, hc_count, lora_rank, n_inject, cfg=None):
    """H2. Returns (lora_act bf16 [M, R], injection bf16 [M, HC] or None)."""
    cfg = cfg or HCF_CFG
    m, d = residual.shape
    hc_dim = d // hc_count
    nr = lora_rank + n_inject
    split_k, block_n, block_k = cfg["split_k"], cfg["block_n"], cfg["block_k"]
    assert w_down.shape[1] == d and w_down.shape[0] >= nr and w_down.stride(1) == 1
    assert d % (split_k * block_k) == 0 and hc_dim % block_k == 0
    grid_n = triton.cdiv(nr, block_n)
    assert grid_n <= _HCF_COUNTERS
    block_m = _hcf_block_m(m)
    ws_n = grid_n * block_n
    ws = torch.empty(split_k, block_m, ws_n, dtype=torch.float32, device=residual.device)
    lora = residual.new_empty(m, lora_rank)
    inj = residual.new_empty(m, n_inject) if n_inject else None
    _hcf_down_kernel[(grid_n, split_k)](
        residual,
        rrms,
        norm_weight,
        w_down,
        ws,
        _hcf_counters(residual.device),
        lora,
        inj,
        m,
        residual.stride(0),
        w_down.stride(0),
        lora.stride(0),
        inj.stride(0) if inj is not None else 0,
        HC_DIM=hc_dim,
        HC=hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        R=lora_rank,
        NR=nr,
        K_SPLIT=d // split_k,
        SPLIT_K=split_k,
        WS_N=ws_n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        DOT_F32=_hcf_interpret(),
    )
    return lora, inj


def hcf_up_gatemix(residual, rrms, norm_weight, lora, w_up, hc_count, cfg=None):
    """H3. Returns block_input bf16 [M, H]."""
    cfg = cfg or HCF_CFG
    m, d = residual.shape
    hc_dim = d // hc_count
    r = lora.shape[1]
    assert w_up.shape == (d, r) and w_up.stride(1) == 1 and lora.stride(1) == 1
    block_j = cfg["block_j"]
    block_k = min(cfg["block_k_up"], triton.next_power_of_2(r))
    out = residual.new_empty(m, hc_dim)
    _hcf_up_kernel[(triton.cdiv(hc_dim, block_j),)](
        residual,
        rrms,
        norm_weight,
        lora,
        w_up,
        out,
        m,
        residual.stride(0),
        lora.stride(0),
        w_up.stride(0),
        out.stride(0),
        HC_DIM=hc_dim,
        HC=hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        R=r,
        BLOCK_M=_hcf_block_m(m),
        BLOCK_J=block_j,
        BLOCK_K=block_k,
        DOT_F32=_hcf_interpret(),
    )
    return out


def hcf_forward(residual, block_output, injection, norm_weight, w_down, w_up, eps, hc_count, lora_rank, use_combine, cfg=None):
    """Fused GatedResidual.mix (block_output None) / combine_and_mix.

    Returns (hidden_states, block_input, injection_or_None), like stock.
    """
    new_res, rrms = hcf_combine_stats(residual, block_output, injection, norm_weight, eps, hc_count)
    lora, inj = hcf_down_inject_silu(
        new_res, rrms, norm_weight, w_down, hc_count, lora_rank, hc_count if use_combine else 0, cfg
    )
    block_input = hcf_up_gatemix(new_res, rrms, norm_weight, lora, w_up, hc_count, cfg)
    return new_res, block_input, inj


# ---------------------------------------------------------------- dispatch
def _hcf_weights(module):
    """(norm_w, w_down, w_up) if the module is the plain bf16 layout, else None."""
    try:
        cfg = module.config
        hc, h, r = module.hc_count, module.hidden_size, module.lora_rank
        d = hc * h
        norm_w = module.hc_norm.weight
        if module.use_combine:
            w_down = module.input_mix_weight_down_block_inject.weight
            rows = r + hc + module.pad_size
        else:
            w_down = module.input_mix_weight_down.weight
            rows = r
        w_up = module.input_mix_weight_up.weight
    except AttributeError:
        return None
    ok = (
        cfg.params_dtype == torch.bfloat16
        and all(
            isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16 and t.is_contiguous()
            for t in (norm_w, w_down, w_up)
        )
        and norm_w.numel() in (h, d)
        and tuple(w_down.shape) == (rows, d)
        and tuple(w_up.shape) == (d, r)
    )
    return (norm_w, w_down, w_up) if ok else None


def _hcf_key(module, block_output, injection):
    return (
        module.hc_count,
        module.hidden_size,
        module.lora_rank,
        bool(module.use_combine),
        block_output is not None,
        injection is not None,
        module.hc_norm.weight.numel(),
    )


def _hcf_run(module, weights, hidden_states, block_output, injection):
    norm_w, w_down, w_up = weights
    return hcf_forward(
        hidden_states,
        block_output,
        injection,
        norm_w,
        w_down,
        w_up,
        module.config.rms_norm_eps,
        module.hc_count,
        module.lora_rank,
        module.use_combine,
    )


def _hcf_ref64(module, weights, hidden_states, block_output, injection):
    """fp64 reference of the chain (no intermediate rounding) for class A."""
    norm_w, w_down, w_up = weights
    hc, h, r = module.hc_count, module.hidden_size, module.lora_rank
    m = hidden_states.shape[0]
    x = hidden_states.double().view(m, hc, h)
    if block_output is not None:
        b = block_output.double()[:, None, :]
        if injection is not None:
            b = b * (2.0 * torch.sigmoid(injection.double() / hc))[:, :, None]
        x = x + b
    rrms = torch.rsqrt(x.square().mean(-1, keepdim=True) + module.config.rms_norm_eps)
    w = norm_w.double().view(-1, h) if norm_w.numel() == h * hc else norm_w.double().view(1, h)
    xn = (x * rrms * (1.0 + w)).reshape(m, hc * h)
    down = xn @ w_down.double().T
    lora = down[:, :r] / hc
    lora = lora * torch.sigmoid(lora)
    inj = down[:, r : r + hc] if module.use_combine else None
    gate = (lora @ w_up.double().T).view(m, hc, h)
    bi = (torch.sigmoid(gate) * xn.view(m, hc, h)).sum(1) / hc
    return bi, inj


def _hcf_rel(a, ref):
    return ((a.double() - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def _hcf_elem_bad(stock, fused, ref) -> int:
    """Gross-error guard, per element: |fused - ref64| may not exceed
    2*|stock - ref64| + ~2 bf16 ulps of (|ref64| + RMS(ref64)). The ulp term
    covers elements where bf16 rounding of lora/gate moves both paths by ~1%
    (e.g. outlier channels). The class A criterion itself is the pooled L2
    error vs fp64 in hcf_self_test."""
    s, f, r = stock.double(), fused.double(), ref.double()
    tol = 2.0 * (s - r).abs() + 2.0**-6 * (r.abs() + r.square().mean().sqrt())
    return int(((f - r).abs() > tol).sum())


def _hcf_inputs(dist, m, hc, h, device, gen):
    d = hc * h

    def rn(*shape):
        return torch.randn(*shape, generator=gen, device="cpu").to(device)

    if dist == 0:  # plain normal
        hs, bo, inj = rn(m, d), rn(m, h), rn(m, hc)
    elif dist == 1:  # heavy-tailed, per-row scale spread
        hs = rn(m, d) * torch.exp(rn(m, 1)) * torch.exp(0.5 * rn(1, d))
        bo, inj = rn(m, h) * 8.0, rn(m, hc) * 6.0
    else:  # small magnitude with a few outlier channels
        hs = rn(m, d) * 1e-2
        hs[:, :: max(1, d // 7)] = 30.0
        bo, inj = rn(m, h) * 1e-3, rn(m, hc) * 0.1
    bf = torch.bfloat16
    return hs.to(bf), bo.to(bf), inj.to(bf)


def hcf_self_test(module, with_block, with_inj, device, stock_fn=None, ms=HCF_SELF_TEST_MS):
    """Fused vs stock on the module's real weights. Returns (ok, detail).

    Residual: bitwise equal. block_input / injection: class A: the L2 error
    vs an fp64 reference, pooled over all cases, <= 2x stock's own pooled
    error, and no element fails the _hcf_elem_bad gross-error guard.
    On CUDA also replays a captured graph (M=4 and M=32) and requires it to
    equal eager fused bitwise, twice (the arrival counters must self-reset).
    stock_fn(hs, bo, inj) defaults to module.mix / module.combine_and_mix
    (dispatch is suppressed while the self-test runs).
    """
    weights = _hcf_weights(module)
    if weights is None:
        return False, "unsupported weight layout"
    hc, h = module.hc_count, module.hidden_size
    if stock_fn is None:
        if with_block:
            stock_fn = module.combine_and_mix
        else:
            stock_fn = lambda hs, bo, inj: module.mix(hs)  # noqa: E731
    gen = torch.Generator().manual_seed(0x4B35)
    err2 = {}  # name -> [stock sq err, fused sq err]
    _HCF_STATE["busy"] = True
    try:
        with torch.no_grad():
            for dist in range(3):
                for m in ms:
                    hs, bo, inj = _hcf_inputs(dist, m, hc, h, device, gen)
                    bo = bo if with_block else None
                    inj = inj if (with_block and with_inj) else None
                    s_res, s_bi, s_inj = stock_fn(hs, bo, inj)
                    f_res, f_bi, f_inj = _hcf_run(module, weights, hs, bo, inj)
                    tag = f"dist={dist} M={m}"
                    if not torch.equal(s_res, f_res):
                        return False, f"{tag}: residual not bitwise equal"
                    if not bool(torch.isfinite(f_bi).all()):
                        return False, f"{tag}: non-finite block_input"
                    r_bi, r_inj = _hcf_ref64(module, weights, hs, bo, inj)
                    pairs = [("block_input", s_bi, f_bi, r_bi)]
                    if module.use_combine:
                        pairs.append(("injection", s_inj, f_inj, r_inj))
                    for name, s, f, r in pairs:
                        bad = _hcf_elem_bad(s, f, r)
                        if bad:
                            return False, f"{tag}: {name} {bad} elements with gross error"
                        acc = err2.setdefault(name, [0.0, 0.0])
                        acc[0] += float((s.double() - r).square().sum())
                        acc[1] += float((f.double() - r).square().sum())
            ratio = {k: (v[1] / v[0]) ** 0.5 if v[0] > 0 else (0.0 if v[1] == 0 else 9e9) for k, v in err2.items()}
            for name, q in ratio.items():
                if q > 2.0:
                    return False, f"{name} pooled err vs fp64 {q:.2f}x stock (> 2x)"
            for gm in (4, 32) if torch.device(device).type == "cuda" else ():
                hs, bo, inj = _hcf_inputs(0, gm, hc, h, device, gen)
                bo = bo if with_block else None
                inj = inj if (with_block and with_inj) else None
                eager = _hcf_run(module, weights, hs, bo, inj)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    _hcf_run(module, weights, hs, bo, inj)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                # thread_local: this may run lazily mid-serve; do not make other
                # threads' CUDA calls fail while the test graph is captured.
                with torch.cuda.graph(g, capture_error_mode="thread_local"):
                    graphed = _hcf_run(module, weights, hs, bo, inj)
                for _ in range(2):
                    g.replay()
                    torch.cuda.synchronize()
                    for a, b in zip(eager, graphed):
                        if a is not None and not torch.equal(a, b):
                            return False, f"M={gm}: graph replay != eager fused"
                del g
    except Exception as e:  # noqa: BLE001 - fail closed
        return False, f"{type(e).__name__}: {e}"[:300]
    finally:
        _HCF_STATE["busy"] = False
    return True, "pooled err/stock " + ", ".join(f"{k}={v:.2f}" for k, v in ratio.items())


def hcf_dispatch(module, hidden_states, block_output, injection):
    """Fused result tuple, or None to run the stock path."""
    st = _HCF_STATE
    if not st["enabled"] or st["failed"] or st["busy"] or not _HCF_HAS_TRITON:
        return None
    m = hidden_states.shape[0]
    if not (hidden_states.is_cuda and 1 <= m <= HCF_MAX_M and hidden_states.dim() == 2):
        return None
    if hidden_states.dtype != torch.bfloat16 or hidden_states.stride(1) != 1:
        return None
    if block_output is not None and (
        block_output.dtype != torch.bfloat16 or block_output.stride(1) != 1
    ):
        return None
    if injection is not None and (block_output is None or injection.stride(1) != 1):
        return None
    weights = _hcf_weights(module)
    if weights is None:
        return None
    key = _hcf_key(module, block_output, injection)
    if key not in st["tested"]:
        if torch.cuda.is_current_stream_capturing():
            return None  # never self-test inside a capture; stock for this graph
        _hcf_counters(hidden_states.device)
        ok, detail = hcf_self_test(
            module, block_output is not None, injection is not None, hidden_states.device
        )
        if not ok:
            st["failed"] = True
            _hcf_logger.warning("K5 hc_fused self-test FAILED %s: %s; stock HC path", key, detail)
            return None
        st["tested"].add(key)
        _hcf_logger.info("K5 hc_fused self-test passed %s: %s", key, detail)
    return _hcf_run(module, weights, hidden_states, block_output, injection)
