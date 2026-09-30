# R11-OVERLAY
# base_image_digest: sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
# upstream_file: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
# upstream_file_sha256: b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a
# upstream_PR: none (local K3; deferred-commit pattern of in-tree KDA RecoverSSM)
# generator: docker/v030/apply_gdn_lazy_overlay.py (do not hand-edit)
# embedded: docker/v030/gdn_lazy.py sha256 83b5c9e1d2e5b9fc56c0fc61904c9df4bdcd64d87851db424b55a25d016b7ce5
# features: K3 lazy GDN state commit for MTP verify, opt-in via VLLM_QWEN38_GDN_LAZY=1,
#   default off, self-tested (bitwise vs stock) at layer init; pair with
#   docker/v030/gdn_lazy_attn.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import os
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops import (
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

# Optional ROCm AITER Triton kernels for the GDN decode path.
# Availability is checked centrally via rocm_aiter_ops; the actual function
# references are imported here so that they can be called without per-call
# import overhead.
GDN_AITER_TRITON_AVAILABLE = (
    rocm_aiter_ops.are_gdn_triton_kernels_available()
    or rocm_aiter_ops.is_rdna_gdn_triton_kernels_available()
)

if GDN_AITER_TRITON_AVAILABLE:
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token as gdn_aiter_fused_reshape_causal_conv1d_update_single_token,  # noqa: E501
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule,  # noqa: E501
    )

logger = init_logger(__name__)

MAX_FUSED_GDN_MTP_TOKENS = 8
FUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)


def _resolve_gdn_prefill_backend(
    vllm_config: VllmConfig,
) -> tuple[str, Literal["triton", "flashinfer", "cutedsl"]]:
    """Resolve GDN prefill backend.

    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) - no further constraints;
      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``;
      - Blackwell (SM12.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``.

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;
    """
    additional_config = vllm_config.additional_config
    backend_cfg = (
        additional_config.get("gdn_prefill_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    backend = str(backend_cfg).strip().lower()

    if not current_platform.is_cuda():
        return backend, "triton"

    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )

    supports_flashinfer = False
    supports_cutedsl = False

    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True
    elif (
        current_platform.is_device_capability_family(120)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        # The in-tree CuteDSL kernel targets SM100 only, so it stays off here.
        supports_flashinfer = True

    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"


def _log_gdn_backend_decision(
    vllm_config: VllmConfig,
    requested_backend: str,
    active_backend: str,
) -> None:
    """Log the GDN prefill backend choice in the attention-selector style."""
    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )

    if current_platform.is_cpu():
        logger.info_once(
            "Using %s GDN prefill kernel (head_k_dim=%s).",
            "CPU",
            head_k_dim,
        )
        return

    chosen = {
        "flashinfer": "FlashInfer",
        "cutedsl": "CuteDSL",
        "triton": "Triton/FLA",
    }[active_backend]
    logger.info_once(
        "Using %s GDN prefill kernel (requested=%s, head_k_dim=%s).",
        chosen,
        requested_backend,
        head_k_dim,
    )
    if active_backend == "flashinfer" and current_platform.is_device_capability(90):
        logger.warning_once(
            "FlashInfer GDN prefill is JIT-compiled; first run may take a "
            "while. Set --gdn-prefill-backend triton to skip JIT.",
        )


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(torch.int64)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        backend, active_backend = _resolve_gdn_prefill_backend(vllm_config)
        self.gdn_prefill_backend = active_backend

        if backend in ("flashinfer", "cutedsl") and active_backend != backend:
            logger.warning_once(
                "GDN prefill backend '%s' is selected but cannot use this "
                "kernel on the current platform. Falling back to Triton/FLA.",
                backend,
            )
        _log_gdn_backend_decision(vllm_config, backend, active_backend)

        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        else:
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )

    def forward_cutedsl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            chunk_gated_delta_rule_cutedsl,
        )

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        assert cu_seqlens is not None
        assert chunk_indices is not None
        assert chunk_offsets is not None

        o, final_state = chunk_gated_delta_rule_cutedsl(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        if not output_final_state:
            final_state = None
        return o, final_state


@PluggableLayer.register("qwen_gated_delta_net_attention")
class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        gqa_interleaved_layout=False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.gqa_interleaved_layout = gqa_interleaved_layout
        if current_platform.is_xpu():
            self._forward_method = self.forward_xpu
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.mamba.ops.cpu.gdn_attention import (
                register_cpu_gdn_attention_ops,
            )

            register_cpu_gdn_attention_ops()
            self._forward_method = self.forward_cpu
        elif current_platform.is_rocm():
            self._forward_method = self.forward_hip
        else:
            self._forward_method = self.forward_cuda

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),
        # in_proj_qkv and in_proj_z are created separately instead.
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )

        # ba_proj doesn't support blockwise fp8 quantization.
        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint
        # layouts, so we use a factory method to create the projection.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )
        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                query_key_settings,
                query_key_settings,
                value_settings,
            ],
            self.tp_size,
            self.tp_rank,
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "swish", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
            device=current_platform.current_device(),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=reduce_results,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        self.gdn_prefill_backend = self.chunk_gated_delta_rule.gdn_prefill_backend
        self._prefill_kernels_warmed_up = False
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )
        self.gdn_decode_kernel = envs.VLLM_GDN_DECODE_KERNEL.strip().lower()
        if self.gdn_decode_kernel == "cuda" and current_platform.is_cuda_alike():
            reason = self._fused_gdn_decode_unsupported_reason(vllm_config)
            if reason is not None:
                if "VLLM_GDN_DECODE_KERNEL" in os.environ:
                    raise ValueError(
                        f"VLLM_GDN_DECODE_KERNEL=cuda is not supported: {reason}"
                    )
                logger.info_once(
                    "Falling back to the Triton GDN decode path: %s", reason
                )
                self.gdn_decode_kernel = "triton"
        elif current_platform.is_cpu():
            self.gdn_decode_kernel = "CPU"

        self.enable_fused_gdn_decode = self.gdn_decode_kernel == "cuda"
        logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)
        gdl_layer_init(self)  # K3: self-test before any cudagraph capture

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _fused_gdn_decode_unsupported_reason(
        self, vllm_config: VllmConfig
    ) -> str | None:
        conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()
        if (
            self.gqa_interleaved_layout
            or self.head_k_dim != 128
            or self.head_v_dim != 128
            or self.norm.activation not in ("silu", "sigmoid")
            or vllm_config.model_config.dtype != torch.bfloat16
            or conv_state_dtype != torch.bfloat16
            or recurrent_state_dtype not in FUSED_GDN_STATE_DTYPES
            or not current_platform.has_device_capability(80)
        ):
            return (
                "the fused CUDA kernel requires a BF16 GDN model with "
                "K=V=128, SiLU or sigmoid gating, non-interleaved GQA "
                "layout, BF16 convolution cache, BF16 or FP32 recurrent "
                "state, and a GPU with compute capability 8.0+"
            )
        if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
            return "torch.ops._C.fused_gdn_decode_post_conv_mtp is not built"
        return None

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), qkvz weights are
        # stored as a single fused tensor with interleaved GQA layout, so we
        # use one output shard to preserve the interleaving across TP ranks.
        # When gqa_interleaved_layout=False (Qwen3.5), the checkpoint has
        # separate q, k, v, z weights, so we use 4 independent output sizes.
        output_sizes = (
            [sum((key_dim, key_dim, value_dim, value_dim))]
            if self.gqa_interleaved_layout
            else [key_dim, key_dim, value_dim, value_dim]
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), in_proj_ba is stored
        # as a single fused weight [b_g0, a_g0, b_g1, a_g1, ...] interleaved
        # by key-head group; a single output shard preserves this across TP.
        # When gqa_interleaved_layout=False (Qwen3.5), in_proj_b and in_proj_a
        # are separate checkpoint weights, so we use 2 independent output sizes.
        output_sizes = (
            [num_v_heads * 2] if self.gqa_interleaved_layout else [num_v_heads] * 2
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=self.maybe_disable_tp(quant_config),
        )

    def maybe_disable_tp(self, quant_config: QuantizationConfig | None) -> bool:
        """Whether to replicate ba_proj instead of TP-sharding it.

        Marlin requires output_size_per_partition >= MIN_THREAD_N=64, which
        the Qwen3.5 non-interleaved [num_v_heads]*2 layout violates at TP>=2
        (e.g. num_v_heads=64, TP=4 -> 16). Replicating the projection keeps
        each rank above the Marlin threshold; forward() then slices b/a to
        the local TP partition. Qwen3-Next's interleaved [num_v_heads*2]
        layout is unaffected and stays TP-sharded.

        See https://github.com/vllm-project/vllm/issues/35924
        """
        return (
            current_platform.is_cuda()
            and not self.gqa_interleaved_layout
            and isinstance(quant_config, (AutoAWQConfig, AutoGPTQConfig, INCConfig))
        )

    def split_ba(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, a = ba.chunk(2, dim=-1)
        if self.disable_tp_for_ba_proj and self.tp_size > 1:
            # ba_proj is replicated for Marlin; slice b/a to local TP rank.
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]
        return b, a

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    @torch.compile(fullgraph=True)
    def prepare_gdn_attention_core_inputs(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_tokens: int,
    ):
        """
        Derives mixed_qkv, z, b, a from projected qkvz/ba for the GDN custom op.

        For gqa_interleaved_layout (Qwen3-Next): unpack the interleaved
        [ng, (hk + hk + np/ng*hv + np/ng*hv)] layout into contiguous qkv.
        For non-interleaved layout (Qwen3.5): simple split along last dim.
        """
        if not self.gqa_interleaved_layout:
            # Qwen3.5: weights are in [q, k, v, z] order
            assert num_tokens == mixed_qkvz.shape[0]
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z_flat = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            n = mixed_qkvz.shape[0]
            z_out = z_flat.reshape(n, -1, self.head_v_dim)
            b, a = mixed_ba.chunk(2, dim=-1)
            return mixed_qkv, z_out, b, a

        # Qwen3-Next: interleaved GQA layout
        base_shape_qkvz = mixed_qkvz.size()[:-1]
        base_shape_ba = mixed_ba.size()[:-1]
        ng = self.num_k_heads // self.tp_size

        new_tensor_shape_qkvz = base_shape_qkvz + (
            ng,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = base_shape_ba + (
            ng,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

        mixed_qkv_logical = torch.cat(
            [
                query.reshape(num_tokens, -1),
                key.reshape(num_tokens, -1),
                value.reshape(num_tokens, -1),
            ],
            dim=-1,
        )

        # The split above produces non-contiguous views into the interleaved
        # buffer.  Concatenating everything into a single flat tensor forces a
        # contiguous copy, then slicing back out gives contiguous q/k/v/z/b/a
        # tensors that downstream kernels require.  Doing this in one cat+slice
        # keeps torch.compile in a single Triton graph instead of emitting
        # separate copy kernels per tensor.  The original code used
        # rearrange(...).contiguous() on each tensor individually.
        fused = torch.cat(
            [
                mixed_qkv_logical.reshape(-1),
                z.reshape(-1),
                b.reshape(-1),
                a.reshape(-1),
            ],
            dim=0,
        )

        curr = 0
        qkv_numel = mixed_qkv_logical.numel()
        z_numel = z.numel()
        b_numel = b.numel()
        a_numel = a.numel()

        mixed_qkv_out = fused[curr : curr + qkv_numel].view(num_tokens, -1)
        curr += qkv_numel

        z_out = fused[curr : curr + z_numel].view(
            num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim
        )
        curr += z_numel

        b_out = fused[curr : curr + b_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )
        curr += b_numel

        a_out = fused[curr : curr + a_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )

        return mixed_qkv_out, z_out, b_out, a_out

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

        The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
        followed by ``.contiguous()`` on each tensor.  This version flattens
        all three splits into a single buffer via ``torch.cat`` so that
        torch.compile emits one Triton copy kernel instead of three separate
        contiguous() calls.
        """
        if mixed_qkv is None:
            return None, None, None

        seq_len = mixed_qkv.shape[0]
        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size

        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        fused = torch.cat(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
        )

        q_size = seq_len * q_dim
        k_size = seq_len * k_dim

        q_contig = fused[0:q_size]
        k_contig = fused[q_size : q_size + k_size]
        v_contig = fused[q_size + k_size :]

        query = q_contig.view(1, seq_len, -1, self.head_k_dim)
        key = k_contig.view(1, seq_len, -1, self.head_k_dim)
        value = v_contig.view(1, seq_len, -1, self.head_v_dim)

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_method(hidden_states)

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        output, _ = self.out_proj(core_attn_out)
        return output

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """ROCm forward using AITER Triton fused projection+attention when
        available, otherwise falling back to the generic CUDA path."""
        if GDN_AITER_TRITON_AVAILABLE:
            num_tokens = hidden_states.size(0)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = projected_states_qkvz.view(num_tokens, -1)
            projected_states_ba = projected_states_ba.view(num_tokens, -1)
            core_attn_out = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=projected_states_qkvz.dtype,
                device=projected_states_qkvz.device,
            )

            torch.ops.vllm.qwen_gdn_attention_core(
                projected_states_qkvz,
                projected_states_ba,
                z,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
                use_aiter=True,
            )

            return self._output_projection(core_attn_out, z)
        else:
            return self.forward_cuda(hidden_states)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        use_fused_gdn_decode = (
            self.enable_fused_gdn_decode
            and hidden_states.dtype == torch.bfloat16
            and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
        )
        if use_fused_gdn_decode:
            core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                mixed_qkvz,
                ba,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
            )
            output, _ = self.out_proj(core_attn_out.flatten(-2))
            return output

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self.split_ba(ba)

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b.contiguous(),
            a.contiguous(),
            core_attn_out,
            layer_name=_encode_layer_name(self.prefix),
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        return self._output_projection(core_attn_out, z)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert not hasattr(self, "in_proj_qkv"), "lora isn't supported on CPU."

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)

        num_tokens = hidden_states.size(0)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.cpu_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        """Warm up GDN prefill kernels during V1 profiling.

        During V1 profile runs, ``_forward_core`` returns early because
        ``attn_metadata`` is ``None``, so the autotuned kernels used by
        ``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        ``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        vLLM allocates KV cache using most of the remaining GPU memory.
        When the first real inference triggers the autotuner it OOMs
        because there is not enough memory left for benchmarking.

        This method runs minimal forward passes through
        ``chunk_gated_delta_rule`` with small dummy tensors to force
        autotuning while GPU memory is still plentiful.  The autotuner
        results are cached globally, so only the first layer incurs
        actual benchmarking cost.

        All kernels including ``chunk_fwd_kernel_o`` now use a fixed
        ``BT = chunk_size`` (64).  A single warmup pass with T = 64
        is sufficient to populate the autotuner cache.

        The decode path uses ``gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule``
        which has fixed kernel parameters (no autotuning), so only the
        prefill (chunked) path needs warming up.
        """
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True

        device = qkv_or_qkvz.device
        dtype = qkv_or_qkvz.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Mirror the real
        # prefill path here: build q/k/v/g/beta via fused_post_conv_prep and
        # then run chunk_gated_delta_rule with in-kernel L2 norm disabled.
        T = FLA_CHUNK_SIZE
        dummy_mixed_qkv = torch.randn(
            T, qkv_or_qkvz.shape[-1] - v_dim, device=device, dtype=dtype
        )
        dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=dummy_mixed_qkv,
            a=dummy_a,
            b=dummy_b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
        state = torch.zeros(
            1,
            num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

        # CuteDSL kernels require metadata
        chunk_indices = None
        chunk_offsets = None
        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, T)

        try:
            self.chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
        except Exception:
            logger.warning(
                "GDN prefill kernel warmup (T=%d) failed for "
                "layer %s. First inference may OOM due to "
                "autotuner.",
                T,
                self.prefix,
                exc_info=True,
            )
        else:
            logger.debug(
                "GDN prefill kernel warmup (T=%d) completed for layer %s",
                T,
                self.prefix,
            )
        finally:
            del (
                dummy_mixed_qkv,
                q,
                k,
                v,
                dummy_a,
                dummy_b,
                g,
                beta,
                state,
                cu_seqlens,
                chunk_indices,
                chunk_offsets,
            )

        torch.accelerator.empty_cache()

    def _forward_core_rocm(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """ROCm AITER fast path: conv1d + recurrent attention from packed
        qkvz/ba layout.

        For decode-only (no spec, no prefill) interleaved-GQA layouts,
        dispatches directly to ``_forward_core_decode_aiter``. Otherwise unpacks
        the packed layout and falls through to ``_forward_core``.

        Args:
            qkvz: packed [q, k, v, z] projection (num_tokens, qkvz_dim)
            ba:   packed [b, a] gating vectors    (num_tokens, 2*num_heads)
            z_out: **output** buffer for z        (num_tokens, num_heads,
                   head_dim); mutated in-place.
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            v_dim = core_attn_out.shape[-1] * core_attn_out.shape[-2]
            self._warmup_prefill_kernels(qkvz, v_dim)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # The AITER fused reshape/conv kernel expects Qwen3-Next's interleaved
        # GQA layout. Qwen3.5 uses a non-interleaved q/k/v/z layout and must use
        # the generic path below to split/rearrange inputs correctly.
        if (
            self.gqa_interleaved_layout
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_aiter(
                qkvz=qkvz,
                ba=ba,
                z_out=z_out,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        core_attn_out.zero_()
        num_tokens_all = qkvz.shape[0]
        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(
            qkvz, ba, num_tokens_all
        )
        z_out[:] = z
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
        )

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)
        gdl_fixup(self, attn_metadata)  # K3: pending rings -> stock layout

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                a_spec = a
                b_spec = b
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        # Split mixed non-spec-decode+prefill to process independently
        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            if split_non_spec:
                conv_output_prefill = mixed_qkv_non_spec[num_decode_tokens:]
                a_prefill = a_non_spec[num_decode_tokens:]
                b_prefill = b_non_spec[num_decode_tokens:]
            else:
                conv_output_prefill = mixed_qkv_non_spec
                a_prefill = a_non_spec
                b_prefill = b_non_spec

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = fused_post_conv_prep(
                conv_output=conv_output_prefill,
                a=a_prefill,
                b=b_prefill,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=self.num_k_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a_spec,
                    b=b_spec,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_spec_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process non-spec-decode part
        if split_non_spec:
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec[:num_decode_tokens]  # type: ignore[index]
            )
            core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a[:num_decode_tokens],
                b=b[:num_decode_tokens],
                dt_bias=self.dt_bias,
                q=query_decode,
                k=key_decode,
                v=value_decode,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_decode = None

        # 2.3: Process the remaining part (prefill chunk, or non-spec decode-only)
        if attn_metadata.num_prefills > 0:
            # State indices, initial-state mask and cu_seqlens for the chunk
            # kernel are precomputed by the metadata builder (the prefill tail
            # when decodes are peeled off, else the full non-spec batch), so they
            # don't need to be re-derived per layer.
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            initial_state = ssm_state[prefill_state_indices]
            initial_state[~prefill_has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=attn_metadata.prefill_query_start_loc,
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
            # Init cache
            ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)

            if split_non_spec:
                # Stitch the peeled decode outputs in front of the prefill
                # outputs (decode-first order).
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec], dim=1
                )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_core_decode_aiter(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_non_spec, b, a = (
            gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
                qkvz,
                attn_metadata.num_actual_tokens,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
                ba,
                z_out,
                core_attn_out,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        )

        # 2. Recurrent attention
        gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            qkv=mixed_qkv_non_spec,
            key_dim=self.key_dim // self.tp_size,
            value_dim=self.value_dim // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],  # type: ignore[index]
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=core_attn_out.reshape(-1),
        )

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """
        Core attention computation with a packed non-spec decode fast path.
        """
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )
        return

    def _forward_core_decode_spec_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        state_indices = attn_metadata.spec_state_indices_tensor
        cu_seqlens = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert state_indices is not None
        assert cu_seqlens is not None
        assert num_accepted_tokens is not None

        num_requests = attn_metadata.num_spec_decodes
        num_actual_tokens = attn_metadata.num_actual_tokens
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv = causal_conv1d_update(
            mixed_qkv[:num_actual_tokens],
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=state_indices[:num_requests, 0],
            num_accepted_tokens=num_accepted_tokens[:num_requests],
            query_start_loc=cu_seqlens[: num_requests + 1],
            max_query_len=state_indices.size(1),
            validate_data=False,
        )
        self._forward_core_decode_spec_post_conv_fused_norm(
            mixed_qkv=mixed_qkv,
            b=b[:num_actual_tokens],
            a=a[:num_actual_tokens],
            output_gate=output_gate[:num_actual_tokens],
            core_attn_out=core_attn_out[:num_actual_tokens],
            attn_metadata=attn_metadata,
        )

    def _forward_core_decode_spec_post_conv_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        state_indices = attn_metadata.spec_state_indices_tensor
        cu_seqlens = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert state_indices is not None
        assert cu_seqlens is not None
        assert num_accepted_tokens is not None

        num_requests = attn_metadata.num_spec_decodes
        if gdl_decode(self, mixed_qkv, a, b, output_gate, core_attn_out, attn_metadata):
            return  # K3 lazy commit (docker/v030/gdn_lazy.py)
        ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_indices=state_indices[:num_requests],
            cu_seqlens=cu_seqlens[: num_requests + 1],
            num_accepted_tokens=num_accepted_tokens[:num_requests],
            state=self.kv_cache[1],
            output_gate=output_gate,
            norm_weight=self.norm.weight,
            out=core_attn_out,
            scale=self.head_k_dim**-0.5,
            norm_eps=self.layer_norm_epsilon,
            output_gate_activation=self.norm.activation,
        )

    def _forward_core_fused_norm_packed(
        self,
        mixed_qkvz: torch.Tensor,
        ba: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkvz[:, :qkv_size], 0)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)
        mixed_qkv, output_gate_flat = mixed_qkvz.split(
            [qkv_size, self.value_dim // self.tp_size], dim=-1
        )
        output_gate = output_gate_flat.reshape(
            output_gate_flat.size(0), -1, self.head_v_dim
        )
        b, a = self.split_ba(ba)
        self._forward_core_fused_norm(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            output_gate=output_gate,
            core_attn_out=core_attn_out,
        )

    def _can_use_fused_gdn_mtp_decode(
        self, attn_metadata: GDNAttentionMetadata
    ) -> bool:
        state_indices = attn_metadata.spec_state_indices_tensor
        return (
            attn_metadata.spec_sequence_masks is not None
            and attn_metadata.num_decodes == 0
            and attn_metadata.num_spec_decodes > 0
            and self.kv_cache[1].dtype in FUSED_GDN_STATE_DTYPES
            and self.gdn_decode_kernel == "cuda"
            and self.num_v_heads % self.num_k_heads == 0
            and self.num_v_heads // self.num_k_heads in (1, 2, 3, 4, 8)
            and state_indices is not None
            and state_indices.size(1) <= MAX_FUSED_GDN_MTP_TOKENS
            and hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")
        )

    def _rms_norm_gated_cuda(
        self,
        x: torch.Tensor,
        output_gate: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
            layer_norm_fwd,
        )

        x_shape = x.shape
        assert output_gate.shape == x_shape
        assert out.shape == x_shape
        x_2d = x.reshape(-1, x_shape[-1])
        output_gate_2d = output_gate.reshape(-1, x_shape[-1])
        out_2d = out.reshape(-1, x_shape[-1])
        assert x_2d.stride(-1) == 1
        assert output_gate_2d.stride(-1) == 1
        assert out_2d.stride(-1) == 1
        layer_norm_fwd(
            x_2d,
            self.norm.weight.contiguous(),
            self.norm.bias,
            self.norm.eps,
            z=output_gate_2d,
            out=out_2d,
            group_size=(
                x_shape[-1] if self.norm.group_size is None else self.norm.group_size
            ),
            norm_before_gate=self.norm.norm_before_gate,
            is_rms_norm=True,
            activation=self.norm.activation,
        )

    def _forward_core_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata = None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.prefix)
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata, GDNAttentionMetadata)
        if (
            self._can_use_fused_gdn_mtp_decode(attn_metadata)
            and attn_metadata.num_prefills == 0
        ):
            self._forward_core_decode_spec_fused_norm(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                output_gate=output_gate,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )
            return
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b.contiguous(),
            a=a.contiguous(),
            core_attn_out=core_attn_out,
        )
        num_actual_tokens = attn_metadata.num_actual_tokens
        self._rms_norm_gated_cuda(
            core_attn_out[:num_actual_tokens],
            output_gate[:num_actual_tokens],
            core_attn_out[:num_actual_tokens],
        )


@eager_break_during_capture
def qwen_gdn_attention_core(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    use_aiter: bool = False,
) -> None:
    """Custom op dispatching to _forward_core or _forward_core_rocm.

    Handles conv1d + recurrent attention only; input/output projections
    are performed by the caller.

    When ``use_aiter=False`` (standard path):
        qkv_or_qkvz is [q, k, v], b_or_ba is b, a_or_z_out is a (read-only).
    When ``use_aiter=True`` (AITER Triton path, ROCm only):
        qkv_or_qkvz is [q, k, v, z], b_or_ba is [b, a], a_or_z_out is the
        z output buffer (mutated in-place).

    ``core_attn_out`` is always mutated in-place.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if use_aiter:
        self._forward_core_rocm(
            qkvz=qkv_or_qkvz,
            ba=b_or_ba,
            z_out=a_or_z_out,
            core_attn_out=core_attn_out,
        )
    else:
        self._forward_core(
            mixed_qkv=qkv_or_qkvz,
            b=b_or_ba,
            a=a_or_z_out,
            core_attn_out=core_attn_out,
        )


direct_register_custom_op(
    op_name="qwen_gdn_attention_core",
    op_func=qwen_gdn_attention_core,
    mutates_args=["a_or_z_out", "core_attn_out"],
)


@eager_break_during_capture
def qwen_gdn_attention_core_fused_norm_packed(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core_fused_norm_packed(
        mixed_qkvz=mixed_qkvz,
        ba=ba,
        core_attn_out=core_attn_out,
    )


direct_register_custom_op(
    op_name="qwen_gdn_attention_core_fused_norm_packed",
    op_func=qwen_gdn_attention_core_fused_norm_packed,
    mutates_args=["core_attn_out"],
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output


# ---- K3 embedded from docker/v030/gdn_lazy.py (do not hand-edit) ----
# K3 GDN MTP decode with lazy state commit (plan section 4 K3, L3; finding F08).
#
# Source of truth for the kernels. docker/v030/apply_gdn_lazy_overlay.py embeds
# this file verbatim at the end of the v0.30.0 qwen_gdn_linear_attn.py overlay
# (docker/v030/gdn_lazy_linear_attn.py), so it must stay a plain module: no
# __future__ import, every top-level name prefixed _gdl / gdl / GDL.
# Design, slot scheme and evidence: docker/v030/K3.md.
#
# Stock v0.30 verify step (fused_gdn_decode_post_conv_mtp_kernel<float,3,true>,
# grid (req, HV), 256 threads): read the committed fp32 state from
# state_indices[req, a-1], run the T <= k+1 verify tokens, store the state after
# EVERY token t to state_indices[req, t] (1 read + T writes of 64 KiB per head).
#
# K3, same grid, one kernel per layer per step:
#   lazy row  : load S_c from column 0, replay the n <= T_prev accepted records of
#               the previous step from the ring, store S_c ONCE to column 0, run
#               the T verify tokens producing outputs only, append T records to
#               the ring (1 read + 1 write + ~4 KiB of ring traffic per head).
#   eager row : exactly the stock layout (after replaying any pending records):
#               candidate t stored to column t. Used when a 1600-token align
#               boundary is within 2k+1 tokens (so the V2 align pre-copy and
#               checkpoint copy only ever see the stock layout), when the ring
#               column is NULL, or when lazy is off but pending rings exist.
#   ring      : the exact stock-kernel INPUTS of each verify token, BF16 (post-
#               conv k row, v row, raw a, raw b; 516 B per head per token), held
#               in the tail (rows 120..127) of the head's region of the COLUMN-1
#               state slot, which the lazy layout never uses for state.
#   header    : int32 per (layer, state block): number of pending records whose
#               committed base S_c sits in that block (column 0). 0 = stock
#               layout. Written by a separate tiny kernel after the decode
#               kernel so every CTA of the step reads the same value.
#   fixup     : before ANY stock reader of the state (mixed batches, non-spec
#               rows, prefills), _gdl_materialize replays pending rings into the
#               exact stock layout (candidate t -> column t) and zeroes headers;
#               prefilling rows only zero headers (stale ring of a dead request).
#
# Numerics: the per-token update, l2norm, gating, butterfly reductions (xor
# 16..1 order, rebuilt as a split tree), gated RMSNorm and bf16 roundings follow
# the stock SASS op-for-op (FTZ inline PTX: mul/add/sub/fma .ftz, ex2.approx.ftz,
# rcp.approx.ftz, rsqrt.approx.ftz, libdevice log1pf). Target class A0 vs the
# stock CUDA kernel; the load-time self-test REQUIRES bitwise equality on GPU
# (outputs every step + committed state), else K3 stays off (fail closed).
# Replay reuses the same jit functions as the in-step update, so lazy == eager
# bitwise by construction (proven on CPU under TRITON_INTERPRET in
# tests/test_gdn_lazy.py, where the PTX helpers fall back to plain ops).
#
# Default OFF: VLLM_QWEN38_GDN_LAZY=1 on BOTH ranks, CUDA, fp32 SSM state,
# K=V=128, W=k+1 in 2..8, mamba_cache_mode none/align, and the self-test passed.
import logging
import os

import torch

try:
    import triton
    import triton.language as tl

    _GDL_HAS_TRITON = True
except Exception:  # noqa: BLE001 - no triton means stock path
    _GDL_HAS_TRITON = False

GDL_ENV = "VLLM_QWEN38_GDN_LAZY"
GDL_DIM = 128  # K = V = 128 (stock kernel requirement)
GDL_BV = 32  # V rows per chunk (stock kChunkV)
GDL_TMAX = 8  # stock kMaxMtpTokens
# Ring at the tail of each head's 64 KiB region of the column-1 slot, in bf16
# elements: TMAX records of [k(128) | v(128)], then TMAX a, then TMAX b.
GDL_HEAD_BF16 = GDL_DIM * GDL_DIM * 2
GDL_RING_BF16 = GDL_TMAX * 2 * GDL_DIM + 2 * GDL_TMAX
GDL_RING_OFF = GDL_HEAD_BF16 - GDL_RING_BF16  # 30704 -> fp32 row 119.9 (chunk 3)
GDL_LOG2E = 1.4426950216293334961  # float32(log2 e), the stock __expf constant
GDL_MODE_AUTO = 0
GDL_MODE_EAGER = 1
# Reduction implementation (see _gdl_red32): 0 = warp reduce (GPU default),
# 1 = explicit split tree (CPU default; GDL_RED=1 forces it on GPU).
GDL_RED_ENV = "VLLM_QWEN38_GDN_LAZY_RED"

_gdl_logger = logging.getLogger("vllm.qwen38.gdn_lazy")
_GDL_STATE = {
    "env": os.environ.get(GDL_ENV, "0").strip().lower(),
    "failed": False,  # self-test failed: never start a lazy row
    "tested": set(),  # variant keys that passed the self-test
    "used": False,  # a lazy row may exist: fixups must run from now on
    "detail": "",
}


def gdl_enabled() -> bool:
    return _GDL_STATE["env"] in ("1", "force") and _GDL_HAS_TRITON


def _gdl_interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def _gdl_red() -> int:
    v = os.environ.get(GDL_RED_ENV)
    if v in ("0", "1"):
        return int(v)
    return 1 if _gdl_interpret() else 0


if _GDL_HAS_TRITON:
    from triton.language.extra import libdevice as _gdl_libdevice

    # ---------------------------------------------------------------- FTZ ops
    # GPU: inline PTX with .ftz, exactly the stock SASS (FMUL/FADD/FFMA.FTZ,
    # MUFU.EX2/RCP/RSQ) and never contracted or reassociated by LLVM.
    # INTERP (CPU tests): plain ops; lazy vs eager stays bitwise because both
    # paths call the same helpers.
    @triton.jit
    def _gdl_mul(x, y, INTERP: tl.constexpr):
        if INTERP:
            return x * y
        else:
            x, y = tl.broadcast(x, y)
            return tl.inline_asm_elementwise(
                "mul.ftz.f32 $0, $1, $2;", "=f,f,f", [x, y],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_add(x, y, INTERP: tl.constexpr):
        if INTERP:
            return x + y
        else:
            x, y = tl.broadcast(x, y)
            return tl.inline_asm_elementwise(
                "add.ftz.f32 $0, $1, $2;", "=f,f,f", [x, y],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_sub(x, y, INTERP: tl.constexpr):
        if INTERP:
            return x - y
        else:
            x, y = tl.broadcast(x, y)
            return tl.inline_asm_elementwise(
                "sub.ftz.f32 $0, $1, $2;", "=f,f,f", [x, y],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_fma(x, y, z, INTERP: tl.constexpr):
        if INTERP:
            return x * y + z
        else:
            x, y = tl.broadcast(x, y)
            x, z = tl.broadcast(x, z)
            x, y = tl.broadcast(x, y)
            return tl.inline_asm_elementwise(
                "fma.rn.ftz.f32 $0, $1, $2, $3;", "=f,f,f,f", [x, y, z],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_ex2(x, INTERP: tl.constexpr):
        if INTERP:
            return tl.exp2(x)
        else:
            return tl.inline_asm_elementwise(
                "ex2.approx.ftz.f32 $0, $1;", "=f,f", [x],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_rcp(x, INTERP: tl.constexpr):
        if INTERP:
            return 1.0 / x
        else:
            return tl.inline_asm_elementwise(
                "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_rsq(x, INTERP: tl.constexpr):
        if INTERP:
            return 1.0 / tl.sqrt(x)
        else:
            return tl.inline_asm_elementwise(
                "rsqrt.approx.ftz.f32 $0, $1;", "=f,f", [x],
                dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_log1p(x, INTERP: tl.constexpr):
        if INTERP:
            return tl.log(1.0 + x)
        else:
            # libdevice log1pf with nvvm-reflect FTZ: same polynomial as the
            # stock SASS; its 3 non-ftz ops only differ on subnormal inputs,
            # which ex2.approx.ftz never produces.
            return _gdl_libdevice.log1p(x)

    @triton.jit
    def _gdl_c(x, C: tl.constexpr):
        """fp32 constant broadcast to x's shape (inline asm needs tensors)."""
        return tl.full(x.shape, C, tl.float32)

    @triton.jit
    def _gdl_bf16(x, INTERP: tl.constexpr):
        """fp32 -> bf16 round-to-nearest-even (F2FP.BF16.F32.PACK_AB). The CPU
        interpreter truncates, so there the rounding is done on the bits."""
        if INTERP:
            b = x.to(tl.uint32, bitcast=True)
            b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
            return b.to(tl.float32, bitcast=True).to(tl.bfloat16)
        else:
            return x.to(tl.bfloat16)

    # ---------------------------------------------------------- reductions
    @triton.jit
    def _gdl_addc_gpu(a, b):
        return tl.inline_asm_elementwise(
            "add.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b],
            dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gdl_addc_cpu(a, b):
        return a + b

    @triton.jit
    def _gdl_red32(x, INTERP: tl.constexpr, RED: tl.constexpr):
        """[R, 32] -> [R] in the warp_reduce_sum order.
        RED=0: tl.reduce over axis 1. With the natural layout of a row-major
          [R, 128] tile (4 contiguous k per lane, 32 lanes per row) the lane axis
          holds one element per lane and Triton lowers the reduce to shfl.bfly
          16, 8, 4, 2, 1 - the stock order. Layout is the compiler's choice, so
          the self-test's bitwise gate is what proves it (fail closed).
        RED=1: explicit split tree (order guaranteed by construction; slow on
          GPU, used by the CPU tests and as a harness cross-check)."""
        if RED == 1:
            return _gdl_bfly(x, INTERP)
        else:
            if INTERP:
                return tl.reduce(x, 1, _gdl_addc_cpu)
            else:
                return tl.reduce(x, 1, _gdl_addc_gpu)

    @triton.jit
    def _gdl_bfly(x, INTERP: tl.constexpr):
        """[R, 32] -> [R]: the warp_reduce_sum xor-butterfly (16, 8, 4, 2, 1)
        as an explicit split tree. fp add is commutative, so lane j's result
        (and every lane's) equals this tree bit for bit."""
        R: tl.constexpr = x.shape[0]
        x = tl.permute(tl.reshape(x, [R, 2, 16]), (0, 2, 1))
        lo, hi = tl.split(x)
        x = _gdl_add(lo, hi, INTERP)
        x = tl.permute(tl.reshape(x, [R, 2, 8]), (0, 2, 1))
        lo, hi = tl.split(x)
        x = _gdl_add(lo, hi, INTERP)
        x = tl.permute(tl.reshape(x, [R, 2, 4]), (0, 2, 1))
        lo, hi = tl.split(x)
        x = _gdl_add(lo, hi, INTERP)
        x = tl.permute(tl.reshape(x, [R, 2, 2]), (0, 2, 1))
        lo, hi = tl.split(x)
        x = _gdl_add(lo, hi, INTERP)
        lo, hi = tl.split(x)
        return _gdl_add(lo, hi, INTERP)

    @triton.jit
    def _gdl_quad(v):
        """[128] -> 4 x [1, 32], v_i[j] = v[i*32 + j] (lane j holds dims
        j, j+32, j+64, j+96: the stock l2norm / RMSNorm mapping)."""
        t = tl.reshape(tl.permute(tl.reshape(v, [4, 32]), (1, 0)), [32, 2, 2])
        e, o = tl.split(t)
        v0, v2 = tl.split(e)
        v1, v3 = tl.split(o)
        return (tl.reshape(v0, [1, 32]), tl.reshape(v1, [1, 32]),
                tl.reshape(v2, [1, 32]), tl.reshape(v3, [1, 32]))

    @triton.jit
    def _gdl_lane4(x):
        """[R, 128] -> 4 x [R, 32], x_i[r, l] = x[r, 4l + i] (stock k_base =
        lane * 4 mapping of the state update)."""
        R: tl.constexpr = x.shape[0]
        e, o = tl.split(tl.reshape(x, [R, 32, 2, 2]))
        x0, x2 = tl.split(e)
        x1, x3 = tl.split(o)
        return x0, x1, x2, x3

    @triton.jit
    def _gdl_unlane4(x0, x1, x2, x3):
        R: tl.constexpr = x0.shape[0]
        t = tl.join(tl.join(x0, x2), tl.join(x1, x3))
        return tl.reshape(t, [R, 128])

    @triton.jit
    def _gdl_sumsq(v, INTERP: tl.constexpr, RED: tl.constexpr):
        """sum of squares of a [128] vector in the stock order -> [1]."""
        v0, v1, v2, v3 = _gdl_quad(v)
        s = _gdl_fma(v0, v0, tl.zeros_like(v0), INTERP)
        s = _gdl_fma(v1, v1, s, INTERP)
        s = _gdl_fma(v2, v2, s, INTERP)
        s = _gdl_fma(v3, v3, s, INTERP)
        return _gdl_red32(s, INTERP, RED)

    @triton.jit
    def _gdl_dot4(w0, w1, w2, w3, h0, h1, h2, h3, INTERP: tl.constexpr,
                  RED: tl.constexpr):
        """per-row dot(h[r, :], w) in the stock order: per lane
        fma(w_i, h_i, .) for i = 0..3 from 0, then the butterfly. -> [R]"""
        acc = _gdl_fma(w0, h0, tl.zeros_like(h0), INTERP)
        acc = _gdl_fma(w1, h1, acc, INTERP)
        acc = _gdl_fma(w2, h2, acc, INTERP)
        acc = _gdl_fma(w3, h3, acc, INTERP)
        return _gdl_red32(acc, INTERP, RED)

    # --------------------------------------------------------- token pieces
    @triton.jit
    def _gdl_qk(q_raw, k_raw, scale, INTERP: tl.constexpr, RED: tl.constexpr):
        """l2norm(q) * scale and l2norm(k) in lane4 layout ([1, 32] each)."""
        sq = _gdl_sumsq(q_raw, INTERP, RED)
        sk = _gdl_sumsq(k_raw, INTERP, RED)
        qs = _gdl_mul(_gdl_rsq(_gdl_add(sq, _gdl_c(sq, 1.0e-6), INTERP), INTERP),
                      scale, INTERP)
        ks = _gdl_rsq(_gdl_add(sk, _gdl_c(sk, 1.0e-6), INTERP), INTERP)
        q = _gdl_mul(q_raw, qs, INTERP)
        k = _gdl_mul(k_raw, ks, INTERP)
        q0, q1, q2, q3 = _gdl_lane4(tl.reshape(q, [1, 128]))
        k0, k1, k2, k3 = _gdl_lane4(tl.reshape(k, [1, 128]))
        return q0, q1, q2, q3, k0, k1, k2, k3

    @triton.jit
    def _gdl_k(k_raw, INTERP: tl.constexpr, RED: tl.constexpr):
        sk = _gdl_sumsq(k_raw, INTERP, RED)
        ks = _gdl_rsq(_gdl_add(sk, _gdl_c(sk, 1.0e-6), INTERP), INTERP)
        k = _gdl_mul(k_raw, ks, INTERP)
        return _gdl_lane4(tl.reshape(k, [1, 128]))

    @triton.jit
    def _gdl_gates(a_val, b_val, alog, dtb, INTERP: tl.constexpr):
        """decay = __expf(-__expf(A_log) * softplus(a + dt_bias)),
        beta = 1 / (1 + __expf(-b)); stock SASS order, all [1]."""
        x = _gdl_add(a_val, dtb, INTERP)
        ex = _gdl_ex2(_gdl_mul(x, _gdl_c(x, 1.4426950216293334961), INTERP), INTERP)
        sp = tl.where(x > 20.0, x, _gdl_log1p(ex, INTERP))
        e = _gdl_ex2(_gdl_mul(alog, _gdl_c(alog, 1.4426950216293334961), INTERP), INTERP)
        t = _gdl_mul(e, sp, INTERP)
        decay = _gdl_ex2(_gdl_mul(t, _gdl_c(t, -1.4426950216293334961), INTERP), INTERP)
        eb = _gdl_ex2(_gdl_mul(b_val, _gdl_c(b_val, -1.4426950216293334961), INTERP),
                      INTERP)
        beta = _gdl_rcp(_gdl_add(eb, _gdl_c(eb, 1.0), INTERP), INTERP)
        return decay, beta

    @triton.jit
    def _gdl_update(h0, h1, h2, h3, k0, k1, k2, k3, v, decay, beta,
                    INTERP: tl.constexpr, RED: tl.constexpr):
        """One delta-rule token on a [BV, 128] tile held as 4 x [BV, 32].
        h *= decay; hk = h.k; delta = (v - hk) * beta; h += k * delta."""
        h0 = _gdl_mul(h0, decay, INTERP)
        h1 = _gdl_mul(h1, decay, INTERP)
        h2 = _gdl_mul(h2, decay, INTERP)
        h3 = _gdl_mul(h3, decay, INTERP)
        hk = _gdl_dot4(k0, k1, k2, k3, h0, h1, h2, h3, INTERP, RED)
        delta = _gdl_mul(beta, _gdl_sub(v, hk, INTERP), INTERP)
        d = tl.reshape(delta, [delta.shape[0], 1])
        h0 = _gdl_fma(k0, d, h0, INTERP)
        h1 = _gdl_fma(k1, d, h1, INTERP)
        h2 = _gdl_fma(k2, d, h2, INTERP)
        h3 = _gdl_fma(k3, d, h3, INTERP)
        return h0, h1, h2, h3

    @triton.jit
    def _gdl_load1(ptr):
        return tl.load(ptr + tl.zeros([1], dtype=tl.int32)).to(tl.float32)

    @triton.jit
    def _gdl_replay_one(h0, h1, h2, h3, ring_rec, ring_ab, t, rows, alog, dtb,
                        INTERP: tl.constexpr, TMAX: tl.constexpr, RED: tl.constexpr):
        k_raw = tl.load(ring_rec + tl.arange(0, 128)).to(tl.float32)
        v = tl.load(ring_rec + 128 + rows).to(tl.float32)
        a_val = _gdl_load1(ring_ab + t)
        b_val = _gdl_load1(ring_ab + TMAX + t)
        k0, k1, k2, k3 = _gdl_k(k_raw, INTERP, RED)
        decay, beta = _gdl_gates(a_val, b_val, alog, dtb, INTERP)
        return _gdl_update(h0, h1, h2, h3, k0, k1, k2, k3, v, decay, beta, INTERP, RED)

    @triton.jit
    def _gdl_row_plan(si_row, W, T, acc, hdr_ptr, seq_len, BS, MODE: tl.constexpr,
                      TMAX: tl.constexpr):
        """Per-request decisions shared by the decode and header kernels.
        Returns (valid, src_slot, n_replay, lazy, col0, ring_slot)."""
        col0 = tl.load(si_row)
        ring_slot = tl.load(si_row + 1, mask=W > 1, other=0)
        r = tl.load(hdr_ptr + col0, mask=col0 > 0, other=0)
        pending = r > 0
        acc_ok = (acc > 0) & (acc <= W)
        stock_src = tl.load(si_row + acc - 1, mask=acc_ok, other=0)
        src = tl.where(pending, col0, stock_src)
        n_rep = tl.where(pending, tl.minimum(tl.maximum(acc, 1), r), 0)
        valid = (src > 0) & (T <= TMAX)
        if MODE == 0:
            lazy = ring_slot > 0
        else:
            lazy = ring_slot < 0  # never
        nc = seq_len - T
        lazy = lazy & ((BS <= 0) | ((nc // tl.maximum(BS, 1))
                                    == ((nc + 2 * W - 1) // tl.maximum(BS, 1))))
        return valid, src, n_rep, lazy, col0, ring_slot

    # ------------------------------------------------------------- kernels
    @triton.jit
    def _gdl_decode_kernel(
        mixed_ptr, stride_mixed,
        a_ptr, stride_a, b_ptr, stride_b,
        alog_ptr, dtb_ptr,
        si_ptr, stride_si,
        cu_ptr, acc_ptr, seqlen_ptr,
        state_ptr, ring_ptr, stride_slot, stride_slot_ring,
        hdr_ptr,
        gate_ptr, stride_gate,
        nw_ptr, out_ptr,
        H, HV, W, BS, scale, eps,
        G: tl.constexpr, SIGMOID: tl.constexpr, MODE: tl.constexpr,
        INTERP: tl.constexpr, TMAX: tl.constexpr, BV: tl.constexpr,
        RING_OFF: tl.constexpr, RED: tl.constexpr,
    ):
        req = tl.program_id(0)
        hv = tl.program_id(1)
        bos = tl.load(cu_ptr + req)
        eos = tl.load(cu_ptr + req + 1)
        T = eos - bos
        if T <= 0:
            return
        acc = tl.load(acc_ptr + req)
        seq_len = tl.load(seqlen_ptr + req)
        si_row = si_ptr + req.to(tl.int64) * stride_si
        valid, src, n_rep, lazy, col0, ring_slot = _gdl_row_plan(
            si_row, W, T, acc, hdr_ptr, seq_len, BS, MODE, TMAX)
        cols = tl.arange(0, 128)
        out_row = out_ptr + (bos.to(tl.int64) * HV + hv) * 128
        if valid == 0:
            for t in range(0, T):
                tl.store(out_row + t * HV * 128 + cols,
                         tl.zeros([128], dtype=tl.float32).to(tl.bfloat16))
            return
        kh = hv // G
        alog = _gdl_load1(alog_ptr + hv)
        dtb = _gdl_load1(dtb_ptr + hv)
        head_off = hv.to(tl.int64) * 128 * 128
        src_base = state_ptr + src.to(tl.int64) * stride_slot + head_off
        col0_base = state_ptr + col0.to(tl.int64) * stride_slot + head_off
        ring_head = (ring_ptr + ring_slot.to(tl.int64) * stride_slot_ring
                     + hv.to(tl.int64) * (2 * 128 * 128) + RING_OFF)
        ring_ab = ring_head + TMAX * 256
        rr = tl.arange(0, BV)
        for c in tl.static_range(128 // BV):
            rows = c * BV + rr
            tile_off = rows[:, None] * 128 + cols[None, :]
            h0, h1, h2, h3 = _gdl_lane4(tl.load(src_base + tile_off))
            for t in range(0, n_rep):
                h0, h1, h2, h3 = _gdl_replay_one(
                    h0, h1, h2, h3, ring_head + t * 256, ring_ab, t, rows,
                    alog, dtb, INTERP, TMAX, RED)
            if not INTERP:
                tl.debug_barrier()  # ring reads before any column-1 store
            if lazy:
                tl.store(col0_base + tile_off, _gdl_unlane4(h0, h1, h2, h3))
            for t in range(0, T):
                tok = bos + t
                mrow = mixed_ptr + tok.to(tl.int64) * stride_mixed
                q_raw = tl.load(mrow + kh * 128 + cols).to(tl.float32)
                k_raw = tl.load(mrow + H * 128 + kh * 128 + cols).to(tl.float32)
                v = tl.load(mrow + 2 * H * 128 + hv * 128 + rows).to(tl.float32)
                a_val = _gdl_load1(a_ptr + tok.to(tl.int64) * stride_a + hv)
                b_val = _gdl_load1(b_ptr + tok.to(tl.int64) * stride_b + hv)
                q0, q1, q2, q3, k0, k1, k2, k3 = _gdl_qk(q_raw, k_raw, scale, INTERP, RED)
                decay, beta = _gdl_gates(a_val, b_val, alog, dtb, INTERP)
                h0, h1, h2, h3 = _gdl_update(h0, h1, h2, h3, k0, k1, k2, k3, v,
                                             decay, beta, INTERP, RED)
                o = _gdl_dot4(q0, q1, q2, q3, h0, h1, h2, h3, INTERP, RED)
                tl.store(out_row + t * HV * 128 + rows, _gdl_bf16(o, INTERP))
                if not lazy:
                    dst = tl.load(si_row + t)
                    if dst > 0:
                        dst_base = (state_ptr + dst.to(tl.int64) * stride_slot
                                    + head_off)
                        tl.store(dst_base + tile_off, _gdl_unlane4(h0, h1, h2, h3))
        if not INTERP:
            tl.debug_barrier()  # raw o of all chunks visible; last ring read done
        # gated RMSNorm epilogue on the bf16-rounded outputs (stock shared_out)
        w = tl.load(nw_ptr + cols).to(tl.float32)
        for t in range(0, T):
            tok = bos + t
            o = tl.load(out_row + t * HV * 128 + cols).to(tl.float32)
            s = _gdl_sumsq(o, INTERP, RED)
            rstd = _gdl_rsq(_gdl_fma(s, _gdl_c(s, 0.0078125), eps, INTERP), INTERP)
            y = _gdl_mul(_gdl_mul(o, rstd, INTERP), w, INTERP)
            gi = tl.load(gate_ptr + tok.to(tl.int64) * stride_gate + hv * 128
                         + cols).to(tl.float32)
            eg = _gdl_ex2(_gdl_mul(gi, _gdl_c(gi, -1.4426950216293334961), INTERP),
                          INTERP)
            sig = _gdl_rcp(_gdl_add(eg, _gdl_c(eg, 1.0), INTERP), INTERP)
            if SIGMOID:
                gate = sig
            else:
                gate = _gdl_mul(gi, sig, INTERP)
            y = _gdl_mul(gate, y, INTERP)
            tl.store(out_row + t * HV * 128 + cols, _gdl_bf16(y, INTERP))
        if lazy:
            # append this step's exact inputs; replayed next step
            for t in range(0, T):
                tok = bos + t
                mrow = mixed_ptr + tok.to(tl.int64) * stride_mixed
                rec = ring_head + t * 256
                tl.store(rec + cols, tl.load(mrow + H * 128 + kh * 128 + cols))
                tl.store(rec + 128 + cols, tl.load(mrow + 2 * H * 128 + hv * 128 + cols))
                tl.store(ring_ab + t, tl.load(a_ptr + tok.to(tl.int64) * stride_a + hv))
                tl.store(ring_ab + TMAX + t,
                         tl.load(b_ptr + tok.to(tl.int64) * stride_b + hv))

    @triton.jit
    def _gdl_header_kernel(si_ptr, stride_si, cu_ptr, acc_ptr, seqlen_ptr, hdr_ptr,
                           W, BS, MODE: tl.constexpr, TMAX: tl.constexpr):
        req = tl.program_id(0)
        T = tl.load(cu_ptr + req + 1) - tl.load(cu_ptr + req)
        if T <= 0:
            return
        acc = tl.load(acc_ptr + req)
        seq_len = tl.load(seqlen_ptr + req)
        si_row = si_ptr + req.to(tl.int64) * stride_si
        valid, src, n_rep, lazy, col0, ring_slot = _gdl_row_plan(
            si_row, W, T, acc, hdr_ptr, seq_len, BS, MODE, TMAX)
        if valid:
            if lazy:
                tl.store(hdr_ptr + col0, T)
            else:
                for t in range(0, W):
                    s = tl.load(si_row + t)
                    if s > 0:
                        tl.store(hdr_ptr + s, 0)

    @triton.jit
    def _gdl_materialize_kernel(
        idx_ptr, stride_idx, flag_ptr,
        alog_ptr, dtb_ptr,
        state_ptr, ring_ptr, stride_slot, stride_slot_ring,
        hdr_ptr, W,
        INTERP: tl.constexpr, TMAX: tl.constexpr, BV: tl.constexpr,
        RING_OFF: tl.constexpr, RED: tl.constexpr,
    ):
        """Replay a pending ring into the stock layout: the state after record
        t goes to column t (t < r), exactly what the stock kernel stored."""
        row = tl.program_id(0)
        hv = tl.program_id(1)
        irow = idx_ptr + row.to(tl.int64) * stride_idx
        col0 = tl.load(irow)
        flag = tl.load(flag_ptr + row)
        if (flag == 0) | (col0 <= 0):
            return
        r = tl.load(hdr_ptr + col0)
        if r <= 0:
            return
        ring_slot = tl.load(irow + 1)
        alog = _gdl_load1(alog_ptr + hv)
        dtb = _gdl_load1(dtb_ptr + hv)
        head_off = hv.to(tl.int64) * 128 * 128
        col0_base = state_ptr + col0.to(tl.int64) * stride_slot + head_off
        ring_head = (ring_ptr + ring_slot.to(tl.int64) * stride_slot_ring
                     + hv.to(tl.int64) * (2 * 128 * 128) + RING_OFF)
        ring_ab = ring_head + TMAX * 256
        cols = tl.arange(0, 128)
        rr = tl.arange(0, BV)
        for c in tl.static_range(128 // BV):
            rows = c * BV + rr
            tile_off = rows[:, None] * 128 + cols[None, :]
            h0, h1, h2, h3 = _gdl_lane4(tl.load(col0_base + tile_off))
            s0 = h0
            s1 = h1
            s2 = h2
            s3 = h3
            for t in range(0, r):
                h0, h1, h2, h3 = _gdl_replay_one(
                    h0, h1, h2, h3, ring_head + t * 256, ring_ab, t, rows,
                    alog, dtb, INTERP, TMAX, RED)
                if t == 1:  # column 1 holds the ring: store it after all reads
                    s0 = h0
                    s1 = h1
                    s2 = h2
                    s3 = h3
                else:
                    dst = tl.load(irow + t)
                    if dst > 0:
                        tl.store(state_ptr + dst.to(tl.int64) * stride_slot + head_off
                                 + tile_off, _gdl_unlane4(h0, h1, h2, h3))
            if not INTERP:
                tl.debug_barrier()
            if r > 1:
                if ring_slot > 0:
                    tl.store(state_ptr + ring_slot.to(tl.int64) * stride_slot
                             + head_off + tile_off, _gdl_unlane4(s0, s1, s2, s3))

    @triton.jit
    def _gdl_zero_hdr_kernel(idx_ptr, stride_idx, hdr_ptr, W):
        row = tl.program_id(0)
        irow = idx_ptr + row.to(tl.int64) * stride_idx
        for t in range(0, W):
            s = tl.load(irow + t)
            if s > 0:
                tl.store(hdr_ptr + s, 0)


# ------------------------------------------------------------------ launchers
def _gdl_state_ok(state: torch.Tensor) -> bool:
    try:
        state.view(torch.bfloat16)
    except RuntimeError:
        return False
    return (
        state.dtype == torch.float32
        and state.dim() == 4
        and state.shape[2] == GDL_DIM
        and state.shape[3] == GDL_DIM
        and state.stride(3) == 1
        and state.stride(2) == GDL_DIM
        and state.stride(1) == GDL_DIM * GDL_DIM
    )


def gdl_decode_launch(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens,
                      num_accepted, seq_lens, state, hdr, output_gate, norm_weight,
                      out, num_k_heads, scale, eps, sigmoid, block_size=0,
                      mode=GDL_MODE_AUTO):
    """One K3 verify step (decode kernel + header kernel). Shapes as the stock
    ops.fused_gdn_decode_post_conv_mtp; seq_lens [N] (num_computed + T) is only
    read when block_size > 0; hdr int32 [num_slots]."""
    n = state_indices.shape[0]
    W = state_indices.shape[1]
    HV = state.shape[1]
    G = HV // num_k_heads
    interp = _gdl_interpret()
    ring = state.view(torch.bfloat16)
    if n == 0:
        return
    _gdl_decode_kernel[(n, HV)](
        mixed_qkv, mixed_qkv.stride(0),
        a, a.stride(0), b, b.stride(0),
        A_log, dt_bias,
        state_indices, state_indices.stride(0),
        cu_seqlens, num_accepted, seq_lens,
        state, ring, state.stride(0), ring.stride(0),
        hdr,
        output_gate, output_gate.stride(0),
        norm_weight, out,
        num_k_heads, HV, W, int(block_size), float(scale), float(eps),
        G=G, SIGMOID=bool(sigmoid), MODE=int(mode), INTERP=interp,
        TMAX=GDL_TMAX, BV=GDL_BV, RING_OFF=GDL_RING_OFF, RED=_gdl_red(),
        num_warps=4,
    )
    _gdl_header_kernel[(n,)](
        state_indices, state_indices.stride(0), cu_seqlens, num_accepted, seq_lens,
        hdr, W, int(block_size), MODE=int(mode), TMAX=GDL_TMAX, num_warps=1,
    )


def gdl_materialize_launch(idx, flags, A_log, dt_bias, state, hdr):
    """idx int32 [n, W] (state columns per row), flags int32 [n]: 1 = replay a
    pending ring into the stock layout, 0 = only drop the headers (prefilling
    row: any ring there belongs to a dead request). Always zeroes the headers
    of all W columns afterwards."""
    n = idx.shape[0]
    if n == 0:
        return
    HV = state.shape[1]
    ring = state.view(torch.bfloat16)
    _gdl_materialize_kernel[(n, HV)](
        idx, idx.stride(0), flags, A_log, dt_bias,
        state, ring, state.stride(0), ring.stride(0), hdr, idx.shape[1],
        INTERP=_gdl_interpret(), TMAX=GDL_TMAX, BV=GDL_BV, RING_OFF=GDL_RING_OFF,
        RED=_gdl_red(), num_warps=4,
    )
    _gdl_zero_hdr_kernel[(n,)](idx, idx.stride(0), hdr, idx.shape[1], num_warps=1)


# ------------------------------------------------------------------ self-test
def _gdl_problem(device, gen, n=3, H=2, G=3, W=4, dt_dtype=torch.float32,
                 nw_dtype=torch.float32, slots_extra=2):
    """Random 'captured-like' verify inputs: post-conv qkv with SiLU-like
    magnitudes, a/b gate logits, a real-scale A_log / dt_bias."""
    HV = H * G
    D = GDL_DIM
    T_tokens = n * W

    def rn(*shape, s=1.0):
        # explicit cpu/fp32: model init runs under a cuda device context and a
        # bf16 default dtype
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cpu") * s

    p = {
        "H": H, "HV": HV, "W": W, "n": n,
        "mixed": (rn(T_tokens, 2 * H * D + HV * D, s=0.6)).to(torch.bfloat16).to(device),
        "a": rn(T_tokens, HV, s=2.0).to(torch.bfloat16).to(device),
        "b": rn(T_tokens, HV, s=2.0).to(torch.bfloat16).to(device),
        "A_log": (rn(HV, s=0.8) + 0.5).to(device),
        "dt_bias": rn(HV, s=1.0).to(dt_dtype).to(device),
        "gate": rn(T_tokens, HV, D, s=1.5).to(torch.bfloat16).to(device),
        "norm_w": (1.0 + rn(D, s=0.1)).to(nw_dtype).to(device),
        "num_slots": 1 + n * W + slots_extra,
    }
    return p


def gdl_self_test(device, G=3, sigmoid=True, dt_dtype=torch.float32,
                  nw_dtype=torch.float32, stock_fn=None, steps=4, W=4):
    """K3 lazy vs the stock kernel over `steps` verify steps with random
    acceptance, including T < W rows and a padded (T = 0) row. Returns (ok,
    detail). Requires bitwise equal outputs every step and bitwise equal
    committed states at the end (after a materialize). stock_fn defaults to
    torch.ops._C.fused_gdn_decode_post_conv_mtp via vllm._custom_ops."""
    if stock_fn is None:
        from vllm import _custom_ops as _ops

        def stock_fn(**kw):
            _ops.fused_gdn_decode_post_conv_mtp(**kw)
    gen = torch.Generator(device="cpu").manual_seed(0x4B33)

    def rn(*shape, s=1.0):
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cpu") * s

    H = 2
    n = 3
    p = _gdl_problem(device, gen, n=n, H=H, G=G, W=W, dt_dtype=dt_dtype,
                     nw_dtype=nw_dtype)
    HV, D = p["HV"], GDL_DIM
    num_slots = p["num_slots"]
    st0 = rn(num_slots, HV, D, D, s=0.05).to(device)
    s_stock = st0.clone()
    s_lazy = st0.clone()
    hdr = torch.zeros(num_slots, dtype=torch.int32, device=device)
    si = torch.arange(1, 1 + n * W, dtype=torch.int32, device="cpu").view(n, W).to(device)
    acc = torch.ones(n, dtype=torch.int32, device=device)
    eps = 1e-6
    scale = D ** -0.5
    try:
        for step in range(steps):
            # row lengths: full, short, and one padded row on odd steps
            lens = [W, max(1, W - 1 - (step % 2)), 0 if step % 2 else W]
            cum = [0]
            for x in lens:
                cum.append(cum[-1] + x)
            cu = torch.tensor(cum, dtype=torch.int32, device=device)
            ntok = cum[-1]
            mixed = rn(ntok, p["mixed"].shape[1], s=0.6).to(torch.bfloat16).to(device)
            a = rn(ntok, HV, s=2.0).to(torch.bfloat16).to(device)
            b = rn(ntok, HV, s=2.0).to(torch.bfloat16).to(device)
            gate = rn(ntok, HV, D, s=1.5).to(torch.bfloat16).to(device)
            seq = torch.full((n,), 100 + step * W, dtype=torch.int32, device=device)
            o_s = torch.zeros(ntok, HV, D, dtype=torch.bfloat16, device=device)
            o_l = torch.zeros_like(o_s)
            kw = dict(mixed_qkv=mixed, a=a, b=b, A_log=p["A_log"], dt_bias=p["dt_bias"],
                      state_indices=si, cu_seqlens=cu, num_accepted_tokens=acc,
                      state=s_stock, output_gate=gate, norm_weight=p["norm_w"], out=o_s,
                      scale=scale, norm_eps=eps,
                      output_gate_activation="sigmoid" if sigmoid else "silu")
            stock_fn(**kw)
            # step 2 runs eager rows (replay pending records, stock layout)
            gdl_decode_launch(mixed, a, b, p["A_log"], p["dt_bias"], si, cu, acc, seq,
                              s_lazy, hdr, gate, p["norm_w"], o_l, H, scale, eps,
                              sigmoid, block_size=0,
                              mode=GDL_MODE_EAGER if step == 2 else GDL_MODE_AUTO)
            if not torch.equal(o_s.view(torch.int16), o_l.view(torch.int16)):
                bad = int((o_s.view(torch.int16) != o_l.view(torch.int16)).sum())
                return False, f"step {step}: {bad} output elements differ"
            new_acc = []
            for r_, T in enumerate(lens):
                new_acc.append(int(torch.randint(1, max(T, 1) + 1, (1,), generator=gen,
                                                 device="cpu"))
                               if T > 0 else int(acc[r_]))
            acc = torch.tensor(new_acc, dtype=torch.int32, device=device)
        # committed state: materialize, then compare column a-1 of every row
        flags = torch.ones(n, dtype=torch.int32, device=device)
        gdl_materialize_launch(si, flags, p["A_log"], p["dt_bias"], s_lazy, hdr)
        for r_ in range(n):
            col = int(si[r_, int(acc[r_]) - 1])
            if not torch.equal(s_stock[col].view(torch.int32), s_lazy[col].view(torch.int32)):
                bad = int((s_stock[col] != s_lazy[col]).sum())
                return False, f"committed state row {r_}: {bad} elements differ"
        if torch.device(device).type == "cuda":
            # graph replay of one lazy step equals eager, twice
            gs = s_lazy.clone()
            gh = hdr.clone()
            cu = torch.tensor([0, W, 2 * W, 3 * W], dtype=torch.int32, device=device)
            ntok = 3 * W
            mixed = rn(ntok, p["mixed"].shape[1], s=0.6).to(torch.bfloat16).to(device)
            args = (mixed, mixed[:, :HV], mixed[:, HV:2 * HV], p["A_log"], p["dt_bias"],
                    si, cu, acc, torch.full((n,), 64, dtype=torch.int32, device=device))
            gate = rn(ntok, HV, D).to(torch.bfloat16).to(device)
            o_e = torch.zeros(ntok, HV, D, dtype=torch.bfloat16, device=device)
            ref_s, ref_h = gs.clone(), gh.clone()
            gdl_decode_launch(*args, ref_s, ref_h, gate, p["norm_w"], o_e, H, scale, eps,
                              sigmoid)
            o_g = torch.zeros_like(o_e)
            ws, wh = gs.clone(), gh.clone()
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                gdl_decode_launch(*args, ws, wh, gate, p["norm_w"], o_g, H, scale, eps,
                                  sigmoid)
            torch.cuda.current_stream().wait_stream(s_)
            g = torch.cuda.CUDAGraph()
            cs, ch = gs.clone(), gh.clone()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                gdl_decode_launch(*args, cs, ch, gate, p["norm_w"], o_g, H, scale, eps,
                                  sigmoid)
            for _ in range(2):
                cs.copy_(gs)
                ch.copy_(gh)
                g.replay()
                torch.cuda.synchronize()
                if not (torch.equal(o_g, o_e) and torch.equal(cs, ref_s)
                        and torch.equal(ch, ref_h)):
                    return False, "graph replay != eager"
            del g
    except Exception as e:  # noqa: BLE001 - fail closed
        return False, f"{type(e).__name__}: {e}"[:300]
    return True, f"bitwise over {steps} steps, W={W}, G={G}"


# ------------------------------------------------------------ layer dispatch
def _gdl_layer_key(layer):
    return (
        layer.num_v_heads // layer.num_k_heads,
        layer.norm.activation,
        layer.dt_bias.dtype,
        layer.norm.weight.dtype,
    )


def gdl_layer_init(layer) -> None:
    """Run the self-test once per variant at layer construction (before the
    FULL cudagraph capture), so captured decode graphs contain K3."""
    st = _GDL_STATE
    if not gdl_enabled() or st["failed"]:
        return
    if getattr(layer, "gdn_decode_kernel", None) != "cuda" or not torch.cuda.is_available():
        return
    if layer.head_k_dim != GDL_DIM or layer.head_v_dim != GDL_DIM:
        return
    key = _gdl_layer_key(layer)
    if key in st["tested"]:
        return
    G, act, dt_dtype, nw_dtype = key
    if st["env"] == "force":
        ok, detail = True, "self-test skipped (force)"
    else:
        ok, detail = gdl_self_test(torch.device("cuda", torch.cuda.current_device()),
                                   G=G, sigmoid=(act == "sigmoid"), dt_dtype=dt_dtype,
                                   nw_dtype=nw_dtype)
    st["detail"] = detail
    if not ok:
        st["failed"] = True
        _gdl_logger.warning("K3 gdn_lazy self-test FAILED %s: %s; stock GDN decode", key, detail)
        return
    st["tested"].add(key)
    _gdl_logger.info("K3 gdn_lazy self-test passed %s: %s", key, detail)


def _gdl_hdr(layer, md):
    hdrs = getattr(md, "lazy_hdr", None)
    if not hdrs:
        return None
    hdr = hdrs.get(layer.prefix)
    if hdr is None or hdr.numel() < layer.kv_cache[1].shape[0]:
        return None
    return hdr


def gdl_decode(layer, mixed_qkv, a, b, output_gate, out, md) -> bool:
    """Replace the stock fused MTP decode for a pure spec batch. Returns True
    if K3 ran (lazy or eager rows), False to run the stock op."""
    st = _GDL_STATE
    if not gdl_enabled():
        return False
    used = getattr(layer, "_gdl_used", False)
    state = layer.kv_cache[1]
    hdr = _gdl_hdr(layer, md)
    if hdr is None or not _gdl_state_ok(state):
        if used:
            raise RuntimeError("K3: pending GDN rings but no header table; refusing stock read")
        return False
    active = (not st["failed"]) and _gdl_layer_key(layer) in st["tested"]
    if not st.get("logged"):
        st["logged"] = True
        _gdl_logger.info("K3 gdn_lazy dispatch: %s (key %s, tested %s)",
                         "ACTIVE" if active else "inactive", _gdl_layer_key(layer),
                         sorted(map(str, st["tested"])))
    if not active and not used:
        return False
    n = md.num_spec_decodes
    si = md.spec_state_indices_tensor[:n]
    W = si.shape[1]
    if not (2 <= W <= GDL_TMAX):
        if used:
            raise RuntimeError("K3: unsupported width with pending rings")
        return False
    seq = getattr(md, "lazy_seq_lens", None)
    bs = int(getattr(md, "lazy_block_size", -1))
    mode = GDL_MODE_AUTO if active else GDL_MODE_EAGER
    if seq is None or bs < 0:
        # no boundary information (or mamba_cache_mode 'all'): never start a
        # ring, but still replay pending ones into the stock layout
        seq = md.num_accepted_tokens
        bs = 0
        mode = GDL_MODE_EAGER
    if mode == GDL_MODE_AUTO:
        layer._gdl_used = True
        st["used"] = True
    gdl_decode_launch(
        mixed_qkv, a, b, layer.A_log, layer.dt_bias, si,
        md.spec_query_start_loc[: n + 1], md.num_accepted_tokens[:n], seq[:n],
        state, hdr, output_gate, layer.norm.weight, out,
        layer.num_k_heads // layer.tp_size, layer.head_k_dim ** -0.5,
        layer.layer_norm_epsilon, layer.norm.activation == "sigmoid",
        block_size=bs, mode=mode,
    )
    return True


def gdl_fixup(layer, md) -> None:
    """Before any stock reader of the GDN state in this step: materialize
    pending rings of spec rows and of non-prefilling non-spec rows into the
    stock layout; drop headers of prefilling rows."""
    if not getattr(layer, "_gdl_used", False):
        return
    hdr = _gdl_hdr(layer, md)
    state = layer.kv_cache[1]
    if hdr is None:
        raise RuntimeError("K3: pending GDN rings but no header table")
    idx_parts, flag_parts = [], []
    if md.spec_state_indices_tensor is not None and md.num_spec_decodes > 0:
        si = md.spec_state_indices_tensor[: md.num_spec_decodes]
        idx_parts.append(si)
        flag_parts.append(torch.ones(si.shape[0], dtype=torch.int32, device=si.device))
    ns = getattr(md, "lazy_ns_state_indices", None)
    nsp = getattr(md, "lazy_ns_prefilling", None)
    if ns is not None and nsp is not None and ns.shape[0] > 0:
        W = idx_parts[0].shape[1] if idx_parts else ns.shape[1]
        idx_parts.append(ns[:, :W])
        flag_parts.append((~nsp).to(torch.int32))
    if not idx_parts:
        return
    idx = torch.cat([p.to(torch.int32) for p in idx_parts]).contiguous()
    flags = torch.cat(flag_parts).contiguous()
    gdl_materialize_launch(idx, flags, layer.A_log, layer.dt_bias, state, hdr)
