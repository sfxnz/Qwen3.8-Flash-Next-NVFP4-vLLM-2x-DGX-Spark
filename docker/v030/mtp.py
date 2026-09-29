# R11-OVERLAY
# base_image_digest: sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
# upstream_file: vllm/models/qwen4_exp/nvidia/mtp.py
# upstream_file_sha256: 928c450001ee42cae9f6b0d5e9dc477b63019af2e90baa77505659ea3334eb88
# upstream_PR: none (local; L1a mirrors upstream Qwen3_5MTP LocalArgmaxMixin wiring, L1b is FR-Spec arXiv:2502.14856 style, L1b' reuses v0.30 Marlin FP8 / CUTLASS scaled_mm ops)
# generator: docker/v030/apply_mtp_overlay.py (do not hand-edit)
# features: L1a get_top_tokens via LocalArgmaxMixin (always on);
#   L1b reduced-vocab draft head, opt-in via VLLM_QWEN38_DRAFT_VOCAB=<json>, default off;
#   L1b' draft-only FP8 lm_head copy, opt-in via VLLM_QWEN38_DRAFT_HEAD_FP8=1, default off
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp MTP (Multi-Token Predictor) model.

The MTP draft model reuses the Qwen4Exp backbone (PLE/HC/MoE) but:
  - drops all multi-modal handling (text-only),
  - forces PLE off while keeping the main model's HC stream count,
  - fuses the backbone hidden and the new-token embedding via
    ``residual_linear_shared`` (fc_embedding + shared fc_hidden) instead of
    the ``Linear(2H, H)`` + repeat used by other MTP variants,
  - emits TWO hidden streams per step (scheme A): a single stream [T, H]
    (final-mixer collapsed, fed to the LM head) and a pre-final-mixer
    multi stream [T, hc_count*H] (fed to the next draft step).
"""

import json
import os
from collections.abc import Iterable

import regex as re
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import LocalArgmaxMixin, SupportsPP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)

from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .model import (
    _EXTRA_WEIGHTS_MAPPER,
    _QWEN4_EXP_IGNORED_MISSING_SUFFIXES,
    Qwen4ExpDecoderLayer,
    Qwen4ExpMixtureOfExperts,
    Qwen4ExpSparseMoeBlock,
)

logger = init_logger(__name__)

# L1b reduced-vocab and L1b' FP8 draft heads (recipe overlay). Default off.
_DRAFT_VOCAB_ENV = "VLLM_QWEN38_DRAFT_VOCAB"
_DRAFT_VOCAB_MIN_IDS = 1024
_DRAFT_HEAD_FP8_ENV = "VLLM_QWEN38_DRAFT_HEAD_FP8"
_FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def _load_draft_vocab(path: str, vocab_size: int) -> list[int]:
    """Read and validate a draft_vocab_<V>.json id list (sorted, unique)."""
    with open(path) as f:
        doc = json.load(f)
    ids = doc.get("ids") if isinstance(doc, dict) else doc
    if isinstance(doc, dict) and doc.get("vocab_size", vocab_size) != vocab_size:
        raise ValueError(
            f"draft vocab built for vocab_size={doc.get('vocab_size')}, "
            f"model has {vocab_size}"
        )
    if not isinstance(ids, list) or len(ids) < _DRAFT_VOCAB_MIN_IDS:
        raise ValueError(f"draft vocab needs >= {_DRAFT_VOCAB_MIN_IDS} ids")
    prev = -1
    for i in ids:
        if type(i) is not int or not prev < i < vocab_size:
            raise ValueError("draft vocab ids must be sorted unique ints < vocab")
        prev = i
    return ids


def _draft_vocab_rank_slice(ids: list[int], tp_size: int, tp_rank: int) -> list[int]:
    """Balanced contiguous split of the sorted ids; rank 0 gets the lowest.

    Keeping rank order == id order makes the cross-rank argmax (first max
    wins) pick the lowest id on ties, like a full-vocab argmax.
    """
    per = -(-len(ids) // tp_size)
    return ids[tp_rank * per : (tp_rank + 1) * per]


def _draft_head_fp8_kernels(value: str) -> tuple[str, ...]:
    """FP8 draft-head kernels to try, in order, for a VLLM_QWEN38_DRAFT_HEAD_FP8 value."""
    v = value.strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return ()
    if v in ("1", "on", "true", "yes", "auto"):
        return ("marlin", "w8a8")
    if v in ("marlin", "w8a16"):
        return ("marlin",)
    if v in ("w8a8", "cutlass"):
        return ("w8a8",)
    raise ValueError(f"{_DRAFT_HEAD_FP8_ENV}={value!r}: use 1|auto, marlin or w8a8")


def _quantize_rows_fp8(
    w: torch.Tensor, chunk: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric E4M3: w[i] ~= q[i] * scale[i] (scale fp32, [N]).

    The scale is rounded to w.dtype first, so Marlin (which keeps scales in
    the activation dtype) and CUTLASS (fp32 scales) dequantize identically.
    Chunked to bound the fp32 transient.
    """
    q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
    for s in range(0, w.shape[0], chunk):
        blk = w[s : s + chunk].float()
        sc = (blk.abs().amax(dim=1) / _FP8_MAX).to(w.dtype).float()
        sc = torch.where(sc > 0, sc, torch.ones_like(sc))  # all-zero rows
        q[s : s + chunk] = (
            (blk / sc[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        )
        scale[s : s + chunk] = sc
    return q, scale


class _Fp8DraftHead:
    """Draft-only FP8 copy of lm_head rows [N, K]; logits(h) -> [M, N].

    kernel "marlin": W8A16, FP8 weights dequantized in the Marlin GEMM,
    activations stay BF16. kernel "w8a8": per-token dynamic FP8 activations
    + CUTLASS scaled_mm with the per-row weight scale.
    """

    def __init__(self, rows: torch.Tensor, kernel: str) -> None:
        q, scale = _quantize_rows_fp8(rows)
        self.kernel = kernel
        self.n, self.k = rows.shape
        self.dtype = rows.dtype
        self.nbytes = q.numel() * q.element_size() + scale.numel() * 4
        if kernel == "marlin":
            from vllm.model_executor.layers.quantization.utils import (
                marlin_utils_fp8 as mu,
            )

            layer = nn.Module()
            layer.output_size_per_partition = self.n
            layer.input_size_per_partition = self.k
            layer.orig_dtype = rows.dtype
            layer.weight = nn.Parameter(q, requires_grad=False)
            layer.weight_scale = nn.Parameter(scale, requires_grad=False)
            mu.prepare_fp8_layer_for_marlin(layer, size_k_first=False)
            self.weight = layer.weight.data
            self.scale = layer.weight_scale.data
            self.workspace = layer.workspace
            self._marlin = mu.apply_fp8_marlin_linear
        elif kernel == "w8a8":
            from vllm import _custom_ops as ops

            self._ops = ops
            self.weight = q.t()  # [K, N] column-major, as cutlass_scaled_mm wants
            self.scale = scale.view(-1, 1)
        else:
            raise ValueError(f"unknown FP8 draft-head kernel {kernel!r}")

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.kernel == "marlin":
            return self._marlin(
                h, self.weight, self.scale, self.workspace, self.n, self.k, None
            )
        hq, hs = self._ops.scaled_fp8_quant(h, use_per_token_if_dynamic=True)
        return self._ops.cutlass_scaled_mm(hq, self.weight, hs, self.scale, self.dtype)

    def check(self, rows: torch.Tensor) -> float:
        """Raise unless logits() matches the BF16 rows to FP8 accuracy."""
        h = torch.randn(4, self.k, dtype=rows.dtype, device=rows.device)
        ref = F.linear(h, rows).float()
        got = self.logits(h).float()
        if got.shape != ref.shape or not bool(torch.isfinite(got).all()):
            raise RuntimeError(f"bad output {tuple(got.shape)}")
        err = float((got - ref).abs().max() / ref.abs().max().clamp(min=1e-6))
        if err > 0.1:
            raise RuntimeError(f"max rel. error {err:.3f} > 0.1 vs BF16")
        return err


def _build_fp8_draft_head(
    rows: torch.Tensor, kernels: tuple[str, ...]
) -> tuple[_Fp8DraftHead | None, str]:
    """First kernel that builds and passes its self-check, else (None, why)."""
    errs = []
    for kernel in kernels:
        try:
            head = _Fp8DraftHead(rows, kernel)
            head.check(rows)
            return head, ""
        except Exception as e:  # noqa: BLE001 - try the next kernel
            errs.append(f"{kernel}: {type(e).__name__}: {e}")
    return None, "; ".join(errs)


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
) -> list[str]:
    remapped: list[str] = []
    for name in ignored_layers:
        if name.startswith("mtp."):
            new_name = re.sub(
                r"(?<=\.layers\.)\d+",
                lambda m: str(mtp_start_layer_idx + int(m.group(0))),
                name,
            )
            remapped.append(new_name)
        else:
            remapped.append(name)
    return remapped


def _remap_quantized_layers(
    quantized_layers: dict[str, dict],
    mtp_start_layer_idx: int,
) -> dict[str, dict]:
    """Map checkpoint MTP layer indices to standalone draft indices."""
    return {
        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: layer_info
        for name, layer_info in quantized_layers.items()
    }


def _remap_mtp_weight_name(name: str) -> str | None:
    """Map Qwen4Exp checkpoint paths into the standalone draft model."""

    for checkpoint_prefix in (
        "model.language_model.",
        "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen4ExpMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )
        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)
        if quantized_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "quantized_layers",
                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
            )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config


class Qwen4ExpMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen4ExpTextConfig = model_config.hf_text_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # residual_linear_shared fusion: fc_embedding projects the token
            # embedding, fc_hidden (shared across HC branches) projects the
            # backbone hidden; the embedding is added as a residual to every
            # branch (see mtp_residual_linear_shared.md).
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            self.layers = nn.ModuleList(
                Qwen4ExpDecoderLayer(
                    draft_vllm_config,
                    layer_type="full_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                )
                for idx in range(self.num_mtp_layers)
            )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen4ExpSparseMoeBlock,
            "mlp",
        )

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )

    def _iter_qsa_attentions(self):
        """Yield MTP attention modules that own a QSA indexer."""

        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and getattr(attention, "indexer", None) is not None
            ):
                yield attention

    def set_skip_topk(self, skip: bool) -> None:
        """Select on MTP step 0 and reuse its QSA indices on later steps."""

        for attention in self._iter_qsa_attentions():
            attention.indexer.skip_topk = skip

    def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
        """Keep each request's target-aligned step-0 sparse-index row."""

        num_rows = row_indices.numel()
        for attention in self._iter_qsa_attentions():
            buffer = attention.topk_indices_buffer
            selected = buffer.index_select(0, row_indices)
            buffer[:num_rows].copy_(selected)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        hc_count = self.hc_count
        hidden_size = self.hidden_size
        prev_block_output: torch.Tensor | None = None

        if get_pp_group().is_first_rank:
            assert hidden_states is not None
            if inputs_embeds is None:
                assert input_ids is not None
                inputs_embeds = self.embed_input_ids(input_ids)
            # Embedding branch: pre-norm -> fc_embedding -> [T, H].
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            inputs_embeds = self.fc_embedding(inputs_embeds)

            # Backbone hidden is multi-stream [T, hc_count*H] (scheme A:
            # the main model truly emits the pre-final-mixer multi stream
            # on the first step; subsequent steps reuse the prior draft
            # step's multi stream).
            num_tokens = hidden_states.shape[0]
            hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
            hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
                num_tokens, hc_count, hidden_size
            )
            hidden_states = self.fc_hidden(hidden_states)
            hidden_states = hidden_states.flatten(-2)
            prev_block_output = inputs_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=prev_block_output,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        if not get_pp_group().is_last_rank:
            # As in the target model, PP carries a materialized tensor rather
            # than the delayed hidden/output/injection tuple.
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
            return IntermediateTensors({"hidden_states": hidden_states})

        # Last PP rank finalize. Keep both:
        #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
        #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
        #       for the next draft step (zero extra compute, just kept).
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=self.is_fused_shared_expert_enabled,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)


class Qwen4ExpMTP(
    LocalArgmaxMixin, nn.Module, SupportsPP, Qwen4ExpMixtureOfExperts
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4ExpMTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        # L1b/L1b': filled by load_weights only when a draft head is enabled.
        self.register_buffer("draft_vocab_weight", None, persistent=False)
        self.register_buffer("draft_vocab_ids", None, persistent=False)
        self.draft_head_fp8: _Fp8DraftHead | None = None
        self.draft_head_full = False  # FP8 rows are this rank's whole shard
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        if self.draft_head_fp8 is not None and self.draft_head_full:
            # L1b': full-vocab FP8 draft logits, gathered like the stock head
            # (probabilistic drafting, or greedy without local argmax).
            logits = self.draft_head_fp8.logits(hidden_states)
            if get_tensor_model_parallel_world_size() > 1:
                logits = self.logits_processor._gather_logits(logits)
            if logits is not None:
                logits = logits[..., : self.logits_processor.org_vocab_size]
            return logits
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ids = self.draft_vocab_ids
        if ids is None:
            return super().get_top_tokens(hidden_states)
        # L1b/L1b': this rank's rows (V'/tp or the whole shard, BF16 or FP8),
        # local argmax, then an [M, 2] all-gather.
        logits = self._draft_head_logits(hidden_states)
        local_idx = logits.argmax(dim=-1, keepdim=True)
        local_val = logits.gather(-1, local_idx).squeeze(-1)
        top = ids[local_idx.squeeze(-1)]
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return top
        # float32 carries ids < 2**24 exactly (vocab is 248,320).
        pair = torch.stack([local_val.float(), top.float()], dim=-1)
        gathered = tensor_model_parallel_all_gather(pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        best = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        return gathered[:, :, 1].gather(-1, best).squeeze(-1).to(torch.int64)

    def _draft_head_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.draft_head_fp8 is not None:
            return self.draft_head_fp8.logits(hidden_states)
        return F.linear(hidden_states, self.draft_vocab_weight)

    def _draft_vocab_plan(self) -> list[int]:
        """Ids this rank keeps for the L1b head; raises if L1b cannot run."""
        path = os.environ.get(_DRAFT_VOCAB_ENV, "").strip()
        spec = self.vllm_config.speculative_config
        if spec is None or not spec.use_local_argmax_reduction:
            raise ValueError("needs speculative_config use_local_argmax_reduction")
        if spec.draft_sample_method == "probabilistic":
            raise ValueError("incompatible with draft_sample_method=probabilistic")
        self._check_draft_head_supported()
        ids = _load_draft_vocab(path, self.config.vocab_size)
        return _draft_vocab_rank_slice(
            ids,
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )

    def _check_draft_head_supported(self) -> None:
        if get_pp_group().world_size != 1:
            raise ValueError("pipeline parallel is not supported")
        if self.config.tie_word_embeddings:
            raise ValueError("tied embeddings are not supported")
        lp = self.logits_processor
        if lp.scale != 1.0 or lp.soft_cap is not None or lp.logits_as_input:
            raise ValueError("logits scale/soft_cap is not supported")

    def _full_head_rows(self) -> tuple[torch.Tensor, range]:
        """This rank's whole lm_head shard (L1b' without L1b); raises if unusable."""
        self._check_draft_head_supported()
        si = self.lm_head.shard_indices
        if si.num_org_vocab_padding or si.num_added_elements_padded:
            raise ValueError("padded or added-vocab lm_head shard is not supported")
        w = self.lm_head.weight.data
        if w.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"lm_head dtype {w.dtype} is not BF16/FP16")
        if w.shape[0] != si.num_org_elements:
            raise ValueError(f"lm_head shard has {w.shape[0]} rows")
        return w, range(si.org_vocab_start_index, si.org_vocab_end_index)

    @staticmethod
    def _tp_agree(ok: bool, err: str) -> tuple[bool, str]:
        """MIN over TP ranks on the CPU group; every rank must call it."""
        if get_tensor_model_parallel_world_size() > 1:
            flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
            torch.distributed.all_reduce(
                flag,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().cpu_group,
            )
            if ok and int(flag.item()) == 0:
                return False, "another TP rank could not build it"
        return ok, err

    def _finish_draft_head(
        self,
        local_ids: list[int] | None,
        rows: torch.Tensor | None,
        err: str,
        requested: bool,
        fp8_kernels: tuple[str, ...],
        fp8_err: str,
    ) -> None:
        """Install L1b and/or L1b' on every TP rank, or on none (fail closed).

        Every TP rank joins both agreements even with the envs unset, so a
        rank that lacks them makes the others fall back instead of hanging.
        """
        # L1b: the reduced BF16 rows captured while loading.
        head_dtype = self.logits_processor.head_dtype
        ok = local_ids is not None and rows is not None
        if ok and head_dtype not in (None, rows.dtype):
            ok, err = False, f"head_dtype {head_dtype} != lm_head {rows.dtype}"
        ok, err = self._tp_agree(ok, err)
        if not ok:
            if requested:
                logger.warning(
                    "%s set but reduced draft head disabled (%s); using the "
                    "full lm_head for drafting.",
                    _DRAFT_VOCAB_ENV,
                    err or "lm_head.weight not seen while loading",
                )
            local_ids = rows = None
        else:
            # Read the device only here: on a PP>1 non-last rank lm_head is a
            # PPMissingLayer without .weight (the plan already failed closed).
            rows = rows.to(self.lm_head.weight.device)

        # L1b': FP8 copy of the reduced rows, else of this rank's full shard.
        fp8_requested = bool(fp8_kernels or fp8_err)
        head, full = None, False
        if fp8_kernels:
            try:
                if rows is None:
                    rows, local_ids = self._full_head_rows()
                    full = True
                if head_dtype not in (None, rows.dtype):
                    raise ValueError(f"head_dtype {head_dtype} != lm_head {rows.dtype}")
                head, fp8_err = _build_fp8_draft_head(rows, fp8_kernels)
            except Exception as e:  # fail closed to BF16
                fp8_err = f"{type(e).__name__}: {e}"
        fp8_ok, fp8_err = self._tp_agree(head is not None, fp8_err)
        if not fp8_ok:
            head = None
            if fp8_requested:
                logger.warning(
                    "%s set but FP8 draft head disabled (%s); drafting with "
                    "BF16 rows.",
                    _DRAFT_HEAD_FP8_ENV,
                    fp8_err or "not requested on every rank",
                )
            if full:  # never keep the draft's own BF16 shard alive
                local_ids = rows = None
        if local_ids is None or rows is None:
            return

        device = rows.device
        if isinstance(local_ids, range):
            ids = torch.arange(
                local_ids.start, local_ids.stop, dtype=torch.int64, device=device
            )
        else:
            ids = torch.tensor(local_ids, dtype=torch.int64, device=device)
        self.draft_vocab_ids = ids
        if head is not None:
            self.draft_head_fp8, self.draft_head_full = head, full
            what = f"FP8 {head.kernel}"
            mib = head.nbytes / 2**20
        else:
            self.draft_vocab_weight = rows.contiguous()
            what = f"{rows.dtype}"
            mib = rows.numel() * rows.element_size() / 2**20
        logger.info(
            "Draft head (%s): %d rows on this rank (ids %d..%d), %s, %.1f MiB; "
            "the target keeps its BF16 lm_head for verify.",
            "L1b'" if full else ("L1b+L1b'" if head is not None else "L1b"),
            len(local_ids),
            local_ids[0],
            local_ids[-1],
            what,
            mib,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        draft_vocab = bool(os.environ.get(_DRAFT_VOCAB_ENV, "").strip())
        fp8_kernels: tuple[str, ...] = ()
        fp8_err = ""
        try:
            fp8_kernels = _draft_head_fp8_kernels(
                os.environ.get(_DRAFT_HEAD_FP8_ENV, "")
            )
        except ValueError as e:
            fp8_err = str(e)
        local_ids: list[int] | None = None
        captured: dict[str, torch.Tensor] = {}
        err = ""
        if draft_vocab:
            try:
                local_ids = self._draft_vocab_plan()
            except Exception as e:  # fail closed to the stock head
                err = f"{type(e).__name__}: {e}"

        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    if local_ids is not None and remapped_name == "lm_head.weight":
                        index = torch.tensor(local_ids, device=weight.device)
                        captured["rows"] = weight.index_select(0, index)
                    yield remapped_name, weight

        mapper = WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(remap_weight_names(), mapper=mapper)
        requested = draft_vocab or fp8_kernels or fp8_err
        if requested or get_tensor_model_parallel_world_size() > 1:
            self._finish_draft_head(
                local_ids,
                captured.get("rows"),
                err,
                draft_vocab,
                fp8_kernels,
                fp8_err,
            )
        return loaded


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]
