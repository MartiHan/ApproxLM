import re
from contextlib import nullcontext
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List

import torch.nn as nn
import streamlit as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification

from approxlm.adapters.huggingface.evaluation import run_qualitative_decoder_evaluation
from approxlm.application.runtime import build_layer_specs, is_fp32_baseline, mode_to_lut_path, run_experiment_backend
from approxlm.application.defaults import DEFAULT_MODEL, DEFAULT_DATASET, DEFAULT_DECODER_MODEL, DEFAULT_DECODER_DATASET, APPROX_OPTIONS, DEFAULT_TRACE_ENABLED, DEFAULT_ATTENTION_MODE, BLOCK_INDEX_PATTERN


CUSTOM_LUT_OPTION = "custom LUT"


try:
    from accelerate import init_empty_weights
except Exception:  # pragma: no cover
    init_empty_weights = None


def build_xlmr_architecture(num_encoder_layers: int = 12) -> List[Dict[str, str]]:
    architecture = [
        _group_linear_layer_name("classifier.dense"),
        _group_linear_layer_name("classifier.out_proj"),
    ]
    for i in range(num_encoder_layers):
        prefix = f"roberta.encoder.layer.{i}"
        architecture.extend(
            [
                _group_linear_layer_name(f"{prefix}.attention.self.query"),
                _group_linear_layer_name(f"{prefix}.attention.self.key"),
                _group_linear_layer_name(f"{prefix}.attention.self.value"),
                _group_linear_layer_name(f"{prefix}.attention.output.dense"),
                _group_linear_layer_name(f"{prefix}.intermediate.dense"),
                _group_linear_layer_name(f"{prefix}.output.dense"),
            ]
        )
    return architecture


def build_decoder_only_architecture(num_decoder_layers: int = 24) -> List[Dict[str, str]]:
    architecture: List[Dict[str, str]] = []
    for i in range(num_decoder_layers):
        prefix = f"model.layers.{i}"
        architecture.extend(
            [
                _group_linear_layer_name(f"{prefix}.self_attn.q_proj"),
                _group_linear_layer_name(f"{prefix}.self_attn.k_proj"),
                _group_linear_layer_name(f"{prefix}.self_attn.v_proj"),
                _group_linear_layer_name(f"{prefix}.self_attn.o_proj"),
                _group_linear_layer_name(f"{prefix}.mlp.gate_proj"),
                _group_linear_layer_name(f"{prefix}.mlp.up_proj"),
                _group_linear_layer_name(f"{prefix}.mlp.down_proj"),
            ]
        )
    return architecture


def _fallback_architecture(task_type: str) -> List[Dict[str, Any]]:
    return build_decoder_only_architecture(24) if task_type == "decoder_only" else build_xlmr_architecture(12)


def _group_linear_layer_name(layer_name: str) -> Dict[str, Any]:
    match = BLOCK_INDEX_PATTERN.match(layer_name)
    if not match:
        return {
            "group": "head",
            "layer": layer_name,
            "block_family": None,
            "block_index": None,
            "suffix": layer_name,
        }

    prefix, block_index, suffix = match.groups()
    prefix_parts = [part for part in prefix.split(".") if part]
    family = prefix_parts[-1] if prefix_parts else "block"
    return {
        "group": f"{family}_{block_index}",
        "layer": layer_name,
        "block_family": family,
        "block_index": int(block_index),
        "suffix": suffix,
    }


def _sort_architecture(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(item: Dict[str, Any]):
        block_index = item.get("block_index")
        return (
            item.get("block_family") is None,
            item.get("block_family") or "",
            float("inf") if block_index is None else block_index,
            item["layer"],
        )

    return sorted(items, key=sort_key)


def _is_linear_like_module(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True

    class_name = module.__class__.__name__.lower()
    if "linear" in class_name or "quantlinear" in class_name:
        return True

    has_feature_shape = hasattr(module, "in_features") and hasattr(module, "out_features")
    has_weight_like_state = hasattr(module, "weight") or hasattr(module, "qweight")
    return bool(has_feature_shape and has_weight_like_state)


@lru_cache(maxsize=16)
def discover_model_architecture(
    model_name: str,
    task_type: str,
    trust_remote_code: bool = False,
) -> List[Dict[str, Any]]:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if hasattr(config, "quantization_config"):
        config.quantization_config = None
    auto_model_cls = AutoModelForCausalLM if task_type == "decoder_only" else AutoModelForSequenceClassification
    context = init_empty_weights() if init_empty_weights is not None else nullcontext()

    with context:
        model = auto_model_cls.from_config(config, trust_remote_code=trust_remote_code)

    architecture = [
        _group_linear_layer_name(name)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    return _sort_architecture(architecture) or _fallback_architecture(task_type)


def refresh_architecture_state(
    *,
    model_name: str,
    task_type: str,
    state_key: str,
    arch_state_key: str,
    error_state_key: str,
    trust_remote_code: bool = False,
) -> None:
    existing_modes = dict(st.session_state.get(state_key, {}))
    architecture = discover_model_architecture(
        model_name=model_name,
        task_type=task_type,
        trust_remote_code=trust_remote_code,
    )
    st.session_state[arch_state_key] = architecture
    st.session_state[state_key] = {
        item["layer"]: existing_modes.get(item["layer"], "fp32")
        for item in architecture
    }
    st.session_state[error_state_key] = None


def render_lut_mode_selector(
    label: str,
    *,
    current: str | None,
    key: str,
    disabled: bool = False,
) -> str:
    current = current or "fp32"
    is_custom = current not in APPROX_OPTIONS
    options = list(APPROX_OPTIONS)
    options.append(CUSTOM_LUT_OPTION)
    selected = st.selectbox(
        label,
        options,
        index=options.index(CUSTOM_LUT_OPTION) if is_custom else options.index(current),
        key=key,
        disabled=disabled,
        help="Select a preset or choose custom LUT to enter a LUT name/path. Names are resolved as <name>.npy in the current working directory, then packaged resources.",
    )
    if selected != CUSTOM_LUT_OPTION:
        return selected

    custom_value = st.text_input(
        f"{label} custom LUT",
        value=current if is_custom else "",
        key=f"{key}_custom",
        disabled=disabled,
        placeholder="my_lut or path/to/my_lut.npy",
        label_visibility="collapsed",
    ).strip()
    return custom_value or (current if is_custom else "fp32")


def run_qualitative_backend(
    config: Dict[str, Any],
    progress_callback=None,
) -> Dict[str, Any]:
    if is_fp32_baseline(config["layer_modes"]):
        layer_specs = None
    else:
        layer_specs = build_layer_specs(config["layer_modes"], task_type="decoder_only")
        if not layer_specs:
            raise RuntimeError("No quantized or approximate decoder layers were selected.")

    return run_qualitative_decoder_evaluation(
        model_name=config["model_url"],
        prompt=config["prompt"],
        ground_truth=config["ground_truth"],
        dataset_name=config.get("dataset_url", DEFAULT_DECODER_DATASET),
        split_name=config.get("split_name", "test"),
        dataset_config=config.get("dataset_config", "wikitext-2-raw-v1"),
        dataset_format=config.get("dataset_format", "wikitext"),
        layer_specs=layer_specs,
        batch_size=config.get("batch_size", 1),
        max_input_length=config.get("max_input_length", 1024),
        max_new_tokens=config.get("max_new_tokens", 64),
        calibration_percentile=config.get("calibration_percentile", 99.9),
        calibration_batches=config.get("calibration_batches", 16),
        trust_remote_code=config.get("trust_remote_code", False),
        do_sample=config.get("do_sample", False),
        temperature=config.get("temperature", 0.7),
        top_k=config.get("top_k", 40),
        backend_quantize=config.get("backend_quantize", True),
        top_token_count=config.get("top_token_count", 20),
        progress_callback=progress_callback,
    )


def default_experiment_name() -> str:
    return datetime.now().strftime("exp_%Y%m%d_%H%M%S")


def init_state() -> None:
    st.session_state.setdefault("selected_experiment_id", None)
    st.session_state.setdefault("arch_mode", "bulk")
    st.session_state.setdefault("layer_modes", {})
    st.session_state.setdefault("classification_architecture", None)
    st.session_state.setdefault("decoder_architecture", None)
    st.session_state.setdefault("qualitative_architecture", None)
    st.session_state.setdefault("classification_architecture_error", None)
    st.session_state.setdefault("decoder_architecture_error", None)
    st.session_state.setdefault("qualitative_architecture_error", None)
