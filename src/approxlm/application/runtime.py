from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from approxlm.adapters.huggingface.evaluation import (
    run_baseline_experiment,
    run_decoder_only_experiment,
    run_quantized_per_layer_experiment,
)
from approxlm.adapters.persistence.sqlite import save_experiment, save_traces_to_npz
from approxlm.application.defaults import (
    DEFAULT_ATTENTION_MODE,
    DEFAULT_DATASET,
    DEFAULT_DECODER_DATASET,
    DEFAULT_TRACE_ENABLED,
)
from approxlm.application.luts import resolve_lut_path
from approxlm.domain.specs import LayerQuantSpec


def is_fp32_baseline(layer_modes: Dict[str, str]) -> bool:
    return all(mode in {None, "None", "fp32"} for mode in layer_modes.values())


def mode_to_lut_path(mode: str | None) -> str | None:
    return resolve_lut_path(mode)


def build_layer_specs(layer_modes: Dict[str, str], task_type: str = "classification") -> Dict[str, LayerQuantSpec]:
    specs: Dict[str, LayerQuantSpec] = {}
    act_per_channel = task_type == "decoder_only"
    for layer_name, mode in layer_modes.items():
        if mode in {None, "None", "fp32"}:
            continue

        specs[layer_name] = LayerQuantSpec(
            act_data_type="int8",
            weight_data_type="int8",
            act_symmetric=True,
            weight_symmetric=True,
            act_per_channel=act_per_channel,
            weight_per_channel=True,
            approx_lut_path=mode_to_lut_path(mode),
        )
    return specs


def run_experiment_backend(
    config: Dict[str, Any],
    progress_callback=None,
    trace_enabled: bool = DEFAULT_TRACE_ENABLED,
    attention_mode: str = DEFAULT_ATTENTION_MODE,
) -> Dict[str, Any]:
    task_type = config.get("task_type", "classification")
    if task_type == "decoder_only":
        if is_fp32_baseline(config["layer_modes"]):
            layer_specs = None
        else:
            layer_specs = build_layer_specs(config["layer_modes"], task_type=task_type)
            if not layer_specs:
                raise RuntimeError("No quantized or approximate decoder layers were selected.")

        return run_decoder_only_experiment(
            model_name=config["model_url"],
            dataset_name=config.get("dataset_url", DEFAULT_DECODER_DATASET),
            split_name=config.get("split_name", "train"),
            dataset_config=config.get("dataset_config"),
            dataset_format=config.get("dataset_format", "auto"),
            layer_specs=layer_specs,
            batch_size=config.get("batch_size", 4),
            perplexity_stride=config.get("perplexity_stride"),
            wikitext_token_limit=config.get("wikitext_token_limit"),
            max_input_length=config.get("max_input_length", 1024),
            max_new_tokens=config.get("max_new_tokens", 128),
            max_samples=config.get("max_samples", 100),
            calibration_percentile=config.get("calibration_percentile", 99.9),
            calibration_batches=config.get("calibration_batches", 16),
            trust_remote_code=config.get("trust_remote_code", False),
            do_sample=config.get("do_sample", False),
            temperature=config.get("temperature", 0.7),
            top_k=config.get("top_k", 40),
            bertscore_model=config.get("bertscore_model", "bert-base-uncased"),
            backend_quantize=config.get("backend_quantize", True),
            trace_enabled=trace_enabled,
            profile_layer_modes=config["layer_modes"],
            progress_callback=progress_callback,
        )

    common_kwargs = {
        "model_name": config["model_url"],
        "dataset_name": config.get("dataset_url", DEFAULT_DATASET),
        "dataset_revision": config.get("dataset_revision", "refs/convert/parquet"),
        "dataset_data_dir": config.get("dataset_data_dir", "en-US"),
        "text_col": config.get("text_col", "utt"),
        "label_col": config.get("label_col", "intent"),
        "max_length": config.get("max_length", 128),
        "split_name": config.get("split_name", "test"),
        "trace_enabled": trace_enabled,
        "attention_mode": attention_mode,
        "progress_callback": progress_callback,
    }

    if is_fp32_baseline(config["layer_modes"]):
        return run_baseline_experiment(
            batch_size=config.get("batch_size", 256),
            profile_layer_modes=config["layer_modes"],
            **common_kwargs,
        )

    layer_specs = build_layer_specs(config["layer_modes"], task_type=task_type)
    if not layer_specs:
        raise RuntimeError("No quantized or approximate layers were selected.")

    return run_quantized_per_layer_experiment(
        layer_specs=layer_specs,
        batch_size=config.get("batch_size", 256),
        calibration_percentile=config.get("calibration_percentile", 99.9),
        calibration_batches=config.get("calibration_batches", 50),
        backend_quantize=config.get("backend_quantize", True),
        profile_layer_modes=config["layer_modes"],
        **common_kwargs,
    )


def execute_and_store_experiment(
    *,
    experiment_name: str,
    config: Dict[str, Any],
    progress_callback=None,
) -> Dict[str, Any]:
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result = run_experiment_backend(
        config,
        progress_callback=progress_callback,
        trace_enabled=config.get("trace_enabled", DEFAULT_TRACE_ENABLED),
        attention_mode=config.get("attention_mode", DEFAULT_ATTENTION_MODE),
    )
    traces = result.get("traces", {})
    trace_file_path = save_traces_to_npz(experiment_id, traces) if config.get("trace_enabled") and traces else None
    num_traced_samples = int(len(traces["sample_id"])) if traces and "sample_id" in traces else None

    record = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_url": config["model_url"],
        "dataset_url": config.get("dataset_url", DEFAULT_DATASET),
        "config": config,
        "metrics": result.get("metrics", {}),
        "matmul_profile": result.get("matmul_profile", {}),
        "trace_file_path": trace_file_path,
        "trace_enabled": config.get("trace_enabled", False) and trace_file_path is not None,
        "attention_mode": config.get("attention_mode") if config.get("trace_enabled") else None,
        "num_traced_samples": num_traced_samples,
    }
    save_experiment(record)
    record["traces"] = traces
    return record
