import json
from pathlib import Path
from typing import Any, Dict, List

from approxlm.application.defaults import DEFAULT_ATTENTION_MODE, DEFAULT_DATASET, DEFAULT_MODEL

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_PRESET = "encoder_only_mul8s_1KVA_block_sweep"
PRESET_SUFFIX = "_block_sweep"
ATTENTION_SUFFIXES = {
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
}
NON_ATTENTION_SUFFIXES = {
    "intermediate.dense",
    "output.dense",
}
ENCODER_LAYER_SWEEP_SUFFIXES = (
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
)
DECODER_ATTENTION_SUFFIXES = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
}
DECODER_NON_ATTENTION_SUFFIXES = {
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}
DECODER_LAYER_SCOPE_ALIASES = {
    "gate": "mlp.gate_proj",
    "gate_proj": "mlp.gate_proj",
    "ff_gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "up_proj": "mlp.up_proj",
    "ff_up": "mlp.up_proj",
    "down": "mlp.down_proj",
    "down_proj": "mlp.down_proj",
    "ff_down": "mlp.down_proj",
}
DECODER_FF_SWEEP_SUFFIXES = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
DECODER_COMBINED_LAYER_SCOPES = {
    "gate_up": ("mlp.gate_proj", "mlp.up_proj"),
    "gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
    "up_gate": ("mlp.gate_proj", "mlp.up_proj"),
    "up_gate_proj": ("mlp.gate_proj", "mlp.up_proj"),
}
DEFAULT_CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "f1")
DEFAULT_DECODER_METRICS = ("bleu", "rouge_l", "perplexity", "bertscore_precision", "bertscore_recall", "bertscore_f1")


def _load_config_file(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Dispatcher config not found: {path}")

    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(raw_text)
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not available, so YAML dispatcher configs cannot be loaded.")
        payload = yaml.safe_load(raw_text)
        return payload or {}
    raise ValueError(f"Unsupported dispatcher config format: {suffix}")


def _build_encoder_layer_modes(
    *,
    num_encoder_layers: int,
    default_mode: str = "fp32",
    classifier_mode: str = "int8_exact",
) -> Dict[str, str]:
    layer_modes = {
        "classifier.dense": classifier_mode,
        "classifier.out_proj": classifier_mode,
    }
    for block_index in range(num_encoder_layers):
        prefix = f"roberta.encoder.layer.{block_index}"
        for suffix in (
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ):
            layer_modes[f"{prefix}.{suffix}"] = default_mode
    return layer_modes


def _build_decoder_layer_modes(
    *,
    num_decoder_layers: int,
    default_mode: str = "fp32",
) -> Dict[str, str]:
    layer_modes: Dict[str, str] = {}
    for block_index in range(num_decoder_layers):
        prefix = f"model.layers.{block_index}"
        for suffix in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            layer_modes[f"{prefix}.{suffix}"] = default_mode
    return layer_modes


def _encoder_block_sweep_from_preset(config: Dict[str, Any]) -> Dict[str, Any]:
    num_encoder_layers = int(config.get("num_encoder_layers", 12))
    approx_mode = config.get("approx_mode", "mul8s_1KVA")
    baseline_mode = config.get("baseline_mode", "int8_exact")
    trace_enabled = bool(config.get("trace_enabled", True))
    attention_mode = config.get("attention_mode", DEFAULT_ATTENTION_MODE)

    base_config = {
        "task_type": "classification",
        "model_url": config.get("model_url", DEFAULT_MODEL),
        "dataset_url": config.get("dataset_url", DEFAULT_DATASET),
        "trace_enabled": trace_enabled,
        "attention_mode": attention_mode,
    }

    experiments: List[Dict[str, Any]] = []

    baseline_layers = _build_encoder_layer_modes(
        num_encoder_layers=num_encoder_layers,
        default_mode=baseline_mode,
        classifier_mode=baseline_mode,
    )
    experiments.append(
        {
            "name": "baseline_int8_exact_all",
            "label": "baseline:int8_exact:all",
            "layer_modes": baseline_layers,
            "tags": ["baseline", "encoder_only", baseline_mode],
        }
    )

    approx_all_layers = dict(baseline_layers)
    for layer_name in approx_all_layers:
        if layer_name.startswith("roberta.encoder.layer."):
            approx_all_layers[layer_name] = approx_mode
    experiments.append(
        {
            "name": f"{approx_mode}_all_encoder_layers",
            "label": f"{approx_mode}:all",
            "layer_modes": approx_all_layers,
            "tags": ["all_layers", "encoder_only", approx_mode],
        }
    )

    for block_index in range(num_encoder_layers):
        single_block_layers = dict(baseline_layers)
        prefix = f"roberta.encoder.layer.{block_index}."
        for layer_name in single_block_layers:
            if layer_name.startswith(prefix):
                single_block_layers[layer_name] = approx_mode
        experiments.append(
            {
                "name": f"{approx_mode}_encoder_block_{block_index}",
                "label": f"{approx_mode}:encoder_{block_index}",
                "layer_modes": single_block_layers,
                "tags": ["single_block", "encoder_only", approx_mode],
                "encoder_block_index": block_index,
            }
        )

    return {
        "dispatcher_name": config.get("dispatcher_name", DEFAULT_PRESET),
        "description": config.get("description", "Encoder-only sweep with baseline, full approximation, and per-block approximation."),
        "base_config": base_config,
        "experiments": experiments,
    }


def _is_supported_encoder_block_sweep_preset(preset_name: str) -> bool:
    return preset_name.startswith("encoder_only_") and preset_name.endswith(PRESET_SUFFIX)


def _infer_approx_mode_from_preset(preset_name: str) -> str:
    if not _is_supported_encoder_block_sweep_preset(preset_name):
        raise ValueError(f"Unsupported dispatcher preset: {preset_name}")
    return preset_name[len("encoder_only_") : -len(PRESET_SUFFIX)]


def _normalize_block_selector(blocks: Any, *, num_encoder_layers: int) -> List[int]:
    if blocks in (None, "all"):
        return list(range(num_encoder_layers))
    if isinstance(blocks, int):
        return [blocks]
    if isinstance(blocks, list):
        normalized = [int(block) for block in blocks]
        invalid = [block for block in normalized if block < 0 or block >= num_encoder_layers]
        if invalid:
            raise ValueError(f"Invalid encoder block indices: {invalid}")
        return normalized
    raise ValueError(f"Unsupported encoder_blocks value: {blocks}")


def _matches_encoder_layer_scope(layer_name: str, layer_scope: str) -> bool:
    if layer_scope == "all":
        return True

    suffix = layer_name.split("roberta.encoder.layer.", 1)[-1].split(".", 1)[-1]
    if layer_scope == "attention":
        return suffix in ATTENTION_SUFFIXES
    if layer_scope == "non_attention":
        return suffix in NON_ATTENTION_SUFFIXES
    if layer_scope in ATTENTION_SUFFIXES or layer_scope in NON_ATTENTION_SUFFIXES:
        return suffix == layer_scope
    raise ValueError(f"Unsupported encoder_layer_scope: {layer_scope}")


def _encoder_single_layer_scopes(layer_scope: str) -> List[str]:
    if layer_scope == "all":
        return list(ENCODER_LAYER_SWEEP_SUFFIXES)
    if layer_scope == "attention":
        return [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ]
    if layer_scope == "non_attention":
        return [
            "intermediate.dense",
            "output.dense",
        ]
    if layer_scope in ATTENTION_SUFFIXES or layer_scope in NON_ATTENTION_SUFFIXES:
        return [layer_scope]
    raise ValueError(f"Unsupported encoder_layer_scope for single-layer sweep: {layer_scope}")


def _matches_decoder_layer_scope(layer_name: str, layer_scope: str) -> bool:
    if layer_scope == "all":
        return True

    suffix = layer_name.split("model.layers.", 1)[-1].split(".", 1)[-1]
    normalized_scope = DECODER_LAYER_SCOPE_ALIASES.get(layer_scope, layer_scope)
    combined_scope = DECODER_COMBINED_LAYER_SCOPES.get(normalized_scope)
    if combined_scope is not None:
        return suffix in combined_scope
    if layer_scope == "attention":
        return suffix in DECODER_ATTENTION_SUFFIXES
    if layer_scope == "non_attention":
        return suffix in DECODER_NON_ATTENTION_SUFFIXES
    if normalized_scope in DECODER_ATTENTION_SUFFIXES or normalized_scope in DECODER_NON_ATTENTION_SUFFIXES:
        return suffix == normalized_scope
    raise ValueError(f"Unsupported decoder_layer_scope: {layer_scope}")


def _decoder_single_layer_scopes(layer_scope: str) -> List[str]:
    normalized_scope = DECODER_LAYER_SCOPE_ALIASES.get(layer_scope, layer_scope)
    combined_scope = DECODER_COMBINED_LAYER_SCOPES.get(normalized_scope)
    if combined_scope is not None:
        return list(combined_scope)
    if normalized_scope == "all":
        return [*DECODER_ATTENTION_SUFFIXES, *DECODER_FF_SWEEP_SUFFIXES]
    if normalized_scope == "attention":
        return sorted(DECODER_ATTENTION_SUFFIXES)
    if normalized_scope == "non_attention":
        return list(DECODER_FF_SWEEP_SUFFIXES)
    if normalized_scope in DECODER_ATTENTION_SUFFIXES or normalized_scope in DECODER_NON_ATTENTION_SUFFIXES:
        return [normalized_scope]
    raise ValueError(f"Unsupported decoder_layer_scope for single-layer sweep: {layer_scope}")


def _decoder_layer_scope_slug(layer_scope: str) -> str:
    return layer_scope.replace(".", "_")


def _encoder_layer_scope_slug(layer_scope: str) -> str:
    return layer_scope.replace(".", "_")


def _encoder_layer_scope_label(layer_scope: str) -> str:
    return layer_scope.removeprefix("attention.self.").removeprefix("attention.output.")


def _decoder_layer_scope_label(layer_scope: str) -> str:
    return layer_scope.removeprefix("mlp.").removeprefix("self_attn.")


def _apply_encoder_mode(
    layer_modes: Dict[str, str],
    *,
    encoder_mode: str,
    encoder_blocks: Any,
    num_encoder_layers: int,
    encoder_layer_scope: str = "all",
) -> None:
    for block_index in _normalize_block_selector(encoder_blocks, num_encoder_layers=num_encoder_layers):
        prefix = f"roberta.encoder.layer.{block_index}."
        for layer_name in layer_modes:
            if layer_name.startswith(prefix) and _matches_encoder_layer_scope(layer_name, encoder_layer_scope):
                layer_modes[layer_name] = encoder_mode


def _apply_decoder_mode(
    layer_modes: Dict[str, str],
    *,
    decoder_mode: str,
    decoder_blocks: Any,
    num_decoder_layers: int,
    decoder_layer_scope: str = "all",
) -> None:
    for block_index in _normalize_block_selector(decoder_blocks, num_encoder_layers=num_decoder_layers):
        prefix = f"model.layers.{block_index}."
        for layer_name in layer_modes:
            if layer_name.startswith(prefix) and _matches_decoder_layer_scope(layer_name, decoder_layer_scope):
                layer_modes[layer_name] = decoder_mode


def _expand_encoder_only_experiment(
    experiment: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    num_encoder_layers = int(experiment.get("num_encoder_layers", defaults.get("num_encoder_layers", 12)))
    default_mode = experiment.get("default_mode", defaults.get("default_mode", "fp32"))
    classifier_mode = experiment.get("classifier_mode", defaults.get("classifier_mode", "int8_exact"))

    if "layer_modes" in experiment:
        layer_modes = dict(experiment["layer_modes"])
    else:
        layer_modes = _build_encoder_layer_modes(
            num_encoder_layers=num_encoder_layers,
            default_mode=default_mode,
            classifier_mode=classifier_mode,
        )

        encoder_mode = experiment.get("encoder_mode")
        if encoder_mode is not None:
            _apply_encoder_mode(
                layer_modes,
                encoder_mode=encoder_mode,
                encoder_blocks=experiment.get("encoder_blocks", "all"),
                num_encoder_layers=num_encoder_layers,
                encoder_layer_scope=experiment.get("encoder_layer_scope", defaults.get("encoder_layer_scope", "all")),
            )

        for layer_name, mode in defaults.get("layer_mode_overrides", {}).items():
            layer_modes[layer_name] = mode

    for layer_name, mode in experiment.get("layer_mode_overrides", {}).items():
        layer_modes[layer_name] = mode

    expanded = dict(experiment)
    expanded["layer_modes"] = layer_modes
    expanded.setdefault("task_type", "classification")
    return expanded


def _expand_decoder_only_experiment(
    experiment: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    num_decoder_layers = int(experiment.get("num_decoder_layers", defaults.get("num_decoder_layers", 24)))
    default_mode = experiment.get("default_mode", defaults.get("default_mode", "fp32"))
    attention_default_mode = experiment.get("attention_default_mode", defaults.get("attention_default_mode"))
    non_attention_default_mode = experiment.get("non_attention_default_mode", defaults.get("non_attention_default_mode"))

    if "layer_modes" in experiment:
        layer_modes = dict(experiment["layer_modes"])
    else:
        layer_modes = _build_decoder_layer_modes(
            num_decoder_layers=num_decoder_layers,
            default_mode=default_mode,
        )
        if attention_default_mode is not None:
            _apply_decoder_mode(
                layer_modes,
                decoder_mode=attention_default_mode,
                decoder_blocks="all",
                num_decoder_layers=num_decoder_layers,
                decoder_layer_scope="attention",
            )
        if non_attention_default_mode is not None:
            _apply_decoder_mode(
                layer_modes,
                decoder_mode=non_attention_default_mode,
                decoder_blocks="all",
                num_decoder_layers=num_decoder_layers,
                decoder_layer_scope="non_attention",
            )
        decoder_mode = experiment.get("decoder_mode")
        if decoder_mode is not None:
            _apply_decoder_mode(
                layer_modes,
                decoder_mode=decoder_mode,
                decoder_blocks=experiment.get("decoder_blocks", "all"),
                num_decoder_layers=num_decoder_layers,
                decoder_layer_scope=experiment.get("decoder_layer_scope", defaults.get("decoder_layer_scope", "all")),
            )
        for layer_name, mode in defaults.get("layer_mode_overrides", {}).items():
            layer_modes[layer_name] = mode

    for layer_name, mode in experiment.get("layer_mode_overrides", {}).items():
        layer_modes[layer_name] = mode

    expanded = dict(experiment)
    expanded["layer_modes"] = layer_modes
    expanded.setdefault("task_type", "decoder_only")
    return expanded


def _expand_encoder_only_sweep(
    sweep: Dict[str, Any],
    defaults: Dict[str, Any],
) -> List[Dict[str, Any]]:
    multiplier = sweep["multiplier"]
    num_encoder_layers = int(sweep.get("num_encoder_layers", defaults.get("num_encoder_layers", 12)))
    encoder_layer_scope = sweep.get("encoder_layer_scope", defaults.get("encoder_layer_scope", "all"))
    target_suffix = "all_encs"
    scope_suffix = f"{encoder_layer_scope}_" if encoder_layer_scope != "all" else ""
    base_tags = list(sweep.get("tags", []))
    base_tags.extend(["encoder_only", multiplier, encoder_layer_scope])

    experiments: List[Dict[str, Any]] = []
    if sweep.get("include_all_encoder_layers", True):
        experiments.append(
            {
                "name": f"{multiplier}_{scope_suffix}{target_suffix}",
                "label": f"{multiplier}:{encoder_layer_scope}:all_encs" if encoder_layer_scope != "all" else f"{multiplier}:all_encs",
                "tags": ["all_layers", *base_tags],
                "encoder_mode": multiplier,
                "encoder_blocks": "all",
                "encoder_layer_scope": encoder_layer_scope,
                "multiplier": multiplier,
                "encoder_target": "all encs",
                "plot_order": num_encoder_layers,
            }
        )

    if sweep.get("include_single_encoder_blocks", True):
        for block_index in range(num_encoder_layers):
            experiments.append(
                {
                    "name": f"{multiplier}_{scope_suffix}encoder_block_{block_index}",
                    "label": (
                        f"{multiplier}:{encoder_layer_scope}:enc{block_index}"
                        if encoder_layer_scope != "all"
                        else f"{multiplier}:enc{block_index}"
                    ),
                    "tags": ["single_block", *base_tags],
                    "encoder_mode": multiplier,
                    "encoder_blocks": [block_index],
                    "encoder_layer_scope": encoder_layer_scope,
                    "multiplier": multiplier,
                    "encoder_target": f"enc{block_index}",
                    "plot_order": block_index,
                    "encoder_block_index": block_index,
                }
            )
    if sweep.get("include_single_encoder_layers", False):
        single_layer_scopes = _encoder_single_layer_scopes(encoder_layer_scope)
        for block_index in range(num_encoder_layers):
            for layer_offset, single_layer_scope in enumerate(single_layer_scopes):
                layer_slug = _encoder_layer_scope_slug(single_layer_scope)
                layer_label = _encoder_layer_scope_label(single_layer_scope)
                experiments.append(
                    {
                        "name": f"{multiplier}_encoder_block_{block_index}_{layer_slug}",
                        "label": f"{multiplier}:enc{block_index}:{layer_label}",
                        "tags": ["single_layer", *base_tags, single_layer_scope],
                        "encoder_mode": multiplier,
                        "encoder_blocks": [block_index],
                        "encoder_layer_scope": single_layer_scope,
                        "multiplier": multiplier,
                        "encoder_target": f"enc{block_index}:{layer_label}",
                        "plot_order": block_index * len(single_layer_scopes) + layer_offset,
                        "encoder_block_index": block_index,
                        "encoder_layer_name": single_layer_scope,
                    }
                )
    return experiments


def _expand_decoder_only_sweep(
    sweep: Dict[str, Any],
    defaults: Dict[str, Any],
) -> List[Dict[str, Any]]:
    multiplier = sweep["multiplier"]
    num_decoder_layers = int(sweep.get("num_decoder_layers", defaults.get("num_decoder_layers", 24)))
    decoder_layer_scope = sweep.get("decoder_layer_scope", defaults.get("decoder_layer_scope", "all"))
    first_decoder_block = int(sweep.get("start_decoder_block", 0))
    last_decoder_block = int(sweep.get("end_decoder_block", num_decoder_layers - 1))
    if first_decoder_block < 0 or last_decoder_block >= num_decoder_layers or first_decoder_block > last_decoder_block:
        raise ValueError(
            "Invalid decoder sweep range: "
            f"start_decoder_block={first_decoder_block}, "
            f"end_decoder_block={last_decoder_block}, "
            f"num_decoder_layers={num_decoder_layers}"
        )
    scope_suffix = f"{decoder_layer_scope}_" if decoder_layer_scope != "all" else ""
    base_tags = list(sweep.get("tags", []))
    base_tags.extend(["decoder_only", multiplier, decoder_layer_scope])

    experiments: List[Dict[str, Any]] = []
    if sweep.get("include_all_decoder_layers", True):
        experiments.append(
            {
                "name": f"{multiplier}_{scope_suffix}all_decs",
                "label": f"{multiplier}:{decoder_layer_scope}:all_decs" if decoder_layer_scope != "all" else f"{multiplier}:all_decs",
                "tags": ["all_layers", *base_tags],
                "decoder_mode": multiplier,
                "decoder_blocks": "all",
                "decoder_layer_scope": decoder_layer_scope,
                "multiplier": multiplier,
                "plot_target": "all decs",
                "plot_order": num_decoder_layers,
            }
        )
    if sweep.get("include_single_decoder_blocks", True):
        for block_index in range(first_decoder_block, last_decoder_block + 1):
            experiments.append(
                {
                    "name": f"{multiplier}_{scope_suffix}decoder_block_{block_index}",
                    "label": (
                        f"{multiplier}:{decoder_layer_scope}:dec{block_index}"
                        if decoder_layer_scope != "all"
                        else f"{multiplier}:dec{block_index}"
                    ),
                    "tags": ["single_block", *base_tags],
                    "decoder_mode": multiplier,
                    "decoder_blocks": [block_index],
                    "decoder_layer_scope": decoder_layer_scope,
                    "multiplier": multiplier,
                    "plot_target": f"dec{block_index}",
                    "plot_order": block_index,
                    "decoder_block_index": block_index,
                }
            )
    if sweep.get("include_single_decoder_layers", False):
        single_layer_scopes = _decoder_single_layer_scopes(decoder_layer_scope)
        for block_index in range(first_decoder_block, last_decoder_block + 1):
            for layer_offset, single_layer_scope in enumerate(single_layer_scopes):
                layer_slug = _decoder_layer_scope_slug(single_layer_scope)
                layer_label = _decoder_layer_scope_label(single_layer_scope)
                experiments.append(
                    {
                        "name": f"{multiplier}_decoder_block_{block_index}_{layer_slug}",
                        "label": f"{multiplier}:dec{block_index}:{layer_label}",
                        "tags": ["single_layer", *base_tags, single_layer_scope],
                        "decoder_mode": multiplier,
                        "decoder_blocks": [block_index],
                        "decoder_layer_scope": single_layer_scope,
                        "multiplier": multiplier,
                        "plot_target": f"dec{block_index}:{layer_label}",
                        "plot_order": block_index * len(single_layer_scopes) + layer_offset,
                        "decoder_block_index": block_index,
                        "decoder_layer_name": single_layer_scope,
                    }
                )
    return experiments


def _expand_custom_dispatcher_config(config: Dict[str, Any]) -> Dict[str, Any]:
    base_config = dict(config.get("base_config", {}))
    defaults = dict(config.get("defaults", {}))
    architecture = defaults.get("architecture", "encoder_only")
    if architecture == "encoder_only":
        base_config.setdefault("task_type", "classification")
        base_config.setdefault("model_url", config.get("model_url", DEFAULT_MODEL))
        base_config.setdefault("dataset_url", config.get("dataset_url", DEFAULT_DATASET))
        base_config.setdefault("trace_enabled", config.get("trace_enabled", True))
        base_config.setdefault("attention_mode", config.get("attention_mode", DEFAULT_ATTENTION_MODE))
    elif architecture == "decoder_only":
        base_config.setdefault("task_type", "decoder_only")
    else:
        raise ValueError(f"Unsupported custom dispatcher architecture: {architecture}")

    expanded_experiments: List[Dict[str, Any]] = []
    for experiment in config.get("experiments", []):
        if architecture == "encoder_only":
            expanded_experiments.append(_expand_encoder_only_experiment(experiment, defaults))
        else:
            expanded_experiments.append(_expand_decoder_only_experiment(experiment, defaults))
    for sweep in config.get("sweeps", []):
        if architecture == "encoder_only":
            for experiment in _expand_encoder_only_sweep(sweep, defaults):
                expanded_experiments.append(_expand_encoder_only_experiment(experiment, defaults))
        else:
            for experiment in _expand_decoder_only_sweep(sweep, defaults):
                expanded_experiments.append(_expand_decoder_only_experiment(experiment, defaults))

    return {
        "dispatcher_name": config.get("dispatcher_name", "custom_dispatcher"),
        "description": config.get("description", ""),
        "base_config": base_config,
        "metrics_to_plot": config.get("metrics_to_plot"),
        "architecture": architecture,
        "experiments": expanded_experiments,
    }


def load_dispatcher_config(config_path: str | Path) -> Dict[str, Any]:
    config = _load_config_file(config_path)
    preset_name = config.get("preset")
    if preset_name:
        if not _is_supported_encoder_block_sweep_preset(preset_name):
            raise ValueError(f"Unsupported dispatcher preset: {preset_name}")
        config = dict(config)
        config.setdefault("approx_mode", _infer_approx_mode_from_preset(preset_name))
        config.setdefault("dispatcher_name", preset_name)
        return _encoder_block_sweep_from_preset(config)

    if "base_config" not in config or ("experiments" not in config and "sweeps" not in config):
        raise ValueError("Dispatcher config must define either a supported preset or base_config plus experiments and/or sweeps.")

    return _expand_custom_dispatcher_config(config)


def extract_accuracy(metrics: Dict[str, Any]) -> float:
    accuracy = metrics.get("accuracy")
    if accuracy is None:
        raise ValueError("Experiment metrics do not contain accuracy.")
    return float(accuracy)


def extract_classification_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    macro_avg = metrics.get("macro avg", {})
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "precision": float(macro_avg.get("precision", 0.0)),
        "recall": float(macro_avg.get("recall", 0.0)),
        "f1": float(macro_avg.get("f1-score", 0.0)),
    }


def extract_numeric_metrics(
    metrics: Dict[str, Any],
    metric_names: List[str] | tuple[str, ...] | None = None,
) -> Dict[str, float]:
    if metric_names is None:
        metric_names = [key for key, value in metrics.items() if isinstance(value, (int, float))]
    out: Dict[str, float] = {}
    for metric_name in metric_names:
        value = metrics.get(metric_name)
        if isinstance(value, (int, float)):
            out[metric_name] = float(value)
    return out


def build_dispatcher_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("No dispatcher experiment records were provided.")

    baseline_record = None
    for record in records:
        tags = record.get("config", {}).get("dispatcher_tags", [])
        if "baseline" in tags:
            baseline_record = record
            break
    if baseline_record is None:
        baseline_record = records[0]

    baseline_config = baseline_record.get("config", {})
    task_type = baseline_config.get("task_type", "classification")
    metrics_to_plot = baseline_config.get("metrics_to_plot")
    if metrics_to_plot is None:
        metrics_to_plot = DEFAULT_CLASSIFICATION_METRICS if task_type == "classification" else DEFAULT_DECODER_METRICS
    if task_type == "classification":
        baseline_metrics = extract_classification_metrics(baseline_record["metrics"])
    else:
        baseline_metrics = extract_numeric_metrics(baseline_record["metrics"], metrics_to_plot)
    summary_rows = []
    for record in records:
        config = record.get("config", {})
        if task_type == "classification":
            metric_values = extract_classification_metrics(record["metrics"])
        else:
            metric_values = extract_numeric_metrics(record["metrics"], metrics_to_plot)
        row = {
            "experiment_id": record["experiment_id"],
            "experiment_name": record["experiment_name"],
            "config_label": config.get("dispatcher_experiment_label", record["experiment_name"]),
            "is_baseline": record["experiment_id"] == baseline_record["experiment_id"],
            "order": int(config.get("dispatcher_order", 0)),
            "multiplier": config.get("multiplier"),
            "plot_target": config.get("plot_target", config.get("encoder_target")),
            "plot_order": int(config.get("plot_order", 999)),
            "layer_scope": config.get("encoder_layer_scope", config.get("decoder_layer_scope")),
        }
        row.update(metric_values)
        for metric_name in baseline_metrics:
            if metric_name in metric_values:
                row[f"{metric_name}_drop"] = baseline_metrics[metric_name] - metric_values[metric_name]
        summary_rows.append(row)

    summary_rows.sort(key=lambda row: row["order"])
    return {
        "baseline_experiment_id": baseline_record["experiment_id"],
        "baseline_experiment_name": baseline_record["experiment_name"],
        "baseline_metrics": baseline_metrics,
        "metric_names": list(baseline_metrics.keys()),
        "task_type": task_type,
        "rows": summary_rows,
    }
