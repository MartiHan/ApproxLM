from __future__ import annotations

from typing import Any, Dict, Iterable, List

from approxlm.application.defaults import APPROX_OPTIONS, BLOCK_INDEX_PATTERN

EXACT_MODES = {None, "None", "fp32", "int8_exact", "mul8s_exact"}
APPROXIMATE_MODES = {mode for mode in APPROX_OPTIONS if mode not in EXACT_MODES}


def normalize_mode(mode: Any) -> str | None:
    if mode in (None, "None"):
        return None
    return str(mode)


def is_approximate_mode(mode: Any) -> bool:
    return normalize_mode(mode) in APPROXIMATE_MODES


def is_exact_mode(mode: Any) -> bool:
    return normalize_mode(mode) in EXACT_MODES


def _group_name(layer_name: str) -> str:
    match = BLOCK_INDEX_PATTERN.match(layer_name)
    if not match:
        return "head"
    prefix, block_index, _ = match.groups()
    prefix_parts = [part for part in prefix.split(".") if part]
    family = prefix_parts[-1] if prefix_parts else "block"
    return f"{family}_{block_index}"


def summarize_layer_modes(layer_modes: Dict[str, Any]) -> Dict[str, Any]:
    total = len(layer_modes)
    approximate_layers: List[str] = []
    exact_layers: List[str] = []
    fp32_layers: List[str] = []
    int8_exact_layers: List[str] = []
    mul8s_exact_layers: List[str] = []
    per_group: Dict[str, Dict[str, int]] = {}

    for layer_name, raw_mode in layer_modes.items():
        mode = normalize_mode(raw_mode)
        group = _group_name(layer_name)
        group_counts = per_group.setdefault(
            group,
            {
                "total": 0,
                "approximate": 0,
                "exact": 0,
                "fp32": 0,
                "int8_exact": 0,
                "mul8s_exact": 0,
            },
        )
        group_counts["total"] += 1

        if is_approximate_mode(mode):
            approximate_layers.append(layer_name)
            group_counts["approximate"] += 1
            continue

        exact_layers.append(layer_name)
        group_counts["exact"] += 1
        if mode in (None, "fp32"):
            fp32_layers.append(layer_name)
            group_counts["fp32"] += 1
        elif mode == "int8_exact":
            int8_exact_layers.append(layer_name)
            group_counts["int8_exact"] += 1
        elif mode == "mul8s_exact":
            mul8s_exact_layers.append(layer_name)
            group_counts["mul8s_exact"] += 1

    return {
        "total_linear_layers": total,
        "approximate_matmul_count": len(approximate_layers),
        "exact_matmul_count": len(exact_layers),
        "fp32_matmul_count": len(fp32_layers),
        "int8_exact_matmul_count": len(int8_exact_layers),
        "mul8s_exact_matmul_count": len(mul8s_exact_layers),
        "approximate_fraction": (len(approximate_layers) / total) if total else 0.0,
        "exact_fraction": (len(exact_layers) / total) if total else 0.0,
        "approximate_layers": approximate_layers,
        "exact_layers": exact_layers,
        "fp32_layers": fp32_layers,
        "int8_exact_layers": int8_exact_layers,
        "mul8s_exact_layers": mul8s_exact_layers,
        "per_group_counts": per_group,
    }


def summarize_experiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    layer_modes = config.get("layer_modes")
    if not isinstance(layer_modes, dict):
        raise ValueError("Experiment config does not contain a valid layer_modes mapping.")
    return summarize_layer_modes(layer_modes)


def summarize_dispatcher_configs(configs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for index, config in enumerate(configs):
        summary = summarize_experiment_config(config)
        summary["index"] = index
        summary["experiment_name"] = config.get("dispatcher_experiment_name") or config.get("experiment_name")
        summary["experiment_label"] = config.get("dispatcher_experiment_label")
        summaries.append(summary)
    return summaries
