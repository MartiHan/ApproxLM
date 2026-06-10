from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn as nn

from approxlm.adapters.pytorch.matmul import FloatWeightMatmulReplacementLinear, MatmulReplacementLinear
from approxlm.adapters.pytorch.quantized_linear import QuantizedLinearBackend

APPROX_OPTIONS = {
    "fp32",
    "int8_exact",
    "mul8s_exact",
    "mul8s_1KVA",
    "mul8s_1KVB",
    "mul8s_1KR6",
    "mul8s_1KR3",
}
EXACT_MODES = {None, "None", "fp32", "int8_exact", "mul8s_exact"}
APPROXIMATE_MODES = {mode for mode in APPROX_OPTIONS if mode not in EXACT_MODES}
LAYER_INDEX_PATTERN = re.compile(r"(\d+)")


def _normalize_mode(mode: Any) -> str | None:
    if hasattr(mode, "approx_lut_path"):
        approx_lut_path = getattr(mode, "approx_lut_path", None)
        if approx_lut_path:
            return str(approx_lut_path)
        return "int8_exact"
    if mode in (None, "None"):
        return None
    return str(mode)


def _mode_bucket(mode: Any, module: nn.Module | None = None) -> str:
    if module is not None and isinstance(module, (QuantizedLinearBackend, MatmulReplacementLinear, FloatWeightMatmulReplacementLinear)) and module.approx_lut_path:
        return "approximate"

    if hasattr(mode, "approx_lut_path") and getattr(mode, "approx_lut_path", None):
        return "approximate"

    normalized = _normalize_mode(mode)
    if normalized in APPROXIMATE_MODES:
        return "approximate"
    return "exact"


def _is_profiled_linear(module: nn.Module) -> bool:
    if isinstance(module, (nn.Linear, QuantizedLinearBackend, MatmulReplacementLinear, FloatWeightMatmulReplacementLinear)):
        return True
    class_name = module.__class__.__name__.lower()
    if "linear" in class_name or "quantlinear" in class_name:
        return True
    has_feature_shape = hasattr(module, "in_features") and hasattr(module, "out_features")
    has_weight_like_state = hasattr(module, "weight") or hasattr(module, "qweight") or hasattr(module, "int8_weight_nk")
    return bool(has_feature_shape and has_weight_like_state)


def _natural_layer_sort_key(layer_name: str) -> List[Any]:
    return [
        int(part) if part.isdigit() else part
        for part in LAYER_INDEX_PATTERN.split(layer_name)
    ]


@dataclass
class LayerMatmulStats:
    layer_name: str
    mode: str | None
    execution_type: str
    call_count: int = 0
    scalar_multiplications: int = 0
    input_rows: int = 0
    in_features: int = 0
    out_features: int = 0


@dataclass
class ForwardMatmulProfiler:
    model: nn.Module
    layer_modes: Dict[str, Any]
    handles: List[Any] = field(default_factory=list)
    layer_stats: Dict[str, LayerMatmulStats] = field(default_factory=dict)

    def attach(self) -> None:
        selected_names = set(self.layer_modes)
        for name, module in self.model.named_modules():
            if name not in selected_names or not _is_profiled_linear(module):
                continue
            self.layer_stats[name] = LayerMatmulStats(
                layer_name=name,
                mode=_normalize_mode(self.layer_modes.get(name)),
                execution_type=_mode_bucket(self.layer_modes.get(name), module),
            )
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs, _output) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim == 0:
                return
            in_features = int(x.shape[-1])
            rows = 1
            for dim in x.shape[:-1]:
                rows *= int(dim)

            out_features = getattr(module, "out_features", None)
            if out_features is None and hasattr(module, "weight") and torch.is_tensor(module.weight):
                out_features = int(module.weight.shape[0])
            if out_features is None:
                return

            stats = self.layer_stats[layer_name]
            stats.call_count += 1
            stats.input_rows += rows
            stats.in_features = in_features
            stats.out_features = int(out_features)
            stats.scalar_multiplications += rows * in_features * int(out_features)

        return hook

    def summary(self) -> Dict[str, Any]:
        rows = []
        total_scalar_mults = 0
        approximate_scalar_mults = 0
        exact_scalar_mults = 0
        total_calls = 0
        approximate_calls = 0
        exact_calls = 0

        for layer_name in sorted(self.layer_stats, key=_natural_layer_sort_key):
            stats = self.layer_stats[layer_name]
            row = {
                "layer_name": stats.layer_name,
                "mode": stats.mode,
                "execution_type": stats.execution_type,
                "call_count": stats.call_count,
                "input_rows": stats.input_rows,
                "in_features": stats.in_features,
                "out_features": stats.out_features,
                "mac_operations": stats.scalar_multiplications,
                "scalar_multiplications": stats.scalar_multiplications,
            }
            rows.append(row)

            total_scalar_mults += stats.scalar_multiplications
            total_calls += stats.call_count
            if stats.execution_type == "approximate":
                approximate_scalar_mults += stats.scalar_multiplications
                approximate_calls += stats.call_count
            else:
                exact_scalar_mults += stats.scalar_multiplications
                exact_calls += stats.call_count

        for row in rows:
            fraction = (row["mac_operations"] / total_scalar_mults) if total_scalar_mults else 0.0
            row["mac_contribution_fraction"] = fraction
            row["mac_contribution_percent"] = 100.0 * fraction

        return {
            "total_selected_layers": len(self.layer_stats),
            "total_linear_calls": total_calls,
            "approximate_linear_calls": approximate_calls,
            "exact_linear_calls": exact_calls,
            "total_mac_operations": total_scalar_mults,
            "approximate_mac_operations": approximate_scalar_mults,
            "exact_mac_operations": exact_scalar_mults,
            "total_scalar_multiplications": total_scalar_mults,
            "approximate_scalar_multiplications": approximate_scalar_mults,
            "exact_scalar_multiplications": exact_scalar_mults,
            "approximate_mac_fraction": (approximate_scalar_mults / total_scalar_mults) if total_scalar_mults else 0.0,
            "exact_mac_fraction": (exact_scalar_mults / total_scalar_mults) if total_scalar_mults else 0.0,
            "approximate_scalar_fraction": (approximate_scalar_mults / total_scalar_mults) if total_scalar_mults else 0.0,
            "exact_scalar_fraction": (exact_scalar_mults / total_scalar_mults) if total_scalar_mults else 0.0,
            "per_layer": rows,
        }
