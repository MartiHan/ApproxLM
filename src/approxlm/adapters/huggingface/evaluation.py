from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gc
from datasets import DatasetDict, load_dataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from approxlm.adapters.pytorch.matmul import FloatWeightMatmulReplacementLinear, MatmulReplacementLinear
from approxlm.adapters.pytorch.quantized_linear import QuantizedLinearBackend, weight_stats_from_linear
from approxlm.adapters.huggingface.datasets import MassiveTorchDataset, SequenceClassificationCollator
from approxlm.domain.specs import (
    DatasetSpec,
    DecoderOnlyDatasetSpec,
    DecoderOnlyRuntimeSpec,
    LayerQuantSpec,

    ModelSpec,
    RuntimeSpec,

    TraceSpec,
    normalize_layer_spec,
)
from approxlm.adapters.pytorch.profiler import ForwardMatmulProfiler
from approxlm.quantization.builder import QuantParamsBuilder
from approxlm.quantization.calibration.base import CalibStats
from approxlm.quantization.calibration.histogram import HistogramCalibrator
from approxlm.adapters.huggingface.decoder_eval import (
    build_chat_prompt,
    build_records,
    evaluate_decoder_only_model,
    evaluate_qualitative_prompt,
    infer_dataset_format,
    load_generation_model,
)


def _build_parent_map(model: nn.Module) -> Dict[str, Tuple[nn.Module, str]]:
    parent_map: Dict[str, Tuple[nn.Module, str]] = {}
    for parent_name, parent in model.named_modules():
        for child_name, _child in parent.named_children():
            parent_map[f"{parent_name}.{child_name}".lstrip(".")] = (parent, child_name)
    return parent_map


def _is_linear_like_module(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True

    class_name = module.__class__.__name__.lower()
    if "linear" in class_name or "quantlinear" in class_name:
        return True

    has_feature_shape = hasattr(module, "in_features") and hasattr(module, "out_features")
    has_weight_like_state = hasattr(module, "weight") or hasattr(module, "qweight") or hasattr(module, "int8_weight_nk")
    return bool(has_feature_shape and has_weight_like_state)


def _is_prequantized_model(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "quantization_config", None) is not None:
        return True

    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        if "quantlinear" in class_name or "gptq" in class_name or "awq" in class_name:
            return True
        if hasattr(module, "qweight"):
            return True
    return False


def _layer_specs_with_lut(layer_specs: Dict[str, LayerQuantSpec]) -> Dict[str, LayerQuantSpec]:
    return {
        name: spec
        for name, spec in layer_specs.items()
        if normalize_layer_spec(spec).approx_lut_path is not None
    }


def _print_matmul_profile_mac_contributions(profile: Dict[str, Any]) -> None:
    rows = profile.get("per_layer") or []
    if not rows:
        return

    print("Decoder-only linear MAC operation contribution:")
    print(f"{'layer_name':<64} {'mac_operations':>18} {'mac_%':>10}")
    for row in rows:
        print(
            f"{row.get('layer_name', ''):<64} "
            f"{int(row.get('mac_operations', 0)):>18} "
            f"{float(row.get('mac_contribution_percent', 0.0)):>9.4f}%"
        )


def _module_device(module: nn.Module) -> torch.device:
    for attr in ("weight", "qweight", "int8_weight_nk", "bias"):
        value = getattr(module, attr, None)
        if torch.is_tensor(value):
            return value.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _materialize_linear_like_module(module: nn.Module, batch_size: int = 256) -> nn.Linear:
    if isinstance(module, nn.Linear):
        linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
        linear.weight.data.copy_(module.weight.detach().to(torch.float32))
        if module.bias is not None:
            linear.bias.data.copy_(module.bias.detach().to(torch.float32))
        return linear

    if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
        raise RuntimeError(f"Cannot materialize linear-like module of type {module.__class__.__name__}.")

    module_device = _module_device(module)
    input_dtype = None
    for attr in ("weight", "qweight", "bias"):
        value = getattr(module, attr, None)
        if torch.is_tensor(value) and torch.is_floating_point(value):
            input_dtype = value.dtype
            break
    if input_dtype is None:
        input_dtype = torch.float16 if module_device.type == "cuda" else torch.float32

    training = module.training
    module.eval()
    rows: List[torch.Tensor] = []
    eye = torch.eye(module.in_features, device=module_device, dtype=input_dtype)
    bias = getattr(module, "bias", None)

    with torch.no_grad():
        for start in range(0, module.in_features, batch_size):
            stop = min(start + batch_size, module.in_features)
            chunk = eye[start:stop]
            out = module(chunk)
            if bias is not None:
                out = out - bias
            rows.append(out.detach().to(torch.float32).cpu())

    if training:
        module.train()

    weight = torch.cat(rows, dim=0).transpose(0, 1).contiguous()
    linear = nn.Linear(module.in_features, module.out_features, bias=bias is not None)
    linear.weight.data.copy_(weight)
    if bias is not None:
        linear.bias.data.copy_(bias.detach().to(torch.float32).cpu())
    return linear


def _prequantized_int8_state(module: nn.Module) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
    int8_weight = getattr(module, "int8_weight_nk", None)
    if not torch.is_tensor(int8_weight) or int8_weight.ndim != 2:
        return None

    weight_scale = getattr(module, "int8_channel_scale", None)
    if not torch.is_tensor(weight_scale):
        weight_scale = None
    bias = getattr(module, "bias", None)
    if not torch.is_tensor(bias):
        bias = None
    return int8_weight.detach(), weight_scale.detach() if weight_scale is not None else None, bias.detach() if bias is not None else None


def _requantize_weight_per_output_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    weight_fp = weight.detach().to(torch.float32)
    channel_max = weight_fp.abs().amax(dim=1)
    channel_scale = (channel_max / 127.0).clamp_min(1e-10)
    weight_int8 = torch.round(weight_fp / channel_scale.view(-1, 1)).clamp(-128, 127).to(torch.int8)
    return weight_int8, channel_scale


def _linear_from_gptq_dequantized_weight(module: nn.Module) -> Optional[nn.Linear]:
    required = ("qweight", "qzeros", "scales", "g_idx", "bits", "pack_dtype_bits", "maxq")
    if not all(hasattr(module, attr) for attr in required):
        return None

    try:
        from gptqmodel.nn_modules.triton_utils.dequant import dequant
    except Exception:
        return None

    weight_kn = dequant(
        torch.float32,
        module.qweight,
        module.scales,
        module.qzeros,
        module.g_idx,
        int(module.bits),
        int(module.pack_dtype_bits),
        int(module.maxq),
    )
    weight = weight_kn.transpose(0, 1).contiguous()
    bias = getattr(module, "bias", None)
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    linear.weight.data.copy_(weight.detach().cpu())
    if bias is not None:
        linear.bias.data.copy_(bias.detach().to(torch.float32).cpu())
    return linear


def replace_selected_linears_with_quantized_backend_per_layer(
    model: nn.Module,
    act_stats_by_name: Dict[str, CalibStats],
    layer_specs: Dict[str, LayerQuantSpec],
) -> List[str]:
    parent_map = _build_parent_map(model)
    replaced: List[str] = []

    for name, module in list(model.named_modules()):
        if not _is_linear_like_module(module) or name not in layer_specs or name not in act_stats_by_name:
            continue

        spec = normalize_layer_spec(layer_specs[name])
        float_linear = _materialize_linear_like_module(module)
        act_builder = QuantParamsBuilder(
            data_type=spec.act_data_type,
            symmetric=spec.act_symmetric,
            per_channel=spec.act_per_channel,
        )
        weight_builder = QuantParamsBuilder(
            data_type=spec.weight_data_type,
            symmetric=spec.weight_symmetric,
            per_channel=spec.weight_per_channel,
        )
        act_qparams = act_builder.build(act_stats_by_name[name])
        weight_qparams = weight_builder.build(
            weight_stats_from_linear(
                float_linear,
                per_channel=spec.weight_per_channel,
                symmetric=spec.weight_symmetric,
                data_type=spec.weight_data_type,
            )
        )

        parent, attr = parent_map[name]
        setattr(
            parent,
            attr,
            QuantizedLinearBackend(
                float_linear=float_linear,
                act_quant_params=act_qparams,
                weight_quant_params=weight_qparams,
                approx_lut_path=spec.approx_lut_path,
            ).to(_module_device(module)),
        )
        replaced.append(name)

    return replaced


def replace_selected_linears_with_matmul_replacement(
    model: nn.Module,
    layer_specs: Dict[str, LayerQuantSpec],
) -> List[str]:
    parent_map = _build_parent_map(model)
    replaced: List[str] = []

    for name in layer_specs:
        if name not in parent_map:
            continue

        parent, attr = parent_map[name]
        module = getattr(parent, attr)
        if not _is_linear_like_module(module):
            continue
        spec = normalize_layer_spec(layer_specs[name])

        int8_state = _prequantized_int8_state(module)
        float_linear = None
        replacement_cls = MatmulReplacementLinear
        if int8_state is None:
            float_linear = _linear_from_gptq_dequantized_weight(module)
            if float_linear is not None:
                replacement_cls = MatmulReplacementLinear
        if int8_state is None and float_linear is None:
            float_linear = _materialize_linear_like_module(module)
            replacement_cls = FloatWeightMatmulReplacementLinear

        if int8_state is not None:
            weight_int8, weight_scale, bias = int8_state
            replacement = MatmulReplacementLinear(
                weight_int8=weight_int8,
                weight_scale=weight_scale,
                bias=bias,
                approx_lut_path=spec.approx_lut_path,
            )
        elif replacement_cls is FloatWeightMatmulReplacementLinear:
            replacement = FloatWeightMatmulReplacementLinear(
                weight=float_linear.weight.detach(),
                bias=float_linear.bias.detach() if float_linear.bias is not None else None,
                approx_lut_path=spec.approx_lut_path,
            )
        else:
            weight_int8, weight_scale = _requantize_weight_per_output_int8(float_linear.weight.detach())
            replacement = MatmulReplacementLinear(
                weight_int8=weight_int8,
                weight_scale=weight_scale,
                bias=float_linear.bias.detach() if float_linear.bias is not None else None,
                approx_lut_path=spec.approx_lut_path,
            )
        module_device = _module_device(module)
        setattr(parent, attr, replacement.to(module_device))
        del module, replacement, float_linear, int8_state
        gc.collect()
        if module_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        replaced.append(name)

    return replaced


class BaselineIntentEvaluator:
    def __init__(
        self,
        model_spec: ModelSpec,
        dataset_spec: DatasetSpec,
        runtime_spec: RuntimeSpec,
        layer_specs: Optional[Dict[str, LayerQuantSpec]] = None,
        profile_layer_modes: Optional[Dict[str, Any]] = None,
        calibration_percentile: float = 99.9,
        calibration_batches: int = 50,
        trace_spec: Optional[TraceSpec] = None,
        backend_quantize: bool = True,
    ):
        self.layer_specs = {name: normalize_layer_spec(spec) for name, spec in (layer_specs or {}).items()}
        self.profile_layer_modes = dict(profile_layer_modes or self.layer_specs)
        self.model_spec = model_spec
        self.dataset_spec = dataset_spec
        self.runtime_spec = runtime_spec
        self.calibration_percentile = calibration_percentile
        self.calibration_batches = calibration_batches
        self.trace_spec = trace_spec or TraceSpec()
        self.backend_quantize = bool(backend_quantize)
        self.device = self._resolve_device(runtime_spec.device)
        self.tokenizer = None
        self.model = None
        self.dataset_dict: Optional[DatasetDict] = None
        self.progress_callback = None
        self.model_is_prequantized = False
        self.forward_matmul_profiler: Optional[ForwardMatmulProfiler] = None

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        return torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _emit_progress(self, progress: float, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(progress, message)

    def _empty_trace_store(self) -> Dict[str, List[Any]]:
        return {
            "sample_id": [],
            "true_label_id": [],
            "pred_label_id": [],
            "true_label_str": [],
            "pred_label_str": [],
            "logits": [],
            "cls_by_layer": [],
            "input_ids": [],
            "attention_mask": [],
            "text": [],
        }

    def _finalize_trace_store(self, traces: Dict[str, List[Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, values in traces.items():
            if not values:
                out[key] = []
            elif key in {"sample_id", "true_label_id", "pred_label_id"}:
                out[key] = np.asarray(values, dtype=np.int64)
            elif key in {"true_label_str", "pred_label_str", "text"}:
                out[key] = np.asarray(values, dtype=object)
            else:
                try:
                    out[key] = np.stack(values, axis=0)
                except ValueError:
                    out[key] = np.asarray(values, dtype=object)
        return out

    def _extract_cls_by_layer(self, hidden_states) -> Optional[torch.Tensor]:
        if hidden_states is None or len(hidden_states) <= 1:
            return None
        cls_layers = [state[:, 0, :].detach().to(torch.float32).cpu() for state in hidden_states[1:] if state is not None]
        return torch.stack(cls_layers, dim=1) if cls_layers else None

    def load_model(self) -> None:
        self._emit_progress(0.10, "Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_spec.name)
        self._emit_progress(0.25, "Loading sequence classification model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_spec.name)
        self.model.to(self.device)
        self.model.eval()
        self.model_is_prequantized = _is_prequantized_model(self.model)
        self._emit_progress(0.35, f"Model ready on {self.device}.")

    def load_dataset(self) -> None:
        self._emit_progress(0.40, "Loading dataset split metadata...")
        self.dataset_dict = load_dataset(
            self.dataset_spec.name,
            revision=self.dataset_spec.revision,
            data_dir=self.dataset_spec.data_dir,
        )
        self._emit_progress(0.50, "Dataset loaded.")

    def _build_loader(self, split_name: str) -> DataLoader:
        if self.dataset_dict is None or self.tokenizer is None:
            raise RuntimeError("Tokenizer and dataset must be loaded first.")

        torch_ds = MassiveTorchDataset(
            hf_ds=self.dataset_dict[split_name],
            text_col=self.dataset_spec.text_col,
            label_col=self.dataset_spec.label_col,
        )
        collator = SequenceClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.runtime_spec.max_length,
            pad_to_max_length=self.trace_spec.enabled and self.trace_spec.pad_to_max_length,
            store_text=self.trace_spec.enabled and self.trace_spec.store_text,
        )
        return DataLoader(
            torch_ds,
            batch_size=self.runtime_spec.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.runtime_spec.num_workers,
            pin_memory=self.runtime_spec.pin_memory and self.device.type == "cuda",
        )

    def _selected_linear_modules_by_name(self) -> Dict[str, nn.Module]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        selected_names = set(self.layer_specs)
        return {
            name: module
            for name, module in self.model.named_modules()
            if _is_linear_like_module(module) and name in selected_names
        }

    @staticmethod
    def group_layers_for_calibration(layer_specs: Dict[str, LayerQuantSpec]) -> Dict[Tuple[str, bool, bool], List[str]]:
        groups: Dict[Tuple[str, bool, bool], List[str]] = {}
        for layer_name, spec in layer_specs.items():
            groups.setdefault((spec.act_data_type, spec.act_symmetric, spec.act_per_channel), []).append(layer_name)
        return groups

    def prepare_model_for_inference(self) -> None:
        if self.model is None or self.dataset_dict is None or not self.layer_specs:
            return
        active_layer_specs = self.layer_specs
        if self.model_is_prequantized:
            if self.backend_quantize:
                active_layer_specs = _layer_specs_with_lut(self.layer_specs)
                if not active_layer_specs:
                    self._emit_progress(0.76, "Model is already quantized; skipping additional backend quantization.")
                    return
                raise RuntimeError(
                    "LUT replacement for a prequantized GPTQ model must use the existing GPTQ quantized tensors. "
                    "Disable backend quantization to use scale 1 and zero bias instead of W8A8 "
                    "calibration/requantization on an already quantized model."
                )

        original_layer_specs = self.layer_specs
        try:
            self.layer_specs = active_layer_specs
            if self.backend_quantize:
                selected_modules = self._selected_linear_modules_by_name()
                calib_loader = self._build_loader(self.dataset_spec.val_split)
                grouped = self.group_layers_for_calibration(active_layer_specs)
                act_stats_by_name: Dict[str, CalibStats] = {}
                self._emit_progress(0.52, f"Calibrating {len(active_layer_specs)} selected layers...")

                group_items = list(grouped.items())
                total_groups = max(len(group_items), 1)
                for group_idx, ((act_data_type, act_symmetric, act_per_channel), layer_names) in enumerate(group_items, start=1):
                    calibrator = HistogramCalibrator(
                        data_type=act_data_type,
                        per_channel=act_per_channel,
                        symmetric=act_symmetric,
                        calib_dataset=calib_loader,
                        num_batches=self.calibration_batches,
                        layers_to_calibrate=[selected_modules[name] for name in layer_names if name in selected_modules],
                        percentile=self.calibration_percentile,
                        num_bins=2048,
                        channel_axis=-1,
                    )
                    act_stats_by_name.update(calibrator.compute_calib_stats(self.model))
                    self._emit_progress(
                        0.52 + (0.72 - 0.52) * (group_idx / total_groups),
                        f"Calibration group {group_idx}/{total_groups} finished.",
                    )
                replaced = replace_selected_linears_with_quantized_backend_per_layer(
                    model=self.model,
                    act_stats_by_name=act_stats_by_name,
                    layer_specs=active_layer_specs,
                )
            else:
                if not any(name in _build_parent_map(self.model) for name in active_layer_specs):
                    raise RuntimeError(
                        "None of the selected layer names matched modules in the loaded model."
                    )
                self._emit_progress(
                    0.72,
                    f"Installing identity matmul replacements for {len(active_layer_specs)} selected layers.",
                )
                replaced = replace_selected_linears_with_matmul_replacement(
                    model=self.model,
                    layer_specs=active_layer_specs,
                )
        finally:
            self.layer_specs = original_layer_specs

        self.model.to(self.device)
        self.model.eval()
        backend_label = "quantized backend" if self.backend_quantize else "matmul replacement"
        self._emit_progress(0.76, f"{backend_label.title()} modules installed for {len(replaced)} layers.")

    def _label_names(self, split_name: str) -> List[str]:
        if self.dataset_dict is None:
            raise RuntimeError("Dataset is not loaded.")
        return list(self.dataset_dict[split_name].features[self.dataset_spec.label_col].names)

    def _true_labels_as_strings(self, split_name: str) -> List[str]:
        if self.dataset_dict is None:
            raise RuntimeError("Dataset is not loaded.")
        split = self.dataset_dict[split_name]
        label_feat = split.features[self.dataset_spec.label_col]
        return [label_feat.int2str(idx) for idx in split[self.dataset_spec.label_col]]

    @torch.inference_mode()
    def predict_split(self, split_name: str) -> Dict[str, Any]:
        if self.model is None or self.dataset_dict is None:
            raise RuntimeError("Model and dataset must be loaded first.")

        loader = self._build_loader(split_name)
        y_pred_str: List[str] = []
        y_true_str = self._true_labels_as_strings(split_name)
        traces = self._empty_trace_store() if self.trace_spec.enabled else None
        forward_keys = {"input_ids", "attention_mask", "token_type_ids"}
        num_batches = max(len(loader), 1)
        self._emit_progress(0.78, f"Running inference on {split_name} split...")

        collect_hidden = self.trace_spec.enabled
        matmul_profile = {}
        if self.model is not None and self.profile_layer_modes:
            self.forward_matmul_profiler = ForwardMatmulProfiler(self.model, self.profile_layer_modes)
            self.forward_matmul_profiler.attach()

        try:
            for batch_idx, batch in enumerate(loader, start=1):
                inputs = {key: value.to(self.device) for key, value in batch.items() if key in forward_keys and torch.is_tensor(value)}
                outputs = self.model(
                    **inputs,
                    output_hidden_states=collect_hidden,
                    output_attentions=False,
                    return_dict=True,
                )
                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
                pred_labels = [self.model.config.id2label.get(pred_id, f"LABEL_{pred_id}") for pred_id in pred_ids]
                y_pred_str.extend(pred_labels)

                if self.trace_spec.enabled and traces is not None:
                    self._append_batch_traces(traces, batch, outputs, pred_ids, pred_labels, collect_hidden)

                self._emit_progress(
                    min(0.78 + 0.17 * (batch_idx / num_batches), 0.90),
                    f"Inference progress: {batch_idx}/{num_batches} batches processed.",
                )
        finally:
            if self.forward_matmul_profiler is not None:
                matmul_profile = self.forward_matmul_profiler.summary()
                self.forward_matmul_profiler.detach()
            self.forward_matmul_profiler = None

        result = {"y_true_str": y_true_str, "y_pred_str": y_pred_str}
        if traces is not None:
            result["traces"] = self._finalize_trace_store(traces)
        result["matmul_profile"] = matmul_profile
        return result

    def _append_batch_traces(
        self,
        traces: Dict[str, List[Any]],
        batch: Dict[str, Any],
        outputs,
        pred_ids: List[int],
        pred_labels: List[str],
        collect_hidden: bool,
    ) -> None:
        logits_cpu = outputs.logits.detach().to(torch.float32).cpu().numpy()
        cls_by_layer = self._extract_cls_by_layer(outputs.hidden_states).numpy() if collect_hidden and outputs.hidden_states is not None else None
        input_ids = batch["input_ids"].cpu().numpy() if self.trace_spec.store_input_ids else None
        attention_mask = batch["attention_mask"].cpu().numpy() if self.trace_spec.store_attention_mask else None
        texts = batch["text"] if self.trace_spec.store_text and "text" in batch else None
        sample_ids = batch["sample_id"].cpu().tolist()
        true_label_ids = batch["labels"].cpu().tolist()

        for idx, sample_id in enumerate(sample_ids):
            traces["sample_id"].append(int(sample_id))
            traces["true_label_id"].append(int(true_label_ids[idx]))
            traces["pred_label_id"].append(int(pred_ids[idx]))
            traces["true_label_str"].append(self.model.config.id2label.get(int(true_label_ids[idx]), str(true_label_ids[idx])))
            traces["pred_label_str"].append(pred_labels[idx])
            traces["logits"].append(logits_cpu[idx])
            if cls_by_layer is not None:
                traces["cls_by_layer"].append(cls_by_layer[idx])
            if input_ids is not None:
                traces["input_ids"].append(input_ids[idx])
            if attention_mask is not None:
                traces["attention_mask"].append(attention_mask[idx])
            if texts is not None:
                traces["text"].append(texts[idx])

    def evaluate_split(self, split_name: str = "test") -> Dict[str, Any]:
        pred_out = self.predict_split(split_name)
        labels_order = sorted(self._label_names(split_name))
        self._emit_progress(0.96, "Computing classification report...")
        report_dict = classification_report(
            pred_out["y_true_str"],
            pred_out["y_pred_str"],
            labels=labels_order,
            target_names=labels_order,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "label"})
        self._emit_progress(1.0, "Evaluation finished.")
        result = {
            "split": split_name,
            "device": str(self.device),
            "num_examples": len(pred_out["y_true_str"]),
            "metrics": report_dict,
            "report_df": report_df,
            "matmul_profile": pred_out.get("matmul_profile"),
        }
        if "traces" in pred_out:
            result["traces"] = pred_out["traces"]
        return result

    def run_baseline(self, split_name: str = "test", progress_callback=None) -> Dict[str, Any]:
        self.progress_callback = progress_callback
        self._emit_progress(0.02, "Starting baseline evaluation...")
        self.load_model()
        self.load_dataset()
        self.prepare_model_for_inference()
        return self.evaluate_split(split_name=split_name)


class DecoderOnlyCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        prompts = [item["prompt"] for item in batch]
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }


class DecoderOnlyEvaluator:
    def __init__(
        self,
        model_spec: ModelSpec,
        dataset_spec: DecoderOnlyDatasetSpec,
        runtime_spec: DecoderOnlyRuntimeSpec,
        layer_specs: Optional[Dict[str, LayerQuantSpec]] = None,
        profile_layer_modes: Optional[Dict[str, Any]] = None,
        calibration_percentile: float = 99.9,
        calibration_batches: int = 16,
        trace_spec: Optional[TraceSpec] = None,
        backend_quantize: bool = True,
    ):
        self.model_spec = model_spec
        self.dataset_spec = dataset_spec
        self.runtime_spec = runtime_spec
        self.layer_specs = {name: normalize_layer_spec(spec) for name, spec in (layer_specs or {}).items()}
        self.profile_layer_modes = dict(profile_layer_modes or self.layer_specs)
        self.calibration_percentile = calibration_percentile
        self.calibration_batches = calibration_batches
        self.trace_spec = trace_spec or TraceSpec()
        self.backend_quantize = bool(backend_quantize)
        self.device = BaselineIntentEvaluator._resolve_device(runtime_spec.device)
        self.tokenizer = None
        self.model = None
        self.records: List[Dict[str, str]] = []
        self.progress_callback = None
        self.model_is_prequantized = False
        self.forward_matmul_profiler: Optional[ForwardMatmulProfiler] = None

    def _emit_progress(self, progress: float, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(progress, message)

    def load_model(self) -> None:
        self._emit_progress(0.05, "Loading decoder-only tokenizer...")
        self._emit_progress(0.18, "Loading decoder-only model...")
        self.tokenizer, self.model = load_generation_model(
            self.model_spec.name,
            self.runtime_spec.trust_remote_code,
            self.device,
        )
        self.model_is_prequantized = _is_prequantized_model(self.model)
        self._emit_progress(0.28, f"Decoder-only model ready on {self.device}.")

    def load_dataset(self) -> None:
        self._emit_progress(0.32, "Loading decoder-only evaluation records...")
        dataset_format = infer_dataset_format(self.dataset_spec.name, self.dataset_spec.format)
        if dataset_format == "wikitext" and (not self.layer_specs or not self.backend_quantize):
            self.records = []
            self._emit_progress(0.38, "WikiText corpus perplexity will stream from the dataset at inference time.")
            return

        max_records = self.runtime_spec.max_samples
        if dataset_format == "wikitext" and self.layer_specs and self.backend_quantize:
            max_records = max(1, self.runtime_spec.batch_size * self.calibration_batches)
        self.records = build_records(
            self.dataset_spec.name,
            self.dataset_spec.split,
            max_records,
            dataset_config=self.dataset_spec.config,
            dataset_format=self.dataset_spec.format,
        )
        self._emit_progress(0.38, f"Loaded {len(self.records)} evaluation samples.")

    def _calibration_loader(self) -> DataLoader:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded first.")
        prompts = [
            {"prompt": build_chat_prompt(self.tokenizer, record["instruction"], record["input"])}
            for record in self.records
        ]
        return DataLoader(
            prompts,
            batch_size=self.runtime_spec.batch_size,
            shuffle=False,
            collate_fn=DecoderOnlyCollator(self.tokenizer, self.runtime_spec.max_input_length),
            num_workers=self.runtime_spec.num_workers,
            pin_memory=self.runtime_spec.pin_memory and self.device.type == "cuda",
        )

    def _selected_linear_modules_by_name(self) -> Dict[str, nn.Module]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        selected_names = set(self.layer_specs)
        return {
            name: module
            for name, module in self.model.named_modules()
            if _is_linear_like_module(module) and name in selected_names
        }

    def prepare_model_for_inference(self) -> None:
        if self.model is None or self.tokenizer is None or not self.layer_specs:
            return
        active_layer_specs = self.layer_specs
        if self.model_is_prequantized:
            if self.backend_quantize:
                active_layer_specs = _layer_specs_with_lut(self.layer_specs)
                if not active_layer_specs:
                    self._emit_progress(0.66, "Model is already quantized; skipping additional backend quantization.")
                    return
                raise RuntimeError(
                    "LUT replacement for a prequantized GPTQ model must use the existing GPTQ quantized tensors. "
                    "Disable backend quantization to use scale 1 and zero bias instead of W8A8 "
                    "calibration/requantization on an already quantized model."
                )

        original_layer_specs = self.layer_specs
        try:
            self.layer_specs = active_layer_specs
            if self.backend_quantize:
                selected_modules = self._selected_linear_modules_by_name()
                if not selected_modules:
                    raise RuntimeError(
                        "None of the selected decoder-only layer names matched linear-like modules in the loaded model."
                    )
                calib_loader = self._calibration_loader()
                grouped = BaselineIntentEvaluator.group_layers_for_calibration(active_layer_specs)
                act_stats_by_name: Dict[str, CalibStats] = {}
                self._emit_progress(0.42, f"Calibrating {len(active_layer_specs)} decoder-only layers...")

                group_items = list(grouped.items())
                total_groups = max(len(group_items), 1)
                for group_idx, ((act_data_type, act_symmetric, act_per_channel), layer_names) in enumerate(group_items, start=1):
                    calibrator = HistogramCalibrator(
                        data_type=act_data_type,
                        per_channel=act_per_channel,
                        symmetric=act_symmetric,
                        calib_dataset=calib_loader,
                        num_batches=self.calibration_batches,
                        layers_to_calibrate=[selected_modules[name] for name in layer_names if name in selected_modules],
                        percentile=self.calibration_percentile,
                        num_bins=2048,
                        channel_axis=-1,
                    )
                    act_stats_by_name.update(calibrator.compute_calib_stats(self.model))
                    self._emit_progress(
                        0.42 + (0.62 - 0.42) * (group_idx / total_groups),
                        f"Calibration group {group_idx}/{total_groups} finished.",
                    )
                replaced = replace_selected_linears_with_quantized_backend_per_layer(
                    model=self.model,
                    act_stats_by_name=act_stats_by_name,
                    layer_specs=active_layer_specs,
                )
            else:
                if not any(name in _build_parent_map(self.model) for name in active_layer_specs):
                    raise RuntimeError(
                        "None of the selected decoder-only layer names matched modules in the loaded model."
                    )
                self._emit_progress(
                    0.62,
                    f"Installing identity matmul replacements for {len(active_layer_specs)} decoder-only layers.",
                )
                replaced = replace_selected_linears_with_matmul_replacement(
                    model=self.model,
                    layer_specs=active_layer_specs,
                )
        finally:
            self.layer_specs = original_layer_specs

        if not hasattr(self.model, "hf_device_map"):
            self.model.to(self.device)
        self.model.eval()
        backend_label = "quantized backend" if self.backend_quantize else "matmul replacement"
        self._emit_progress(0.66, f"{backend_label.title()} modules installed for {len(replaced)} decoder-only layers.")

    def evaluate(self, progress_callback=None) -> Dict[str, Any]:
        self.progress_callback = progress_callback
        self._emit_progress(0.02, "Starting decoder-only evaluation...")
        self.load_model()
        self.load_dataset()
        self.prepare_model_for_inference()
        matmul_profile = {}
        if self.model is not None and self.profile_layer_modes:
            self.forward_matmul_profiler = ForwardMatmulProfiler(self.model, self.profile_layer_modes)
            self.forward_matmul_profiler.attach()

        def scaled_progress(progress: float, message: str) -> None:
            self._emit_progress(0.68 + 0.32 * progress, message)

        try:
            result = evaluate_decoder_only_model(
                model_name=self.model_spec.name,
                dataset_name=self.dataset_spec.name,
                dataset_config=self.dataset_spec.config,
                dataset_format=self.dataset_spec.format,
                split=self.dataset_spec.split,
                max_samples=self.runtime_spec.max_samples,
                batch_size=self.runtime_spec.batch_size,
                perplexity_stride=self.runtime_spec.perplexity_stride,
                wikitext_token_limit=self.runtime_spec.wikitext_token_limit,
                max_input_length=self.runtime_spec.max_input_length,
                max_new_tokens=self.runtime_spec.max_new_tokens,
                device=str(self.device),
                trust_remote_code=self.runtime_spec.trust_remote_code,
                do_sample=self.runtime_spec.do_sample,
                temperature=self.runtime_spec.temperature,
                top_k=self.runtime_spec.top_k,
                bertscore_model=self.runtime_spec.bertscore_model,
                progress_callback=scaled_progress,
                generation_tokenizer=self.tokenizer,
                generation_model=self.model,
                trace_enabled=self.trace_spec.enabled,
            )
        finally:
            if self.forward_matmul_profiler is not None:
                matmul_profile = self.forward_matmul_profiler.summary()
                self.forward_matmul_profiler.detach()
            self.forward_matmul_profiler = None
        _print_matmul_profile_mac_contributions(matmul_profile)
        result["matmul_profile"] = matmul_profile
        return result

    def evaluate_qualitative(
        self,
        prompt: str,
        ground_truth: str,
        top_token_count: int = 20,
        progress_callback=None,
    ) -> Dict[str, Any]:
        self.progress_callback = progress_callback
        self._emit_progress(0.02, "Starting qualitative evaluation...")
        self.load_model()
        self.load_dataset()
        self.prepare_model_for_inference()
        self._emit_progress(0.72, "Scoring qualitative prompt...")
        result = evaluate_qualitative_prompt(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            ground_truth=ground_truth,
            device=self.device,
            max_input_length=self.runtime_spec.max_input_length,
            max_new_tokens=self.runtime_spec.max_new_tokens,
            do_sample=self.runtime_spec.do_sample,
            temperature=self.runtime_spec.temperature,
            top_k=self.runtime_spec.top_k,
            top_token_count=top_token_count,
        )
        self._emit_progress(1.0, "Qualitative evaluation finished.")
        result["matmul_profile"] = {}
        return result


def run_baseline_experiment(
    model_name: str,
    dataset_name: str,
    dataset_revision: str,
    dataset_data_dir: str,
    text_col: str,
    label_col: str,
    batch_size: int = 256,
    max_length: int = 128,
    split_name: str = "test",
    trace_enabled: bool = False,
    attention_mode: str = "cls_row",
    profile_layer_modes: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    evaluator = BaselineIntentEvaluator(
        model_spec=ModelSpec(name=model_name),
        dataset_spec=DatasetSpec(
            name=dataset_name,
            revision=dataset_revision,
            data_dir=dataset_data_dir,
            text_col=text_col,
            label_col=label_col,
        ),
        runtime_spec=RuntimeSpec(batch_size=batch_size, max_length=max_length),
        profile_layer_modes=profile_layer_modes,
        trace_spec=TraceSpec(enabled=trace_enabled, attention_mode=attention_mode),
    )
    return evaluator.run_baseline(split_name=split_name, progress_callback=progress_callback)


def run_quantized_per_layer_experiment(
    model_name: str,
    dataset_name: str,
    dataset_revision: str,
    dataset_data_dir: str,
    text_col: str,
    label_col: str,
    layer_specs: Dict[str, LayerQuantSpec],
    batch_size: int = 256,
    max_length: int = 128,
    split_name: str = "test",
    calibration_percentile: float = 99.9,
    calibration_batches: int = 50,
    trace_enabled: bool = False,
    attention_mode: str = "cls_row",
    backend_quantize: bool = True,
    profile_layer_modes: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    evaluator = BaselineIntentEvaluator(
        model_spec=ModelSpec(name=model_name),
        dataset_spec=DatasetSpec(
            name=dataset_name,
            revision=dataset_revision,
            data_dir=dataset_data_dir,
            text_col=text_col,
            label_col=label_col,
        ),
        runtime_spec=RuntimeSpec(
            batch_size=batch_size,
            max_length=max_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
        layer_specs=layer_specs,
        profile_layer_modes=profile_layer_modes,
        calibration_percentile=calibration_percentile,
        calibration_batches=calibration_batches,
        trace_spec=TraceSpec(enabled=trace_enabled, attention_mode=attention_mode),
        backend_quantize=backend_quantize,
    )
    return evaluator.run_baseline(split_name=split_name, progress_callback=progress_callback)


def run_decoder_only_experiment(
    model_name: str,
    dataset_name: str,
    split_name: str,
    dataset_config: Optional[str] = None,
    dataset_format: str = "auto",
    layer_specs: Optional[Dict[str, LayerQuantSpec]] = None,
    batch_size: int = 8,
    perplexity_stride: Optional[int] = None,
    wikitext_token_limit: Optional[int] = None,
    max_input_length: int = 1024,
    max_new_tokens: int = 128,
    max_samples: int = 100,
    calibration_percentile: float = 99.9,
    calibration_batches: int = 16,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_k: int = 40,
    bertscore_model: str = "bert-base-uncased",
    backend_quantize: bool = True,
    trace_enabled: bool = False,
    profile_layer_modes: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    evaluator = DecoderOnlyEvaluator(
        model_spec=ModelSpec(name=model_name),
        dataset_spec=DecoderOnlyDatasetSpec(
            name=dataset_name,
            split=split_name,
            config=dataset_config,
            format=dataset_format,
        ),
        runtime_spec=DecoderOnlyRuntimeSpec(
            batch_size=batch_size,
            perplexity_stride=perplexity_stride,
            wikitext_token_limit=wikitext_token_limit,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            max_samples=max_samples,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            trust_remote_code=trust_remote_code,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            bertscore_model=bertscore_model,
        ),
        layer_specs=layer_specs,
        profile_layer_modes=profile_layer_modes,
        calibration_percentile=calibration_percentile,
        calibration_batches=calibration_batches,
        trace_spec=TraceSpec(enabled=trace_enabled),
        backend_quantize=backend_quantize,
    )
    return evaluator.evaluate(progress_callback=progress_callback)


def run_qualitative_decoder_evaluation(
    model_name: str,
    prompt: str,
    ground_truth: str,
    dataset_name: str = "wikitext",
    split_name: str = "test",
    dataset_config: Optional[str] = "wikitext-2-raw-v1",
    dataset_format: str = "wikitext",
    layer_specs: Optional[Dict[str, LayerQuantSpec]] = None,
    batch_size: int = 1,
    max_input_length: int = 1024,
    max_new_tokens: int = 64,
    calibration_percentile: float = 99.9,
    calibration_batches: int = 16,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_k: int = 40,
    backend_quantize: bool = True,
    top_token_count: int = 20,
    progress_callback=None,
) -> Dict[str, Any]:
    evaluator = DecoderOnlyEvaluator(
        model_spec=ModelSpec(name=model_name),
        dataset_spec=DecoderOnlyDatasetSpec(
            name=dataset_name,
            split=split_name,
            config=dataset_config,
            format=dataset_format,
        ),
        runtime_spec=DecoderOnlyRuntimeSpec(
            batch_size=batch_size,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            max_samples=max(1, batch_size * calibration_batches),
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            trust_remote_code=trust_remote_code,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        ),
        layer_specs=layer_specs,
        calibration_percentile=calibration_percentile,
        calibration_batches=calibration_batches,
        trace_spec=TraceSpec(enabled=False),
        backend_quantize=backend_quantize,
    )
    return evaluator.evaluate_qualitative(
        prompt=prompt,
        ground_truth=ground_truth,
        top_token_count=top_token_count,
        progress_callback=progress_callback,
    )
