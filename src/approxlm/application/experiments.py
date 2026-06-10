from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
from approxlm.domain.config import ExperimentConfig
from approxlm.domain.specs import LayerQuantSpec
from approxlm.adapters.huggingface.evaluation import run_baseline_experiment, run_quantized_per_layer_experiment, run_decoder_only_experiment, run_qualitative_decoder_evaluation

FP32_MODES={None,'None','fp32'}
def is_fp32_baseline(layer_modes: Dict[str,str]) -> bool: return all(v in FP32_MODES for v in layer_modes.values())

def build_layer_specs(config: ExperimentConfig) -> Dict[str,LayerQuantSpec]:
    specs={}
    q=config.quantization
    for name,mode in config.layer_modes.items():
        if mode in FP32_MODES: continue
        lut=None if mode in {'int8_exact','exact'} else config.resolve_lut(mode)
        specs[name]=LayerQuantSpec(
            act_data_type=q.activation.format, weight_data_type=q.weight.format,
            act_symmetric=q.activation.symmetric, weight_symmetric=q.weight.symmetric,
            act_per_channel=q.activation.per_channel, weight_per_channel=q.weight.per_channel,
            approx_lut_path=lut,
        )
    return specs

def run_experiment(config: ExperimentConfig, progress_callback=None) -> Dict[str,Any]:
    specs=None if is_fp32_baseline(config.layer_modes) else build_layer_specs(config)
    if config.model.task_type=='classification':
        common=dict(model_name=config.model.hf_id,dataset_name=config.dataset.name,
            dataset_revision=config.dataset.revision or 'refs/convert/parquet',dataset_data_dir=config.dataset.data_dir or 'en-US',
            text_col=config.dataset.text_col,label_col=config.dataset.label_col,max_length=config.runtime.max_length,
            split_name=config.dataset.split,trace_enabled=config.trace.enabled,attention_mode=config.trace.attention_mode,
            progress_callback=progress_callback,profile_layer_modes=config.layer_modes)
        if specs is None: return run_baseline_experiment(batch_size=config.runtime.batch_size,**common)
        return run_quantized_per_layer_experiment(layer_specs=specs,batch_size=config.runtime.batch_size,
            calibration_percentile=config.quantization.calibration.percentile,calibration_batches=config.quantization.calibration.batches,
            backend_quantize=config.runtime.backend_quantize,**common)
    return run_decoder_only_experiment(model_name=config.model.hf_id,dataset_name=config.dataset.name,
        split_name=config.dataset.split,dataset_config=config.dataset.config,dataset_format=config.dataset.format,
        layer_specs=specs,batch_size=config.runtime.batch_size,perplexity_stride=config.runtime.perplexity_stride,
        wikitext_token_limit=config.runtime.wikitext_token_limit,max_input_length=config.runtime.max_input_length,
        max_new_tokens=config.runtime.max_new_tokens,max_samples=config.runtime.max_samples,
        calibration_percentile=config.quantization.calibration.percentile,calibration_batches=config.quantization.calibration.batches,
        trust_remote_code=config.model.trust_remote_code,do_sample=config.runtime.do_sample,temperature=config.runtime.temperature,
        top_k=config.runtime.top_k,bertscore_model=config.runtime.bertscore_model,backend_quantize=config.runtime.backend_quantize,
        trace_enabled=config.trace.enabled,profile_layer_modes=config.layer_modes,progress_callback=progress_callback)
