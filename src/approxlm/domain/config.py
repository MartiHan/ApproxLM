from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from approxlm.application.luts import resolve_lut_path

TaskType = Literal['classification','decoder_only']

@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    attention_mode: str = 'cls_row'
    store_input_ids: bool = True
    store_attention_mask: bool = True
    store_text: bool = False
    pad_to_max_length: bool = True

@dataclass(frozen=True)
class QuantizerConfig:
    format: str = 'int8'
    symmetric: bool = True
    per_channel: bool = False

@dataclass(frozen=True)
class CalibrationConfig:
    method: str = 'histogram'
    percentile: float = 99.9
    batches: int = 50

@dataclass(frozen=True)
class QuantizationConfig:
    activation: QuantizerConfig = field(default_factory=QuantizerConfig)
    weight: QuantizerConfig = field(default_factory=lambda: QuantizerConfig(per_channel=True))
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

@dataclass(frozen=True)
class ModelConfig:
    hf_id: str
    task_type: TaskType
    trust_remote_code: bool = False

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str
    revision: Optional[str] = None
    data_dir: Optional[str] = None
    config: Optional[str] = None
    format: str = 'auto'
    text_col: str = 'utt'
    label_col: str = 'intent'
    instruction_col: str = 'instruction'
    input_col: str = 'input'
    reference_col: str = 'output'

@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int = 256
    max_length: int = 128
    max_samples: int = 100
    max_input_length: int = 1024
    max_new_tokens: int = 128
    perplexity_stride: Optional[int] = None
    wikitext_token_limit: Optional[int] = None
    device: Optional[str] = None
    backend_quantize: bool = True
    do_sample: bool = False
    temperature: float = 0.7
    top_k: int = 40
    bertscore_model: str = 'bert-base-uncased'

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    dataset: DatasetConfig
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    layer_modes: Dict[str, str] = field(default_factory=dict)
    lut_directory: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve_lut(self, mode: str) -> str:
        return resolve_lut_path(mode, lut_directory=self.lut_directory)
