from __future__ import annotations

from dataclasses import dataclass
from typing import Optional



@dataclass(frozen=True)
class TraceSpec:
    enabled: bool = False
    attention_mode: str = "cls_row"
    store_input_ids: bool = True
    store_attention_mask: bool = True
    store_text: bool = False
    pad_to_max_length: bool = True


@dataclass(frozen=True)
class LayerQuantSpec:
    act_data_type: str = "int8"
    weight_data_type: str = "int8"
    act_symmetric: bool = True
    weight_symmetric: bool = True
    act_per_channel: bool = False
    weight_per_channel: bool = True
    approx_lut_path: Optional[str] = None

    @property
    def is_lut(self) -> bool:
        return self.approx_lut_path is not None


@dataclass(frozen=True)
class DatasetSpec:
    name: str = "AmazonScience/massive"
    revision: str = "refs/convert/parquet"
    data_dir: str = "en-US"
    text_col: str = "utt"
    label_col: str = "intent"
    test_split: str = "test"
    val_split: str = "validation"


@dataclass(frozen=True)
class ModelSpec:
    name: str = "qanastek/XLMRoberta-Alexa-Intents-Classification"


@dataclass(frozen=True)
class DecoderOnlyDatasetSpec:
    name: str = "tatsu-lab/alpaca"
    split: str = "train"
    config: Optional[str] = None
    format: str = "auto"
    instruction_col: str = "instruction"
    input_col: str = "input"
    reference_col: str = "output"


@dataclass(frozen=True)
class RuntimeSpec:
    batch_size: int = 256
    max_length: int = 128
    num_workers: int = 0
    pin_memory: bool = True
    device: Optional[str] = None


@dataclass(frozen=True)
class DecoderOnlyRuntimeSpec:
    batch_size: int = 8
    perplexity_stride: Optional[int] = None
    wikitext_token_limit: Optional[int] = None
    max_input_length: int = 1024
    max_new_tokens: int = 128
    max_samples: int = 100
    num_workers: int = 0
    pin_memory: bool = True
    device: Optional[str] = None
    trust_remote_code: bool = False
    do_sample: bool = False
    temperature: float = 0.7
    top_k: int = 40
    bertscore_model: str = "bert-base-uncased"


def normalize_layer_spec(spec: LayerQuantSpec) -> LayerQuantSpec:
    if spec.approx_lut_path is None:
        return spec
    return LayerQuantSpec(
        act_data_type="int8",
        weight_data_type="int8",
        act_symmetric=True,
        weight_symmetric=True,
        act_per_channel=spec.act_per_channel,
        weight_per_channel=spec.weight_per_channel,
        approx_lut_path=spec.approx_lut_path,
    )


