from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml
from approxlm.domain.config import *

def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    q=raw.get('quantization',{})
    def qz(section, default_per_channel=False):
        d=q.get(section,{})
        return QuantizerConfig(format=d.get('format','int8'),symmetric=d.get('symmetric',True),per_channel=d.get('per_channel',default_per_channel))
    cal=q.get('calibration',{})
    return ExperimentConfig(
        name=raw['name'], model=ModelConfig(**raw['model']), dataset=DatasetConfig(**raw['dataset']),
        quantization=QuantizationConfig(activation=qz('activation'),weight=qz('weight',True),calibration=CalibrationConfig(**cal)),
        runtime=RuntimeConfig(**raw.get('runtime',{})),trace=TraceConfig(**raw.get('trace',{})),
        layer_modes=raw.get('layer_modes',{}),lut_directory=raw.get('lut_directory'),metadata=raw.get('metadata',{}))
