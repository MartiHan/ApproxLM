from __future__ import annotations
from typing import Any, Dict, Protocol
from approxlm.domain.config import ExperimentConfig
class ExperimentEvaluatorPort(Protocol):
    def evaluate(self, config: ExperimentConfig, progress_callback=None) -> Dict[str, Any]: ...
