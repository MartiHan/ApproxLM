from __future__ import annotations
from typing import Any, Protocol
from approxlm.domain.config import ModelConfig
class ModelLoaderPort(Protocol):
    def load(self, config: ModelConfig) -> Any: ...
