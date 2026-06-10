from __future__ import annotations
from typing import Any, Protocol
from approxlm.domain.config import DatasetConfig
class DatasetLoaderPort(Protocol):
    def load(self, config: DatasetConfig) -> Any: ...
