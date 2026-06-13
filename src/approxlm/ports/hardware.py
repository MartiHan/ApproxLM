from __future__ import annotations

from typing import Protocol

from approxlm.domain.hardware import (
    HardwareReport,
    HardwareSynthesisConfig,
)


class HardwareCharacterizationPort(Protocol):
    """Application-facing port for hardware characterization."""

    def characterize(
        self,
        config: HardwareSynthesisConfig,
    ) -> HardwareReport:
        ...