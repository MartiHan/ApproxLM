from __future__ import annotations

from approxlm.domain.hardware import (
    HardwareReport,
    HardwareSynthesisConfig,
)
from approxlm.ports.hardware import HardwareCharacterizationPort


def characterize_hardware(
    config: HardwareSynthesisConfig,
    *,
    adapter: HardwareCharacterizationPort,
) -> HardwareReport:
    """Execute one hardware-characterization use case."""

    return adapter.characterize(config)