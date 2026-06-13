from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


PowerActivityMode = Literal["global", "vcd"]


@dataclass(frozen=True)
class ElectricalConstraints:
    """Electrical and timing assumptions used by OpenSTA."""

    virtual_clock_period_ns: float = 10.0
    input_delay_ns: float = 0.0
    output_delay_ns: float = 0.0

    driving_cell: str | None = None
    output_load_pf: float | None = None


@dataclass(frozen=True)
class PowerActivityConfig:
    """Power activity source for OpenSTA."""

    mode: PowerActivityMode = "global"

    global_activity: float = 0.1
    global_duty: float = 0.5

    vcd_path: Path | None = None
    vcd_scope: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "vcd" and self.vcd_path is None:
            raise ValueError(
                "Power activity mode 'vcd' requires vcd_path."
            )

        if not 0.0 <= self.global_activity <= 1.0:
            raise ValueError("global_activity must be between 0 and 1.")

        if not 0.0 <= self.global_duty <= 1.0:
            raise ValueError("global_duty must be between 0 and 1.")


@dataclass(frozen=True)
class HardwareSynthesisConfig:
    """Complete input configuration for one hardware run."""

    top: str
    sources: tuple[Path, ...]
    liberty: Path
    output_dir: Path

    yosys_binary: str = "yosys"
    opensta_binary: str = "sta"

    flatten: bool = True
    use_system_verilog: bool = False
    use_dfflibmap: bool = False

    constraints: ElectricalConstraints = field(
        default_factory=ElectricalConstraints
    )
    power_activity: PowerActivityConfig = field(
        default_factory=PowerActivityConfig
    )

    keep_intermediate_files: bool = True

    def __post_init__(self) -> None:
        if not self.top.strip():
            raise ValueError("top must not be empty.")

        if not self.sources:
            raise ValueError("At least one Verilog source is required.")


@dataclass(frozen=True)
class HardwareReport:
    """Parsed hardware characterization results."""

    top: str

    area_um2: float | None
    cell_count: int | None

    max_delay_ns: float | None
    min_delay_ns: float | None
    worst_slack_ns: float | None

    total_power_uw: float | None
    internal_power_uw: float | None
    switching_power_uw: float | None
    leakage_power_uw: float | None

    output_dir: Path
    mapped_netlist: Path
    mapped_json: Path

    synthesis_script: Path
    synthesis_log: Path
    synthesis_stats: Path

    timing_script: Path
    timing_log: Path

    yosys_version: str | None
    opensta_version: str | None

    assumptions: dict[str, object]