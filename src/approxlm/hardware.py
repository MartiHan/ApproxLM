from __future__ import annotations

from pathlib import Path

from approxlm.adapters.hardware.characterization import (
    YosysOpenSTACharacterizationAdapter,
)
from approxlm.application.characterize_hardware import (
    characterize_hardware,
)
from approxlm.domain.hardware import (
    ElectricalConstraints,
    HardwareReport,
    HardwareSynthesisConfig,
    PowerActivityConfig,
)


def characterize_verilog(
    *,
    verilog: str | Path,
    top: str,
    liberty: str | Path,
    extra_sources: list[str | Path] | None = None,
    output_dir: str | Path = "hardware/runs",
    yosys_binary: str = "yosys",
    opensta_binary: str = "sta",
    driving_cell: str | None = None,
    output_load_pf: float | None = None,
    clock_period_ns: float = 10.0,
    activity: float = 0.1,
    duty: float = 0.5,
    vcd_path: str | Path | None = None,
    vcd_scope: str | None = None,
    use_system_verilog: bool = False,
) -> HardwareReport:
    """Characterize one Verilog design with Yosys and OpenSTA."""

    source_paths = [
        *(Path(path) for path in (extra_sources or [])),
        Path(verilog),
    ]

    if vcd_path is None:
        power_activity = PowerActivityConfig(
            mode="global",
            global_activity=activity,
            global_duty=duty,
        )
    else:
        power_activity = PowerActivityConfig(
            mode="vcd",
            vcd_path=Path(vcd_path),
            vcd_scope=vcd_scope,
        )

    config = HardwareSynthesisConfig(
        top=top,
        sources=tuple(source_paths),
        liberty=Path(liberty),
        output_dir=Path(output_dir) / top,
        yosys_binary=yosys_binary,
        opensta_binary=opensta_binary,
        use_system_verilog=use_system_verilog,
        constraints=ElectricalConstraints(
            virtual_clock_period_ns=clock_period_ns,
            driving_cell=driving_cell,
            output_load_pf=output_load_pf,
        ),
        power_activity=power_activity,
    )

    adapter = YosysOpenSTACharacterizationAdapter()

    return characterize_hardware(
        config,
        adapter=adapter,
    )