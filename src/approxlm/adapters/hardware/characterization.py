from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from approxlm.adapters.hardware.opensta import run_opensta
from approxlm.adapters.hardware.parsers import (
    parse_opensta_power,
    parse_opensta_timing,
    parse_yosys_stats,
)
from approxlm.adapters.hardware.scripts import (
    build_opensta_script,
    build_yosys_script,
)
from approxlm.adapters.hardware.toolchain import (
    require_hardware_toolchain,
)
from approxlm.adapters.hardware.yosys import run_yosys
from approxlm.domain.hardware import (
    HardwareReport,
    HardwareSynthesisConfig,
)


def _validate_input_files(
    config: HardwareSynthesisConfig,
) -> None:
    missing_sources = [
        source
        for source in config.sources
        if not source.is_file()
    ]

    if missing_sources:
        formatted = "\n".join(
            f"  - {path}" for path in missing_sources
        )
        raise FileNotFoundError(
            "The following Verilog sources do not exist:\n"
            f"{formatted}"
        )

    if not config.liberty.is_file():
        raise FileNotFoundError(
            f"Liberty file does not exist: {config.liberty}"
        )

    if config.power_activity.mode == "vcd":
        vcd_path = config.power_activity.vcd_path

        if vcd_path is None or not vcd_path.is_file():
            raise FileNotFoundError(
                f"VCD file does not exist: {vcd_path}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


class YosysOpenSTACharacterizationAdapter:
    """Hardware port implemented with Yosys and OpenSTA."""

    def characterize(
        self,
        config: HardwareSynthesisConfig,
    ) -> HardwareReport:
        _validate_input_files(config)

        statuses = require_hardware_toolchain(
            yosys_binary=config.yosys_binary,
            opensta_binary=config.opensta_binary,
        )

        output_dir = config.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        synthesis_script = output_dir / "synthesis.ys"
        synthesis_log = output_dir / "synthesis.log"
        synthesis_stats = output_dir / "synthesis_stats.txt"

        mapped_netlist = output_dir / f"{config.top}_mapped.v"
        mapped_json = output_dir / f"{config.top}_mapped.json"

        timing_script = output_dir / "timing.tcl"
        timing_log = output_dir / "timing.log"

        synthesis_script.write_text(
            build_yosys_script(
                config,
                mapped_netlist=mapped_netlist,
                mapped_json=mapped_json,
                stats_path=synthesis_stats,
            ),
            encoding="utf-8",
        )

        run_yosys(
            synthesis_script,
            executable=config.yosys_binary,
            log_path=synthesis_log,
        )

        if not mapped_netlist.is_file():
            raise RuntimeError(
                "Yosys completed without producing the mapped netlist: "
                f"{mapped_netlist}"
            )

        timing_script.write_text(
            build_opensta_script(
                config,
                mapped_netlist=mapped_netlist,
            ),
            encoding="utf-8",
        )

        run_opensta(
            timing_script,
            executable=config.opensta_binary,
            log_path=timing_log,
        )

        yosys_text = synthesis_stats.read_text(
            encoding="utf-8",
            errors="replace",
        )

        timing_text = timing_log.read_text(
            encoding="utf-8",
            errors="replace",
        )

        synthesis_metrics = parse_yosys_stats(yosys_text)
        timing_metrics = parse_opensta_timing(timing_text)
        power_metrics = parse_opensta_power(timing_text)

        assumptions = {
            "liberty": str(config.liberty.resolve()),
            "liberty_sha256": _sha256(config.liberty),
            "sources": {
                str(source.resolve()): _sha256(source)
                for source in config.sources
            },
            "constraints": asdict(config.constraints),
            "power_activity": {
                **asdict(config.power_activity),
                "vcd_path": (
                    str(config.power_activity.vcd_path.resolve())
                    if config.power_activity.vcd_path
                    else None
                ),
            },
            "flatten": config.flatten,
            "use_system_verilog": config.use_system_verilog,
            "use_dfflibmap": config.use_dfflibmap,
        }

        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                assumptions,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return HardwareReport(
            top=config.top,
            area_um2=synthesis_metrics.area_um2,
            cell_count=synthesis_metrics.cell_count,
            max_delay_ns=timing_metrics.max_delay_ns,
            min_delay_ns=timing_metrics.min_delay_ns,
            worst_slack_ns=timing_metrics.worst_slack_ns,
            total_power_uw=power_metrics.total_power_uw,
            internal_power_uw=power_metrics.internal_power_uw,
            switching_power_uw=power_metrics.switching_power_uw,
            leakage_power_uw=power_metrics.leakage_power_uw,
            output_dir=output_dir,
            mapped_netlist=mapped_netlist,
            mapped_json=mapped_json,
            synthesis_script=synthesis_script,
            synthesis_log=synthesis_log,
            synthesis_stats=synthesis_stats,
            timing_script=timing_script,
            timing_log=timing_log,
            yosys_version=statuses["yosys"].version,
            opensta_version=statuses["opensta"].version,
            assumptions=assumptions,
        )