from __future__ import annotations

from pathlib import Path

from approxlm.domain.hardware import HardwareSynthesisConfig


def _tcl_quote(path: Path | str) -> str:
    value = str(path)
    return "{" + value.replace("}", "\\}") + "}"


def _yosys_quote(path: Path | str) -> str:
    value = str(path)
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_yosys_script(
    config: HardwareSynthesisConfig,
    *,
    mapped_netlist: Path,
    mapped_json: Path,
    stats_path: Path,
) -> str:
    read_option = "-sv " if config.use_system_verilog else ""

    source_lines = "\n".join(
        f"read_verilog {read_option}{_yosys_quote(source.resolve())}"
        for source in config.sources
    )

    flatten_option = " -flatten" if config.flatten else ""

    dfflibmap_command = ""
    if config.use_dfflibmap:
        dfflibmap_command = (
            "dfflibmap "
            f"-liberty {_yosys_quote(config.liberty.resolve())}\n"
        )

    top = config.top
    liberty = _yosys_quote(config.liberty.resolve())
    netlist = _yosys_quote(mapped_netlist.resolve())
    json_path = _yosys_quote(mapped_json.resolve())
    stats = _yosys_quote(stats_path.resolve())

    return f"""# Generated automatically by ApproxLM.

{source_lines}

hierarchy -check -top {top}
synth -top {top}{flatten_option} -noabc
techmap
{dfflibmap_command}\
abc -liberty {liberty}

tee -o {stats} log APPROXLM_YOSYS_STATS_BEGIN
tee -a {stats} stat -liberty {liberty}
tee -a {stats} log APPROXLM_YOSYS_STATS_END

write_verilog -noattr -noexpr -nodec {netlist}
write_json {json_path}
"""


def build_opensta_script(
    config: HardwareSynthesisConfig,
    *,
    mapped_netlist: Path,
) -> str:
    constraints = config.constraints
    activity = config.power_activity

    lines: list[str] = [
        "# Generated automatically by ApproxLM.",
        "",
        f"read_liberty {_tcl_quote(config.liberty.resolve())}",
        f"read_verilog {_tcl_quote(mapped_netlist.resolve())}",
        f"link_design {config.top}",
        "",
        (
            "create_clock -name approxlm_virtual_clock "
            f"-period {constraints.virtual_clock_period_ns}"
        ),
        (
            "set_input_delay "
            f"{constraints.input_delay_ns} "
            "-clock approxlm_virtual_clock [all_inputs]"
        ),
        (
            "set_output_delay "
            f"{constraints.output_delay_ns} "
            "-clock approxlm_virtual_clock [all_outputs]"
        ),
    ]

    if constraints.driving_cell:
        lines.extend(
            [
                "",
                (
                    "set_driving_cell "
                    f"-lib_cell {constraints.driving_cell} "
                    "[all_inputs]"
                ),
            ]
        )

    if constraints.output_load_pf is not None:
        lines.append(
            f"set_load {constraints.output_load_pf} [all_outputs]"
        )

    lines.extend(
        [
            "",
            "check_setup",
            "",
            'puts "AXLM_MAX_TIMING_BEGIN"',
            (
                "report_checks "
                "-from [all_inputs] "
                "-to [all_outputs] "
                "-path_delay max "
                "-group_count 1 "
                "-endpoint_count 1 "
                "-format full "
                "-digits 6"
            ),
            'puts "AXLM_MAX_TIMING_END"',
            "",
            'puts "AXLM_MIN_TIMING_BEGIN"',
            (
                "report_checks "
                "-from [all_inputs] "
                "-to [all_outputs] "
                "-path_delay min "
                "-group_count 1 "
                "-endpoint_count 1 "
                "-format full "
                "-digits 6"
            ),
            'puts "AXLM_MIN_TIMING_END"',
            "",
            'puts "AXLM_SLACK_BEGIN"',
            "report_worst_slack -max -digits 6",
            'puts "AXLM_SLACK_END"',
            "",
        ]
    )

    if activity.mode == "global":
        lines.append(
            "set_power_activity "
            f"-global -activity {activity.global_activity} "
            f"-duty {activity.global_duty}"
        )
    elif activity.mode == "vcd":
        scope_option = (
            f"-scope {activity.vcd_scope} "
            if activity.vcd_scope
            else ""
        )

        lines.append(
            f"read_vcd {scope_option}"
            f"{_tcl_quote(activity.vcd_path.resolve())}"
        )
    else:
        raise ValueError(
            f"Unsupported power activity mode: {activity.mode}"
        )

    lines.extend(
        [
            "",
            'puts "AXLM_POWER_BEGIN"',
            "report_power",
            'puts "AXLM_POWER_END"',
            "",
            "exit",
        ]
    )

    return "\n".join(lines) + "\n"
