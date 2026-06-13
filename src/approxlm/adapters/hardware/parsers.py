from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SynthesisMetrics:
    area_um2: float | None
    cell_count: int | None


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class TimingMetrics:
    max_delay_ns: float | None
    min_delay_ns: float | None
    max_required_time_ns: float | None
    worst_slack_ns: float | None


def _extract_section(
    text: str,
    begin_marker: str,
    end_marker: str,
) -> str:
    begin = text.find(begin_marker, 0)
    if begin == -1:
        return ""

    begin += len(begin_marker)
    end = text.find(end_marker, begin)

    if end == -1:
        return ""

    return text[begin:end]


def _find_last_float(
    text: str,
    patterns: tuple[str, ...],
) -> float | None:
    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if matches:
            value = matches[-1]

            if isinstance(value, tuple):
                value = value[0]

            return float(value)

    return None


def _parse_arrival_time(section: str) -> float | None:
    return _find_last_float(
        section,
        (
            rf"({_FLOAT})\s+data arrival time",
            rf"data arrival time\s*[:=]?\s*({_FLOAT})",
        ),
    )


def _parse_required_time(section: str) -> float | None:
    return _find_last_float(
        section,
        (
            rf"({_FLOAT})\s+data required time",
            rf"data required time\s*[:=]?\s*({_FLOAT})",
        ),
    )


def _parse_slack(section: str) -> float | None:
    return _find_last_float(
        section,
        (
            rf"({_FLOAT})\s+slack(?:\s+\([A-Z]+\))?",
            rf"worst slack\s*[:=]?\s*({_FLOAT})",
        ),
    )


def parse_opensta_timing(text: str) -> TimingMetrics:
    max_section = _extract_section(
        text,
        "AXLM_MAX_TIMING_BEGIN",
        "AXLM_MAX_TIMING_END",
    )

    min_section = _extract_section(
        text,
        "AXLM_MIN_TIMING_BEGIN",
        "AXLM_MIN_TIMING_END",
    )

    slack_section = _extract_section(
        text,
        "AXLM_SLACK_BEGIN",
        "AXLM_SLACK_END",
    )

    return TimingMetrics(
        max_delay_ns=_parse_arrival_time(max_section),
        min_delay_ns=_parse_arrival_time(min_section),
        max_required_time_ns=_parse_required_time(max_section),
        worst_slack_ns=(
            _parse_slack(slack_section)
            or _parse_slack(max_section)
        ),
    )


@dataclass(frozen=True)
class PowerMetrics:
    total_power_uw: float | None
    internal_power_uw: float | None
    switching_power_uw: float | None
    leakage_power_uw: float | None


def parse_yosys_stats(text: str) -> SynthesisMetrics:
    match = re.search(
        rf"Chip area for top module\b.*?:\s*({_FLOAT})",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError("No top-module area was found in the Yosys report.")

    area = float(match.group(1))

    cell_count_patterns = (
        r"Number of cells:\s*(\d+)",
        r"cells\s+(\d+)",
    )

    cell_count: int | None = None

    for pattern in cell_count_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            cell_count = int(matches[-1])
            break

    return SynthesisMetrics(
        area_um2=area,
        cell_count=cell_count,
    )


def _power_to_uw(value: float, unit: str) -> float:
    normalized = unit.lower()

    factors = {
        "w": 1_000_000.0,
        "mw": 1_000.0,
        "uw": 1.0,
        "µw": 1.0,
        "nw": 0.001,
        "pw": 0.000001,
    }

    try:
        return value * factors[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported power unit: {unit}") from exc


def parse_opensta_power(text: str) -> PowerMetrics:
    section = _extract_section(
        text,
        "AXLM_POWER_BEGIN",
        "AXLM_POWER_END",
    )

    # OpenSTA output varies by version. This parser first attempts
    # labeled totals, then falls back to a final Total row.
    labeled_patterns = {
        "internal": rf"Internal\s+Power\s*[:=]\s*({_FLOAT})\s*([munpµ]?W)",
        "switching": rf"Switching\s+Power\s*[:=]\s*({_FLOAT})\s*([munpµ]?W)",
        "leakage": rf"Leakage\s+Power\s*[:=]\s*({_FLOAT})\s*([munpµ]?W)",
        "total": rf"Total\s+Power\s*[:=]\s*({_FLOAT})\s*([munpµ]?W)",
    }

    values: dict[str, float | None] = {
        "internal": None,
        "switching": None,
        "leakage": None,
        "total": None,
    }

    for key, pattern in labeled_patterns.items():
        matches = re.findall(
            pattern,
            section,
            flags=re.IGNORECASE,
        )
        if matches:
            value, unit = matches[-1]
            values[key] = _power_to_uw(float(value), unit)

    # Common tabular form:
    # Total  1.23e-04  2.34e-04  5.67e-06  3.62e-04
    if values["total"] is None:
        total_row = re.findall(
            rf"^\s*Total\s+"
            rf"({_FLOAT})\s+"
            rf"({_FLOAT})\s+"
            rf"({_FLOAT})\s+"
            rf"({_FLOAT})",
            section,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if total_row:
            internal, switching, leakage, total = total_row[-1]

            # Most OpenSTA report_power tables use watts unless the
            # report header states otherwise.
            values["internal"] = float(internal) * 1_000_000.0
            values["switching"] = float(switching) * 1_000_000.0
            values["leakage"] = float(leakage) * 1_000_000.0
            values["total"] = float(total) * 1_000_000.0

    return PowerMetrics(
        total_power_uw=values["total"],
        internal_power_uw=values["internal"],
        switching_power_uw=values["switching"],
        leakage_power_uw=values["leakage"],
    )
