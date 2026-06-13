from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from approxlm.adapters.hardware.errors import (
    HardwareToolNotFoundError,
)


@dataclass(frozen=True)
class ToolStatus:
    name: str
    executable: str
    resolved_path: str | None
    available: bool
    version: str | None


def _run_version_command(
    executable: str,
    candidate_arguments: tuple[tuple[str, ...], ...],
) -> str | None:
    for arguments in candidate_arguments:
        try:
            result = subprocess.run(
                [executable, *arguments],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue

        output = "\n".join(
            part.strip()
            for part in (result.stdout, result.stderr)
            if part and part.strip()
        )

        if output:
            return output.splitlines()[0]

    return None


def get_yosys_status(executable: str = "yosys") -> ToolStatus:
    resolved = shutil.which(executable)

    if resolved is None:
        return ToolStatus(
            name="Yosys",
            executable=executable,
            resolved_path=None,
            available=False,
            version=None,
        )

    version = _run_version_command(
        resolved,
        (
            ("-V",),
            ("-version",),
        ),
    )

    return ToolStatus(
        name="Yosys",
        executable=executable,
        resolved_path=resolved,
        available=True,
        version=version,
    )


def get_opensta_status(executable: str = "sta") -> ToolStatus:
    resolved = shutil.which(executable)

    if resolved is None:
        return ToolStatus(
            name="OpenSTA",
            executable=executable,
            resolved_path=None,
            available=False,
            version=None,
        )

    version = _run_version_command(
        resolved,
        (
            ("-version",),
            ("-help",),
        ),
    )

    return ToolStatus(
        name="OpenSTA",
        executable=executable,
        resolved_path=resolved,
        available=True,
        version=version,
    )


def check_hardware_toolchain(
    *,
    yosys_binary: str = "yosys",
    opensta_binary: str = "sta",
) -> dict[str, ToolStatus]:
    return {
        "yosys": get_yosys_status(yosys_binary),
        "opensta": get_opensta_status(opensta_binary),
    }


def require_hardware_toolchain(
    *,
    yosys_binary: str = "yosys",
    opensta_binary: str = "sta",
) -> dict[str, ToolStatus]:
    statuses = check_hardware_toolchain(
        yosys_binary=yosys_binary,
        opensta_binary=opensta_binary,
    )

    missing = [
        status.name
        for status in statuses.values()
        if not status.available
    ]

    if missing:
        names = ", ".join(missing)
        raise HardwareToolNotFoundError(
            f"Missing hardware tools: {names}. "
            "Install Yosys and OpenSTA and ensure their executables "
            "are available on PATH. The remaining ApproxLM functionality "
            "does not require these tools."
        )

    return statuses
