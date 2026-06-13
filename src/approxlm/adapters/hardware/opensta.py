from __future__ import annotations

import subprocess
from pathlib import Path

from approxlm.adapters.hardware.errors import (
    HardwareToolExecutionError,
)


def run_opensta(
    script_path: Path,
    *,
    executable: str,
    log_path: Path,
) -> None:
    result = subprocess.run(
        [executable, str(script_path)],
        cwd=str(script_path.parent),
        text=True,
        capture_output=True,
        check=False,
    )

    combined_output = "\n".join(
        part
        for part in (result.stdout, result.stderr)
        if part
    )

    log_path.write_text(
        combined_output,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise HardwareToolExecutionError(
            "OpenSTA failed with exit code "
            f"{result.returncode}. See: {log_path}"
        )