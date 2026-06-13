from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Iterable


LUT_RESOURCE_PACKAGE = "approxlm.resources.luts"
EXACT_LUT_MODES = {None, "None", "fp32", "int8_exact", "exact"}


def lut_filename(mode: str) -> str:
    return mode if mode.endswith(".npy") else f"{mode}.npy"


def _with_npy_suffix(path: Path) -> Path:
    return path if path.suffix else path.with_suffix(".npy")


def resolve_lut_path(
    mode: str | None,
    *,
    lut_directory: str | None = None,
    search_cwd: bool = True,
) -> str | None:
    if mode in EXACT_LUT_MODES:
        return None
    if mode is None:
        return None

    mode_path = Path(mode).expanduser()
    if mode_path.is_absolute() or mode_path.parent != Path("."):
        path = _with_npy_suffix(mode_path)
        return str(path if path.is_absolute() else Path.cwd().joinpath(path))

    filename = lut_filename(mode)
    candidates: list[Path] = []
    if lut_directory:
        candidates.append(Path(lut_directory).expanduser().joinpath(filename))
    if search_cwd:
        candidates.append(Path.cwd().joinpath(filename))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    resource = resources.files(LUT_RESOURCE_PACKAGE).joinpath(filename)
    if resource.is_file():
        return str(resource)

    checked = [str(candidate) for candidate in candidates]
    checked.append(f"{LUT_RESOURCE_PACKAGE}/{filename}")
    raise FileNotFoundError(f"Could not find LUT '{filename}'. Checked: {', '.join(checked)}")


def options_with_current(options: Iterable[str], current: str | None) -> list[str]:
    values = list(options)
    if current not in {None, ""} and current not in values:
        values.append(str(current))
    return values
