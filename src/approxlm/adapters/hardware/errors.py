class HardwareCharacterizationError(RuntimeError):
    """Base error for the optional hardware flow."""


class HardwareToolNotFoundError(HardwareCharacterizationError):
    """Raised when Yosys or OpenSTA is missing."""


class HardwareToolExecutionError(HardwareCharacterizationError):
    """Raised when an external tool returns a nonzero exit code."""


class HardwareReportParseError(HardwareCharacterizationError):
    """Raised when a generated report cannot be parsed."""