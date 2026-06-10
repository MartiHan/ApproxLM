from __future__ import annotations

from typing import Union

import torch

from approxlm.quantization.calibration.base import CalibStats
from approxlm.quantization.params import QuantParams
from approxlm.domain.quantization import get_quantization_format


class QuantParamsBuilder:
    """
    Build quantization parameters from calibration statistics.

    Supports:
      - symmetric / asymmetric
      - signed / unsigned
      - per-tensor / per-channel

    Notes:
      - For symmetric quantization:
            scale = amax / qmax
            zero_point = 0       for signed
            zero_point = midpoint for unsigned
      - For asymmetric quantization:
            scale = (amax - amin) / (qmax - qmin)
            zero_point = qmin - round(amin / scale)
    """

    def __init__(
        self,
        data_type: str,
        symmetric: bool,
        per_channel: bool,
        eps: float = 1e-8,
    ):
        self.data_type = data_type.lower()
        self.symmetric = bool(symmetric)
        self.per_channel = bool(per_channel)
        self.eps = float(eps)

        self.qmin, self.qmax = self._get_qrange(self.data_type)
        self.is_signed = self.qmin < 0

    def build(self, stats: CalibStats) -> QuantParams:
        amax = self._to_tensor(stats.amax)
        amin = self._to_tensor(stats.amin)

        if self.symmetric:
            return self._build_symmetric(amax=amax)
        return self._build_asymmetric(amin=amin, amax=amax)

    # =========================================================
    # Symmetric
    # =========================================================
    def _build_symmetric(self, amax: torch.Tensor) -> QuantParams:
        amax = torch.clamp(amax, min=self.eps)

        if self.is_signed:
            # signed symmetric, e.g. int8: [-127, 127]
            denom = float(self.qmax)
            scale = amax / denom
            zero_point = torch.zeros_like(scale, dtype=torch.int64)
        else:
            # unsigned symmetric is less common, but if used, center around midpoint
            # e.g. uint8: midpoint ~ 128
            midpoint = (self.qmin + self.qmax) // 2
            denom = float(self.qmax - midpoint)
            scale = amax / max(denom, 1.0)
            zero_point = torch.full_like(scale, midpoint, dtype=torch.int64)

        return QuantParams(
            scale=self._maybe_scalar(scale),
            zero_point=self._maybe_scalar(zero_point),
            qmin=self.qmin,
            qmax=self.qmax,
            symmetric=True,
            per_channel=self.per_channel,
            data_type=self.data_type,
        )

    # =========================================================
    # Asymmetric
    # =========================================================
    def _build_asymmetric(self, amin: torch.Tensor, amax: torch.Tensor) -> QuantParams:
        # ensure proper ordering
        amin = torch.minimum(amin, amax)
        amax = torch.maximum(amax, amin + self.eps)

        scale = (amax - amin) / float(self.qmax - self.qmin)
        scale = torch.clamp(scale, min=self.eps)

        zero_point = self.qmin - torch.round(amin / scale)
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax).to(torch.int64)

        return QuantParams(
            scale=self._maybe_scalar(scale),
            zero_point=self._maybe_scalar(zero_point),
            qmin=self.qmin,
            qmax=self.qmax,
            symmetric=False,
            per_channel=self.per_channel,
            data_type=self.data_type,
        )

    # =========================================================
    # Helpers
    # =========================================================
    @staticmethod
    def _to_tensor(x: Union[float, int, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.detach().clone().to(torch.float32)
        return torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def _maybe_scalar(x: torch.Tensor):
        if x.numel() == 1:
            val = x.item()
            if isinstance(val, float):
                return float(val)
            if isinstance(val, int):
                return int(val)
            return val
        return x

    @staticmethod
    def _get_qrange(data_type: str) -> tuple[int, int]:
        fmt = get_quantization_format(data_type)
        return fmt.qmin, fmt.qmax
