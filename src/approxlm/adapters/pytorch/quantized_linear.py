from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from approxlm.adapters.pytorch.matmul import (
    approx_int8_matmul_torch_lut,
    approx_int8_matmul_triton_lut,
    get_cached_mul8_lut_flat,
    int32_accum_matmul,
    is_exact_mul8_lut,
    triton,
)
from approxlm.quantization.calibration.base import CalibStats
from approxlm.domain.quantization import get_quantization_format


def dequantize_linear_output(
    y: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_channel: bool,
) -> torch.Tensor:
    if act_scale.ndim != 0:
        raise RuntimeError("This backend supports only per-tensor activation dequantization.")
    if weight_per_channel:
        if weight_scale.ndim != 1:
            raise RuntimeError("Per-channel weight scale must be 1D [out_features].")
        return y.to(torch.float32) * (act_scale * weight_scale.unsqueeze(0))
    if weight_scale.ndim != 0:
        raise RuntimeError("Per-tensor weight scale must be scalar.")
    return y.to(torch.float32) * (act_scale * weight_scale)


def _as_float_scale_tensor(scale: Any) -> torch.Tensor:
    if torch.is_tensor(scale):
        return scale.detach().clone().to(torch.float32)
    return torch.tensor(scale, dtype=torch.float32)


def _activation_scale_for_input(act_scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if act_scale.ndim == 0:
        return act_scale
    if act_scale.ndim != 1:
        raise RuntimeError("Per-channel activation scale must be 1D [in_features].")
    if act_scale.numel() != x.shape[-1]:
        raise RuntimeError(
            f"Per-channel activation scale has {act_scale.numel()} values, "
            f"but input last dimension is {x.shape[-1]}."
        )
    return act_scale.view(*([1] * (x.ndim - 1)), -1)


def _apply_weight_scale(y: torch.Tensor, weight_scale: torch.Tensor, weight_per_channel: bool) -> torch.Tensor:
    if weight_per_channel:
        if weight_scale.ndim != 1:
            raise RuntimeError("Per-channel weight scale must be 1D [out_features].")
        return y * weight_scale.unsqueeze(0)
    if weight_scale.ndim != 0:
        raise RuntimeError("Per-tensor weight scale must be scalar.")
    return y * weight_scale


def _linear_output_with_per_channel_activation_scale(
    x_q: torch.Tensor,
    weight_q: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_channel: bool,
    approx_lut_path: Optional[str],
) -> torch.Tensor:
    if act_scale.ndim != 1:
        raise RuntimeError("Per-channel activation scale must be 1D [in_features].")
    if act_scale.numel() != x_q.shape[1]:
        raise RuntimeError(
            f"Per-channel activation scale has {act_scale.numel()} values, "
            f"but quantized input has {x_q.shape[1]} features."
        )

    act_scale = act_scale.to(device=x_q.device, dtype=torch.float32).contiguous()
    if approx_lut_path:
        lut = get_cached_mul8_lut_flat(approx_lut_path, x_q.device)
        y = _approx_int8_matmul_torch_lut_per_channel_scaled(x_q, weight_q, lut, act_scale)
    else:
        y = torch.matmul(x_q.to(torch.float32) * act_scale.unsqueeze(0), weight_q.to(torch.float32).t())

    return _apply_weight_scale(y, weight_scale.to(device=y.device, dtype=y.dtype), weight_per_channel)


def _approx_int8_matmul_torch_lut_per_channel_scaled(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    lut_flat: torch.Tensor,
    act_scale: torch.Tensor,
    chunk_k: int = 32,
) -> torch.Tensor:
    m, k = a_int8.shape
    n = b_int8.shape[0]
    out = torch.zeros((m, n), device=a_int8.device, dtype=torch.float32)

    a_codes = a_int8.to(torch.int16) + 128
    b_codes = b_int8.to(torch.int16) + 128
    lut_flat = lut_flat.to(device=a_int8.device)

    for start in range(0, k, chunk_k):
        stop = min(start + chunk_k, k)
        a_chunk = a_codes[:, start:stop].to(torch.int32)[:, None, :]
        b_chunk = b_codes[:, start:stop].to(torch.int32)[None, :, :]
        lut_idx = ((a_chunk << 8) | b_chunk).reshape(-1)
        products = lut_flat[lut_idx].reshape(m, n, stop - start).to(torch.float32)
        out += (products * act_scale[start:stop].view(1, 1, -1)).sum(dim=-1)

    return out


def weight_stats_from_linear(
    linear: nn.Linear,
    per_channel: bool = True,
    symmetric: bool = True,
    data_type: str = "int8",
) -> CalibStats:
    weight = linear.weight.detach().to(torch.float32)
    if per_channel:
        if symmetric:
            amax = weight.abs().amax(dim=1).clamp(min=1e-8)
            amin = -amax
        else:
            amax = weight.amax(dim=1)
            amin = weight.amin(dim=1)
    else:
        if symmetric:
            amax = weight.abs().amax().clamp(min=1e-8)
            amin = -amax
        else:
            amax = weight.amax()
            amin = weight.amin()

    fmt = get_quantization_format(data_type)
    qmin, qmax = fmt.qmin, fmt.qmax
    return CalibStats(amax=amax, amin=amin, qmin=qmin, qmax=qmax)


class QuantizedLinearBackend(nn.Module):
    def __init__(
        self,
        float_linear: nn.Linear,
        act_quant_params: Any,
        weight_quant_params: Any,
        approx_lut_path: Optional[str] = None,
    ):
        super().__init__()
        self.approx_lut_path = approx_lut_path
        self.act_format = get_quantization_format(act_quant_params.data_type)
        self.weight_format = get_quantization_format(weight_quant_params.data_type)
        if self.approx_lut_path and (self.act_format.name != "int8" or self.weight_format.name != "int8"):
            raise ValueError("The LUT backend currently supports signed int8 operands only. Exact quantized backends are extensible to other registered formats.")
        self.in_features = float_linear.in_features
        self.out_features = float_linear.out_features
        self.act_per_channel = bool(act_quant_params.per_channel)
        self.weight_per_channel = bool(weight_quant_params.per_channel)

        weight_fp = float_linear.weight.detach().to(torch.float32)
        weight_scale = _as_float_scale_tensor(weight_quant_params.scale)
        if self.weight_per_channel:
            weight_scale = weight_scale.view(-1).to(torch.float32)
            if weight_scale.numel() != self.out_features:
                raise RuntimeError(
                    f"Per-channel weight scale must have {self.out_features} values, "
                    f"got {weight_scale.numel()}."
                )
            weight_q = torch.round(weight_fp / weight_scale.view(-1, 1)).clamp(self.weight_format.qmin, self.weight_format.qmax).to(self.weight_format.storage_dtype)
            self.register_buffer("w_scale", weight_scale)
        else:
            if weight_scale.ndim != 0:
                if weight_scale.numel() != 1:
                    raise RuntimeError("Per-tensor weight scale must be scalar.")
                weight_scale = weight_scale.reshape(())
            weight_q = torch.round(weight_fp / weight_scale).clamp(self.weight_format.qmin, self.weight_format.qmax).to(self.weight_format.storage_dtype)
            self.register_buffer("w_scale", weight_scale)

        self.register_buffer("w_q", weight_q)
        act_scale = _as_float_scale_tensor(act_quant_params.scale)
        if self.act_per_channel:
            act_scale = act_scale.view(-1)
            if act_scale.numel() != self.in_features:
                raise RuntimeError(
                    f"Per-channel activation scale must have {self.in_features} values, "
                    f"got {act_scale.numel()}."
                )
        elif act_scale.ndim != 0:
            if act_scale.numel() != 1:
                raise RuntimeError("Per-tensor activation scale must be scalar.")
            act_scale = act_scale.reshape(())
        self.register_buffer("act_scale", act_scale)
        if float_linear.bias is not None:
            self.register_buffer("bias", float_linear.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = torch.round(x / _activation_scale_for_input(self.act_scale, x)).clamp(-128, 127).to(torch.int8)
        x_2d = x_q.view(-1, x_q.shape[-1]).contiguous()
        w_2d = self.w_q.contiguous()

        if self.act_per_channel:
            y_fp = _linear_output_with_per_channel_activation_scale(
                x_q=x_2d,
                weight_q=w_2d,
                act_scale=self.act_scale,
                weight_scale=self.w_scale,
                weight_per_channel=self.weight_per_channel,
                approx_lut_path=self.approx_lut_path,
            )
        else:
            if self.approx_lut_path:
                lut = get_cached_mul8_lut_flat(self.approx_lut_path, x.device)
                if x.is_cuda and triton is not None:
                    try:
                        y_int = approx_int8_matmul_triton_lut(x_2d, w_2d, lut)
                    except (AttributeError, RuntimeError):
                        y_int = approx_int8_matmul_torch_lut(x_2d, w_2d, lut)
                else:
                    y_int = approx_int8_matmul_torch_lut(x_2d, w_2d, lut)
            else:
                if x_2d.dtype == torch.int8 and w_2d.dtype == torch.int8:
                    y_int = self._safe_int_mm(x_2d, w_2d.transpose(0, 1))
                else:
                    y_int = torch.matmul(x_2d.to(torch.int32), w_2d.to(torch.int32).transpose(0, 1))

            y_fp = dequantize_linear_output(y_int, self.act_scale, self.w_scale, self.weight_per_channel)
        y_fp = y_fp.view(*x_q.shape[:-1], self.out_features)
        if self.bias is not None:
            y_fp = y_fp + self.bias
        return y_fp.to(x.dtype)

    @staticmethod
    def _safe_int_mm(a_int8: torch.Tensor, b_int8_t: torch.Tensor) -> torch.Tensor:
        m, k = a_int8.shape
        n = b_int8_t.shape[1]

        # torch._int_mm has backend-dependent shape constraints and can fail on
        # small decoder-only generation batches such as M=1 during autoregressive decoding.
        if m <= 16 or n <= 16 or k <= 16:
            return int32_accum_matmul(a_int8, b_int8_t)

        try:
            return torch._int_mm(a_int8, b_int8_t)
        except RuntimeError:
            return int32_accum_matmul(a_int8, b_int8_t)
