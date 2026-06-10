from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


_APPROX_LUT_CACHE: dict[tuple[str, str], torch.Tensor] = {}
_EXACT_LUT_CACHE: dict[str, bool] = {}


def is_exact_mul8_lut(npy_path: str | Path) -> bool:
    path = str(npy_path)
    if path not in _EXACT_LUT_CACHE:
        lut = np.load(path)
        if lut.shape != (256, 256):
            _EXACT_LUT_CACHE[path] = False
        else:
            values = np.arange(256, dtype=np.int16) - 128
            expected = (values[:, None].astype(np.int32) * values[None, :].astype(np.int32)).astype(np.int16)
            _EXACT_LUT_CACHE[path] = bool(np.array_equal(lut, expected))
    return _EXACT_LUT_CACHE[path]


def load_mul8_lut_flat(npy_path: str | Path, device: torch.device | str = "cuda") -> torch.Tensor:
    lut = np.load(str(npy_path))
    if lut.shape != (256, 256):
        raise RuntimeError(f"Expected LUT shape (256, 256), got {lut.shape}")
    return torch.from_numpy(lut.astype(np.int16, copy=False)).reshape(-1).to(device=device)


def get_cached_mul8_lut_flat(npy_path: str | Path, device: torch.device) -> torch.Tensor:
    key = (str(device), str(npy_path))
    if key not in _APPROX_LUT_CACHE:
        _APPROX_LUT_CACHE[key] = load_mul8_lut_flat(npy_path=npy_path, device=device)
    return _APPROX_LUT_CACHE[key]


if triton is not None:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _approx_int8_gemm_lut_kernel(
        a_ptr, b_t_ptr, lut_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            for kk in tl.static_range(0, BLOCK_K):
                k_idx = k0 + kk
                k_mask = k_idx < K
                a_mask = (offs_m < M) & k_mask
                b_mask = (offs_n < N) & k_mask
                a_k = tl.load(a_ptr + offs_m * stride_am + k_idx * stride_ak, mask=a_mask, other=0).to(tl.int16)
                b_k = tl.load(b_t_ptr + offs_n * stride_bn + k_idx * stride_bk, mask=b_mask, other=0).to(tl.int16)
                lut_idx = ((a_k + 128).to(tl.int32)[:, None] << 8) | (b_k + 128).to(tl.int32)[None, :]
                out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N) & k_mask
                acc += tl.load(lut_ptr + lut_idx, mask=out_mask, other=0).to(tl.int32)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def approx_int8_matmul_triton_lut(
    a_int8: torch.Tensor,
    b_int8_t: torch.Tensor,
    lut_flat: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    m, k = a_int8.shape
    n = b_int8_t.shape[0]
    c_int32 = torch.empty((m, n), device=a_int8.device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    _approx_int8_gemm_lut_kernel[grid](
        a_int8, b_int8_t, lut_flat, c_int32,
        m, n, k,
        a_int8.stride(0), a_int8.stride(1),
        b_int8_t.stride(0), b_int8_t.stride(1),
        c_int32.stride(0), c_int32.stride(1),
    )
    return c_int32


def approx_int8_matmul_torch_lut(
    a_int8: torch.Tensor,
    b_int8_t: torch.Tensor,
    lut_flat: torch.Tensor,
    chunk_k: int = 32,
) -> torch.Tensor:
    m, k = a_int8.shape
    n = b_int8_t.shape[0]
    out = torch.zeros((m, n), device=a_int8.device, dtype=torch.int32)

    a_codes = a_int8.to(torch.int16) + 128
    b_codes = b_int8_t.to(torch.int16) + 128

    for start in range(0, k, chunk_k):
        stop = min(start + chunk_k, k)
        a_chunk = a_codes[:, start:stop].to(torch.int32)[:, None, :]
        b_chunk = b_codes[:, start:stop].to(torch.int32)[None, :, :]
        lut_idx = ((a_chunk << 8) | b_chunk).reshape(-1)
        out += lut_flat[lut_idx].reshape(m, n, stop - start).to(torch.int32).sum(dim=-1)

    return out


def int32_accum_matmul(a_int8: torch.Tensor, b_int8_t: torch.Tensor, chunk_k: int = 256) -> torch.Tensor:
    a_i32 = a_int8.to(torch.int32)
    b_i32 = b_int8_t.to(torch.int32)
    try:
        return torch.matmul(a_i32, b_i32)
    except RuntimeError:
        out = torch.zeros(
            (a_int8.shape[0], b_int8_t.shape[1]),
            device=a_int8.device,
            dtype=torch.int32,
        )
        for start in range(0, a_int8.shape[1], chunk_k):
            stop = min(start + chunk_k, a_int8.shape[1])
            products = a_i32[:, start:stop, None] * b_i32[None, start:stop, :]
            out += products.sum(dim=1, dtype=torch.int32)
        return out


class MatmulReplacementLinear(nn.Module):
    def __init__(
        self,
        weight_int8: torch.Tensor,
        weight_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        approx_lut_path: Optional[str] = None,
    ):
        super().__init__()
        if weight_int8.ndim != 2:
            raise RuntimeError(f"Expected int8 weight shape [out_features, in_features], got {tuple(weight_int8.shape)}")
        self.approx_lut_path = approx_lut_path
        self.in_features = int(weight_int8.shape[1])
        self.out_features = int(weight_int8.shape[0])
        self.register_buffer("w_q", weight_int8.detach().to(torch.int8).contiguous())
        if weight_scale is not None:
            scale = weight_scale.detach().to(torch.float32)
            if scale.ndim == 0:
                pass
            elif scale.ndim == 1 and scale.numel() == self.out_features:
                scale = scale.contiguous()
            else:
                raise RuntimeError(
                    f"Expected scalar or per-output weight scale [{self.out_features}], got {tuple(scale.shape)}"
                )
            self.register_buffer("w_scale", scale)
        else:
            self.w_scale = None
        if bias is not None:
            self.register_buffer("bias", bias.detach().contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = torch.round(x).clamp(-128, 127).to(torch.int8)
        x_2d = x_q.view(-1, x_q.shape[-1]).contiguous()
        w_2d = self.w_q.contiguous()

        if self.approx_lut_path:
            if is_exact_mul8_lut(self.approx_lut_path):
                y_int = int32_accum_matmul(x_2d, w_2d.transpose(0, 1))
            else:
                lut = get_cached_mul8_lut_flat(self.approx_lut_path, x.device)
                if x.is_cuda and triton is not None:
                    try:
                        y_int = approx_int8_matmul_triton_lut(x_2d, w_2d, lut)
                    except (AttributeError, RuntimeError):
                        y_int = approx_int8_matmul_torch_lut(x_2d, w_2d, lut)
                else:
                    y_int = approx_int8_matmul_torch_lut(x_2d, w_2d, lut)
        else:
            y_int = int32_accum_matmul(x_2d, w_2d.transpose(0, 1))

        y = y_int.to(torch.float32).view(*x_q.shape[:-1], self.out_features)
        if self.w_scale is not None:
            y = y * self.w_scale.to(device=y.device, dtype=y.dtype)
        if self.bias is not None:
            y = y + self.bias.to(device=y.device, dtype=y.dtype)
        return y.to(x.dtype)


class FloatWeightMatmulReplacementLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        approx_lut_path: Optional[str] = None,
    ):
        super().__init__()
        if weight.ndim != 2:
            raise RuntimeError(f"Expected weight shape [out_features, in_features], got {tuple(weight.shape)}")
        self.approx_lut_path = approx_lut_path
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.register_buffer("weight", weight.detach().contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.detach().contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approx_lut_path is None:
            return F.linear(x, self.weight.to(dtype=x.dtype), self.bias.to(dtype=x.dtype) if self.bias is not None else None)

        x_q = torch.round(x).clamp(-128, 127).to(torch.int8)
        w_q = torch.round(self.weight).clamp(-128, 127).to(torch.int8)
        x_2d = x_q.view(-1, x_q.shape[-1]).contiguous()

        if is_exact_mul8_lut(self.approx_lut_path):
            y_int = int32_accum_matmul(x_2d, w_q.t().contiguous())
        else:
            lut = get_cached_mul8_lut_flat(self.approx_lut_path, x.device)
            if x.is_cuda and triton is not None:
                try:
                    y_int = approx_int8_matmul_triton_lut(x_2d, w_q.contiguous(), lut)
                except (AttributeError, RuntimeError):
                    y_int = approx_int8_matmul_torch_lut(x_2d, w_q.contiguous(), lut)
            else:
                y_int = approx_int8_matmul_torch_lut(x_2d, w_q.contiguous(), lut)

        y = y_int.to(torch.float32).view(*x_q.shape[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        return y.to(x.dtype)
