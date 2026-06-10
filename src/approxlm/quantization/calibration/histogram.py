from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from collections.abc import Mapping
from approxlm.quantization.calibration.base import BaseCalibrator, CalibStats
from approxlm.domain.quantization import get_quantization_format


class HistogramCalibrator(BaseCalibrator):
    """
    Histogram / percentile calibrator.

    Supports:
      - per-tensor (sometimes called per-layer in your UI)
      - per-channel

    Conventions:
      - For activations of Linear layers, the channel axis is usually the last dim.
      - If per_channel=False, one scalar range is produced per layer.
      - If per_channel=True, one range per channel is produced.

    Notes:
      - This implementation currently uses percentile over collected samples.
      - Histogram collection is kept available for inspection / future extensions.
    """

    def __init__(
        self,
        *args,
        percentile: float = 99.9,
        num_bins: int = 2048,
        channel_axis: int = -1,
        max_samples_per_batch: int = 20_000,
        max_samples_per_layer: int = 200_000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.percentile = float(percentile)
        self.num_bins = int(num_bins)
        self.channel_axis = int(channel_axis)
        self.max_samples_per_batch = int(max_samples_per_batch)
        self.max_samples_per_layer = int(max_samples_per_layer)

        # Per-layer storage.
        # If per_channel=False: list of 1D tensors [N]
        # If per_channel=True : list of 2D tensors [N, C]
        self._samples = defaultdict(list)

    # =========================================================
    # Hook factory
    # =========================================================
    def make_hook(self, layer_name: str) -> Callable:
        def hook(_module, inputs, _output):
            if not inputs:
                return

            x = inputs[0]
            if not torch.is_tensor(x):
                return

            x = x.detach()

            if self.per_channel:
                values = self._prepare_per_channel_samples(x)
            else:
                values = self._prepare_per_tensor_samples(x)

            self._samples[layer_name].append(values.cpu())

        return hook

    # =========================================================
    # Public API
    # =========================================================
    def compute_calib_stats(self, model: nn.Module) -> Dict[str, CalibStats]:
        model.eval()

        handles = []
        target_module_ids = {id(m) for m in self.layers_to_calibrate}

        for name, module in model.named_modules():
            if id(module) in target_module_ids:
                handles.append(module.register_forward_hook(self.make_hook(name)))

        device = next(model.parameters()).device
        forward_keys = {"input_ids", "attention_mask", "token_type_ids"}

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.calib_dataset):
                if batch_idx >= self.num_batches:
                    break

                if not isinstance(batch, Mapping):
                    raise TypeError(
                        f"Calibration loader must yield mapping-like batches, got {type(batch).__name__}."
                    )

                inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in forward_keys and torch.is_tensor(v)
                }

                if not inputs:
                    raise RuntimeError(
                        "Calibration batch did not contain any valid model inputs "
                        f"from keys {sorted(forward_keys)}. Got keys: {list(batch.keys())}"
                    )

                model(**inputs)

        for h in handles:
            h.remove()

        results: Dict[str, CalibStats] = {}

        for layer_name, chunks in self._samples.items():
            if not chunks:
                continue

            if self.per_channel:
                stats = self._compute_per_channel_stats(chunks)
            else:
                stats = self._compute_per_tensor_stats(chunks)

            qmin, qmax = self._get_qrange()

            results[layer_name] = CalibStats(
                amax=stats["amax"],
                amin=stats["amin"],
                qmin=qmin,
                qmax=qmax,
            )

        self.calib_stats = results
        return results

    # =========================================================
    # Histogram helper
    # =========================================================
    def collect_histogram(self, values: torch.Tensor) -> torch.Tensor:
        """
        Build a histogram over non-negative magnitudes.

        For per-tensor:
            values shape [N]
        For per-channel:
            values shape [N, C]
            -> returns [C, num_bins]
        """
        values = values.detach().cpu()

        if values.numel() == 0:
            if values.ndim == 2:
                return torch.zeros((values.shape[1], self.num_bins), dtype=torch.float32)
            return torch.zeros((self.num_bins,), dtype=torch.float32)

        if values.ndim == 1:
            vmax = float(values.max().item()) if values.numel() > 0 else 1.0
            vmax = max(vmax, 1e-8)
            return torch.histc(values, bins=self.num_bins, min=0, max=vmax)

        if values.ndim == 2:
            hists = []
            for c in range(values.shape[1]):
                vc = values[:, c]
                vmax = float(vc.max().item()) if vc.numel() > 0 else 1.0
                vmax = max(vmax, 1e-8)
                hists.append(torch.histc(vc, bins=self.num_bins, min=0, max=vmax))
            return torch.stack(hists, dim=0)

        raise ValueError(f"Unsupported values.ndim={values.ndim} for histogram collection.")

    # =========================================================
    # Internal helpers: sample preparation
    # =========================================================
    def _prepare_per_tensor_samples(self, x: torch.Tensor) -> torch.Tensor:
        values = x.detach().to(torch.float32).abs().reshape(-1)
        return self._deterministic_subsample_1d(values, self.max_samples_per_batch)

    def _prepare_per_channel_samples(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().to(torch.float32).abs()

        channel_axis = self.channel_axis
        if channel_axis < 0:
            channel_axis = x.ndim + channel_axis

        if not (0 <= channel_axis < x.ndim):
            raise ValueError(
                f"Invalid channel_axis={self.channel_axis} for tensor with ndim={x.ndim}"
            )

        # Move channel dim to the end, then flatten all others.
        if channel_axis != x.ndim - 1:
            perm = [d for d in range(x.ndim) if d != channel_axis] + [channel_axis]
            x = x.permute(*perm)

        x = x.reshape(-1, x.shape[-1])  # [N, C]
        return self._deterministic_subsample_2d_rows(x, self.max_samples_per_batch)

    # =========================================================
    # Internal helpers: stats computation
    # =========================================================
    def _compute_per_tensor_stats(self, chunks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        values = torch.cat(chunks, dim=0)  # [N]
        values = self._deterministic_subsample_1d(values, self.max_samples_per_layer)

        amax = torch.quantile(values, self.percentile / 100.0).clamp(min=1e-8)

        if self.symmetric:
            amin = -amax
        else:
            amin = values.min()

        return {"amax": amax, "amin": amin}

    def _compute_per_channel_stats(self, chunks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        values = torch.cat(chunks, dim=0)  # [N, C]
        values = self._deterministic_subsample_2d_rows(values, self.max_samples_per_layer)

        amax = torch.quantile(
            values,
            self.percentile / 100.0,
            dim=0,
        ).clamp(min=1e-8)  # [C]

        if self.symmetric:
            amin = -amax
        else:
            amin = values.min(dim=0).values  # [C]

        return {"amax": amax, "amin": amin}

    # =========================================================
    # Internal helpers: deterministic subsampling
    # =========================================================
    @staticmethod
    def _deterministic_subsample_1d(values: torch.Tensor, cap: int) -> torch.Tensor:
        if values.numel() <= cap:
            return values
        step = max(values.numel() // cap, 1)
        return values[::step][:cap]

    @staticmethod
    def _deterministic_subsample_2d_rows(values: torch.Tensor, cap: int) -> torch.Tensor:
        if values.shape[0] <= cap:
            return values
        step = max(values.shape[0] // cap, 1)
        return values[::step][:cap]

    # =========================================================
    # Internal helpers: integer ranges
    # =========================================================
    def _get_qrange(self) -> tuple[int, int]:
        fmt = get_quantization_format(self.data_type)
        return fmt.qmin, fmt.qmax
