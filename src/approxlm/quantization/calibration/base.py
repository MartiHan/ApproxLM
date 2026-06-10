from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =========================
# Data container
# =========================

@dataclass
class CalibStats:
    amax: torch.Tensor
    amin: torch.Tensor
    qmax: int
    qmin: int


# =========================
# Base calibrator
# =========================

class BaseCalibrator(ABC):
    def __init__(
        self,
        data_type: str,
        per_channel: bool,
        symmetric: bool,
        calib_dataset: DataLoader,
        num_batches: int,
        layers_to_calibrate: List[nn.Module],
    ):
        self.data_type = data_type
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.calib_dataset = calib_dataset
        self.num_batches = num_batches
        self.layers_to_calibrate = layers_to_calibrate

        self.calib_stats: Dict[str, CalibStats] = {}

    @abstractmethod
    def compute_calib_stats(self, model: nn.Module) -> Dict[str, CalibStats]:
        pass

    @abstractmethod
    def make_hook(self, layer_name: str) -> Callable:
        pass