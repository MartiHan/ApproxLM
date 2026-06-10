from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch


@dataclass
class QuantParams:
    scale: Union[float, torch.Tensor]
    zero_point: Union[int, torch.Tensor]
    qmin: int
    qmax: int
    symmetric: bool
    per_channel: bool
    data_type: str