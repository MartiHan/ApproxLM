from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, Iterable
import torch

@dataclass(frozen=True)
class QuantizationFormat:
    """Integer quantization format metadata.

    Approximate LUT execution currently supports signed 8-bit operands, while
    exact quantized execution can be extended by registering another format.
    """
    name: str
    bits: int
    signed: bool = True
    narrow_range: bool = True

    @property
    def qmin(self) -> int:
        if self.signed:
            return -(2 ** (self.bits - 1) - (1 if self.narrow_range else 0))
        return 0

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - (1 if self.signed else 0)) - 1

    @property
    def storage_dtype(self) -> torch.dtype:
        if self.signed:
            if self.bits <= 8: return torch.int8
            if self.bits <= 16: return torch.int16
            return torch.int32
        if self.bits <= 8: return torch.uint8
        # PyTorch has no generally useful uint16 matmul backend.
        return torch.int32

class QuantizationFormatRegistry:
    def __init__(self) -> None:
        self._formats: Dict[str, QuantizationFormat] = {}

    def register(self, fmt: QuantizationFormat, *, replace: bool=False) -> None:
        key=fmt.name.lower()
        if key in self._formats and not replace:
            raise KeyError(f"Quantization format already registered: {fmt.name}")
        self._formats[key]=fmt

    def get(self, name: str) -> QuantizationFormat:
        key=name.lower()
        if key in self._formats: return self._formats[key]
        match=re.fullmatch(r'(u?int)(\d+)', key)
        if not match: raise KeyError(f"Unknown quantization format: {name}")
        signed=match.group(1)=='int'; bits=int(match.group(2))
        if bits < 2 or bits > 32: raise ValueError(f"Unsupported integer width: {bits}")
        fmt=QuantizationFormat(key,bits,signed=signed,narrow_range=signed)
        self._formats[key]=fmt
        return fmt

    def names(self) -> Iterable[str]: return tuple(sorted(self._formats))

FORMATS=QuantizationFormatRegistry()
for _name,_bits,_signed in [('int4',4,True),('uint4',4,False),('int8',8,True),('uint8',8,False),('int12',12,True),('uint12',12,False),('int16',16,True),('uint16',16,False)]:
    FORMATS.register(QuantizationFormat(_name,_bits,_signed,narrow_range=_signed))

def get_quantization_format(name: str) -> QuantizationFormat:
    return FORMATS.get(name)
