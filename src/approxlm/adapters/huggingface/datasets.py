from __future__ import annotations
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset

class MassiveTorchDataset(Dataset):
    def __init__(self, hf_ds, text_col: str, label_col: str):
        self.ds = hf_ds
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.ds[idx]
        return {
            "sample_id": idx,
            "text": example[self.text_col],
            "label_raw": example[self.label_col],
        }


class SequenceClassificationCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 128,
        pad_to_max_length: bool = False,
        store_text: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.store_text = store_text

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        enc = self.tokenizer(
            texts,
            padding="max_length" if self.pad_to_max_length else True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(
            [(-100 if item["label_raw"] is None else int(item["label_raw"])) for item in batch],
            dtype=torch.long,
        )
        enc["sample_id"] = torch.tensor([item["sample_id"] for item in batch], dtype=torch.long)
        if self.store_text:
            enc["text"] = texts
        return enc
