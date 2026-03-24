from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass
class SequenceItem:
    utt_id: str
    codes: torch.Tensor
    phoneme_id: torch.Tensor
    cond_num: torch.Tensor
    note_on_boundary: torch.Tensor
    note_off_boundary: torch.Tensor


class SVSSequenceDataset(Dataset):
    def __init__(self, cache_path: str, max_seq_len: int | None = None):
        payload = torch.load(cache_path, map_location="cpu")
        self.items_raw = payload["items"]
        self.meta = payload["meta"]
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.items_raw)

    def __getitem__(self, idx: int) -> SequenceItem:
        it = self.items_raw[idx]
        codes = it["codes"].long()
        phoneme = it["phoneme_id"].long()
        cond = it["cond_num"].float()
        on_b = it["note_on_boundary"].float()
        off_b = it["note_off_boundary"].float()

        if self.max_seq_len is not None and codes.size(0) > self.max_seq_len:
            codes = codes[: self.max_seq_len]
            phoneme = phoneme[: self.max_seq_len]
            cond = cond[: self.max_seq_len]
            on_b = on_b[: self.max_seq_len]
            off_b = off_b[: self.max_seq_len]

        return SequenceItem(
            utt_id=it["utt_id"],
            codes=codes,
            phoneme_id=phoneme,
            cond_num=cond,
            note_on_boundary=on_b,
            note_off_boundary=off_b,
        )


def collate_sequences(batch: List[SequenceItem]) -> Dict[str, torch.Tensor | List[str]]:
    bsz = len(batch)
    t_max = max(x.codes.size(0) for x in batch)
    f = batch[0].codes.size(1)
    k = batch[0].codes.size(2)
    c = batch[0].cond_num.size(1)

    codes = torch.zeros(bsz, t_max, f, k, dtype=torch.long)
    phoneme_id = torch.zeros(bsz, t_max, dtype=torch.long)
    cond_num = torch.zeros(bsz, t_max, c)
    note_on_boundary = torch.zeros(bsz, t_max)
    note_off_boundary = torch.zeros(bsz, t_max)
    mask = torch.zeros(bsz, t_max)

    utt_ids: List[str] = []
    for i, it in enumerate(batch):
        t = it.codes.size(0)
        codes[i, :t] = it.codes
        phoneme_id[i, :t] = it.phoneme_id
        cond_num[i, :t] = it.cond_num
        note_on_boundary[i, :t] = it.note_on_boundary
        note_off_boundary[i, :t] = it.note_off_boundary
        mask[i, :t] = 1.0
        utt_ids.append(it.utt_id)

    return {
        "utt_id": utt_ids,
        "codes": codes,
        "phoneme_id": phoneme_id,
        "cond_num": cond_num,
        "note_on_boundary": note_on_boundary,
        "note_off_boundary": note_off_boundary,
        "mask": mask,
    }
