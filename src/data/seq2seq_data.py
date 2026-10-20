"""Dữ liệu cho Seq2Seq từ đầu: mã hóa bằng SentencePiece, cắt độ dài, ghép batch với padding động.

Quy ước độ dài (khớp với phép đo ở spm_tokenizer.measure):
  nguồn : ids[:max_input - 1] + [</s>]                 -> tối đa max_input token
  đích  : [<s>] + ids[:max_target - 2] + [</s>]        -> tối đa max_target token
Batch: tgt_in = đích[:-1] (đưa vào decoder), tgt_out = đích[1:] (nhãn). PAD trong nhãn bỏ qua khi tính loss.
"""
from __future__ import annotations

from pathlib import Path

from src.preprocessing.spm_tokenizer import BOS_ID, EOS_ID, PAD_ID, read_jsonl, target_text


def encode_source(ids: list[int], max_input: int) -> list[int]:
    return list(ids[:max_input - 1]) + [EOS_ID]


def encode_target(ids: list[int], max_target: int) -> list[int]:
    return [BOS_ID] + list(ids[:max_target - 2]) + [EOS_ID]


def build_examples(records: list[dict], sp, max_input: int, max_target: int, chunk: int = 1000) -> list[dict]:
    """Mã hóa toàn bộ bản ghi thành [{'id', 'src', 'tgt'}] (danh sách id token, chưa padding)."""
    out = []
    for i in range(0, len(records), chunk):
        part = records[i:i + chunk]
        src = sp.encode([r["article"] for r in part], out_type=int)
        tgt = sp.encode([target_text(r["highlights"]) for r in part], out_type=int)
        for r, s, t in zip(part, src, tgt):
            out.append({"id": r.get("id", ""), "src": encode_source(s, max_input), "tgt": encode_target(t, max_target)})
    return out


def load_examples(data_dir: str | Path, source: str, split: str, sp, max_input: int, max_target: int,
                  limit: int | None = None) -> list[dict]:
    recs = read_jsonl(Path(data_dir) / f"{source}_{split}.jsonl")
    if limit:
        recs = recs[:limit]
    return build_examples(recs, sp, max_input, max_target)


def collate(batch: list[dict], pad_id: int = PAD_ID) -> dict:
    """Đệm động theo độ dài lớn nhất trong batch."""
    import torch
    B = len(batch)
    src_lens = torch.tensor([len(b["src"]) for b in batch], dtype=torch.long)
    S = int(src_lens.max())
    T = max(len(b["tgt"]) for b in batch)
    src = torch.full((B, S), pad_id, dtype=torch.long)
    tgt = torch.full((B, T), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        src[i, :len(b["src"])] = torch.tensor(b["src"], dtype=torch.long)
        tgt[i, :len(b["tgt"])] = torch.tensor(b["tgt"], dtype=torch.long)
    return {"id": [b["id"] for b in batch], "src": src, "src_lens": src_lens,
            "tgt_in": tgt[:, :-1], "tgt_out": tgt[:, 1:]}


def make_loader(examples: list[dict], batch_size: int, shuffle: bool = False, seed: int = 42, num_workers: int = 0):
    """DataLoader trên danh sách ví dụ (list là dataset hợp lệ). Xáo trộn có seed cố định."""
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=g if shuffle else None, num_workers=num_workers)
