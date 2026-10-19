"""Tokenization cho BART / T5, đo độ dài token và tỉ lệ bị cắt, tạo dataset huấn luyện."""
from __future__ import annotations

import numpy as np


def prepare_texts(articles, highlights, prefix: str = "", target_join: str = " "):
    inputs = [prefix + a for a in articles]
    targets = [h.replace("\n", target_join) for h in highlights]
    return inputs, targets


def tokenize_batch(batch: dict, tokenizer, max_input: int, max_target: int, prefix: str = "", target_join: str = " ") -> dict:
    """Dùng với datasets.map(batched=True). Chưa padding: việc đệm làm động bằng DataCollatorForSeq2Seq."""
    inputs, targets = prepare_texts(batch["article"], batch["highlights"], prefix, target_join)
    enc = tokenizer(inputs, max_length=max_input, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target, truncation=True)
    enc["labels"] = labels["input_ids"]
    return dict(enc)


def build_tokenized_dataset(records: list[dict], tokenizer, max_input: int, max_target: int, prefix: str = "", target_join: str = " "):
    from datasets import Dataset
    ds = Dataset.from_list([{"article": r["article"], "highlights": r["highlights"]} for r in records])
    return ds.map(lambda b: tokenize_batch(b, tokenizer, max_input, max_target, prefix, target_join),
                  batched=True, remove_columns=ds.column_names)


def token_lengths(texts: list[str], tokenizer, batch_size: int = 500) -> np.ndarray:
    """Số token của từng văn bản khi KHÔNG cắt (đã gồm token đặc biệt)."""
    out = []
    for i in range(0, len(texts), batch_size):
        ids = tokenizer(texts[i:i + batch_size], truncation=False, verbose=False)["input_ids"]
        out.extend(len(x) for x in ids)
    return np.asarray(out)


def truncation_report(lengths: np.ndarray, thresholds: list[int]) -> list[dict]:
    """Với mỗi ngưỡng: % văn bản bị cắt và % token bị mất trung bình."""
    rows = []
    total = lengths.sum()
    for t in thresholds:
        lost = np.clip(lengths - t, 0, None)
        rows.append({"max_len": t, "pct_truncated": float((lengths > t).mean() * 100),
                     "pct_tokens_lost": float(lost.sum() / total * 100)})
    return rows
