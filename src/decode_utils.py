"""Tiện ích giải mã/đánh giá Seq2Seq không phụ thuộc torch (kiểm thử được ở mọi nơi)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.eda_utils import ngrams, split_sentences, tokenize


def banned_tokens(prefix: list[int], n: int) -> set[int]:
    """Token bị cấm ở bước kế tiếp để KHÔNG tạo n-gram lặp (no_repeat_ngram). n <= 0: không chặn."""
    if n <= 0:
        return set()
    if n == 1:
        return set(prefix)
    if len(prefix) < n:
        return set()
    seen: dict[tuple, set] = {}
    for i in range(len(prefix) - n + 1):
        seen.setdefault(tuple(prefix[i:i + n - 1]), set()).add(prefix[i + n - 1])
    return seen.get(tuple(prefix[-(n - 1):]), set())


def sentences_text(text: str) -> str:
    """Tách câu rồi nối bằng '\\n' (quy ước ROUGE-Lsum của evaluate.py, giống tham chiếu CNN/DM)."""
    return "\n".join(split_sentences(text))


def repeated_trigram_rate(preds: list[str]) -> float:
    """% dự đoán có ít nhất một trigram từ lặp lại (đo lỗi lặp, D9 dự đoán số 2)."""
    if not preds:
        return float("nan")
    rep = 0
    for p in preds:
        g = ngrams(tokenize(p), 3)
        rep += len(g) != len(set(g))
    return 100 * rep / len(preds)


def upsert_rows(path: str | Path, new: pd.DataFrame, keys: tuple[str, ...] = ("source", "system")) -> pd.DataFrame:
    """Ghi bảng CSV: dòng mới thay dòng cũ cùng khóa (source, system), các dòng khác được giữ nguyên."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        merged_keys = new[list(keys)].apply(tuple, axis=1)
        mask = old[list(keys)].apply(tuple, axis=1).isin(set(merged_keys))
        new = pd.concat([old[~mask], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new
