"""Hàm tiện ích cho EDA bài toán tóm tắt (không phụ thuộc mạng)."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

_SENT_SPLIT = re.compile(r'(?<=[.!?])["”\']?\s+(?=["“\']?[A-Z0-9])')
_TOKEN = re.compile(r"\w+(?:'\w+)?")

# Dấu vết "rác" thường gặp trong CNN/DailyMail -> motivation cho bước làm sạch (GĐ3)
BOILERPLATE_PATTERNS = {
    "cnn_prefix": re.compile(r"\(CNN\)"),
    "reuters_prefix": re.compile(r"\(Reuters\)"),
    "dailymail_by": re.compile(r"^\s*By \.", re.M),
    "published_updated": re.compile(r"\b(PUBLISHED|UPDATED|Last updated at)\b"),
    "read_more": re.compile(r"(?i)\bread more\b|\bclick here\b"),
}


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def word_count(text: str) -> int:
    return len(tokenize(text))


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return list(zip(*(tokens[i:] for i in range(n)))) if len(tokens) >= n else []


def novel_ngram_ratio(article: str, summary: str, n: int = 1) -> float:
    """Tỉ lệ n-gram của tóm tắt KHÔNG xuất hiện trong bài. Cao = tóm tắt mang tính abstractive."""
    s = ngrams(tokenize(summary), n)
    if not s:
        return float("nan")
    a = set(ngrams(tokenize(article), n))
    return sum(g not in a for g in s) / len(s)


def compression_ratio(article: str, summary: str) -> float:
    """Số từ bài / số từ tóm tắt."""
    sw = word_count(summary)
    return word_count(article) / sw if sw else float("nan")


def summary_sentences(highlights: str) -> list[str]:
    """CNN/DM tách câu tóm tắt bằng ký tự xuống dòng."""
    parts = [p.strip() for p in highlights.split("\n") if p.strip()]
    return parts if parts else split_sentences(highlights)


def best_match_positions(article: str, highlights: str) -> list[int]:
    """Với mỗi câu tóm tắt, trả về chỉ số câu trong bài có nhiều từ trùng nhất.
    Nếu các chỉ số tập trung ở đầu bài -> dữ liệu có 'lead bias'."""
    art = split_sentences(article)
    if not art:
        return []
    art_sets = [set(tokenize(s)) for s in art]
    out = []
    for hs in summary_sentences(highlights):
        h = set(tokenize(hs))
        if not h:
            continue
        scores = [len(h & a) / len(h) for a in art_sets]
        out.append(int(np.argmax(scores)))
    return out


def max_sentence_overlap(article: str, summary: str) -> float:
    """Tỉ lệ từ của tóm tắt nằm trong MỘT câu bài (lấy câu tốt nhất). ~1.0 nghĩa là tóm tắt chép lại một câu."""
    s = set(tokenize(summary))
    if not s:
        return 0.0
    return max((len(s & set(tokenize(x))) / len(s) for x in split_sentences(article)), default=0.0)


def boilerplate_flags(article: str) -> dict:
    return {k: bool(p.search(article)) for k, p in BOILERPLATE_PATTERNS.items()}


def describe(values: Iterable[float]) -> dict:
    v = np.asarray([x for x in values if x == x], dtype=float)  # bỏ NaN
    if v.size == 0:
        return {}
    return {
        "mean": v.mean(), "std": v.std(), "min": v.min(),
        "p25": np.percentile(v, 25), "median": np.median(v),
        "p75": np.percentile(v, 75), "p95": np.percentile(v, 95), "max": v.max(),
    }
