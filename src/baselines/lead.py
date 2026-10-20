"""Baseline extractive không cần huấn luyện: Lead-k, Random-k và Oracle (cận trên của extractive)."""
from __future__ import annotations

from collections import Counter

import numpy as np

from src.data.eda_utils import ngrams, split_sentences, tokenize


def lead_k(article: str, k: int = 3) -> list[int]:
    """Chỉ số k câu đầu tiên của bài."""
    n = len(split_sentences(article))
    return list(range(min(k, n)))


def random_k(article: str, k: int = 3, seed: int = 42) -> list[int]:
    """k câu ngẫu nhiên (giữ thứ tự gốc). Cận dưới: dữ liệu có lead bias thật hay không."""
    n = len(split_sentences(article))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=min(k, n), replace=False).tolist()) if n else []


def _grams(text: str) -> tuple[Counter, Counter]:
    t = tokenize(text)
    return Counter(ngrams(t, 1)), Counter(ngrams(t, 2))


def _f1(cand: Counter, ref: Counter) -> float:
    overlap = sum(min(c, ref[g]) for g, c in cand.items() if g in ref)
    if not overlap:
        return 0.0
    p, r = overlap / sum(cand.values()), overlap / sum(ref.values())
    return 2 * p * r / (p + r)


def oracle_k(article: str, reference: str, k: int = 3) -> list[int]:
    """Chọn tham lam tối đa k câu để cực đại hóa (ROUGE-1 F + ROUGE-2 F)/2 so với tham chiếu.

    Dừng sớm khi thêm câu không còn cải thiện. Đây là xấp xỉ nhanh (token chữ thường, không stemming)
    nên dùng để so sánh tương đối với Lead/TextRank, không phải con số ROUGE chính thức.
    """
    sents = split_sentences(article)
    ref1, ref2 = _grams(reference.replace("\n", " "))
    if not sents or not ref1:
        return []
    grams = [_grams(s) for s in sents]
    sel: list[int] = []
    cur1, cur2, best = Counter(), Counter(), 0.0
    for _ in range(min(k, len(sents))):
        pick, pick_score = None, best
        for i, (g1, g2) in enumerate(grams):
            if i in sel:
                continue
            s = (_f1(cur1 + g1, ref1) + _f1(cur2 + g2, ref2)) / 2
            if s > pick_score:
                pick, pick_score = i, s
        if pick is None:
            break
        sel.append(pick); cur1 += grams[pick][0]; cur2 += grams[pick][1]; best = pick_score
    return sorted(sel)


def join_sentences(article: str, idx: list[int]) -> str:
    """Ghép các câu đã chọn bằng '\\n' (cần cho ROUGE-Lsum)."""
    sents = split_sentences(article)
    return "\n".join(sents[i] for i in idx)
