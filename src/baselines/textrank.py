"""TextRank (Mihalcea & Tarau, 2004): PageRank trên đồ thị câu, trọng số cạnh = cosine TF-IDF."""
from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.eda_utils import split_sentences


def textrank_scores(sents: list[str], stop_words: str | None = "english", damping: float = 0.85) -> np.ndarray:
    n = len(sents)
    if n <= 1:
        return np.ones(n)
    try:
        X = TfidfVectorizer(stop_words=stop_words).fit_transform(sents)
    except ValueError:                       # toàn bộ từ bị loại (vd. câu chỉ có stop word)
        return np.ones(n)
    sim = (X @ X.T).toarray()                # TF-IDF đã chuẩn hóa L2 -> tích vô hướng = cosine
    np.fill_diagonal(sim, 0.0)
    if sim.sum() == 0:
        return np.ones(n)
    try:
        pr = nx.pagerank(nx.from_numpy_array(sim), alpha=damping, max_iter=200, weight="weight")
    except nx.PowerIterationFailedConvergence:
        return np.ones(n)
    return np.array([pr[i] for i in range(n)])


def textrank_k(article: str, k: int = 3, stop_words: str | None = "english", damping: float = 0.85) -> list[int]:
    """Chỉ số k câu có điểm cao nhất, trả theo thứ tự xuất hiện trong bài.
    Điểm bằng nhau (vd. fallback) -> ưu tiên câu đứng trước (hành xử như Lead-k)."""
    sents = split_sentences(article)
    if not sents:
        return []
    scores = textrank_scores(sents, stop_words, damping)
    order = sorted(range(len(sents)), key=lambda i: (-scores[i], i))
    return sorted(order[:k])
