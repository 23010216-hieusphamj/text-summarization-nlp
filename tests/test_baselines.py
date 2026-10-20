import numpy as np
import pytest

from src.baselines.lead import join_sentences, lead_k, oracle_k, random_k
from src.baselines.textrank import textrank_k
from src.evaluate import bootstrap_ci

ART = ("The council approved a new budget on Monday. Funding for schools will rise by ten percent. "
       "Opposition members criticised the plan. Roads will get repaired next year. "
       "A weather report says rain is likely on Friday.")


def test_lead_k_takes_first_sentences():
    assert lead_k(ART, 3) == [0, 1, 2]
    assert lead_k("One sentence only.", 3) == [0]
    assert lead_k("", 3) == []


def test_random_k_is_reproducible_sorted_and_bounded():
    a, b = random_k(ART, 3, seed=7), random_k(ART, 3, seed=7)
    assert a == b and a == sorted(a) and len(set(a)) == 3 and max(a) < 5
    assert len(random_k("Short one.", 3, seed=1)) == 1


def test_join_sentences_uses_newline():
    assert join_sentences(ART, [0, 1]).count("\n") == 1


def test_oracle_picks_matching_sentence_and_stops_early():
    ref = "Funding for schools will rise by ten percent."
    assert oracle_k(ART, ref, k=3) == [1]          # thêm câu khác chỉ làm precision giảm -> dừng
    assert oracle_k(ART, "", k=3) == []


def test_textrank_returns_k_sorted_indices_and_handles_edge_cases():
    idx = textrank_k(ART, 2)
    assert len(idx) == 2 and idx == sorted(idx)
    assert textrank_k("Just one sentence here.", 3) == [0]
    assert textrank_k("", 3) == []
    assert textrank_k("The. A. Of. In.", 2) == [0, 1]          # toàn stop word -> fallback giống Lead


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    v = np.linspace(0, 1, 200)
    m, lo, hi = bootstrap_ci(v, 500, seed=1)
    assert lo < m < hi and (m, lo, hi) == bootstrap_ci(v, 500, seed=1)
    assert np.isnan(bootstrap_ci([])[0])


def test_paired_tests_detects_clear_difference():
    from src.analysis.compare_baselines import paired_tests
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.4, 200)
    sc = {"lead-1": {m: base + 0.05 for m in ("rouge1", "rouge2", "rougeLsum")},
          "textrank-1": {m: base for m in ("rouge1", "rouge2", "rougeLsum")}}
    df = paired_tests(sc, [("lead-1", "textrank-1")], 300, 1)
    assert df["significant"].all() and (df["diff"] > 0.04).all()
