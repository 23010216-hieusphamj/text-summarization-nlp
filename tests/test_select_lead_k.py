import pandas as pd

from src.analysis.select_lead_k import pick_k, selection_score


def _df(vals: dict[int, tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"k": k, "rouge1": a, "rouge2": b, "rougeLsum": c} for k, (a, b, c) in vals.items()])


def test_selection_score_is_mean_of_r1_r2_rlsum():
    assert abs(selection_score({"rouge1": 0.4, "rouge2": 0.2, "rougeLsum": 0.3}) - 0.3) < 1e-12


def test_pick_k_takes_clear_winner():
    d = _df({1: (0.28, 0.10, 0.24), 2: (0.38, 0.16, 0.34), 3: (0.40, 0.18, 0.36), 4: (0.36, 0.15, 0.33), 5: (0.33, 0.13, 0.30)})
    r = pick_k(d, default_k=1)
    assert r["best_k"] == 3 and r["chosen_k"] == 3


def test_pick_k_tie_prefers_k_closest_to_old_default_then_smaller():
    # k=3 tốt nhất 0,300; k=2 thấp hơn 0,001 (< 0,2 điểm) -> hòa; mốc cũ 2 -> chọn 2
    d = _df({2: (0.299, 0.299, 0.299), 3: (0.300, 0.300, 0.300), 4: (0.250, 0.250, 0.250)})
    assert pick_k(d, default_k=2)["chosen_k"] == 2
    assert pick_k(d, default_k=3)["chosen_k"] == 3
    # cách đều mốc -> chọn k nhỏ hơn
    d2 = _df({2: (0.300, 0.300, 0.300), 4: (0.300, 0.300, 0.300)})
    assert pick_k(d2, default_k=3)["chosen_k"] == 2


def test_pick_k_gap_of_exactly_tolerance_is_not_a_tie():
    d = _df({1: (0.300, 0.300, 0.300), 3: (0.302, 0.302, 0.302)})
    assert pick_k(d, default_k=1, tol=0.002 - 1e-9)["chosen_k"] == 3


def test_pick_k_empty_raises():
    try:
        pick_k(pd.DataFrame(columns=["k", "rouge1", "rouge2", "rougeLsum"]), 3)
    except ValueError:
        return
    raise AssertionError("df rỗng phải báo lỗi")
