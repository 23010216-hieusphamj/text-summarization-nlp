from datasets import Dataset

from src.data.cnn_dm import build_subsets


def _fake(n):
    return Dataset.from_dict({
        "id": [str(i) for i in range(n)],
        "article": [f"article {i}" for i in range(n)],
        "highlights": [f"summary {i}" for i in range(n)],
    })


def test_subset_sizes_and_reproducible():
    raw = {"train": _fake(100), "validation": _fake(20), "test": _fake(10)}
    sizes = {"train": 30, "validation": 5, "test": 50}  # test > len -> lấy hết
    a = build_subsets(raw, sizes, seed=42)
    b = build_subsets(raw, sizes, seed=42)
    assert len(a["train"]) == 30 and len(a["validation"]) == 5 and len(a["test"]) == 10
    assert a["train"]["id"] == b["train"]["id"]          # cùng seed -> cùng kết quả
    assert a["train"]["id"] != build_subsets(raw, sizes, seed=7)["train"]["id"]
