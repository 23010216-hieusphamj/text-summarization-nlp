import csv
import random

import pytest

torch = pytest.importorskip("torch")

from src.preprocessing.spm_tokenizer import BOS_ID, EOS_ID
from src.train_seq2seq import train


def _examples(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        src = [rng.randint(4, 29) for _ in range(rng.randint(6, 12))] + [EOS_ID]
        tgt = [BOS_ID] + [rng.randint(4, 29) for _ in range(rng.randint(3, 6))] + [EOS_ID]
        out.append({"id": str(i), "src": src, "tgt": tgt})
    return out


def _cfg(epochs):
    return {"model": {"emb_dim": 16, "hidden": 16, "enc_layers": 2, "attn_dim": 16, "dropout": 0.0},
            "train": {"batch_size": 8, "lr": 0.005, "epochs": epochs, "clip_norm": 1.0, "patience": 5,
                      "seed": 42, "amp": False, "num_workers": 0, "log_every": 2}}


def _rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def test_train_writes_logs_and_checkpoints_then_resumes(tmp_path):
    tr, va = _examples(32, 0), _examples(8, 1)
    logs, ck = tmp_path / "logs", tmp_path / "ck"
    res = train(_cfg(2), tr, va, 30, "t", logs, ck, device="cpu")
    rows = _rows(logs / "t_log.csv")
    assert len(rows) == 2
    assert all(0 < float(r["train_loss"]) < 10 and 0 < float(r["val_loss"]) < 10 for r in rows)
    assert (ck / "t" / "best.pt").exists() and (ck / "t" / "last.pt").exists()
    assert (logs / "t_steps.csv").exists() and (logs / "t_meta.json").exists()
    assert res["n_params"] > 0 and res["best_epoch"] in (1, 2)
    # tiếp tục: chạy thêm đến epoch 3, log có 3 dòng (không ghi lại epoch cũ)
    train(_cfg(3), tr, va, 30, "t", logs, ck, resume=True, device="cpu")
    assert [r["epoch"] for r in _rows(logs / "t_log.csv")] == ["1", "2", "3"]


def test_existing_log_without_resume_is_refused(tmp_path):
    tr, va = _examples(16, 0), _examples(8, 1)
    train(_cfg(1), tr, va, 30, "t", tmp_path / "l", tmp_path / "c", device="cpu")
    with pytest.raises(FileExistsError):
        train(_cfg(1), tr, va, 30, "t", tmp_path / "l", tmp_path / "c", device="cpu")
