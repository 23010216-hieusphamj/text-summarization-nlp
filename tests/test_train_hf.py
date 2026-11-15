"""Test vòng huấn luyện BART bằng mô hình NHỎ ngẫu nhiên (CPU, dropout = 0 để tất định). Cần torch + transformers + rouge_score."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("rouge_score")

from src import train_hf as th  # noqa: E402


class StubTok:
    pad_token_id, eos_token_id = 1, 2

    def batch_decode(self, out, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return [" ".join(str(int(t)) for t in row if int(t) > 2) for row in out]


def make_model():
    from transformers import BartConfig, BartForConditionalGeneration
    torch.manual_seed(0)
    cfg = BartConfig(vocab_size=64, d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                     decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=64,
                     dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, pad_token_id=1, bos_token_id=0,
                     eos_token_id=2, decoder_start_token_id=2, forced_eos_token_id=None)
    return BartForConditionalGeneration(cfg)


def data(n=16):
    rs = np.random.RandomState(0)
    ex = [{"id": str(i), "input_ids": [0] + rs.randint(3, 64, size=10).tolist() + [2],
           "labels": [0] + rs.randint(3, 64, size=4).tolist() + [2]} for i in range(n)]
    refs = [" ".join(str(t) for t in e["labels"] if t > 2) for e in ex]
    return ex, refs


def cfg(epochs=2):
    return {"run_name": "t", "kind": "bart", "model_name": "tiny", "source": "x", "max_input": 16, "max_target": 8,
            "train": {"seed": 1, "lr": 5e-3, "weight_decay": 0.0, "batch_size": 4, "grad_accum": 2, "epochs": epochs,
                      "warmup_ratio": 0.1, "clip_norm": 1.0, "amp": False, "patience": epochs, "log_every": 1,
                      "save_every_steps": 1},
            "best_by": "S", "val_rouge_n": 8,
            "val_gen": {"num_beams": 1, "max_length": 8, "min_length": 0, "length_penalty": 1.0, "no_repeat_ngram_size": 0}}


def run(tmp_path, name, epochs=2, **kw):
    ex, refs = data()
    c = cfg(epochs)
    return th.train(c, ex, ex[:8], refs[:8], StubTok(), tmp_path / name / "logs", tmp_path / name / "ckpt",
                    device="cpu", model=make_model(), **kw)


def test_train_writes_logs_and_checkpoints_and_loss_goes_down(tmp_path):
    res = run(tmp_path, "a", epochs=3)
    log = pd.read_csv(tmp_path / "a" / "logs" / "t_log.csv")
    assert log["epoch"].tolist() == [1, 2, 3]
    assert {"train_loss", "val_loss", "val_rouge1", "val_S", "epoch_seconds", "train_seconds_total"} <= set(log.columns)
    assert np.isfinite(log["val_loss"]).all() and log["train_loss"].iloc[-1] < log["train_loss"].iloc[0]
    assert (tmp_path / "a" / "ckpt" / "t" / "best.pt").exists() and (tmp_path / "a" / "ckpt" / "t" / "last.pt").exists()
    assert res["global_step"] == 3 * 2 and not res["halted"]
    meta = json.load(open(tmp_path / "a" / "logs" / "t_meta.json", encoding="utf-8"))
    assert meta["effective_batch"] == 8 and meta["steps_per_epoch"] == 2 and meta["seed"] == 1


def test_resume_mid_epoch_matches_uninterrupted_run(tmp_path):
    run(tmp_path, "full", epochs=2)
    part = run(tmp_path, "cut", epochs=2, halt_after_step=1)            # bị ngắt giữa epoch 1
    assert part["halted"] and part["global_step"] == 1
    done = run(tmp_path, "cut", epochs=2, resume=True)                   # tiếp tục từ last.pt
    assert not done["halted"] and done["global_step"] == 4
    a = torch.load(tmp_path / "full" / "ckpt" / "t" / "last.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "cut" / "ckpt" / "t" / "last.pt", map_location="cpu", weights_only=False)
    for k in a["model"]:
        assert torch.allclose(a["model"][k], b["model"][k], atol=1e-5), k
    la, lb = (pd.read_csv(tmp_path / n / "logs" / "t_log.csv") for n in ("full", "cut"))
    assert lb["epoch"].tolist() == [1, 2]                                  # không trùng dòng log
    assert np.allclose(la["train_loss"], lb["train_loss"], rtol=1e-4) and np.allclose(la["val_loss"], lb["val_loss"], rtol=1e-4)


def test_existing_logs_without_resume_raise(tmp_path):
    run(tmp_path, "a", epochs=1)
    with pytest.raises(FileExistsError):
        run(tmp_path, "a", epochs=1)


def test_main_uses_cli_log_dir_and_ckpt_dir_else_config_default(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(th, "load_tokenizer", lambda name, cache=None: StubTok())
    monkeypatch.setattr(th, "load_or_build", lambda *a, **k: (data(4)[0], True))
    monkeypatch.setattr(th, "load_records", lambda *a, **k: [{"id": str(i), "article": "x", "highlights": "y"} for i in range(4)])

    def fake_train(cfg, tr, va, refs, tok, log_dir, ckpt_dir, resume=False, **kw):
        seen.update(log_dir=str(log_dir), ckpt_dir=str(ckpt_dir), resume=resume)
        return {"ok": True}

    monkeypatch.setattr(th, "train", fake_train)
    base = ["--config", "configs/bart_cnndm_in512.yaml", "--data_dir", str(tmp_path), "--cache_dir", str(tmp_path)]
    th.main(base + ["--ckpt_dir", str(tmp_path / "ck"), "--log_dir", str(tmp_path / "lg"), "--resume"])
    assert seen == {"log_dir": str(tmp_path / "lg"), "ckpt_dir": str(tmp_path / "ck"), "resume": True}
    th.main(base)
    assert seen["log_dir"] == "results/logs" and seen["ckpt_dir"] == "checkpoints" and seen["resume"] is False
