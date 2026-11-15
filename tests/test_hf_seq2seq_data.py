"""Test dữ liệu BART/T5 (GĐ6 Bước 3). Dùng tokenizer GIẢ (tách theo khoảng trắng, có <s>=0, <pad>=1, </s>=2 như BART)
nên chạy offline, không cần transformers; riêng make_loader cần torch (importorskip)."""
import json

import numpy as np
import pytest
import yaml

from src.data import hf_seq2seq_data as hd


class FakeTok:
    """Giả lập API tokenizer HuggingFace ở mức tối thiểu mà hf_seq2seq_data dùng."""
    bos_token_id, pad_token_id, eos_token_id = 0, 1, 2

    def __init__(self, offset: int = 10):
        self.vocab, self.offset = {}, offset

    def __len__(self):
        return 1000

    def _ids(self, text, max_length, truncation):
        ids = [self.bos_token_id] + [self.vocab.setdefault(w, self.offset + len(self.vocab)) for w in text.split()] + [self.eos_token_id]
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_token_id]          # giữ </s> sau khi cắt như tokenizer thật
        return ids

    def __call__(self, texts=None, text_target=None, max_length=None, truncation=False, verbose=False):
        src = text_target if text_target is not None else texts
        single = isinstance(src, str)
        out = [self._ids(t, max_length, truncation) for t in ([src] if single else src)]
        ids = out[0] if single else out
        return {"input_ids": ids, "attention_mask": [1] * len(ids) if single else [[1] * len(x) for x in out]}

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, f"<{i}>") for i in ids)


def _write(tmp_path, source="cnn_dm", split="train", n=6):
    d = tmp_path / "clean"
    d.mkdir(exist_ok=True)
    recs = [{"id": f"{source}{i}", "article": "alpha beta gamma delta " * (i + 1), "highlights": "one two.\nthree four."} for i in range(n)]
    with open(d / f"{source}_{split}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return d, recs


def test_pack_unpack_ragged_roundtrip_with_empty_sequence():
    seqs = [[1, 2, 3], [], [7], [4, 5]]
    flat, off = hd.pack_ragged(seqs)
    assert off.tolist() == [0, 3, 3, 4, 6] and hd.unpack_ragged(flat, off) == seqs


def test_tokenize_records_prefix_join_and_truncation_keep_eos():
    tok = FakeTok()
    recs = [{"id": "a", "article": "w1 w2 w3 w4 w5 w6 w7 w8", "highlights": "x y.\nz w."}]
    ex = hd.tokenize_records(recs, tok, max_input=6, max_target=4, prefix="summarize: ", target_join=" ")[0]
    assert len(ex["input_ids"]) == 6 and ex["input_ids"][-1] == tok.eos_token_id     # cắt ở 6, còn </s>
    assert tok.decode(ex["input_ids"][1:3]) == "summarize: w1"                          # tiền tố nằm đầu bài
    assert len(ex["labels"]) == 4 and ex["labels"][-1] == tok.eos_token_id
    full = hd.tokenize_records(recs, tok, 50, 50, "", " ")[0]
    assert tok.decode(full["labels"]) == "<0> x y. z w. <2>"                             # "\n" đã nối bằng " "


def test_load_or_build_uses_cache_and_rebuilds_when_anything_changes(tmp_path):
    d, _ = _write(tmp_path)
    cache = tmp_path / "cache"
    tok = FakeTok()
    args = (d, cache, "cnn_dm", "train", "bart", tok, "fake/bart", 16, 8)
    ex1, hit1 = hd.load_or_build(*args)
    ex2, hit2 = hd.load_or_build(*args)
    assert (hit1, hit2) == (False, True) and ex1 == ex2
    assert hd.load_or_build(*args, prefix="summarize: ")[1] is False                 # đổi tiền tố -> xây lại
    assert hd.load_or_build(*args)[1] is False                                         # meta bị ghi đè bởi lần trước -> xây lại
    assert hd.load_or_build(*args)[1] is True
    with open(d / "cnn_dm_train.jsonl", "a", encoding="utf-8") as f:                  # đổi dữ liệu -> xây lại
        f.write(json.dumps({"id": "new", "article": "x y z", "highlights": "q"}) + "\n")
    ex3, hit3 = hd.load_or_build(*args)
    assert hit3 is False and len(ex3) == len(ex1) + 1
    other = (d, cache, "cnn_dm", "train", "bart", FakeTok(offset=500), "fake/bart", 16, 8)   # tokenizer khác (canary khác)
    assert hd.load_or_build(*other)[1] is False
    ex4, _ = hd.load_or_build(d, cache, "cnn_dm", "train", "bart", tok, "fake/bart", 32, 8)   # max_input khác -> cache riêng
    assert (cache / "cnn_dm_train_bart_in32_tg8.npz").exists() and (cache / "cnn_dm_train_bart_in16_tg8.npz").exists()


def test_collate_pads_inputs_with_pad_id_and_labels_with_minus100():
    batch = [{"id": "a", "input_ids": [0, 5, 2], "labels": [0, 9, 2]}, {"id": "b", "input_ids": [0, 5, 6, 7, 2], "labels": [0, 2]}]
    out = hd.collate_seq2seq(batch, pad_token_id=1)
    assert out["input_ids"].shape == (2, 5) and out["labels"].shape == (2, 3)
    assert out["input_ids"][0].tolist() == [0, 5, 2, 1, 1] and out["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert out["labels"][1].tolist() == [0, 2, -100] and out["input_ids"].dtype == np.int64 and out["id"] == ["a", "b"]


def test_length_table_groups_and_validates_length():
    recs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    df = hd.length_table(recs, np.array([100, 512, 513]), np.array([90, 500, 520]))
    assert df["group_512"].tolist() == ["<=512", "<=512", ">512"]          # đúng 512 token chưa bị cắt
    assert df["gt_512"].tolist() == [False, False, True] and df["gt_1024"].sum() == 0 and "n_tok_t5" in df
    with pytest.raises(ValueError):
        hd.length_table(recs, np.array([1, 2]))


def test_check_examples_reports_truncation_and_eos():
    tok = FakeTok()
    ex = hd.tokenize_records([{"id": str(i), "article": "w " * 20, "highlights": "a b c d e"} for i in range(4)], tok, 8, 4)
    r = hd.check_examples(ex, tok, 8, 4)
    assert r["n"] == 4 and r["pct_input_at_max"] == 100.0 and r["pct_label_at_max"] == 100.0
    assert r["pct_label_ends_eos"] == 100.0 and r["pct_input_ends_eos"] == 100.0 and r["n_empty_labels"] == 0


def test_load_records_limit_and_missing_file(tmp_path):
    d, _ = _write(tmp_path)
    assert len(hd.load_records(d, "cnn_dm", "train", limit=2)) == 2
    with pytest.raises(FileNotFoundError):
        hd.load_records(d, "govuk", "train")


def test_cli_end_to_end_with_fake_tokenizers(tmp_path, monkeypatch):
    d, _ = _write(tmp_path, "cnn_dm", "train")
    _write(tmp_path, "cnn_dm", "validation")
    _write(tmp_path, "cnn_dm", "test")
    pre = tmp_path / "pre.yaml"
    pre.write_text(yaml.safe_dump({"tokenizers": {"bart": "fake/bart", "t5": "fake/t5"}, "t5_prefix": "summarize: ", "target_join": " "}))
    cfg = {"preprocess_config": str(pre), "data_dir": str(d), "cache_dir": str(tmp_path / "cache"), "splits": ["train", "validation"],
           "plan": [{"source": "cnn_dm", "kind": "bart", "max_input": 16, "max_target": 8},
                    {"source": "cnn_dm", "kind": "t5", "max_input": 16, "max_target": 8}],
           "length_tables": {"splits": ["validation", "test"], "thresholds": [512, 1024]}, "output": {"tables_dir": str(tmp_path / "tables")}}
    cp = tmp_path / "hf.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(hd, "load_tokenizer", lambda name, cache=None: FakeTok())
    hd.main(["--config", str(cp)])
    hd.main(["--config", str(cp)])                                             # lần 2: dùng cache
    import pandas as pd
    chk = pd.read_csv(tmp_path / "tables" / "hf_tokenization_check.csv")
    assert len(chk) == 4 and chk["from_cache"].tolist() == [True] * 4              # bảng của lần chạy sau cùng
    lt = pd.read_csv(tmp_path / "tables" / "token_lengths_cnn_dm_test.csv")
    assert list(lt.columns[:3]) == ["id", "n_tok_bart", "n_tok_t5"] and len(lt) == 6
    assert (lt["n_tok_t5"] > lt["n_tok_bart"]).all()                               # tiền tố T5 làm dài hơn


def test_make_loader_returns_tensors_and_is_seeded():
    torch = pytest.importorskip("torch")
    ex = [{"id": str(i), "input_ids": [0, i + 3, 2], "labels": [0, 2]} for i in range(8)]
    a = [b["id"] for b in hd.make_loader(ex, 4, pad_token_id=1, shuffle=True, seed=7)]
    b = [b["id"] for b in hd.make_loader(ex, 4, pad_token_id=1, shuffle=True, seed=7)]
    assert a == b
    batch = next(iter(hd.make_loader(ex, 4, pad_token_id=1)))
    assert isinstance(batch["input_ids"], torch.Tensor) and batch["labels"].dtype == torch.int64
