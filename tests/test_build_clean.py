import json
from pathlib import Path

import yaml

from src.preprocessing import build_clean as bc

DM = ("By . Daily Mail Reporter . PUBLISHED: . 07:12 EST, 28 February 2013 . | . UPDATED: . 07:34 EST, "
      "28 February 2013 . ")


def _w(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")


def _rec(i, art, hl):
    return {"id": str(i), "article": art, "highlights": hl}


def test_end_to_end(tmp_path):
    body = " ".join(f"Sentence number {k} about topic {k}." for k in range(30))
    good_hl = "Police arrested a man in Paris after a long investigation into fraud\nTwo others remain at large"
    inp = tmp_path / "in"
    _w(inp / "cnn_dm_train.jsonl", [
        _rec(1, DM + body, good_hl),
        _rec(2, "too short", good_hl),                    # bài quá ngắn -> bị lọc (train)
        _rec(3, body.replace("topic", "subject"), "tiny"),    # tóm tắt quá ngắn -> bị lọc (train)
    ])
    _w(inp / "cnn_dm_validation.jsonl", [_rec(4, "short val article", "tiny")])       # val không bị lọc
    _w(inp / "cnn_dm_test.jsonl", [_rec(5, DM + body, good_hl)])                      # trùng bài 1 -> rò rỉ
    for s in ("train", "validation", "test"):
        _w(inp / f"govuk_{s}.jsonl", [_rec(f"g{s}", "Gov text.\nSecond paragraph here.", "One line summary here ok")])
    cfg = yaml.safe_load(open("configs/preprocess.yaml"))
    cfg.update(input_dir=str(inp), output_dir=str(tmp_path / "out"), tables_dir=str(tmp_path / "tab"))
    cp = tmp_path / "c.yaml"; cp.write_text(yaml.dump(cfg))

    df = bc.main(str(cp))
    tr = bc.read_jsonl(tmp_path / "out" / "cnn_dm_train.jsonl")
    assert [r["id"] for r in tr] == ["1"]                                  # chỉ còn bài tốt
    assert not tr[0]["article"].startswith("By .") and tr[0]["article"].startswith("Sentence number 0")
    assert "\n" in tr[0]["highlights"]                                      # giữ xuống dòng giữa các câu
    assert len(bc.read_jsonl(tmp_path / "out" / "cnn_dm_validation.jsonl")) == 1
    gv = bc.read_jsonl(tmp_path / "out" / "govuk_train.jsonl")[0]
    assert "\n" not in gv["article"] and gv["source"] == "govuk"
    row = df[(df.source == "cnn_dm") & (df.split == "train")].iloc[0]
    assert row.n_input == 3 and row.n_output == 1 and row.n_removed_by_filter == 2 and row.n_changed_by_cleaning >= 1
    leak = (tmp_path / "tab" / "leakage_check.csv").read_text()
    assert "cnn_dm,test,1" in leak
