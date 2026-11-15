"""Test ablation giải mã bằng mô hình BART NHỎ ngẫu nhiên (CPU). Cần torch + transformers + rouge_score."""
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("rouge_score")

import torch  # noqa: E402

from src import ablate_decode as ad  # noqa: E402
from src.analysis.select_decode import one_at_a_time_configs  # noqa: E402
from src.decode_utils import table_path  # noqa: E402
from tests.test_generate_hf import EV, GEN  # noqa: E402
from tests.test_train_hf import StubTok, data, make_model  # noqa: E402


def _setup(tmp_path):
    ex, refs = data(8)
    recs = [{"id": e["id"], "article": " ".join(str(t) for t in e["input_ids"] if t > 2), "highlights": r} for e, r in zip(ex, refs)]
    dirs = {k: tmp_path / k for k in ("predictions", "tables", "scores")}
    return ex, recs, dirs


def _run(tmp_path, gens, existing=frozenset(), force=False):
    ex, recs, dirs = _setup(tmp_path)
    return ad.run_ablation(make_model(), StubTok(), ex, recs, gens, EV, source="x", system="sys", split_tag="validation",
                           dirs=dirs, batch_size=4, device=torch.device("cpu"), use_amp=False, existing_tags=set(existing),
                           force=force)


def test_runs_each_config_once_skips_existing_and_reports_a_choice(tmp_path):
    gens = one_at_a_time_configs(GEN, {"num_beams": [2], "max_length": [6]})
    done = _run(tmp_path, gens)
    assert len(done) == 3
    tab = pd.read_csv(table_path(tmp_path / "tables", "hf", "x", "validation"))
    assert len(tab) == 3 and set(tab["split"]) == {"validation"}
    assert _run(tmp_path, gens, existing=set(tab["gen_tag"])) == []                  # đã có hết -> bỏ qua
    assert len(_run(tmp_path, gens, existing=set(tab["gen_tag"]), force=True)) == 3  # --force chạy lại
    res = ad.report(tmp_path / "tables", "x", "sys", "validation", GEN, "t")
    assert res["table"]["is_chosen"].sum() == 1 and (tmp_path / "tables" / "decode_selection_t.csv").exists()
