import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from src import hf_train_utils as hu


def _cfg():
    return yaml.safe_load(open("configs/bart_cnndm_in512.yaml", encoding="utf-8"))


def test_shipped_configs_are_valid():
    for f in ("configs/bart_cnndm_in512.yaml", "configs/t5small_cnndm_in512.yaml"):
        hu.validate_cfg(yaml.safe_load(open(f, encoding="utf-8")))


def test_validate_cfg_rejects_missing_and_bad_values():
    c = _cfg()
    for path in (("train", "lr"), ("val_gen", "no_repeat_ngram_size"), ("model_name",)):
        bad = copy.deepcopy(c)
        d = bad
        for k in path[:-1]:
            d = d[k]
        del d[path[-1]]
        with pytest.raises(ValueError):
            hu.validate_cfg(bad)
    bad = copy.deepcopy(c); bad["kind"] = "gpt"
    with pytest.raises(ValueError):
        hu.validate_cfg(bad)
    bad = copy.deepcopy(c); bad["best_by"] = "rouge"
    with pytest.raises(ValueError):
        hu.validate_cfg(bad)
    bad = copy.deepcopy(c); bad["train"]["warmup_ratio"] = 1.0
    with pytest.raises(ValueError):
        hu.validate_cfg(bad)


def test_lr_lambda_warmup_peak_and_linear_decay_to_zero():
    f = [hu.lr_lambda(s, 10, 100) for s in range(100)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0) and f[10] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(f[10:], f[11:])) and f[-1] == pytest.approx(1 / 90)
    assert hu.lr_lambda(100, 10, 100) == 0.0 and hu.lr_lambda(0, 0, 50) == 1.0


def test_make_steps_cover_all_examples_once_and_are_reproducible():
    steps = hu.make_steps(23, 4, 3, seed=42, epoch=1)
    flat = np.concatenate([m for s in steps for m in s])
    assert sorted(flat.tolist()) == list(range(23))
    assert len(steps) == hu.steps_per_epoch(23, 4, 3) == 2 and [len(s) for s in steps] == [3, 3]
    again = hu.make_steps(23, 4, 3, seed=42, epoch=1)
    assert all(np.array_equal(a, b) for s, t in zip(steps, again) for a, b in zip(s, t))
    other = hu.make_steps(23, 4, 3, seed=42, epoch=2)
    assert not np.array_equal(np.concatenate([m for s in steps for m in s]), np.concatenate([m for s in other for m in s]))


def test_steps_per_epoch_rounds_up():
    assert hu.steps_per_epoch(19895, 8, 2) == 1244          # ceil(ceil(19895/8)/2) = ceil(2487/2)
    assert hu.steps_per_epoch(16, 4, 2) == 2 and hu.steps_per_epoch(17, 4, 2) == 3


def test_val_subset_indices_fixed_sorted_and_capped():
    a = hu.val_subset_indices(1000, 300, 42)
    assert a == hu.val_subset_indices(1000, 300, 42) and a == sorted(a) and len(set(a)) == 300
    assert hu.val_subset_indices(50, 300) == list(range(50))


def test_selection_score_and_no_decay_and_generate_kwargs():
    assert hu.selection_score({"rouge1": 0.4, "rouge2": 0.2, "rougeLsum": 0.3}) == pytest.approx(30.0)
    assert hu.is_no_decay("model.encoder.layers.0.self_attn.k_proj.bias")
    assert hu.is_no_decay("model.encoder.layernorm_embedding.weight") and hu.is_no_decay("encoder.block.0.layer_norm.weight")
    assert not hu.is_no_decay("model.encoder.layers.0.fc1.weight")
    kw = hu.generate_kwargs({"num_beams": 4, "max_length": 128, "min_length": 0, "length_penalty": 1.0, "no_repeat_ngram_size": 0})
    assert kw["no_repeat_ngram_size"] == 0 and kw["min_length"] == 0 and kw["do_sample"] is False and kw["early_stopping"] is False


def test_truncate_csv_after_removes_only_later_rows(tmp_path):
    p = tmp_path / "log.csv"
    pd.DataFrame({"epoch": [1, 2, 3], "v": [1.0, 2.0, 3.0]}).to_csv(p, index=False)
    assert hu.truncate_csv_after(p, "epoch", 2) == 1
    assert pd.read_csv(p)["epoch"].tolist() == [1, 2]
    assert hu.truncate_csv_after(p, "epoch", 2) == 0 and hu.truncate_csv_after(tmp_path / "none.csv", "epoch", 1) == 0


def test_gen_tag_is_unique_per_config_and_stable():
    base = {"num_beams": 4, "max_length": 128, "min_length": 0, "length_penalty": 1.0, "no_repeat_ngram_size": 3}
    assert hu.gen_tag(base) == "b4-lp1-nr3-min0-max128"
    assert hu.gen_tag({**base, "early_stopping": True}) == "b4-lp1-nr3-min0-max128-es"
    variants = [{**base, **d} for d in ({}, {"num_beams": 1}, {"length_penalty": 2.0}, {"no_repeat_ngram_size": 0},
                                        {"min_length": 30}, {"max_length": 64})]
    assert len({hu.gen_tag(v) for v in variants}) == len(variants)


def test_scores_path_follows_prediction_naming_and_scores_frame_columns(tmp_path):
    assert hu.scores_path(tmp_path, "cnn_dm", "bart__b4", "test") == tmp_path / "cnn_dm_bart__b4.csv"
    assert hu.scores_path(tmp_path, "cnn_dm", "bart__b4", "validation") == tmp_path / "cnn_dm_bart__b4_validation.csv"
    df = hu.scores_frame(["a", "b"], {"rouge1": np.array([0.5, 0.25]), "bertscore_f1": [0.1, 0.2]},
                         ["one two three", "x"], ["one two", "y z"])
    assert df["id"].tolist() == ["a", "b"] and df["pred_words"].tolist() == [3, 1] and df["ref_words"].tolist() == [2, 2]
    assert df["rouge1"].tolist() == [0.5, 0.25] and "bertscore_f1" in df


def _train_configs():
    import glob
    out = {}
    for f in sorted(glob.glob("configs/*.yaml")):
        c = yaml.safe_load(open(f, encoding="utf-8"))
        if isinstance(c, dict) and "run_name" in c and "train" in c:
            out[f] = c
    return out


def test_all_training_configs_valid_unique_and_consistent_with_d7_d10():
    cfgs = _train_configs()
    assert len(cfgs) >= 7                                                     # A, B, C, D, E (2 mức lr), T5-small GOV.UK
    names = [c["run_name"] for c in cfgs.values()]
    assert len(set(names)) == len(names)                                      # run_name không trùng (log/checkpoint không đè nhau)
    for f, c in cfgs.items():
        hu.validate_cfg(c)
        assert c["max_target"] == (128 if c["source"] == "cnn_dm" else 64), f           # D7
        assert c["train"]["batch_size"] * c["train"]["grad_accum"] == 16, f              # batch hiệu dụng 16 (D10)
        assert c["val_gen"]["max_length"] == c["max_target"], f                          # độ dài giải mã khớp max_target
        assert c["train"]["seed"] == 42 and c["best_by"] == "S", f
        assert (c["kind"] == "t5") == (not c["train"]["amp"]), f                          # T5 fp32, BART fp16


def test_ablation_configs_differ_from_a_only_in_the_intended_keys():
    a = yaml.safe_load(open("configs/bart_cnndm_in512.yaml", encoding="utf-8"))

    def diff(b):
        keys = set()
        for k in set(a) | set(b):
            if k == "train":
                keys |= {f"train.{t}" for t in set(a[k]) | set(b[k]) if a[k].get(t) != b[k].get(t)}
            elif a.get(k) != b.get(k):
                keys.add(k)
        return keys

    for tag, lr in (("lr3e5", 3e-5), ("lr1e4", 1e-4)):
        b = yaml.safe_load(open(f"configs/bart_cnndm_in512_{tag}.yaml", encoding="utf-8"))
        assert diff(b) == {"run_name", "train.lr"} and b["train"]["lr"] == lr
    c = yaml.safe_load(open("configs/bart_cnndm_in1024.yaml", encoding="utf-8"))
    assert diff(c) == {"run_name", "max_input", "train.batch_size", "train.grad_accum", "train.eval_batch_size"}
    assert c["max_input"] == 1024


def test_select_tol_keeps_earlier_epoch_when_gain_below_tolerance():
    from src.train_utils import EarlyStopper
    st = EarlyStopper(3, min_delta=0.2)
    flags = [st.update(-s, e) for e, s in enumerate([33.645, 34.382, 34.237], start=1)]     # log thật của thí nghiệm A
    assert flags == [True, True, False] and st.best_epoch == 2
    st = EarlyStopper(3, min_delta=0.2)
    flags = [st.update(-s, e) for e, s in enumerate([34.0, 34.1, 34.35], start=1)]
    assert flags == [True, False, True] and st.best_epoch == 3                              # 34,1 chưa vượt 34,0 + 0,2; 34,35 thì có
