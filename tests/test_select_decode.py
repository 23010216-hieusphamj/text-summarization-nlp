import pandas as pd
import pytest
import yaml

from src.analysis import select_decode as sd
from src.hf_train_utils import gen_tag

BASE = {"num_beams": 4, "max_length": 128, "min_length": 0, "length_penalty": 1.0, "no_repeat_ngram_size": 3, "early_stopping": False}


def _df(items):
    """items: [(thay đổi so với gốc, S thang 0-1)] -> bảng như hf_*_validation.csv."""
    rows = []
    for change, s in items:
        g = {**BASE, **change}
        rows.append({**g, "gen_tag": gen_tag(g), "rouge1": s, "rouge2": s, "rougeLsum": s})
    return pd.DataFrame(rows)


def test_one_at_a_time_configs_count_unique_and_single_change():
    factors = {"num_beams": [1, 2, 8], "length_penalty": [0.6, 1.5, 2.0], "max_length": [64, 96], "min_length": [56],
               "no_repeat_ngram_size": [0]}
    gens = sd.one_at_a_time_configs(BASE, factors)
    assert len(gens) == 1 + 3 + 3 + 2 + 1 + 1 and gens[0] == BASE
    assert len({gen_tag(g) for g in gens}) == len(gens)
    assert all(len(sd.diff_keys(g, BASE)) == 1 for g in gens[1:])


def test_one_at_a_time_skips_values_equal_to_base_and_rejects_unknown_factor():
    assert len(sd.one_at_a_time_configs(BASE, {"num_beams": [4, 1, 1]})) == 2
    with pytest.raises(ValueError):
        sd.one_at_a_time_configs(BASE, {"temperature": [0.5]})


def test_select_prefers_base_when_gain_below_tolerance_and_variant_when_above():
    r = sd.select_decode(_df([({}, 0.340), ({"num_beams": 8}, 0.341)]), BASE)          # +0,1 điểm < 0,2 -> hòa -> giữ gốc
    assert r["chosen_tag"] == gen_tag(BASE) and r["best_tag"] == gen_tag({**BASE, "num_beams": 8})
    r = sd.select_decode(_df([({}, 0.340), ({"num_beams": 8}, 0.345)]), BASE)          # +0,5 điểm -> chọn biến thể
    assert r["chosen_gen"]["num_beams"] == 8 and r["chosen_tag"] != gen_tag(BASE)


def test_select_among_tied_variants_picks_higher_s_and_requires_base_row():
    r = sd.select_decode(_df([({}, 0.330), ({"length_penalty": 0.6}, 0.3400), ({"max_length": 96}, 0.3390)]), BASE)
    assert r["chosen_gen"]["length_penalty"] == 0.6                                    # cùng 1 tham số đổi, S cao hơn
    with pytest.raises(ValueError):
        sd.select_decode(_df([({"num_beams": 8}, 0.34)]), BASE)


def test_tolerance_is_relative_to_best_not_pairwise():
    # base 0,3380; a 0,3395; b 0,3400: base thấp hơn best 0,20 điểm (không < 0,2) nên KHÔNG hòa dù a-base < 0,2
    r = sd.select_decode(_df([({}, 0.3380), ({"num_beams": 2}, 0.3395), ({"num_beams": 8}, 0.3400)]), BASE)
    assert r["chosen_tag"] != gen_tag(BASE)


def test_combine_winners_merges_only_parameters_that_beat_base_by_tolerance():
    df = _df([({}, 0.340), ({"num_beams": 8}, 0.345), ({"length_penalty": 0.6}, 0.3405),
              ({"max_length": 96}, 0.348), ({"max_length": 64}, 0.330)])
    combo = sd.combine_winners(df, BASE)
    assert combo["num_beams"] == 8 and combo["max_length"] == 96 and combo["length_penalty"] == 1.0
    assert sd.combine_winners(_df([({}, 0.34), ({"num_beams": 8}, 0.3405)]), BASE) is None


def test_gen_flags_string():
    s = sd.gen_flags({**BASE, "num_beams": 8, "length_penalty": 0.6})
    assert s == "--num_beams 8 --max_length 128 --min_length 0 --length_penalty 0.6 --no_repeat_ngram 3 --early_stopping false"


def test_decode_ablation_yaml_is_consistent_with_training_configs():
    cfgs = yaml.safe_load(open("configs/decode_ablation.yaml", encoding="utf-8"))
    assert {"cnn_dm", "govuk", "cnn_dm_t5"} <= set(cfgs)
    for key, e in cfgs.items():
        tc = yaml.safe_load(open(e["run_config"], encoding="utf-8"))
        assert e["ckpt_run"] == tc["run_name"], key
        for k in sd.CORE_KEYS:
            assert float(e["base"][k]) == float(tc["val_gen"][k]), (key, k)           # gốc = val_gen đã dùng ở 7a
        assert bool(e["base"].get("early_stopping", False)) == bool(tc["val_gen"].get("early_stopping", False))
        assert set(e["factors"]) <= set(sd.CORE_KEYS)
        gens = sd.one_at_a_time_configs(e["base"], e["factors"])
        assert len(gens) > 3 and all(g["max_length"] <= tc["max_target"] for g in gens), key    # không vượt max_target của nguồn
