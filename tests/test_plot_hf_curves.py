import json

import pandas as pd
import yaml

from src.analysis import plot_hf_curves as pc


def _make_run(logs, name, best=2, epochs=3):
    pd.DataFrame({"epoch": list(range(1, epochs + 1)), "train_loss": [2.4, 2.1, 1.9][:epochs], "val_loss": [1.9, 1.8, 1.8][:epochs],
                  "val_S": [33.0, 34.0, 33.9][:epochs]}).to_csv(logs / f"{name}_log.csv", index=False)
    (logs / f"{name}_meta.json").write_text(json.dumps({"best_epoch": best}))


def test_plot_group_creates_png_for_several_runs(tmp_path):
    logs = tmp_path / "logs"; logs.mkdir()
    for n in ("r1", "r2"):
        _make_run(logs, n)
    p = pc.plot_group(["r1", "r2"], logs, tmp_path / "fig" / "g.png", "nhóm thử")
    assert p.exists() and p.stat().st_size > 5000


def test_plot_group_tolerates_missing_meta_and_unknown_best_epoch(tmp_path):
    logs = tmp_path / "logs"; logs.mkdir()
    _make_run(logs, "r1", best=99)                                            # best_epoch không có trong log
    pd.DataFrame({"epoch": [1, 2], "train_loss": [2.0, 1.9], "val_loss": [1.8, 1.8], "val_S": [30.0, 31.0]}).to_csv(logs / "r2_log.csv", index=False)
    assert pc.plot_group(["r1", "r2"], logs, tmp_path / "g.png", "t").exists()


def test_main_reads_groups_from_config(tmp_path):
    logs = tmp_path / "logs"; logs.mkdir()
    _make_run(logs, "r1")
    cp = tmp_path / "c.yaml"; cp.write_text(yaml.safe_dump({"curves": {"grp": ["r1"]}}))
    pc.main(["--config", str(cp), "--logs_dir", str(logs), "--out_dir", str(tmp_path / "out")])
    assert (tmp_path / "out" / "hf_curves_grp.png").exists()
