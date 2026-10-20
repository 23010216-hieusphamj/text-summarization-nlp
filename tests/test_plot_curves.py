import pytest

pytest.importorskip("matplotlib")
from src.analysis.plot_curves import plot_run
from src.train_utils import append_row


def test_plot_run_creates_png(tmp_path):
    for e, (tl, vl) in enumerate([(5.0, 4.8), (4.2, 4.5), (3.9, 4.6)], start=1):
        append_row(tmp_path / "r_log.csv", {"epoch": e, "train_loss": tl, "val_loss": vl})
    for s in range(1, 8):
        append_row(tmp_path / "r_steps.csv", {"step": s * 50, "epoch": 1, "loss": 5.0 / s})
    out = plot_run("r", log_dir=str(tmp_path), out_dir=str(tmp_path / "fig"))
    assert out.exists() and out.stat().st_size > 1000


def test_plot_run_without_step_log(tmp_path):
    append_row(tmp_path / "r_log.csv", {"epoch": 1, "train_loss": 5.0, "val_loss": 4.8})
    assert plot_run("r", log_dir=str(tmp_path), out_dir=str(tmp_path / "fig")).exists()
