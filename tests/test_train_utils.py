import csv

from src.train_utils import EarlyStopper, append_row


def test_early_stopper_improves_then_stops():
    es = EarlyStopper(patience=2)
    assert es.update(1.0, 1) is True
    assert es.update(0.8, 2) is True and es.best_epoch == 2
    assert es.update(0.9, 3) is False and not es.should_stop
    assert es.update(0.85, 4) is False and es.should_stop
    assert es.best == 0.8 and es.best_epoch == 2


def test_early_stopper_state_roundtrip():
    es = EarlyStopper(patience=3)
    es.update(1.0, 1); es.update(1.2, 2)
    es2 = EarlyStopper(patience=3)
    es2.load_state_dict(es.state_dict())
    assert (es2.best, es2.best_epoch, es2.bad) == (1.0, 1, 1)


def test_append_row_writes_header_once(tmp_path):
    p = tmp_path / "sub" / "log.csv"
    append_row(p, {"epoch": 1, "loss": 2.5})
    append_row(p, {"epoch": 2, "loss": 2.0})
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
