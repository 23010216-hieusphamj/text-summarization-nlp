"""Tiện ích huấn luyện không phụ thuộc torch: dừng sớm và ghi log CSV."""
from __future__ import annotations

import csv
from pathlib import Path


class EarlyStopper:
    """Dừng khi chỉ số (val loss, càng nhỏ càng tốt) không cải thiện sau `patience` epoch liên tiếp."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.bad = float("inf"), 0, 0

    def update(self, value: float, epoch: int) -> bool:
        """Trả về True nếu cải thiện (đáng lưu checkpoint tốt nhất)."""
        if value < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad = value, epoch, 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad": self.bad}

    def load_state_dict(self, d: dict) -> None:
        self.best, self.best_epoch, self.bad = d["best"], d["best_epoch"], d["bad"]


def append_row(path: str | Path, row: dict) -> None:
    """Thêm một dòng vào CSV; tự ghi tiêu đề nếu file chưa có (an toàn khi tiếp tục huấn luyện)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
