"""Chọn tham số giải mã trên VALIDATION theo quy tắc D10 (không dùng torch, dùng được cả trên máy không có GPU).

  * Đổi từng tham số một quanh cấu hình gốc (`one_at_a_time_configs`), không lưới đầy đủ.
  * Điểm chọn S = trung bình (ROUGE-1, ROUGE-2, ROUGE-Lsum) trên validation. Các cấu hình có S thấp hơn S tốt nhất dưới
    0,2 điểm (thang 0-100) được coi là HÒA; trong nhóm hòa chọn cấu hình ít tham số bị đổi nhất so với cấu hình gốc
    (gần mặc định/đơn giản hơn), còn hòa nữa thì S cao hơn. Ngưỡng so với S TỐT NHẤT, không so từng cặp.
  * Tùy chọn `combine_winners` (mở rộng D10, ghi rõ khi dùng): ghép các giá trị thắng của từng tham số thành MỘT cấu hình
    để chạy thêm; cấu hình ghép chỉ được chọn nếu thắng theo cùng quy tắc.
"""
from __future__ import annotations

import pandas as pd

from src.hf_train_utils import gen_tag

CORE_KEYS = ("num_beams", "max_length", "min_length", "length_penalty", "no_repeat_ngram_size")
TOL = 0.002                       # 0,2 điểm trên thang 0-1 của bảng CSV
FLAG_NAMES = {"num_beams": "--num_beams", "max_length": "--max_length", "min_length": "--min_length",
              "length_penalty": "--length_penalty", "no_repeat_ngram_size": "--no_repeat_ngram"}


def gen_of_row(row) -> dict:
    """Cấu hình giải mã của một dòng bảng hf_*.csv."""
    g = {k: row[k] for k in CORE_KEYS}
    g["num_beams"], g["max_length"], g["min_length"] = int(g["num_beams"]), int(g["max_length"]), int(g["min_length"])
    g["length_penalty"], g["no_repeat_ngram_size"] = float(g["length_penalty"]), int(g["no_repeat_ngram_size"])
    g["early_stopping"] = bool(row["early_stopping"]) if "early_stopping" in row and pd.notna(row["early_stopping"]) else False
    return g


def diff_keys(gen: dict, base: dict) -> list[str]:
    keys = [k for k in CORE_KEYS if float(gen[k]) != float(base[k])]
    if bool(gen.get("early_stopping", False)) != bool(base.get("early_stopping", False)):
        keys.append("early_stopping")
    return keys


def one_at_a_time_configs(base: dict, factors: dict) -> list[dict]:
    """[cấu hình gốc] + mỗi giá trị của mỗi tham số, đổi MỘT tham số một lần, bỏ trùng với gốc/lẫn nhau."""
    unknown = set(factors) - set(CORE_KEYS)
    if unknown:
        raise ValueError(f"Tham số không hợp lệ trong factors: {sorted(unknown)}")
    out, seen = [dict(base)], {gen_tag(base)}
    for key, values in factors.items():
        for v in values:
            g = dict(base)
            g[key] = v
            t = gen_tag(g)
            if t not in seen:
                seen.add(t)
                out.append(g)
    return out


def scored(df: pd.DataFrame, base: dict) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    d["S"] = (d["rouge1"] + d["rouge2"] + d["rougeLsum"]) / 3
    d["n_changed"] = [len(diff_keys(gen_of_row(r), base)) for _, r in d.iterrows()]
    return d


def select_decode(df: pd.DataFrame, base: dict, tol: float = TOL) -> dict:
    """df: các dòng của MỘT hệ thống trên validation (cột gen_tag, tham số giải mã, rouge1/2/Lsum thang 0-1)."""
    d = scored(df, base)
    if d.empty:
        raise ValueError("Không có dòng nào để chọn")
    if not (d["n_changed"] == 0).any():
        raise ValueError("Chưa có dòng của cấu hình gốc (chạy cấu hình gốc trước)")
    best = d["S"].max()
    ties = d[best - d["S"] < tol]
    chosen = ties.sort_values(["n_changed", "S"], ascending=[True, False]).iloc[0]
    d["is_best"] = d["S"] == best
    d["is_chosen"] = d["gen_tag"] == chosen["gen_tag"]
    return {"table": d, "best_tag": d.loc[d["S"].idxmax(), "gen_tag"], "chosen_tag": chosen["gen_tag"],
            "chosen_gen": gen_of_row(chosen), "base_S": float(d.loc[d["n_changed"] == 0, "S"].iloc[0]), "best_S": float(best)}


def combine_winners(df: pd.DataFrame, base: dict, tol: float = TOL) -> dict | None:
    """Ghép giá trị thắng của từng tham số (thắng cấu hình gốc ít nhất `tol`) thành một cấu hình; None nếu không có tham số nào thắng."""
    d = scored(df, base)
    base_S = float(d.loc[d["n_changed"] == 0, "S"].iloc[0])
    combo = dict(base)
    for key in CORE_KEYS:
        rows = [r for _, r in d.iterrows() if diff_keys(gen_of_row(r), base) == [key]]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["S"])
        if best["S"] - base_S >= tol:
            combo[key] = gen_of_row(best)[key]
    return None if diff_keys(combo, base) == [] else combo


def gen_flags(gen: dict) -> str:
    """Chuỗi tham số dòng lệnh cho generate_hf (để chạy test một lần với cấu hình đã chốt)."""
    parts = [f"{FLAG_NAMES[k]} {gen[k]}" for k in CORE_KEYS]
    parts.append(f"--early_stopping {'true' if gen.get('early_stopping', False) else 'false'}")
    return " ".join(parts)
