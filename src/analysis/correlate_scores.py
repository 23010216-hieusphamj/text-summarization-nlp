"""Tương quan từng mẫu giữa ROUGE và BERTScore trên TEST (GĐ7, D11).

Đọc điểm từng mẫu có sẵn trong results/scores/ (không chạy lại mô hình), tính:
  * Spearman (chính) và Pearson (phụ) cho các cặp trong configs/correlation.yaml, kèm CI 95% bootstrap trên mẫu;
  * hiệu hai hệ số trong cùng hệ thống (dùng chung các lần lấy mẫu lại -> hiệu ghép cặp);
  * kiểm tra ghép điểm theo `id` (id trùng, thiếu BERTScore, số mẫu khác kỳ vọng; tùy chọn so với data/clean).

    python -m src.analysis.correlate_scores --scores_dir results/scores --tables_dir results/tables [--data_dir <clean>]

Kết quả: results/tables/correlation_{source}.csv và correlation_{source}.md.
Lưu ý khi đọc: đây là tương quan TRÊN TỪNG MẪU của một hệ thống, không phải tương quan giữa các hệ thống;
S là trung bình (R1, R2, Rlsum) của từng mẫu. Hệ số cao không có nghĩa hai độ đo "đo cùng một thứ".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_jsonl(path: Path) -> list[dict]:
    """Đọc jsonl (tự viết để không kéo theo rouge-score từ src.evaluate)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

S_PARTS = ("rouge1", "rouge2", "rougeLsum")


def add_s(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột S = trung bình (rouge1, rouge2, rougeLsum) của từng mẫu (cùng định nghĩa với D10, nhưng theo mẫu)."""
    out = df.copy()
    out["S"] = out[list(S_PARTS)].mean(axis=1)
    return out


def _rank(a: np.ndarray) -> np.ndarray:
    """Hạng trung bình (xử lý hòa) bằng pandas, để khỏi phụ thuộc scipy."""
    return pd.Series(a).rank(method="average").to_numpy()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Hệ số Pearson; nan nếu một vector hằng hoặc ít hơn 3 mẫu."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Hệ số Spearman = Pearson của hạng trung bình."""
    return pearson(_rank(np.asarray(x, dtype=float)), _rank(np.asarray(y, dtype=float)))


def bootstrap_corr(df: pd.DataFrame, pairs: list, diffs: list, n_boot: int, seed: int) -> dict:
    """Spearman bootstrap cho từng cặp và từng hiệu. Mỗi lần lấy mẫu lại dùng chung cho mọi cặp (nên hiệu là ghép cặp).

    Trả {khóa: mảng n_boot giá trị}; khóa cặp là (a, b), khóa hiệu là ("diff", (a1, b1), (a2, b2)).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    cols = {c: df[c].to_numpy(dtype=float) for p in pairs for c in p}
    out = {tuple(p): np.empty(n_boot) for p in pairs}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        for a, b in pairs:
            out[(a, b)][i] = spearman(cols[a][idx], cols[b][idx])
    for d in diffs:
        p1, p2 = tuple(d[0]), tuple(d[1])
        out[("diff", p1, p2)] = out[p1] - out[p2]
    return out


def _ci(v: np.ndarray) -> tuple[float, float]:
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def check_scores(df: pd.DataFrame, n_expected: int | None = None, clean_ids: list[str] | None = None) -> list[str]:
    """Trả danh sách cảnh báo về việc ghép điểm (rỗng nếu ổn). Dự đoán 12 của D11 yêu cầu kiểm tra việc này."""
    w = []
    if df["id"].duplicated().any():
        w.append(f"có {int(df['id'].duplicated().sum())} id trùng")
    if n_expected is not None and len(df) != n_expected:
        w.append(f"số mẫu {len(df)} khác kỳ vọng {n_expected}")
    for c in ("bertscore_f1", *S_PARTS):
        if c not in df.columns:
            w.append(f"thiếu cột {c}")
        elif df[c].isna().any():
            w.append(f"cột {c} có {int(df[c].isna().sum())} giá trị thiếu")
    if clean_ids is not None:
        a, b = set(df["id"].astype(str)), set(map(str, clean_ids))
        if a != b:
            w.append(f"tập id khác data/clean: thiếu {len(b - a)}, thừa {len(a - b)}")
    return w


def correlation_table(source: str, cfg: dict, scores_dir: Path, clean_ids: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Bảng tương quan của mọi hệ thống của một nguồn. Trả (bảng, cảnh báo)."""
    pairs = [tuple(p) for p in cfg["pairs"]]
    diffs = [[tuple(d[0]), tuple(d[1])] for d in cfg.get("diffs", [])]
    src = cfg[source]
    rows, warns = [], []
    for name, spec in src["systems"].items():
        path = scores_dir / spec["file"]
        if not path.exists():
            warns.append(f"[{source}/{name}] không thấy {path}, bỏ qua")
            continue
        df = add_s(pd.read_csv(path))
        for w in check_scores(df, src.get("n_expected"), clean_ids):
            warns.append(f"[{source}/{name}] {w}")
        if df["bertscore_f1"].isna().all() if "bertscore_f1" in df.columns else True:
            warns.append(f"[{source}/{name}] không có BERTScore, bỏ qua")
            continue
        df = df.dropna(subset=["bertscore_f1", *S_PARTS]).reset_index(drop=True)
        boots = bootstrap_corr(df, pairs, diffs, cfg["n_boot"], cfg["seed"])
        for a, b in pairs:
            lo, hi = _ci(boots[(a, b)])
            rows.append({"source": source, "system": name, "pair": f"{a} ~ {b}", "n": len(df),
                         "spearman": spearman(df[a], df[b]), "spearman_lo": lo, "spearman_hi": hi,
                         "pearson": pearson(df[a], df[b]), "note": spec.get("note", "")})
        for p1, p2 in diffs:
            lo, hi = _ci(boots[("diff", p1, p2)])
            est = spearman(df[p1[0]], df[p1[1]]) - spearman(df[p2[0]], df[p2[1]])
            rows.append({"source": source, "system": name, "pair": f"({p1[0]}~{p1[1]}) - ({p2[0]}~{p2[1]})",
                         "n": len(df), "spearman": est, "spearman_lo": lo, "spearman_hi": hi,
                         "pearson": float("nan"), "note": "hiệu ghép cặp; CI không chứa 0 -> khác biệt"})
    return pd.DataFrame(rows), warns


def to_markdown(df: pd.DataFrame, pair: str = "S ~ bertscore_f1") -> str:
    """Bảng gọn cho README: Spearman [CI] của một cặp, theo hệ thống."""
    sub = df[df["pair"] == pair]
    lines = ["| Hệ thống | n | Spearman [CI 95%] | Pearson |", "|---|---|---|---|"]
    for _, r in sub.iterrows():
        lines.append(f"| {r['system']} | {int(r['n'])} | {r['spearman']:.3f} [{r['spearman_lo']:.3f}, {r['spearman_hi']:.3f}] | {r['pearson']:.3f} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/correlation.yaml")
    ap.add_argument("--scores_dir", default="results/scores")
    ap.add_argument("--tables_dir", default="results/tables")
    ap.add_argument("--data_dir", default=None, help="thư mục clean (jsonl); nếu có thì kiểm tra tập id test")
    ap.add_argument("--sources", nargs="+", default=None)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    out = Path(a.tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    for source in a.sources or [k for k in cfg if isinstance(cfg[k], dict) and "systems" in cfg[k]]:
        ids = None
        if a.data_dir:
            ids = [r["id"] for r in read_jsonl(Path(a.data_dir) / f"{source}_test.jsonl")]
        df, warns = correlation_table(source, cfg, Path(a.scores_dir), ids)
        for w in warns:
            print("[cảnh báo]", w)
        df.to_csv(out / f"correlation_{source}.csv", index=False)
        md = to_markdown(df)
        (out / f"correlation_{source}.md").write_text(md + "\n", encoding="utf-8")
        print(f"\n=== {source}: Spearman(S, BERTScore) theo hệ thống ===\n{md}")
        main_sys = cfg[source].get("main")
        if main_sys:
            print(f"\n--- hệ thống chính ({main_sys}) ---")
            print(df[df["system"] == main_sys][["pair", "spearman", "spearman_lo", "spearman_hi", "pearson"]].to_string(index=False))


if __name__ == "__main__":
    main()
