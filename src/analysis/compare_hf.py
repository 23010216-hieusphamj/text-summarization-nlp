"""So sánh ghép cặp các hệ thống trên TEST (D5/D10): bootstrap GHÉP CẶP 2.000 lần, và điểm theo độ dài bài (≤512 / >512 token).

Điểm từng mẫu lấy từ:
  * mô hình HF (BART/T5): results/scores/{source}_{system}__{gen_tag}.csv do generate_hf ghi sẵn (có BERTScore);
  * Lead-k: tính lại trực tiếp từ bài (xác định, không cần file dự đoán);
  * Seq2Seq (GĐ5): file dự đoán results/predictions/{source}_seq2seq-beam4_nr3.jsonl (sinh lại bằng decode_seq2seq).
Điểm của Lead/Seq2Seq được cache vào results/scores/ (thêm BERTScore bằng --bertscore).

    python -m src.analysis.compare_hf --data_dir <clean> --pred_dir <predictions> --scores_dir <scores> --tables_dir <tables> --bertscore

Khác biệt (A - B) có khoảng tin cậy 95% không chứa 0 -> `significant`. KHÔNG hiệu chỉnh đa so sánh (nhiều cặp, nhiều chỉ số):
đọc `significant` như chỉ dẫn, không như kết luận chặt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.compare_baselines import read_jsonl
from src.baselines.lead import join_sentences, lead_k
from src.evaluate import bootstrap_ci, rouge_per_sample
from src.hf_train_utils import scores_path

METRICS = ("rouge1", "rouge2", "rougeLsum", "bertscore_f1")


def per_sample_scores(ids: list[str], preds: list[str], refs: list[str], ev: dict, bertscore: bool) -> pd.DataFrame:
    """Điểm ROUGE (+ BERTScore) từng mẫu theo cài đặt D5 (lấy từ mục evaluate của configs/baselines.yaml)."""
    m = rouge_per_sample(preds, refs, ev["rouge_stemmer"])
    if bertscore:
        from src.evaluate import bertscore_per_sample
        m["bertscore_f1"] = bertscore_per_sample(preds, refs, ev["bertscore_model"], ev["bertscore_rescale"],
                                                 ev["bertscore_batch_size"])
    df = pd.DataFrame({"id": ids})
    for k, v in m.items():
        df[k] = np.asarray(v, dtype=float)
    return df


def load_system_scores(spec: dict, source: str, split: str, records: list[dict], dirs: dict, ev: dict,
                       bertscore: bool) -> pd.DataFrame | None:
    """Điểm từng mẫu của một hệ thống (đọc cache nếu có, không thì tính). None nếu chưa có dữ liệu."""
    kind = spec["type"]
    if kind == "hf":
        p = scores_path(dirs["scores"], source, f"{spec['system']}__{spec['gen_tag']}", split)
        return pd.read_csv(p) if p.exists() else None
    name = f"lead-{spec['k']}" if kind == "lead" else Path(spec["file"]).stem.removeprefix(f"{source}_")
    cache = scores_path(dirs["scores"], source, name, split)
    if cache.exists():
        df = pd.read_csv(cache)
        if not bertscore or "bertscore_f1" in df.columns:
            return df
    refs = {r["id"]: r["highlights"] for r in records}
    if kind == "lead":
        preds = {r["id"]: join_sentences(r["article"], lead_k(r["article"], spec["k"])) for r in records}
    else:
        f = Path(dirs["predictions"]) / spec["file"]
        if not f.exists():
            return None
        preds = {p["id"]: p["prediction"] for p in read_jsonl(f)}
    ids = [r["id"] for r in records if r["id"] in preds]
    df = per_sample_scores(ids, [preds[i] for i in ids], [refs[i] for i in ids], ev, bertscore)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def align(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Chỉ giữ các id chung của MỌI hệ thống, cùng thứ tự (theo hệ thống đầu tiên) và đặt id làm chỉ mục."""
    if not frames:
        return {}
    first = next(iter(frames.values()))["id"].tolist()
    common = set(first)
    for df in frames.values():
        common &= set(df["id"])
    ids = [i for i in first if i in common]
    return {n: df.set_index("id").loc[ids] for n, df in frames.items()}


def paired_diff_table(frames: dict[str, pd.DataFrame], pairs, metrics=METRICS, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap ghép cặp cho hiệu (A - B) từng chỉ số; frames đã `align` (cùng chỉ mục)."""
    rows = []
    for a, b in pairs:
        if a not in frames or b not in frames:
            continue
        for m in metrics:
            if m not in frames[a].columns or m not in frames[b].columns:
                continue
            d = frames[a][m].to_numpy() - frames[b][m].to_numpy()
            mean, lo, hi = bootstrap_ci(d, n_boot, seed)
            rows.append({"A": a, "B": b, "metric": m, "n": len(d), "mean_A": float(frames[a][m].mean()),
                         "mean_B": float(frames[b][m].mean()), "diff": mean, "diff_lo": lo, "diff_hi": hi,
                         "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def by_group_tables(frames: dict[str, pd.DataFrame], groups: pd.Series, pairs, metrics=METRICS, n_boot: int = 2000,
                    seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Điểm trung bình (kèm CI) theo nhóm (vd. '<=512' / '>512') và hiệu ghép cặp trong từng nhóm."""
    groups = groups.reindex(next(iter(frames.values())).index)
    means, diffs = [], []
    for g in sorted(groups.dropna().unique()):
        sub = {n: df[groups == g] for n, df in frames.items()}
        for n, df in sub.items():
            row = {"system": n, "group": g, "n": len(df)}
            for m in metrics:
                if m in df.columns:
                    row[m], row[f"{m}_lo"], row[f"{m}_hi"] = bootstrap_ci(df[m].to_numpy(), n_boot, seed)
            means.append(row)
        d = paired_diff_table(sub, pairs, metrics, n_boot, seed)
        d.insert(0, "group", g)
        diffs.append(d)
    return pd.DataFrame(means), (pd.concat(diffs, ignore_index=True) if diffs else pd.DataFrame())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/compare.yaml")
    ap.add_argument("--eval_config", default="configs/baselines.yaml")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred_dir", default="results/predictions")
    ap.add_argument("--scores_dir", default="results/scores")
    ap.add_argument("--tables_dir", default="results/tables")
    ap.add_argument("--lengths_dir", default=None, help="thư mục chứa token_lengths_{source}_test.csv (mặc định: tables_dir)")
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--bertscore", action="store_true", help="tính thêm BERTScore từng mẫu cho Lead/Seq2Seq (cần GPU để nhanh)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    ev = yaml.safe_load(open(a.eval_config, encoding="utf-8"))["evaluate"]
    dirs = {"scores": Path(a.scores_dir), "predictions": Path(a.pred_dir)}
    out, lengths_dir = Path(a.tables_dir), Path(a.lengths_dir or a.tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    for source in a.sources or [s for s in cfg if s != "curves"]:
        entry = cfg[source]
        records = read_jsonl(Path(a.data_dir) / f"{source}_{a.split}.jsonl")
        frames = {}
        for name, spec in entry["systems"].items():
            df = load_system_scores(spec, source, a.split, records, dirs, ev, a.bertscore)
            if df is None:
                print(f"[{source}] thiếu dữ liệu của '{name}', bỏ qua")
            else:
                frames[name] = df
        if len(frames) < 2:
            print(f"[{source}] không đủ hệ thống để so sánh")
            continue
        frames = align(frames)
        n = len(next(iter(frames.values())))
        pairs = [tuple(p) for p in entry["pairs"]]
        res = paired_diff_table(frames, pairs, METRICS, a.n_boot, a.seed)
        res.to_csv(out / f"hf_paired_tests_{source}.csv", index=False)
        show = res.copy()
        for c in ("mean_A", "mean_B", "diff", "diff_lo", "diff_hi"):
            show[c] = 100 * show[c]
        print(f"\n=== {source}: bootstrap ghép cặp {a.n_boot} lần trên {n} bài (điểm x100; chưa hiệu chỉnh đa so sánh) ===")
        print(show.round(2).to_string(index=False))
        lp = lengths_dir / f"token_lengths_{source}_{a.split}.csv"
        if lp.exists():
            groups = pd.read_csv(lp).set_index("id")["group_512"]
            sys_len = {k: v for k, v in frames.items() if k in entry.get("by_length_systems", frames)}
            means, diffs = by_group_tables(sys_len, groups, [tuple(p) for p in entry.get("by_length_pairs", [])],
                                           METRICS, a.n_boot, a.seed)
            means.to_csv(out / f"hf_by_length_{source}.csv", index=False)
            diffs.to_csv(out / f"hf_by_length_paired_{source}.csv", index=False)
            print(f"\n--- {source}: điểm theo độ dài bài (token BART) ---")
            cols = ["system", "group", "n"] + [m for m in METRICS if m in means.columns]
            print((means[cols].assign(**{m: 100 * means[m] for m in METRICS if m in means.columns})).round(2).to_string(index=False))
            if not diffs.empty:
                d2 = diffs.copy()
                for c in ("mean_A", "mean_B", "diff", "diff_lo", "diff_hi"):
                    d2[c] = 100 * d2[c]
                print(d2[["group", "A", "B", "metric", "n", "diff", "diff_lo", "diff_hi", "significant"]].round(2).to_string(index=False))
        else:
            print(f"[{source}] chưa có {lp.name}: bỏ qua phần điểm theo độ dài (tạo bằng python -m src.data.hf_seq2seq_data)")


if __name__ == "__main__":
    main()
