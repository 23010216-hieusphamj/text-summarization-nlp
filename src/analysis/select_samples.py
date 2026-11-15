"""Chọn mẫu cho phân tích lỗi và chấm tay + cờ tự động (GĐ7, D11). Không cần torch/mạng.

Đọc điểm TEST đã có (results/scores), dự đoán (results/predictions) và dữ liệu sạch (clean/{source}_test.jsonl); không chạy lại mô hình.

  1. Phân tích lỗi (16 mẫu): CNN/DM 4 S cao nhất + 4 S thấp nhất + 4 ngẫu nhiên phân vị 40-60%; GOV.UK 2 cao nhất + 2 thấp nhất.
  2. Chấm tay (40 mẫu): phân tầng theo tứ phân vị của S (CNN/DM 8/7/8/7, GOV.UK 3/2/3/2, Q1 = thấp nhất), seed 42;
     trộn thứ tự, gán mã mù H01.., chọn ngẫu nhiên 8 mẫu để chấm lại (mã mù R01..).
  3. Cờ tự động trên TOÀN test: lặp trigram mức từ, tỉ lệ số/từ viết hoa trong dự đoán không có trong bài (proxy ảo giác),
     tỉ lệ trigram mới; tổng hợp theo tứ phân vị của S (dự đoán 10, 11 của D11).

    python -m src.analysis.select_samples --data_dir <clean> --scores_dir <scores> --pred_dir <predictions> --out_dir <out>

Đầu ra: <out>/samples_key.csv, error_analysis_samples.{csv,md}, flags_*.csv, và <out>/blind/ (chỉ bài + tóm tắt, KHÔNG có điểm
hay tên hệ thống) cho phiếu chấm tay. Khi chấm tay, không mở file ngoài thư mục blind/ (chứa điểm tự động).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.baselines.lead import join_sentences, lead_k
from src.data.eda_utils import ngrams, novel_ngram_ratio, tokenize, word_count

S_PARTS = ("rouge1", "rouge2", "rougeLsum")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CAP = re.compile(r"\b[A-Z][A-Za-z'\-]*")


# ----------------------------------------------------------------------------- điểm S, tứ phân vị, chọn mẫu
def add_s(df: pd.DataFrame) -> pd.DataFrame:
    """S = trung bình (rouge1, rouge2, rougeLsum) của từng mẫu (D11)."""
    out = df.copy()
    out["S"] = out[list(S_PARTS)].mean(axis=1)
    return out


def quartile_of(df: pd.DataFrame) -> pd.Series:
    """Tứ phân vị 1..4 theo vị trí khi sắp S tăng dần (hòa: id tăng dần). Q1 = S thấp nhất; chỉ số theo `id`."""
    d = df.sort_values(["S", "id"], ascending=[True, True]).reset_index(drop=True)
    n = len(d)
    q = np.minimum(3, (4 * np.arange(n)) // n) + 1
    return pd.Series(q, index=d["id"].to_numpy())


def pick_error_samples(df: pd.DataFrame, n_top: int, n_bottom: int, n_mid: int, seed: int,
                       mid_range: tuple[float, float] = (0.40, 0.60)) -> pd.DataFrame:
    """Chọn mẫu phân tích lỗi. top: S cao nhất; bottom: S thấp nhất (hòa: id tăng dần); mid: ngẫu nhiên trong phân vị `mid_range`.

    Phân vị của mẫu: 0 = S thấp nhất, 1 = cao nhất (theo vị trí sau khi sắp tăng dần). Trả cột id, S, group, rank_desc (1 = S cao nhất).
    """
    d = add_s(df).sort_values(["S", "id"], ascending=[False, True]).reset_index(drop=True)
    n = len(d)
    d["rank_desc"] = np.arange(1, n + 1)
    asc = d.sort_values(["S", "id"], ascending=[True, True]).reset_index(drop=True)
    top = d.iloc[:n_top].assign(group="top")
    bottom = asc.iloc[:n_bottom].assign(group="bottom")
    taken = set(top["id"]) | set(bottom["id"])
    mid = d.iloc[0:0].assign(group="mid")
    if n_mid > 0:
        pct = (n - 1 - np.arange(n)) / max(n - 1, 1)                      # d đang giảm dần: phân vị = (n-1-i)/(n-1)
        lo, hi = mid_range
        cand = d[(pct >= lo) & (pct <= hi) & (~d["id"].isin(taken)).to_numpy()].reset_index(drop=True)
        if len(cand) < n_mid:
            raise ValueError(f"chỉ có {len(cand)} ứng viên trong phân vị {mid_range}, cần {n_mid}")
        pick = np.random.default_rng(seed).choice(len(cand), size=n_mid, replace=False)
        mid = cand.iloc[np.sort(pick)].assign(group="mid")
    out = pd.concat([top, mid, bottom], ignore_index=True)
    return out[["id", "S", "group", "rank_desc"]]


def pick_human_samples(df: pd.DataFrame, counts: list[int], seed: int) -> pd.DataFrame:
    """Chọn mẫu chấm tay phân tầng theo tứ phân vị S (counts[0] cho Q1 thấp nhất ... counts[3] cho Q4). Trả id, S, quartile."""
    d = add_s(df)
    q = quartile_of(d)
    d["quartile"] = d["id"].map(q)
    rng = np.random.default_rng(seed)
    parts = []
    for k, cnt in enumerate(counts, start=1):
        cand = d[d["quartile"] == k].sort_values("id").reset_index(drop=True)
        if len(cand) < cnt:
            raise ValueError(f"tứ phân vị Q{k} chỉ có {len(cand)} mẫu, cần {cnt}")
        parts.append(cand.iloc[np.sort(rng.choice(len(cand), size=cnt, replace=False))])
    return pd.concat(parts, ignore_index=True)[["id", "S", "quartile"]]


# ----------------------------------------------------------------------------- cờ tự động
def repeated_trigram(pred: str) -> bool:
    """Có ít nhất một trigram từ lặp lại trong dự đoán (cùng định nghĩa với decode_utils.repeated_trigram_rate)."""
    g = ngrams(tokenize(pred), 3)
    return len(g) != len(set(g))


def _norm(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'").lower()


def _find(token_lower: str, hay: str, number: bool) -> bool:
    pre, post = (r"(?<![\d.,])", r"(?![\d])") if number else (r"(?<!\w)", r"(?!\w)")
    return re.search(pre + re.escape(token_lower) + post, hay) is not None


def entities(pred: str) -> list[str]:
    """Số và từ viết hoa chữ đầu KHÔNG đứng đầu câu trong dự đoán (không trùng; thứ tự xuất hiện). Proxy, có nhiễu (D11)."""
    text = pred.replace("’", "'").replace("‘", "'")
    found: list[tuple[int, str]] = []
    for m in _NUM.finditer(text):
        t = m.group(0).rstrip(",.")
        if t:
            found.append((m.start(), t))
    for m in _CAP.finditer(text):
        prev = text[:m.start()].rstrip().rstrip("\"“”'([").rstrip()
        if prev == "" or prev[-1] in ".!?\n":
            continue                                                     # đầu câu: viết hoa vì ngữ pháp
        t = re.sub(r"'s$", "", m.group(0)).strip("-'")
        if t:
            found.append((m.start(), t))
    seen, out = set(), []
    for _, t in sorted(found):                                           # theo thứ tự xuất hiện
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def unsupported_stats(article: str, pred: str) -> dict:
    """Trong các thực thể (số, từ viết hoa) của dự đoán, bao nhiêu không xuất hiện trong bài (khớp nguyên từ, không phân biệt hoa thường).

    Số được so không tính dấu phẩy ngăn cách nghìn ("1,000" khớp "1000"). Trả n_entities, n_unsupported, unsupported_rate (nan nếu
    không có thực thể), unsupported (danh sách). Cờ có nhiễu ("10" so với "ten", đổi dạng tên): chỉ để người xác nhận.
    """
    art = _norm(article)
    art_nc = art.replace(",", "")
    ents = entities(pred)
    bad = []
    for e in ents:
        low = e.lower()
        is_num = bool(_NUM.fullmatch(e))
        ok = _find(low, art, is_num) or (is_num and _find(low.replace(",", ""), art_nc, True))
        if not ok:
            bad.append(e)
    return {"n_entities": len(ents), "n_unsupported": len(bad),
            "unsupported_rate": (len(bad) / len(ents)) if ents else float("nan"), "unsupported": bad}



def compute_flags(records: dict[str, dict], preds: dict[str, str], ids: list[str]) -> pd.DataFrame:
    """
    Tính các cờ tự động cho từng mẫu:
    - Lặp trigram.
    - Số và từ viết hoa không xuất hiện trong bài.
    - Tỉ lệ trigram mới.
    - Chênh lệch và tỉ lệ độ dài dự đoán so với tham chiếu.

    length_diff = pred_words - ref_words:
        Dương: dự đoán dài hơn tham chiếu.
        Âm: dự đoán ngắn hơn tham chiếu.
        0: hai bản có cùng số từ.

    length_ratio = pred_words / ref_words:
        > 1: dự đoán dài hơn tham chiếu.
        < 1: dự đoán ngắn hơn tham chiếu.
        = 1: độ dài bằng nhau.
        NaN: tham chiếu có 0 từ.
    """
    rows = []

    for i in ids:
        art = records[i]["article"]
        ref = records[i]["highlights"]
        pred = preds[i]

        # Cờ thực thể và lặp
        u = unsupported_stats(art, pred)

        # Đếm số từ bằng cùng hàm word_count của dự án
        pred_words = word_count(pred)
        ref_words = word_count(ref)
        article_words = word_count(art)

        # Chênh lệch có dấu và tỉ lệ độ dài
        length_diff = pred_words - ref_words
        length_ratio = (
            pred_words / ref_words if ref_words > 0 else float("nan")
        )

        rows.append({
            "id": i,
            "rep_trigram": int(repeated_trigram(pred)),
            "novel_trigram": novel_ngram_ratio(art, pred, 3),
            "n_entities": u["n_entities"],
            "n_unsupported": u["n_unsupported"],
            "unsupported_rate": u["unsupported_rate"],
            "unsupported": "; ".join(u["unsupported"]),
            "pred_words": pred_words,
            "ref_words": ref_words,
            "article_words": article_words,
            "length_diff": length_diff,
            "length_ratio": length_ratio,
        })

    return pd.DataFrame(rows)



def bootstrap_diff_means(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Hiệu trung bình mean(a) - mean(b) và CI 95% bootstrap (hai nhóm độc lập; bỏ nan)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)])
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def flag_summary(flags: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp cờ theo tứ phân vị S (Q1 thấp nhất ... Q4 cao nhất) và 'all'. Dự đoán 10, 11 của D11."""
    d = flags.merge(add_s(scores)[["id", "S"]], on="id", how="inner")
    d["quartile"] = d["id"].map(quartile_of(d))
    rows = []
    for name, g in [("all", d)] + [(f"Q{k}", d[d["quartile"] == k]) for k in (1, 2, 3, 4)]:
        has = g[g["n_entities"] > 0]
        rows.append({"group": name, "n": len(g), "rep_trigram_pct": 100 * g["rep_trigram"].mean(),
                     "novel_trigram_mean": g["novel_trigram"].mean(),
                     "unsupported_rate_mean": has["unsupported_rate"].mean() if len(has) else float("nan"),
                     "n_with_entities": len(has),
                     "pct_with_unsupported": 100 * (has["n_unsupported"] > 0).mean() if len(has) else float("nan")})
    out = pd.DataFrame(rows)
    q1, q4 = d[d["quartile"] == 1]["unsupported_rate"], d[d["quartile"] == 4]["unsupported_rate"]
    diff, lo, hi = bootstrap_diff_means(q1.to_numpy(), q4.to_numpy())
    out["unsupported_rate_Q1_minus_Q4"] = [diff] + [float("nan")] * 4
    out["Q1_minus_Q4_lo"] = [lo] + [float("nan")] * 4
    out["Q1_minus_Q4_hi"] = [hi] + [float("nan")] * 4
    return out


# ----------------------------------------------------------------------------- kiểm tra ghép dữ liệu
def check_alignment(scores: pd.DataFrame, preds: dict[str, str], records: dict[str, dict]) -> list[str]:
    """Lỗi nặng (id điểm không có trong dự đoán hay dữ liệu sạch) -> ValueError; trả danh sách cảnh báo nhẹ.

    Cảnh báo nhẹ: số dự đoán có pred_words khác cột pred_words của file điểm (dấu hiệu file dự đoán không phải lần chạy đã chấm điểm).
    """
    ids = list(scores["id"])
    miss_p = [i for i in ids if i not in preds]
    miss_r = [i for i in ids if i not in records]
    if miss_p or miss_r:
        raise ValueError(f"id trong file điểm không có trong dự đoán ({len(miss_p)}) hoặc dữ liệu sạch ({len(miss_r)})")
    warns = []
    if "pred_words" in scores.columns:
        bad = sum(int(w) != word_count(preds[i]) for i, w in zip(scores["id"], scores["pred_words"]))
        if bad:
            warns.append(f"{bad}/{len(ids)} dự đoán có số từ khác cột pred_words của file điểm (file dự đoán có thể không khớp lần chạy)")
    return warns


# ----------------------------------------------------------------------------- dựng toàn bộ lựa chọn
def build_selection(data: dict[str, dict], cfg: dict) -> dict:
    """data[source] = {'scores': df, 'records': {id: {...}}, 'preds': {id: str}, 'system': str}.

    Trả {'key': df, 'error': df, 'human_blind': list[dict], 'rerate_blind': list[dict]} (không có điểm trong *_blind).
    """
    seed = cfg["seed"]
    key_rows, err_rows, human_pool = [], [], []
    for source, d in data.items():
        sc = add_s(d["scores"])
        ec = cfg["error_counts"][source]
        err = pick_error_samples(sc, ec["top"], ec["bottom"], ec["mid"], seed, tuple(cfg["mid_range"]))
        for _, r in err.iterrows():
            key_rows.append({"purpose": "error", "source": source, "system": d["system"], "id": r["id"], "group": r["group"], "S": r["S"]})
            err_rows.append({"source": source, "system": d["system"], **r.to_dict()})
        hum = pick_human_samples(sc, cfg["human_quartile_counts"][source], seed)
        for _, r in hum.iterrows():
            human_pool.append({"source": source, "system": d["system"], "id": r["id"], "S": r["S"], "quartile": int(r["quartile"])})
    human_pool.sort(key=lambda x: (x["source"], x["id"]))
    order = np.random.default_rng(seed).permutation(len(human_pool))
    shuffled = [human_pool[k] for k in order]
    for pos, it in enumerate(shuffled, start=1):
        it["item_id"] = f"H{pos:02d}"
    n_re = cfg["rerate_n"]
    re_idx = np.sort(np.random.default_rng(seed).choice(len(shuffled), size=n_re, replace=False))
    re_items = [shuffled[k] for k in re_idx]
    re_order = np.random.default_rng(seed + 1).permutation(len(re_items))
    re_items = [re_items[k] for k in re_order]
    for pos, it in enumerate(re_items, start=1):
        it["rerate_id"] = f"R{pos:02d}"
    re_map = {it["item_id"]: it["rerate_id"] for it in re_items}
    for it in shuffled:
        key_rows.append({"purpose": "human", "source": it["source"], "system": it["system"], "id": it["id"], "group": f"Q{it['quartile']}",
                         "S": it["S"], "item_id": it["item_id"], "rerate_id": re_map.get(it["item_id"], "")})

    def blind(it, code_key):
        dd = data[it["source"]]
        return {"item_id": it[code_key], "article": dd["records"][it["id"]]["article"], "summary": dd["preds"][it["id"]]}

    return {"key": pd.DataFrame(key_rows), "error": pd.DataFrame(err_rows),
            "human_blind": [blind(it, "item_id") for it in shuffled], "rerate_blind": [blind(it, "rerate_id") for it in re_items]}


def render_error_md(rows: pd.DataFrame, data: dict[str, dict], cfg: dict) -> str:
    """Markdown dễ đọc cho 16 mẫu phân tích lỗi: bài, tham chiếu, dự đoán, Lead-k, điểm và cờ. CÓ điểm tự động (không mù)."""
    out = ["# Mẫu phân tích lỗi (D11)", "",
           "> Có điểm tự động trong file này. Nếu mẫu cũng nằm trong 40 mẫu chấm tay, hãy chấm tay xong rồi mới đọc.", ""]
    for n, (_, r) in enumerate(rows.iterrows(), start=1):
        dd = data[r["source"]]
        rec, pred = dd["records"][r["id"]], dd["preds"][r["id"]]
        lead = join_sentences(rec["article"], lead_k(rec["article"], cfg[r["source"]]["lead_k"])).replace("\n", " ")
        u = unsupported_stats(rec["article"], pred)
        out += [f"## E{n:02d} · {r['source']} · nhóm {r['group']} · S = {100 * r['S']:.1f} · hạng {int(r['rank_desc'])}/{len(dd['scores'])}",
                f"`id`: {r['id']} · hệ thống: {r['system']}", "",
                f"**Bài ({word_count(rec['article'])} từ):** {rec['article']}", "",
                f"**Tham chiếu ({word_count(rec['highlights'])} từ):** {rec['highlights']}", "",
                f"**Dự đoán ({word_count(pred)} từ):** {pred}", "",
                f"**Lead-{cfg[r['source']]['lead_k']}:** {lead}", "",
                f"Cờ: lặp trigram = {int(repeated_trigram(pred))}; thực thể không có trong bài = {u['n_unsupported']}/{u['n_entities']}"
                + (f" ({', '.join(u['unsupported'])})" if u["unsupported"] else ""), "",
                "Nhãn lỗi (lặp / thiếu thông tin / ảo giác / khác / không lỗi): …  Ghi chú: …", "", "---", ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- vào/ra
def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/samples.yaml")
    ap.add_argument("--data_dir", required=True, help="thư mục clean có {source}_test.jsonl")
    ap.add_argument("--scores_dir", default="results/scores")
    ap.add_argument("--pred_dir", default="results/predictions")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--flags_only",
        action="store_true",
        help="Chỉ tạo lại các file cờ, không chọn lại mẫu hoặc ghi đè phiếu mù."
    )
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    out, blind_dir = Path(a.out_dir), Path(a.out_dir) / "blind"
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for source in ("cnn_dm", "govuk"):
        c = cfg[source]
        records = {r["id"]: r for r in read_jsonl(Path(a.data_dir) / f"{source}_test.jsonl")}
        preds = {r["id"]: r["prediction"] for r in read_jsonl(Path(a.pred_dir) / c["predictions"])}
        scores = pd.read_csv(Path(a.scores_dir) / c["scores"])
        for w in check_alignment(scores, preds, records):
            print(f"[cảnh báo] {source}: {w}")
        data[source] = {"scores": scores, "records": records, "preds": preds, "system": c["system"]}
        flags = compute_flags(records, preds, list(scores["id"]))
        flags.to_csv(out / f"flags_{source}_{c['system']}.csv", index=False)
        summ = flag_summary(flags, scores)
        summ.to_csv(out / f"flags_summary_{source}.csv", index=False)
        print(f"\n=== cờ tự động {source} ({c['system']}), theo tứ phân vị S ===\n{summ.round(3).to_string(index=False)}")
    # Chỉ cập nhật file cờ, giữ nguyên toàn bộ kết quả chọn mẫu đã có.
    if a.flags_only:
        print("\nĐã cập nhật các file cờ tự động cho cả hai nguồn.")
        print("Không tạo lại samples_key.csv hoặc các phiếu chấm mù.")
        return
    sel = build_selection(data, cfg)
    sel["key"].to_csv(out / "samples_key.csv", index=False)
    sel["error"].to_csv(out / "error_analysis_samples.csv", index=False)
    (out / "error_analysis_samples.md").write_text(render_error_md(sel["error"], data, cfg), encoding="utf-8")
    write_jsonl(blind_dir / "human_items_blind.jsonl", sel["human_blind"])
    write_jsonl(blind_dir / "human_rerate_items_blind.jsonl", sel["rerate_blind"])
    k = sel["key"]
    h_ids, e_ids = set(k[k.purpose == "human"]["id"]), set(k[k.purpose == "error"]["id"])
    print(f"\nĐã chọn: {len(e_ids)} mẫu phân tích lỗi, {len(sel['human_blind'])} mẫu chấm tay, {len(sel['rerate_blind'])} mẫu chấm lại.")
    print(f"Số mẫu vừa phân tích lỗi vừa chấm tay: {len(h_ids & e_ids)} (chấm tay trước khi đọc file phân tích lỗi).")


if __name__ == "__main__":
    main()
