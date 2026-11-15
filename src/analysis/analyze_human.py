"""Tổng hợp chấm tay D11 và liên hệ với điểm ROUGE/BERTScore theo từng mẫu.

Ví dụ chạy sau khi hoàn tất lượt chấm đầu:
    python -m src.analysis.analyze_human \
      --human_csv results/gd7/blind/human_rating_completed.csv

Sau khi có phiếu lượt hai do một người chấm khác thực hiện:
    python -m src.analysis.analyze_human \
      --human_csv results/gd7/blind/human_rating_completed.csv \
      --rerate_csv results/gd7/blind/human_rerate_completed.csv

Trong giao thức GĐ7 thực tế, người chấm lượt hai đã đọc rubric, chỉ được cung cấp
bài gốc và tóm tắt cần đánh giá, không được xem điểm/nhận xét lượt một. Vì hai
lượt do hai người khác nhau thực hiện, báo cáo dùng thuật ngữ inter-rater agreement.

Khoảng tin cậy bootstrap được tính bằng cách lấy mẫu lại các mẫu chấm (seed cố định).
Vì 40 mẫu được phân tầng theo tứ phân vị S, kết quả mô tả nhóm mẫu đã chọn, không
phải ước lượng đại diện cho toàn bộ tập test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.correlate_scores import pearson, spearman

RATING_COLUMNS = ("fluency_1_5", "completeness_1_5", "factuality_1_5")
RATING_LABELS = {
    "fluency_1_5": "Trôi chảy (Fluency)",
    "completeness_1_5": "Đầy đủ (Completeness)",
    "factuality_1_5": "Đúng sự thật (Factuality)",
}
SCORE_COLUMNS = ("rouge1", "rouge2", "rougeLsum", "bertscore_f1")
CORRELATION_METRICS = ("S", "rouge1", "rouge2", "rougeLsum", "bertscore_f1")
SCORE_LABELS = {
    "S": "S = TB(ROUGE-1, ROUGE-2, ROUGE-Lsum)",
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeLsum": "ROUGE-Lsum",
    "bertscore_f1": "BERTScore F1",
}


def read_rating_csv(path: str | Path) -> pd.DataFrame:
    """Đọc và kiểm tra một phiếu chấm hoặc phiếu chấm lại."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy phiếu chấm: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    required = {"item_id", *RATING_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Phiếu chấm thiếu cột bắt buộc: {', '.join(missing)}")
    df["item_id"] = df["item_id"].astype(str).str.strip()
    if (df["item_id"] == "").any():
        raise ValueError("Phiếu chấm có item_id rỗng")
    if df["item_id"].duplicated().any():
        dup = df.loc[df["item_id"].duplicated(), "item_id"].tolist()
        raise ValueError(f"Phiếu chấm có item_id trùng: {dup}")
    for column in RATING_COLUMNS:
        raw = df[column].astype(str).str.strip()
        if (raw == "").any():
            missing_ids = df.loc[raw == "", "item_id"].tolist()
            raise ValueError(f"Cột {column} còn điểm trống ở: {missing_ids}")
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = numeric.isna() | (numeric < 1) | (numeric > 5) | (numeric % 1 != 0)
        if invalid.any():
            bad = df.loc[invalid, ["item_id", column]].to_dict("records")
            raise ValueError(f"Cột {column} phải là số nguyên 1–5: {bad}")
        df[column] = numeric.astype(int)
    for column in ("error_type", "notes"):
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].astype(str)
    return df


def read_human_key(path: str | Path) -> pd.DataFrame:
    """Đọc khóa ghép mẫu mù Hxx với id nguồn và mã chấm lại Rxx."""
    key = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype=str)
    required = {"purpose", "source", "system", "id", "S", "item_id", "rerate_id"}
    missing = sorted(required - set(key.columns))
    if missing:
        raise ValueError(f"samples_key.csv thiếu cột: {', '.join(missing)}")
    key = key.loc[key["purpose"] == "human"].copy()
    if key.empty:
        raise ValueError("samples_key.csv không có dòng purpose=human")
    if key["item_id"].duplicated().any():
        raise ValueError("samples_key.csv có item_id chấm tay bị trùng")
    return key


def merge_human_with_scores(
    human: pd.DataFrame,
    key: pd.DataFrame,
    config: dict,
    scores_dir: str | Path,
) -> pd.DataFrame:
    """Ghép điểm người chấm với điểm tự động bằng khóa item_id -> id, không ghép theo thứ tự."""
    scores_dir = Path(scores_dir)
    key_ids = set(key["item_id"].astype(str))
    human_ids = set(human["item_id"].astype(str))
    if human_ids != key_ids:
        missing = sorted(key_ids - human_ids)
        extra = sorted(human_ids - key_ids)
        raise ValueError(
            "Danh sách item_id trong phiếu không khớp samples_key.csv "
            f"(thiếu {missing}; thừa {extra})"
        )

    score_cache: dict[str, pd.DataFrame] = {}
    score_records: dict[tuple[str, str], dict] = {}
    for source in sorted(key["source"].unique()):
        if source not in config or "scores" not in config[source]:
            raise ValueError(f"configs/samples.yaml không khai báo file scores cho nguồn {source}")
        score_path = scores_dir / config[source]["scores"]
        if not score_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file điểm {source}: {score_path}")
        scores = pd.read_csv(score_path, encoding="utf-8-sig")
        needed = {"id", *SCORE_COLUMNS, "pred_words", "ref_words"}
        absent = sorted(needed - set(scores.columns))
        if absent:
            raise ValueError(f"File điểm {score_path} thiếu cột: {', '.join(absent)}")
        if scores["id"].astype(str).duplicated().any():
            raise ValueError(f"File điểm {score_path} có id trùng")
        score_cache[source] = scores
        for record in scores.to_dict("records"):
            score_records[(source, str(record["id"]))] = record

    key_by_item = key.set_index("item_id").to_dict("index")
    merged_rows = []
    for human_row in human.to_dict("records"):
        item_id = str(human_row["item_id"])
        key_row = key_by_item[item_id]
        source = key_row["source"]
        source_id = str(key_row["id"])
        try:
            score_row = score_records[(source, source_id)]
        except KeyError as exc:
            raise ValueError(f"Không tìm thấy điểm tự động cho {item_id} ({source}/{source_id})") from exc
        metric_values = {column: float(score_row[column]) for column in SCORE_COLUMNS}
        derived_s = float(np.mean([metric_values[c] for c in ("rouge1", "rouge2", "rougeLsum")]))
        key_s = float(key_row["S"])
        if not np.isclose(derived_s, key_s, atol=1e-8, rtol=0):
            raise ValueError(
                f"S trong samples_key không khớp điểm nguồn ở {item_id}: "
                f"key={key_s:.10f}, tính lại={derived_s:.10f}"
            )
        merged_rows.append({
            **human_row,
            "source": source,
            "system": key_row["system"],
            "source_id": source_id,
            "quartile": key_row.get("group", ""),
            "S": derived_s,
            **metric_values,
            "pred_words": int(score_row["pred_words"]),
            "ref_words": int(score_row["ref_words"]),
        })
    return pd.DataFrame(merged_rows)


def bootstrap_mean_ci(values, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """CI percentile 95% cho trung bình, bootstrap theo đơn vị mẫu."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_spearman_ci(x, y, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, int]:
    """CI percentile 95% cho Spearman; bỏ lần bootstrap có hạng hằng."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(x), len(x))
        value = spearman(x[indices], y[indices])
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high), len(draws)



def bootstrap_spearman_difference_ci(x, y_a, y_b, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, int]:
    """CI bootstrap ghép cặp cho hiệu Spearman(x, y_a) - Spearman(x, y_b)."""
    x = np.asarray(x, dtype=float)
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y_a) & np.isfinite(y_b)
    x, y_a, y_b = x[valid], y_a[valid], y_b[valid]
    if len(x) < 3:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(x), len(x))
        rho_a = spearman(x[indices], y_a[indices])
        rho_b = spearman(x[indices], y_b[indices])
        if np.isfinite(rho_a) and np.isfinite(rho_b):
            draws.append(rho_a - rho_b)
    if not draws:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high), len(draws)


def compare_completeness_correlations(
    df: pd.DataFrame, n_boot: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """So sánh ghép cặp Spearman(BERTScore, Completeness) với Spearman(ROUGE-2, Completeness)."""
    rows = []
    scopes = [("all", df)] + [(source, part) for source, part in df.groupby("source", sort=True)]
    for offset, (scope, part) in enumerate(scopes):
        x = part["completeness_1_5"].to_numpy(dtype=float)
        bscore = part["bertscore_f1"].to_numpy(dtype=float)
        rouge2 = part["rouge2"].to_numpy(dtype=float)
        rho_bscore = spearman(x, bscore)
        rho_rouge2 = spearman(x, rouge2)
        low, high, valid = bootstrap_spearman_difference_ci(
            x, bscore, rouge2, n_boot=n_boot, seed=seed + 300 + offset
        )
        rows.append({
            "scope": scope,
            "n": len(part),
            "comparison": "Spearman(Completeness, BERTScore) - Spearman(Completeness, ROUGE-2)",
            "bertscore_spearman": rho_bscore,
            "rouge2_spearman": rho_rouge2,
            "difference": rho_bscore - rho_rouge2,
            "difference_ci95_low": low,
            "difference_ci95_high": high,
            "bootstrap_valid": valid,
        })
    return pd.DataFrame(rows)

def summarize_ratings(df: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Trung bình, độ lệch chuẩn và CI của từng tiêu chí theo nguồn và toàn bộ mẫu."""
    rows = []
    scopes = [("all", df)] + [(source, part) for source, part in df.groupby("source", sort=True)]
    seed_offset = 0
    for scope, part in scopes:
        for column in RATING_COLUMNS:
            values = part[column].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed + seed_offset)
            row = {
                "scope": scope,
                "n": len(values),
                "criterion": column,
                "criterion_vi": RATING_LABELS[column],
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "ci95_low": low,
                "ci95_high": high,
            }
            for rating in range(1, 6):
                row[f"score_{rating}_count"] = int(np.sum(values == rating))
            rows.append(row)
            seed_offset += 1
    return pd.DataFrame(rows)


def correlate_human_and_automatic(
    df: pd.DataFrame, n_boot: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Tương quan điểm chấm tay với S, ROUGE-1/2/Lsum và BERTScore."""
    rows = []
    scopes = [("all", df)] + [(source, part) for source, part in df.groupby("source", sort=True)]
    seed_offset = 100
    for scope, part in scopes:
        for rating_col in RATING_COLUMNS:
            x = part[rating_col].to_numpy(dtype=float)
            for metric in CORRELATION_METRICS:
                y = part[metric].to_numpy(dtype=float)
                rho = spearman(x, y)
                r = pearson(x, y)
                low, high, n_valid = bootstrap_spearman_ci(
                    x, y, n_boot=n_boot, seed=seed + seed_offset
                )
                rows.append({
                    "scope": scope,
                    "n": len(part),
                    "criterion": rating_col,
                    "criterion_vi": RATING_LABELS[rating_col],
                    "metric": metric,
                    "metric_vi": SCORE_LABELS[metric],
                    "spearman": rho,
                    "spearman_ci95_low": low,
                    "spearman_ci95_high": high,
                    "bootstrap_valid": n_valid,
                    "pearson": r,
                })
                seed_offset += 1
    return pd.DataFrame(rows)


def summarize_error_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Đếm nhãn lỗi dạng nhiều nhãn, ngăn cách bằng dấu chấm phẩy."""
    rows = []
    scopes = [("all", df)] + [(source, part) for source, part in df.groupby("source", sort=True)]
    for scope, part in scopes:
        counts: dict[str, int] = {}
        for value in part.get("error_type", pd.Series(dtype=str)).fillna("").astype(str):
            for label in (token.strip() for token in value.split(";")):
                if label:
                    counts[label] = counts.get(label, 0) + 1
        for label, count in sorted(counts.items()):
            rows.append({"scope": scope, "n_samples": len(part), "label": label,
                         "count": count, "pct_of_samples": 100 * count / len(part) if len(part) else float("nan")})
    return pd.DataFrame(rows, columns=["scope", "n_samples", "label", "count", "pct_of_samples"])


def quadratic_weighted_kappa(first, second, minimum: int = 1, maximum: int = 5) -> float:
    """Kappa có trọng số bậc hai cho hai người chấm trên cùng các mẫu."""
    a = np.asarray(first, dtype=int)
    b = np.asarray(second, dtype=int)
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    if np.any((a < minimum) | (a > maximum) | (b < minimum) | (b > maximum)):
        raise ValueError(f"Điểm QWK phải nằm trong [{minimum}, {maximum}]")
    labels = np.arange(minimum, maximum + 1)
    observed = np.zeros((len(labels), len(labels)), dtype=float)
    for x, y in zip(a, b):
        observed[x - minimum, y - minimum] += 1
    n = len(a)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    denom = float(maximum - minimum) ** 2
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / denom, observed.shape)
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement == 0:
        return float("nan")
    observed_disagreement = float(np.sum(weights * observed))
    return 1.0 - observed_disagreement / expected_disagreement


def analyze_inter_rater_agreement(
    first_round: pd.DataFrame,
    second_rater: pd.DataFrame,
    key: pd.DataFrame,
) -> pd.DataFrame:
    """Tính độ đồng thuận giữa người chấm lượt 1 và người chấm lượt 2 trên 8 mẫu.

    Ghép cặp bằng samples_key.csv (Hxx -> Rxx), không ghép theo thứ tự dòng.
    Nếu cả hai phiếu có article/summary, kiểm tra nội dung của từng cặp phải giống hệt.
    Đây là inter-rater agreement khi hai lượt do hai người khác nhau thực hiện;
    không diễn giải thành intra-rater/test-retest.
    """
    mapping = key.loc[
        key["rerate_id"].astype(str).str.strip() != "",
        ["item_id", "rerate_id"],
    ].copy()
    if mapping.empty:
        raise ValueError("samples_key.csv không có mẫu được chọn chấm lượt hai")
    if mapping["rerate_id"].duplicated().any() or mapping["item_id"].duplicated().any():
        raise ValueError("samples_key.csv có mã lượt hai hoặc item_id bị trùng")

    expected_first_ids = set(mapping["item_id"].astype(str))
    actual_first_ids = set(first_round["item_id"].astype(str))
    if not expected_first_ids.issubset(actual_first_ids):
        raise ValueError(
            "Phiếu lượt đầu thiếu mã H được chọn chấm lượt hai: "
            f"{sorted(expected_first_ids - actual_first_ids)}"
        )

    expected_second_ids = set(mapping["rerate_id"].astype(str))
    actual_second_ids = set(second_rater["item_id"].astype(str))
    if actual_second_ids != expected_second_ids:
        raise ValueError(
            "Phiếu người chấm thứ hai không khớp các mã R đã chọn "
            f"(thiếu {sorted(expected_second_ids-actual_second_ids)}; "
            f"thừa {sorted(actual_second_ids-expected_second_ids)})"
        )

    first = first_round.set_index("item_id")
    second = second_rater.set_index("item_id")
    has_content = all(
        column in first_round.columns and column in second_rater.columns
        for column in ("article", "summary")
    )
    rows = []
    for column in RATING_COLUMNS:
        first_scores = []
        second_scores = []
        for item_id, second_id in mapping[["item_id", "rerate_id"]].itertuples(index=False, name=None):
            first_id = str(item_id)
            second_id = str(second_id)
            if has_content:
                for content_column in ("article", "summary"):
                    if str(first.loc[first_id, content_column]) != str(second.loc[second_id, content_column]):
                        raise ValueError(
                            f"Nội dung {content_column} không khớp giữa {first_id} và {second_id}"
                        )
            first_scores.append(int(first.loc[first_id, column]))
            second_scores.append(int(second.loc[second_id, column]))

        a = np.asarray(first_scores, dtype=int)
        b = np.asarray(second_scores, dtype=int)
        rows.append({
            "criterion": column,
            "criterion_vi": RATING_LABELS[column],
            "n": len(a),
            "rater_1_mean": float(a.mean()),
            "rater_2_mean": float(b.mean()),
            "mean_difference_rater_2_minus_rater_1": float((b - a).mean()),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(a, b),
            "spearman_inter_rater": spearman(a, b),
            "exact_agreement_pct": float(np.mean(a == b) * 100),
            "within_one_point_pct": float(np.mean(np.abs(a - b) <= 1) * 100),
            "mean_absolute_difference": float(np.mean(np.abs(a - b))),
        })
    return pd.DataFrame(rows)


# Tên cũ giữ tương thích với notebook/script cũ; thuật ngữ đúng của kết quả hiện tại là inter-rater.
analyze_rerating = analyze_inter_rater_agreement


def _fmt(value: float, digits: int = 3) -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def build_markdown_report(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    error_labels: pd.DataFrame,
    comparisons: pd.DataFrame,
    reliability: pd.DataFrame | None = None,
) -> str:
    """Tạo báo cáo Markdown để đưa vào README/báo cáo đồ án."""
    lines = [
        "# Kết quả chấm tay D11 – lượt 1",
        "",
        f"- Tổng số mẫu: **{len(merged)}** (CNN/DM: **{int((merged['source'] == 'cnn_dm').sum())}**, GOV.UK: **{int((merged['source'] == 'govuk').sum())}**).",
        "- Người chấm: Phạm Bá Hiếu; một người chấm toàn bộ mẫu.",
        "- CI 95%: bootstrap 1.000 lần theo mẫu, seed 42.",
        "- Ba tiêu chí: Fluency, Completeness, Factuality (thang 1–5).",
        "",
        "> **Giới hạn quan trọng:** 40 mẫu được chọn phân tầng theo tứ phân vị S (không phải mẫu ngẫu nhiên đơn giản). Các trung bình/CI và tương quan dưới đây mô tả nhóm 40 mẫu đã chấm; không được diễn giải như ước lượng đại diện cho toàn bộ tập test. GOV.UK chỉ có 10 mẫu nên CI đặc biệt rộng.",
        "",
        "## 1. Điểm trung bình",
        "",
        "| Phạm vi | Tiêu chí | n | Trung bình | ĐLC | CI bootstrap 95% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['scope']} | {row['criterion_vi']} | {row['n']} | {row['mean']:.2f} | "
            f"{_fmt(row['std'], 2)} | [{row['ci95_low']:.2f}; {row['ci95_high']:.2f}] |"
        )
    lines += [
        "",
        "### Phân phối điểm trên 40 mẫu",
        "",
        "| Tiêu chí | Điểm 1 | Điểm 2 | Điểm 3 | Điểm 4 | Điểm 5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    overall = summary.loc[summary["scope"] == "all"]
    for row in overall.to_dict("records"):
        lines.append(
            f"| {row['criterion_vi']} | {row['score_1_count']} | {row['score_2_count']} | "
            f"{row['score_3_count']} | {row['score_4_count']} | {row['score_5_count']} |"
        )
    lines += [
        "",
        "## 2. Tương quan giữa điểm chấm tay và chỉ số tự động",
        "",
        "Spearman là hệ số chính; Pearson là hệ số phụ. CI 95% bootstrap 1.000 lần. Các hệ số không được dùng để kết luận nhân quả.",
        "",
        "| Phạm vi | Tiêu chí chấm tay | Chỉ số | n | Spearman [CI 95%] | Pearson |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in correlations.to_dict("records"):
        lines.append(
            f"| {row['scope']} | {row['criterion_vi']} | {row['metric_vi']} | {row['n']} | "
            f"{_fmt(row['spearman'])} [{_fmt(row['spearman_ci95_low'])}; {_fmt(row['spearman_ci95_high'])}] | "
            f"{_fmt(row['pearson'])} |"
        )
    lines += [
        "",
        "### So sánh tương quan cho tiêu chí đầy đủ",
        "",
        "Hiệu được tính ghép cặp trên cùng mẫu: Spearman(Completeness, BERTScore) trừ Spearman(Completeness, ROUGE-2). CI không chứa 0 mới gợi ý chênh lệch theo bootstrap; đây vẫn là phân tích thăm dò trên mẫu phân tầng.",
        "",
        "| Phạm vi | n | rho với BERTScore | rho với ROUGE-2 | Hiệu | CI 95% của hiệu |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons.to_dict("records"):
        lines.append(
            f"| {row['scope']} | {row['n']} | {_fmt(row['bertscore_spearman'])} | {_fmt(row['rouge2_spearman'])} | "
            f"{_fmt(row['difference'])} | [{_fmt(row['difference_ci95_low'])}; {_fmt(row['difference_ci95_high'])}] |"
        )
    lines += [
        "",
        "## 3. Nhãn lỗi do người chấm ghi nhận",
        "",
        "Các nhãn có thể chồng lấp vì một mẫu được phép có nhiều loại lỗi; tỷ lệ tính trên số mẫu của từng phạm vi.",
        "",
        "| Phạm vi | Nhãn | Số mẫu có nhãn | Tỷ lệ trên phạm vi |",
        "|---|---|---:|---:|",
    ]
    for row in error_labels.to_dict("records"):
        lines.append(f"| {row['scope']} | `{row['label']}` | {row['count']} | {row['pct_of_samples']:.1f}% |")
    lines += ["", "## 4. Độ đồng thuận giữa hai người chấm (inter-rater agreement)", ""]
    if reliability is None:
        lines += [
            "Chưa có phiếu chấm của người chấm thứ hai; khi có phiếu lượt hai, báo cáo QWK bậc hai, Spearman, tỷ lệ trùng điểm và sai khác tuyệt đối trên đúng các mẫu ghép cặp theo `samples_key.csv`.",
            "Cần ghi rõ hai lượt do cùng một người hay hai người khác nhau thực hiện để phân biệt intra-rater/test–retest với inter-rater agreement.",
        ]
    else:
        lines += [
            "Lượt 1 do Phạm Bá Hiếu thực hiện; lượt 2 do một thành viên khác trong nhóm thực hiện. Người chấm lượt hai đã đọc cùng rubric, chỉ được cung cấp bài gốc và tóm tắt cần đánh giá, không được xem điểm hoặc nhận xét lượt một trước khi chấm. Vì vậy, đây là độ đồng thuận giữa hai người chấm (inter-rater agreement), không phải tính tự nhất quán/test–retest của cùng một người.",
            "Tám mẫu lượt hai được ghép với lượt đầu bằng khóa Hxx–Rxx trong `samples_key.csv`; mã và nội dung bài/tóm tắt đã được kiểm tra. Các ước lượng chỉ mang tính mô tả/thăm dò do n=8.",
            "",
            "| Tiêu chí | n | TB người 1 | TB người 2 | Chênh TB (2−1) | QWK bậc hai | Spearman | Trùng điểm | Sai khác ≤1 | MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in reliability.to_dict("records"):
            lines.append(
                f"| {row['criterion_vi']} | {row['n']} | {_fmt(row['rater_1_mean'])} | "
                f"{_fmt(row['rater_2_mean'])} | {_fmt(row['mean_difference_rater_2_minus_rater_1'])} | "
                f"{_fmt(row['quadratic_weighted_kappa'])} | {_fmt(row['spearman_inter_rater'])} | "
                f"{row['exact_agreement_pct']:.1f}% | {row['within_one_point_pct']:.1f}% | "
                f"{_fmt(row['mean_absolute_difference'])} |"
            )
        exact_total = float(reliability["exact_agreement_pct"].mean())
        within_total = float(reliability["within_one_point_pct"].mean())
        lines += [
            "",
            f"Trên 24 cặp điểm (8 mẫu × 3 tiêu chí), tỷ lệ trùng điểm trung bình theo tiêu chí là {exact_total:.1f}%; tỷ lệ chênh không quá 1 điểm là {within_total:.1f}%.",
            "Không suy rộng các chỉ số này thành độ đồng thuận của toàn bộ nhóm hoặc toàn bộ tập test.",
        ]
    lines += [
        "",
        "## 5. Diễn giải và hạn chế",
        "",
        "- Điểm trung bình chỉ mô tả 40 mẫu đã chọn; không suy rộng sang toàn bộ 1.000 mẫu CNN/DM hoặc 389 mẫu GOV.UK.",
        "- Mẫu phân tầng theo S làm thay đổi phân phối ROUGE trong mẫu chấm; tương quan trên 40 mẫu có thể khác tương quan tính trên toàn bộ test.",
        "- Lượt đầu do Phạm Bá Hiếu chấm 40 mẫu; lượt hai do một thành viên khác chấm 8 mẫu. Người chấm thứ hai đọc rubric và chỉ xem bài gốc cùng tóm tắt, không được xem điểm/nhận xét lượt một. Phần so sánh là inter-rater agreement trên 8 mẫu, không đánh giá intra-rater/test–retest.",
        "- Tám mẫu được chọn lại từ 40 mẫu ban đầu; cỡ mẫu nhỏ nên QWK/Spearman có thể biến động và chỉ mang tính mô tả/thăm dò.",
        "- Chưa hiệu chỉnh đa so sánh cho nhiều hệ số tương quan; cần diễn giải CI và quy mô mẫu, không chỉ nhìn trị tuyệt đối của hệ số.",
        "- ROUGE/BERTScore đo mức tương đồng với tham chiếu, không trực tiếp xác minh tính đúng sự thật so với bài gốc.",
        "",
    ]
    return "\n".join(lines)


def run_analysis(
    human_csv: str | Path,
    key_csv: str | Path,
    config_path: str | Path,
    scores_dir: str | Path,
    output_dir: str | Path,
    rerate_csv: str | Path | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, pd.DataFrame | None]:
    """Chạy toàn bộ tổng hợp, ghi CSV và Markdown."""
    human = read_rating_csv(human_csv)
    key = read_human_key(key_csv)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    merged = merge_human_with_scores(human, key, config, scores_dir)
    summary = summarize_ratings(merged, n_boot=n_boot, seed=seed)
    correlations = correlate_human_and_automatic(merged, n_boot=n_boot, seed=seed)
    error_labels = summarize_error_labels(merged)
    comparisons = compare_completeness_correlations(merged, n_boot=n_boot, seed=seed)
    reliability = None
    if rerate_csv:
        rerate = read_rating_csv(rerate_csv)
        reliability = analyze_inter_rater_agreement(human, rerate, key)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_out = merged[[
        "item_id", "source", "system", "source_id", "quartile", "S",
        *SCORE_COLUMNS, *RATING_COLUMNS, "error_type", "notes", "pred_words", "ref_words",
    ]]
    merged_out.to_csv(output_dir / "human_scores_joined.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "human_summary.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(output_dir / "human_correlations.csv", index=False, encoding="utf-8-sig")
    error_labels.to_csv(output_dir / "human_error_labels.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(output_dir / "human_correlation_comparisons.csv", index=False, encoding="utf-8-sig")
    if reliability is not None:
        reliability.to_csv(output_dir / "human_inter_rater_agreement.csv", index=False, encoding="utf-8-sig")
    report = build_markdown_report(merged, summary, correlations, error_labels, comparisons, reliability)
    (output_dir / "human_analysis.md").write_text(report, encoding="utf-8")
    return {
        "merged": merged_out,
        "summary": summary,
        "correlations": correlations,
        "error_labels": error_labels,
        "comparisons": comparisons,
        "reliability": reliability,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human_csv", default="results/gd7/blind/human_rating_completed.csv")
    parser.add_argument("--key_csv", default="results/gd7/samples_key.csv")
    parser.add_argument("--config", default="configs/samples.yaml")
    parser.add_argument("--scores_dir", default="results/scores")
    parser.add_argument("--output_dir", default="results/gd7")
    parser.add_argument(
        "--rerate_csv", "--second_rater_csv", dest="rerate_csv", default=None,
        help="phiếu chấm của người thứ hai cho 8 mẫu đã chọn; chưa có thì bỏ qua",
    )
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    results = run_analysis(
        args.human_csv, args.key_csv, args.config, args.scores_dir, args.output_dir,
        rerate_csv=args.rerate_csv, n_boot=args.n_boot, seed=args.seed,
    )
    print(f"Đã phân tích {len(results['merged'])} mẫu chấm tay.")
    print(f"Báo cáo: {Path(args.output_dir) / 'human_analysis.md'}")
    if results["reliability"] is None:
        print("Chưa có phiếu người chấm thứ hai; chưa tính inter-rater agreement.")
    else:
        print(f"Đã tính inter-rater agreement: {Path(args.output_dir) / 'human_inter_rater_agreement.csv'}")


if __name__ == "__main__":
    main()
