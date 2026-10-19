"""Làm sạch, lọc, khử trùng lặp và chia tập cho dữ liệu GOV.UK."""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import OrderedDict

from bs4 import BeautifulSoup

from src.data.eda_utils import compression_ratio, max_sentence_overlap, tokenize, word_count

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Phần cuối thông cáo báo chí không thuộc nội dung chính
TAIL_HEADINGS = re.compile(
    r"^(notes? to editors?|further information|media enquiries|contact(s)?|"
    r"press office|more information|related links?)\b", re.I)
BLOCK_TAGS = ("p", "li", "blockquote")
HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\xa0", " ").replace("\u200b", "")
    t = URL_RE.sub("", t)
    t = EMAIL_RE.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def html_to_text(html: str) -> str:
    """HTML -> văn bản thuần: bỏ bảng/hình, bỏ tiêu đề mục, cắt phần 'Notes to editors'."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "table", "figure", "img", "aside", "nav"]):
        tag.decompose()
    lines = []
    for el in soup.find_all(HEADING_TAGS + BLOCK_TAGS):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in HEADING_TAGS:
            if TAIL_HEADINGS.match(txt):
                break                      # bỏ toàn bộ phần phía sau
            continue                       # tiêu đề mục không đưa vào văn bản
        if el.name == "li" and el.find_parent("li"):
            continue                       # tránh lặp li lồng nhau
        lines.append(txt)
    return normalize_text("\n".join(lines))


def _norm_key(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def _is_english(text: str, lang: str) -> bool:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    try:
        return detect(text[:1000]) == lang
    except Exception:
        return False


def build_records(raw_items: list[dict]) -> list[dict]:
    """raw: {link,title,description,document_type,public_updated_at,body_html} -> bản ghi sạch bước đầu."""
    out = []
    for r in raw_items:
        out.append({
            "id": r["link"], "title": (r.get("title") or "").strip(),
            "article": html_to_text(r.get("body_html") or ""),
            "highlights": normalize_text((r.get("description") or "").strip()),
            "doc_type": r.get("document_type"), "date": r.get("public_updated_at"),
        })
    return out


def apply_filters(records: list[dict], c: dict) -> tuple[list[dict], list[tuple[str, int]]]:
    """Trả về (bản ghi giữ lại, funnel[(tên bước, số còn lại)]) để vẽ biểu đồ phễu."""
    funnel = [("Bài tải về", len(records))]
    steps = OrderedDict([
        ("Có nội dung và tóm tắt", lambda r: bool(r["article"]) and bool(r["highlights"])),
        ("Là tiếng Anh", lambda r: _is_english(r["article"], c["lang"])),
        ("Độ dài bài hợp lệ", lambda r: c["min_article_words"] <= word_count(r["article"]) <= c["max_article_words"]),
        ("Độ dài tóm tắt hợp lệ", lambda r: c["min_summary_words"] <= word_count(r["highlights"]) <= c["max_summary_words"]),
        ("Tỉ lệ nén đủ lớn", lambda r: compression_ratio(r["article"], r["highlights"]) >= c["min_compression"]),
        ("Tóm tắt khác tiêu đề", lambda r: _jaccard(r["highlights"], r["title"]) <= c["max_title_jaccard"]),
        ("Tóm tắt không chép nguyên văn bài", lambda r: _norm_key(r["highlights"]) not in _norm_key(r["article"])),
    ])
    thr = c.get("max_copy_overlap", 0.9)        # đặt null trong config để tắt bước này
    if thr is not None:
        steps["Tóm tắt không gần như chép một câu trong bài"] = \
            lambda r: max_sentence_overlap(r["article"], r["highlights"]) < thr
    cur = records
    for name, pred in steps.items():
        cur = [r for r in cur if pred(r)]
        funnel.append((name, len(cur)))
    # Khử trùng lặp
    cur = _dedupe(cur, lambda r: hashlib.md5(_norm_key(r["article"])[:600].encode()).hexdigest())
    funnel.append(("Khử trùng lặp bài (600 ký tự đầu)", len(cur)))
    cur = _dedupe(cur, lambda r: _norm_key(r["highlights"]))
    funnel.append(("Khử trùng lặp tóm tắt", len(cur)))
    return cur, funnel


def _dedupe(records, keyfn):
    seen, out = set(), []
    for r in records:
        k = keyfn(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def split_records(records: list[dict], ratios: dict, seed: int) -> dict:
    recs = sorted(records, key=lambda r: r["id"])      # thứ tự xác định trước khi xáo
    random.Random(seed).shuffle(recs)
    n = len(recs)
    n_tr = int(n * ratios["train"]); n_va = int(n * ratios["validation"])
    return {"train": recs[:n_tr], "validation": recs[n_tr:n_tr + n_va], "test": recs[n_tr + n_va:]}
