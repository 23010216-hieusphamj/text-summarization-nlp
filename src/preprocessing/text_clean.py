"""Làm sạch văn bản cho CNN/DailyMail và GOV.UK, thống nhất định dạng đầu vào mô hình."""
from __future__ import annotations

import re
import unicodedata

# ---- Mẫu rác của CNN/DailyMail (neo ở đầu bài) -------------------------------------------------
_TS = (r"(?:\d{1,2}:\d{2}(?:\s*[AP]M)?\s*(?:[A-Z]{3,4})?\s*,?\s*)?"
       r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4}")
_DM_HEADER = re.compile(
    rf"^\s*(?:By\s*\.\s*.{{1,120}}?)?\bPUBLISHED\s*:\s*\.?\s*{_TS}\s*\.?\s*"
    rf"(?:\|\s*\.?\s*UPDATED\s*:\s*\.?\s*{_TS}\s*\.?\s*)?")
_DM_UPDATED = re.compile(rf"^\s*(?:By\s*\.\s*.{{1,120}}?)?\bUPDATED\s*:\s*\.?\s*{_TS}\s*\.?\s*")
_DM_LAST_UPDATED = re.compile(
    r"^\s*(?:By\s*\.\s*.{1,120}?)?Last updated at\s+\d{1,2}:\d{2}\s*[AP]M\s+on\s+"
    r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4}\s*\.?\s*")
# Byline dạng "By . Nick Harris for MailOnline . " (tên có thể chứa từ viết thường như for/and)
_DM_BYLINE = re.compile(r"^\s*By\s*\.\s+.{1,80}?\s\.\s+(?=[A-Z\"“‘'(])")
# Địa danh: các từ viết hoa (hoặc từ nối như of/de/la); KHÔNG cho phép từ thường khác (told, in, said...)
# để tránh cắt nhầm câu mở đầu có nhắc "(CNN)".
_PLACE = (r"[A-Z][\w.'’\-]*(?:\s+(?:[A-Z][\w.'’\-]*|of|de|la|del|el|the|di|da|do|van|von|al|and)){0,5}")
_CNN_DATELINE = re.compile(rf"^\s*(?:{_PLACE}(?:,\s*{_PLACE})?\s*)?\(CNN\)\s*(?:--|—|-)?\s*")
_START_PATTERNS = (_DM_HEADER, _DM_UPDATED, _DM_LAST_UPDATED, _DM_BYLINE, _CNN_DATELINE)

# ---- Mẫu rác có thể nằm giữa/cuối bài --------------------------------------------------------
_INLINE_PATTERNS = (
    re.compile(r"(?i)\bscroll down for (?:the )?video\b\s*\.?"),
    re.compile(r"(?i)\be-mail to a friend\b\s*\.?"),
    re.compile(r"(?i)\bshare this article\s+share(?:\s+\d+\s+shares?)?\b\s*\.?"),
)
_URL = re.compile(r"https?://\S+|www\.\S+")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


def normalize_whitespace(text: str, keep_newlines: bool = False) -> str:
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH).replace("\xa0", " ")
    if keep_newlines:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
        return "\n".join(ln for ln in lines if ln)
    return re.sub(r"\s+", " ", text).strip()


def strip_start_boilerplate(text: str, max_passes: int = 5) -> str:
    """Bỏ lặp các tiền tố rác ở đầu bài (byline, PUBLISHED/UPDATED, dateline '(CNN) --') cho đến khi hết."""
    for _ in range(max_passes):
        new = text
        for pat in _START_PATTERNS:
            new = pat.sub("", new, count=1)
        if new == text:
            break
        text = new
    return text


def clean_article(text: str) -> str:
    """Làm sạch bài CNN/DM: bỏ tiền tố, cụm rác, URL; chuẩn hóa khoảng trắng."""
    text = normalize_whitespace(text)
    text = strip_start_boilerplate(text)
    for pat in _INLINE_PATTERNS:
        text = pat.sub(" ", text)
    text = _URL.sub("", text)
    return normalize_whitespace(text)


def clean_highlights(text: str) -> str:
    """Giữ ký tự xuống dòng giữa các câu tóm tắt (cần cho ROUGE-Lsum ở GĐ7)."""
    return normalize_whitespace(text, keep_newlines=True)


def clean_generic_article(text: str) -> str:
    """Dùng cho GOV.UK: dữ liệu đã được làm sạch ở GĐ2, chỉ thống nhất khoảng trắng (đoạn văn -> khoảng trắng)."""
    return normalize_whitespace(text)
