# Hệ thống tóm tắt văn bản tiếng Anh

Bài tập lớn học phần **Xử lý ngôn ngữ tự nhiên** – Đề tài: *Xây dựng hệ thống tóm tắt văn bản tiếng Anh*.

> Trạng thái: Giai đoạn 4 – Baseline (xem [Tiến độ](#tiến-độ))

## 1. Bài toán
- **Đầu vào:** một bài văn bản tiếng Anh (ví dụ: bài báo tin tức).
- **Đầu ra:** bản tóm tắt ngắn, giữ ý chính (abstractive).
- **Hướng tiếp cận:** so sánh baseline extractive (Lead-3, TextRank) với mô hình sinh (Seq2Seq+Attention, BART, T5).

## 2. Workflow tổng thể
![workflow](docs/workflow.png)

## 3. Dữ liệu
| Nguồn | Vai trò | Ghi chú |
|---|---|---|
| CNN/DailyMail (HuggingFace) | Huấn luyện + đánh giá chính |287k/13k/11k mẫu gốc; dùng tập con 20k/1k/1k, seed 42|
| Dữ liệu tự thu thập | Điểm cộng, kiểm tra tổng quát hóa |GOV.UK (OGL v3.0): 3.878 cặp, train/val/test = 3.102 / 387 / 389|

## 4. Cài đặt
```bash
git clone <url-repo>
cd text-summarization-nlp
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Cấu trúc thư mục
```
configs/      cấu hình thí nghiệm (yaml)
data/         dữ liệu (raw/processed không commit)
src/          mã nguồn: crawl, preprocessing, baselines, models, train, evaluate
notebooks/    EDA, phân tích lỗi
results/      figures (đường cong hội tụ), tables, logs
app/          demo
docs/         workflow, decisions.md, báo cáo, slides
tests/        unit test
```

## 6. Kết quả

### Baseline không huấn luyện (Giai đoạn 4)

Đánh giá trên tập test. ROUGE là F1 (có stemming), BERTScore là F1 (roberta-large, có rescale), cả hai nhân 100. Độ dài tính bằng số từ. Số liệu đầy đủ (kèm khoảng tin cậy 95%) nằm trong `results/tables/`.

**CNN/DailyMail** (n = 1.000, tham chiếu trung bình 52,6 từ)

| Hệ thống | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | BERTScore | Độ dài |
|---|---|---|---|---|---|
| Random-3 | 28,5 | 8,2 | 25,4 | 12,6 | 66,7 |
| TextRank-3 | 35,2 | 13,8 | 31,5 | 21,6 | 80,5 |
| **Lead-3** | **40,3** | **17,6** | **36,5** | **26,0** | 77,4 |
| Oracle-3 (cận trên, dùng tham chiếu) | 56,7 | 33,5 | 52,6 | 43,2 | 59,3 |

**GOV.UK** (n = 389, tham chiếu trung bình 21,3 từ)

| Hệ thống | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | BERTScore | Độ dài |
|---|---|---|---|---|---|
| Random-1 | 19,5 | 4,1 | 14,7 | 17,0 | 28,3 |
| TextRank-1 | 24,8 | 8,1 | 18,5 | 22,6 | 42,6 |
| **Lead-1** | **33,1** | **15,0** | **25,5** | **30,0** | 36,6 |
| Oracle-1 (cận trên, dùng tham chiếu) | 43,1 | 23,6 | 34,7 | 40,1 | 29,9 |

Nhận xét ngắn: Lead-3 (CNN/DM) và Lead-1 (GOV.UK) là baseline extractive mạnh nhất, vượt TextRank và Random có ý nghĩa thống kê (kiểm định bootstrap ghép cặp, `results/tables/baselines_paired_tests.csv`). Đây là mốc cần vượt ở Giai đoạn 6. Chi tiết và hạn chế: xem `docs/decisions.md` (D8).

*(Bảng so sánh đầy đủ với BART-base, T5-small và phân tích lỗi sẽ cập nhật ở Giai đoạn 7.)*

## Tiến độ
- [x] GĐ0: Khởi tạo repo, workflow v1, decision log
- [x] GĐ1: Dữ liệu công khai + EDA
- [x] GĐ2: Tự thu thập dữ liệu
- [x] GĐ3: Pipeline tiền xử lý
- [x] GĐ4: Baseline (Lead-3, TextRank)
- [ ] GĐ5: Seq2Seq + Attention
- [ ] GĐ6: Fine-tune BART / T5
- [ ] GĐ7: Đánh giá + phân tích lỗi
- [ ] GĐ8: Demo
- [ ] GĐ9–10: Báo cáo, slides

## Nhóm thực hiện
| Họ tên | Mã SV | Nhiệm vụ |
|---|---|---|
|Phạm Bá Hiếu|23010216| ... |
|Nguyễn Phi Hùng| ... | ... |
|Tiêu Hải Nam|23010214 | ... |
|Lê Sơn Hải| ... | ... |
