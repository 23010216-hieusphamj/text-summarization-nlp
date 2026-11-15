# Độ đồng thuận giữa hai người chấm trên 8 mẫu – GĐ7

## 1. Phạm vi và kiểm tra dữ liệu

Lượt 1 do Phạm Bá Hiếu chấm 40 mẫu; lượt 2 do một thành viên khác trong nhóm chấm 8 mẫu đã chọn. Người chấm lượt hai đã đọc rubric, được cung cấp bài gốc và tóm tắt cần đánh giá, nhưng không được xem điểm hoặc nhận xét lượt một trước khi chấm. Khóa `results/gd7/samples_key.csv` ánh xạ mã H của lượt đầu sang mã R của lượt hai.

- Lượt đầu: 40 mã H duy nhất; lượt hai: 8 mã R duy nhất.
- Ghép đúng 8/8 cặp theo khóa H–R.
- `article` và `summary` của cả 8 cặp khớp hoàn toàn giữa hai phiếu.
- Không thiếu điểm ở ba tiêu chí; tất cả điểm là số nguyên trong thang 1–5.
- Không có mã trùng trong từng phiếu; không có mẫu thừa hoặc thiếu trong phiếu lượt hai.

## 2. Kết quả inter-rater agreement

| Tiêu chí | TB người chấm 1 | TB người chấm 2 | Chênh TB (2−1) | MAE | Trùng điểm | QWK | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fluency | 3,500 | 3,625 | +0,125 | 0,375 | 5/8 (62,5%) | 0,750 | 0,832 |
| Completeness | 3,250 | 2,875 | −0,375 | 0,375 | 5/8 (62,5%) | 0,684 | 0,717 |
| Factuality | 4,125 | 4,125 | 0,000 | 0,000 | 8/8 (100%) | 1,000 | 1,000 |

Trong tổng số 24 cặp điểm (8 mẫu × 3 tiêu chí), 18/24 cặp trùng hoàn toàn (75,0%); cả 24/24 cặp chênh không quá một điểm. Không có cặp nào chênh quá một điểm.

QWK là Cohen's kappa có trọng số bậc hai, phù hợp với thang điểm thứ bậc 1–5. Spearman phản ánh tương quan thứ hạng, không thay thế QWK hoặc tỷ lệ trùng điểm.

## 3. Ánh xạ mẫu và các cặp khác biệt

| Mã lượt hai | Mã lượt một | Fluency (1→2) | Completeness (1→2) | Factuality (1→2) |
|---|---|---:|---:|---:|
| R01 | H33 | 4→4 | 4→3 | 5→5 |
| R02 | H16 | 2→3 | 3→3 | 5→5 |
| R03 | H27 | 3→3 | 3→3 | 3→3 |
| R04 | H28 | 4→4 | 4→3 | 5→5 |
| R05 | H04 | 2→3 | 3→3 | 5→5 |
| R06 | H23 | 4→4 | 4→4 | 5→5 |
| R07 | H03 | 5→5 | 2→1 | 1→1 |
| R08 | H17 | 4→3 | 3→3 | 4→4 |

Khác biệt xuất hiện ở Fluency và Completeness; tất cả chênh lệch chỉ một điểm. Factuality trùng hoàn toàn trên 8 mẫu.

## 4. Sai khác giao thức và giới hạn

D11 ban đầu dự kiến cùng một người chấm chấm lại 8 mẫu sau ít nhất ba ngày để đánh giá intra-rater/test–retest. Thực tế, người chấm lượt hai là một thành viên khác trong nhóm và lượt hai được thực hiện sớm hơn mốc 18/11. Vì vậy, kết quả này phải được gọi là **inter-rater agreement**, không phải intra-rater/test–retest. Đây là thay đổi so với giao thức dự kiến ban đầu và được ghi nhận minh bạch.

Cỡ mẫu chỉ có 8 cặp, nên QWK và Spearman mang tính mô tả/thăm dò, có thể biến động đáng kể. Không suy rộng các chỉ số này thành độ đồng thuận của toàn bộ nhóm hoặc toàn bộ tập test. Người chấm lượt hai được cung cấp rubric, bài gốc và tóm tắt, nhưng không xem điểm hoặc nhận xét lượt một; do đó hai lượt chấm độc lập về mặt không tiếp xúc với đánh giá trước đó. Cách làm này không loại trừ mọi nguồn thiên lệch và cỡ mẫu vẫn nhỏ.

## 5. File dữ liệu và tái lập

- Phiếu lượt đầu: `results/gd7/blind/human_rating_completed.csv`.
- Phiếu lượt hai: `results/gd7/blind/human_rerate_completed.csv`.
- Khóa ghép H–R: `results/gd7/samples_key.csv`.
- Bảng số liệu máy đọc: `results/gd7/human_inter_rater_agreement.csv`.
- Báo cáo tổng hợp: `results/gd7/human_analysis.md`.

Chạy lại bằng lệnh:

```powershell
python -m src.analysis.analyze_human `
  --human_csv results/gd7/blind/human_rating_completed.csv `
  --second_rater_csv results/gd7/blind/human_rerate_completed.csv `
  --key_csv results/gd7/samples_key.csv `
  --config configs/samples.yaml `
  --scores_dir results/scores `
  --output_dir results/gd7
```
