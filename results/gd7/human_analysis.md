# Kết quả chấm tay D11 – lượt 1

- Tổng số mẫu: **40** (CNN/DM: **30**, GOV.UK: **10**).
- Người chấm: Phạm Bá Hiếu; một người chấm toàn bộ mẫu.
- CI 95%: bootstrap 1.000 lần theo mẫu, seed 42.
- Ba tiêu chí: Fluency, Completeness, Factuality (thang 1–5).

> **Giới hạn quan trọng:** 40 mẫu được chọn phân tầng theo tứ phân vị S (không phải mẫu ngẫu nhiên đơn giản). Các trung bình/CI và tương quan dưới đây mô tả nhóm 40 mẫu đã chấm; không được diễn giải như ước lượng đại diện cho toàn bộ tập test. GOV.UK chỉ có 10 mẫu nên CI đặc biệt rộng.

## 1. Điểm trung bình

| Phạm vi | Tiêu chí | n | Trung bình | ĐLC | CI bootstrap 95% |
|---|---|---:|---:|---:|---:|
| all | Trôi chảy (Fluency) | 40 | 3.77 | 0.89 | [3.48; 4.05] |
| all | Đầy đủ (Completeness) | 40 | 3.23 | 0.80 | [3.00; 3.45] |
| all | Đúng sự thật (Factuality) | 40 | 4.50 | 1.11 | [4.15; 4.83] |
| cnn_dm | Trôi chảy (Fluency) | 30 | 3.47 | 0.78 | [3.20; 3.73] |
| cnn_dm | Đầy đủ (Completeness) | 30 | 3.37 | 0.81 | [3.10; 3.63] |
| cnn_dm | Đúng sự thật (Factuality) | 30 | 4.73 | 0.83 | [4.40; 5.00] |
| govuk | Trôi chảy (Fluency) | 10 | 4.70 | 0.48 | [4.40; 5.00] |
| govuk | Đầy đủ (Completeness) | 10 | 2.80 | 0.63 | [2.40; 3.20] |
| govuk | Đúng sự thật (Factuality) | 10 | 3.80 | 1.55 | [2.90; 4.70] |

### Phân phối điểm trên 40 mẫu

| Tiêu chí | Điểm 1 | Điểm 2 | Điểm 3 | Điểm 4 | Điểm 5 |
|---|---:|---:|---:|---:|---:|
| Trôi chảy (Fluency) | 0 | 5 | 6 | 22 | 7 |
| Đầy đủ (Completeness) | 0 | 7 | 19 | 12 | 2 |
| Đúng sự thật (Factuality) | 1 | 4 | 1 | 2 | 32 |

## 2. Tương quan giữa điểm chấm tay và chỉ số tự động

Spearman là hệ số chính; Pearson là hệ số phụ. CI 95% bootstrap 1.000 lần. Các hệ số không được dùng để kết luận nhân quả.

| Phạm vi | Tiêu chí chấm tay | Chỉ số | n | Spearman [CI 95%] | Pearson |
|---|---|---|---:|---:|---:|
| all | Trôi chảy (Fluency) | S = TB(ROUGE-1, ROUGE-2, ROUGE-Lsum) | 40 | -0.239 [-0.554; 0.092] | -0.241 |
| all | Trôi chảy (Fluency) | ROUGE-1 | 40 | -0.194 [-0.517; 0.152] | -0.212 |
| all | Trôi chảy (Fluency) | ROUGE-2 | 40 | -0.167 [-0.489; 0.159] | -0.124 |
| all | Trôi chảy (Fluency) | ROUGE-Lsum | 40 | -0.391 [-0.656; -0.081] | -0.347 |
| all | Trôi chảy (Fluency) | BERTScore F1 | 40 | 0.045 [-0.306; 0.394] | 0.014 |
| all | Đầy đủ (Completeness) | S = TB(ROUGE-1, ROUGE-2, ROUGE-Lsum) | 40 | 0.183 [-0.130; 0.467] | 0.184 |
| all | Đầy đủ (Completeness) | ROUGE-1 | 40 | 0.150 [-0.168; 0.445] | 0.153 |
| all | Đầy đủ (Completeness) | ROUGE-2 | 40 | 0.177 [-0.129; 0.444] | 0.183 |
| all | Đầy đủ (Completeness) | ROUGE-Lsum | 40 | 0.195 [-0.098; 0.453] | 0.189 |
| all | Đầy đủ (Completeness) | BERTScore F1 | 40 | 0.373 [0.042; 0.616] | 0.337 |
| all | Đúng sự thật (Factuality) | S = TB(ROUGE-1, ROUGE-2, ROUGE-Lsum) | 40 | 0.038 [-0.325; 0.344] | 0.008 |
| all | Đúng sự thật (Factuality) | ROUGE-1 | 40 | 0.084 [-0.201; 0.383] | 0.019 |
| all | Đúng sự thật (Factuality) | ROUGE-2 | 40 | -0.009 [-0.359; 0.316] | 0.012 |
| all | Đúng sự thật (Factuality) | ROUGE-Lsum | 40 | 0.019 [-0.318; 0.362] | -0.009 |
| all | Đúng sự thật (Factuality) | BERTScore F1 | 40 | -0.196 [-0.473; 0.124] | -0.166 |
| cnn_dm | Trôi chảy (Fluency) | S = TB(ROUGE-1, ROUGE-2, ROUGE-Lsum) | 30 | -0.227 [-0.562; 0.126] | -0.171 |
| cnn_dm | Trôi chảy (Fluency) | ROUGE-1 | 30 | -0.207 [-0.551; 0.148] | -0.191 |
| cnn_dm | Trôi chảy (Fluency) | ROUGE-2 | 30 | -0.130 [-0.427; 0.212] | -0.060 |
| cnn_dm | Trôi chảy (Fluency) | ROUGE-Lsum | 30 | -0.305 [-0.600; 0.058] | -0.235 |
| cnn_dm | Trôi chảy (Fluency) | BERTScore F1 | 30 | 0.056 [-0.280; 0.380] | 0.021 |
| cnn_dm | Đầy đủ (Completeness) | S = TB(ROUGE-1, ROUGE-2, ROUGE-Lsum) | 30 | 0.181 [-0.226; 0.510] | 0.197 |
| cnn_dm | Đầy đủ (Completeness) | ROUGE-1 | 30 | 0.127 [-0.261; 0.479] | 0.186 |
| cnn_dm | Đầy đủ (Completeness) | ROUGE-2 | 30 | 0.194 [-0.183; 0.516] | 0.207 |
| cnn_dm | Đầy đủ (Completeness) | ROUGE-Lsum | 30 | 0.152 [-0.194; 0.464] | 0.174 |
| cnn_dm | Đầy đủ (Completeness) | BERTScore F1 | 30 | 0.520 [0.207; 0.739] | 0.479 |
| cnn_dm | Đúng sự thật (Factuality) | S = TB(ROUGE-1, ROUGE-2, ROUGE-Lsum) | 30 | -0.175 [-0.444; 0.097] | -0.126 |
| cnn_dm | Đúng sự thật (Factuality) | ROUGE-1 | 30 | -0.109 [-0.442; 0.226] | -0.102 |
| cnn_dm | Đúng sự thật (Factuality) | ROUGE-2 | 30 | -0.201 [-0.475; 0.091] | -0.155 |
| cnn_dm | Đúng sự thật (Factuality) | ROUGE-Lsum | 30 | -0.163 [-0.417; 0.095] | -0.106 |
| cnn_dm | Đúng sự thật (Factuality) | BERTScore F1 | 30 | -0.213 [-0.512; 0.124] | -0.216 |
| govuk | Trôi chảy (Fluency) | S = TB(ROUGE-1, ROUGE-2, ROUGE-Lsum) | 10 | -0.342 [-0.871; 0.329] | -0.230 |
| govuk | Trôi chảy (Fluency) | ROUGE-1 | 10 | -0.076 [-0.658; 0.511] | -0.087 |
| govuk | Trôi chảy (Fluency) | ROUGE-2 | 10 | -0.309 [-0.837; 0.352] | -0.209 |
| govuk | Trôi chảy (Fluency) | ROUGE-Lsum | 10 | -0.494 [-0.885; 0.117] | -0.371 |
| govuk | Trôi chảy (Fluency) | BERTScore F1 | 10 | -0.266 [-0.846; 0.297] | -0.318 |
| govuk | Đầy đủ (Completeness) | S = TB(ROUGE-1, ROUGE-2, ROUGE-Lsum) | 10 | 0.007 [-0.579; 0.559] | -0.025 |
| govuk | Đầy đủ (Completeness) | ROUGE-1 | 10 | -0.063 [-0.586; 0.471] | -0.074 |
| govuk | Đầy đủ (Completeness) | ROUGE-2 | 10 | 0.141 [-0.353; 0.658] | 0.053 |
| govuk | Đầy đủ (Completeness) | ROUGE-Lsum | 10 | 0.055 [-0.577; 0.629] | -0.047 |
| govuk | Đầy đủ (Completeness) | BERTScore F1 | 10 | 0.215 [-0.581; 0.886] | 0.233 |
| govuk | Đúng sự thật (Factuality) | S = TB(ROUGE-1, ROUGE-2, ROUGE-Lsum) | 10 | 0.033 [-0.630; 0.790] | 0.021 |
| govuk | Đúng sự thật (Factuality) | ROUGE-1 | 10 | 0.049 [-0.634; 0.737] | 0.040 |
| govuk | Đúng sự thật (Factuality) | ROUGE-2 | 10 | -0.023 [-0.692; 0.654] | 0.139 |
| govuk | Đúng sự thật (Factuality) | ROUGE-Lsum | 10 | -0.085 [-0.677; 0.620] | -0.130 |
| govuk | Đúng sự thật (Factuality) | BERTScore F1 | 10 | 0.033 [-0.624; 0.770] | -0.062 |

### So sánh tương quan cho tiêu chí đầy đủ

Hiệu được tính ghép cặp trên cùng mẫu: Spearman(Completeness, BERTScore) trừ Spearman(Completeness, ROUGE-2). CI không chứa 0 mới gợi ý chênh lệch theo bootstrap; đây vẫn là phân tích thăm dò trên mẫu phân tầng.

| Phạm vi | n | rho với BERTScore | rho với ROUGE-2 | Hiệu | CI 95% của hiệu |
|---|---:|---:|---:|---:|---:|
| all | 40 | 0.373 | 0.177 | 0.196 | [-0.049; 0.453] |
| cnn_dm | 30 | 0.520 | 0.194 | 0.326 | [0.066; 0.623] |
| govuk | 10 | 0.215 | 0.141 | 0.074 | [-0.711; 0.734] |

## 3. Nhãn lỗi do người chấm ghi nhận

Các nhãn có thể chồng lấp vì một mẫu được phép có nhiều loại lỗi; tỷ lệ tính trên số mẫu của từng phạm vi.

| Phạm vi | Nhãn | Số mẫu có nhãn | Tỷ lệ trên phạm vi |
|---|---|---:|---:|
| all | `hallucination` | 6 | 15.0% |
| all | `missing_information` | 26 | 65.0% |
| all | `none` | 12 | 30.0% |
| all | `other` | 5 | 12.5% |
| cnn_dm | `hallucination` | 3 | 10.0% |
| cnn_dm | `missing_information` | 17 | 56.7% |
| cnn_dm | `none` | 12 | 40.0% |
| cnn_dm | `other` | 5 | 16.7% |
| govuk | `hallucination` | 3 | 30.0% |
| govuk | `missing_information` | 9 | 90.0% |

## 4. Độ đồng thuận giữa hai người chấm (inter-rater agreement)

Lượt 1 do Phạm Bá Hiếu thực hiện; lượt 2 do một thành viên khác trong nhóm thực hiện. Người chấm lượt hai đã đọc cùng rubric, chỉ được cung cấp bài gốc và tóm tắt cần đánh giá, không được xem điểm hoặc nhận xét lượt một trước khi chấm. Vì vậy, đây là độ đồng thuận giữa hai người chấm (inter-rater agreement), không phải tính tự nhất quán/test–retest của cùng một người.
Tám mẫu lượt hai được ghép với lượt đầu bằng khóa Hxx–Rxx trong `samples_key.csv`; mã và nội dung bài/tóm tắt đã được kiểm tra. Các ước lượng chỉ mang tính mô tả/thăm dò do n=8.

| Tiêu chí | n | TB người 1 | TB người 2 | Chênh TB (2−1) | QWK bậc hai | Spearman | Trùng điểm | Sai khác ≤1 | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Trôi chảy (Fluency) | 8 | 3.500 | 3.625 | 0.125 | 0.750 | 0.832 | 62.5% | 100.0% | 0.375 |
| Đầy đủ (Completeness) | 8 | 3.250 | 2.875 | -0.375 | 0.684 | 0.717 | 62.5% | 100.0% | 0.375 |
| Đúng sự thật (Factuality) | 8 | 4.125 | 4.125 | 0.000 | 1.000 | 1.000 | 100.0% | 100.0% | 0.000 |

Trên 24 cặp điểm (8 mẫu × 3 tiêu chí), tỷ lệ trùng điểm trung bình theo tiêu chí là 75.0%; tỷ lệ chênh không quá 1 điểm là 100.0%.
Không suy rộng các chỉ số này thành độ đồng thuận của toàn bộ nhóm hoặc toàn bộ tập test.

## 5. Diễn giải và hạn chế

- Điểm trung bình chỉ mô tả 40 mẫu đã chọn; không suy rộng sang toàn bộ 1.000 mẫu CNN/DM hoặc 389 mẫu GOV.UK.
- Mẫu phân tầng theo S làm thay đổi phân phối ROUGE trong mẫu chấm; tương quan trên 40 mẫu có thể khác tương quan tính trên toàn bộ test.
- Lượt đầu do Phạm Bá Hiếu chấm 40 mẫu; lượt hai do một thành viên khác chấm 8 mẫu. Người chấm thứ hai đọc rubric và chỉ xem bài gốc cùng tóm tắt, không được xem điểm/nhận xét lượt một. Phần so sánh là inter-rater agreement trên 8 mẫu, không đánh giá intra-rater/test–retest.
- Tám mẫu được chọn lại từ 40 mẫu ban đầu; cỡ mẫu nhỏ nên QWK/Spearman có thể biến động và chỉ mang tính mô tả/thăm dò.
- Chưa hiệu chỉnh đa so sánh cho nhiều hệ số tương quan; cần diễn giải CI và quy mô mẫu, không chỉ nhìn trị tuyệt đối của hệ số.
- ROUGE/BERTScore đo mức tương đồng với tham chiếu, không trực tiếp xác minh tính đúng sự thật so với bài gốc.
