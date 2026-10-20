# Nhật ký quyết định (Decision Log)

Mục đích: trả lời câu hỏi của giảng viên *"vì sao chọn phương pháp này mà không phải phương pháp khác, và nó tốt hơn ở điểm nào?"*.
Quy ước: mỗi quyết định ghi **bối cảnh → các lựa chọn → quyết định → lý do → bằng chứng thực nghiệm**.
Dòng "Bằng chứng" được **điền sau khi có kết quả** (GĐ4–7). Nếu số liệu đi ngược giả thuyết ban đầu, sửa quyết định và ghi lại lý do.

---

## D1. Loại tóm tắt: abstractive làm hệ thống chính
- **Lựa chọn:** (a) extractive, (b) abstractive, (c) lai.
- **Quyết định (giả thuyết):** abstractive là hệ thống chính; extractive làm baseline.
- **Lý do:** extractive chỉ chọn câu có sẵn nên khó cô đọng, dễ rời rạc; abstractive diễn đạt lại, gần cách con người tóm tắt.
- **Đánh đổi:** abstractive có rủi ro "ảo giác" (hallucination) và tốn tài nguyên hơn.
- **Bằng chứng:** _(điền sau GĐ7: ROUGE, BERTScore, phân tích lỗi)_

## D2. Dữ liệu huấn luyện chính: CNN/DailyMail
- **Lựa chọn:** CNN/DailyMail, XSum, Gigaword, Multi-News.
- **Quyết định:** CNN/DailyMail (tập con 20.000 train / 1.000 val / 1.000 test, seed 42).
- **Lý do:** benchmark phổ biến nhất, dễ so sánh với công trình khác, đủ lớn để lấy tập con huấn luyện trên GPU miễn phí của Colab.
- **Bằng chứng (EDA, GĐ1, trên tập con train 20.000 mẫu):**
  - **Độ dài:** bài trung vị 637 từ (p95 = 1.353), tóm tắt trung vị 46 từ (p95 = 86), trung bình 3,8 câu tóm tắt; tỉ lệ nén trung vị 13,6 lần.
  - **Mức độ abstractive:** 13,0% unigram, 52,2% bigram và 71,7% trigram của tóm tắt không xuất hiện trong bài. Tóm tắt dùng lại phần lớn từ vựng nhưng sắp xếp, rút gọn, ghép cụm khác đi. Hơn một nửa bigram là mới nên phương pháp chỉ trích nguyên câu (extractive) bị giới hạn về ROUGE-2 → là lý do để thử mô hình sinh (abstractive).
  - **Lead bias:** 25,6% câu tóm tắt có câu nguồn nằm trong 3 câu đầu và 64,0% nằm trong 10 câu đầu. Dữ liệu thiên về đầu bài (so với ước tính ~10% cho 3 câu đầu nếu phân bố đều), nhưng khoảng 36% nội dung tóm tắt nằm từ câu 11 trở đi. Lead-3 có thể là baseline khá mạnh; sẽ kiểm chứng ở GĐ4.
  - **Rác dữ liệu:** 62,4% bài chứa ít nhất một dấu vết cần làm sạch: tiền tố Daily Mail "By ." (34,2%), dòng "PUBLISHED/UPDATED" (22,4%), "(CNN)" (26,0%), "read more" (2,6%), Reuters (0%) → cần làm sạch ở GĐ3 (bỏ quy tắc Reuters).
  - **Ngoại lệ:** có mẫu tóm tắt tới 1.167 từ / 128 câu và bài chỉ 39 từ → cần lọc theo độ dài ở GĐ3.
- **Hệ quả cho cấu hình (đã đo bằng tokenizer BART ở GĐ3, xem D7):**
  - `max_input = 512` token: cắt 79,0% bài CNN/DM (mất 43,7% tổng token). Dùng 512 làm cấu hình mặc định và kiểm chứng bằng ablation 512 vs 1.024 (1.024 chỉ cắt 29,1% bài, mất 12,4% token) ở GĐ6.
  - `max_target = 128` token: bao phủ 98,0% tóm tắt CNN/DM; `64` token cho GOV.UK: bao phủ 99,3%.
- **Đánh đổi:** CNN/DM chủ yếu là tin tức tiếng Anh nên kết quả chưa chắc tổng quát sang miền khác → bổ sung dữ liệu tự thu thập ở GĐ2 để kiểm tra.

## D3. Có thu thập dữ liệu riêng: GOV.UK
- **Quyết định:** tự thu thập bộ tin tức tiếng Anh có tóm tắt do người viết từ GOV.UK qua Content/Search API.
- **Lý do:** điểm cộng theo yêu cầu giảng viên; kiểm tra khả năng tổng quát hóa ngoài CNN/DM.
- **Vì sao GOV.UK, không phải nguồn khác:** giấy phép OGL v3.0 cho phép tái sử dụng với điều kiện ghi nguồn; có API chính thức, không cần scrape HTML; có trường `description` do biên tập viên viết. Guardian bị loại vì khóa miễn phí chỉ cho phi thương mại và điều khoản về dùng cho huấn luyện mô hình chưa rõ ràng.
- **Quy trình:** liệt kê bài theo (loại bài × tháng) → tải nội dung (có cache, thử lại có chờ tăng dần khi gặp 503) → HTML sang văn bản → 8 bộ lọc → khử trùng lặp → chia 80/10/10 (seed 42).
- **Bằng chứng:**
  - **Quy mô:** 5.477 bài thô (2023-01 → 2026-09), không bài nào lỗi sau khi tải; còn 3.878 cặp sau làm sạch (70,8%), chia train/val/test = 3.102 / 387 / 389. Loại bài: news_story 1.636, press_release 1.400, government_response 66 (tập train).
  - **Phễu làm sạch:** bước loại nhiều nhất là tóm tắt chép lại bài: 609 bài chép nguyên văn và 627 bài gần như chép một câu (overlap từ ≥ 0,9), cộng 1.236 bài (22,6% số bài thô). Tiếp theo là độ dài bài không hợp lệ (236 bài) và độ dài tóm tắt (55 bài). Khử trùng lặp loại 51 bài.
  - **Ngưỡng lọc 0,9:** được chọn sau khi xem phân bố trên mẫu thử 300 bài (overlap ≥ 0,7 / 0,8 / 0,9 / 1,0 lần lượt chiếm 42,7% / 31,3% / 21,0% / 14,3%), tức ngưỡng được đặt sau khi quan sát dữ liệu. Tỉ lệ bài bị loại ở bước này trên toàn bộ (13,8%) khớp với trên mẫu thử (13,5%).
  - **Độ dài:** bài trung vị 560,5 từ (p95 = 1.453,9), tóm tắt trung vị 21 từ (p95 = 34, max = 63), 1,04 câu tóm tắt; tỉ lệ nén 26,9 lần (CNN/DM: 637 từ, 46 từ, 3,8 câu, 13,6 lần).
  - **n-gram mới:** 19,0% unigram, 56,0% bigram, 74,1% trigram (CNN/DM: 13,0 / 52,2 / 71,7). Không so sánh trực tiếp được vì ta đã lọc bỏ các tóm tắt gần như chép một câu và tóm tắt GOV.UK ngắn hơn.
  - **Lead bias:** 67,1% câu tóm tắt có nguồn ở 3 câu đầu và 87,6% ở 10 câu đầu (CNN/DM: 25,6% và 64,0%).
- **Hệ quả cho các giai đoạn sau:**
  - Lead-1 / Lead-3 dự kiến là baseline mạnh trên bộ này (kiểm chứng ở GĐ4).
  - `max_target = 64` cho GOV.UK (tóm tắt dài nhất 63 từ). Cắt đầu vào 512 token ít gây mất thông tin hơn ở CNN/DM.
  - Mô hình huấn luyện trên CNN/DM sinh tóm tắt dài hơn nhiều so với tham chiếu GOV.UK → báo cáo kèm độ dài tóm tắt và bổ sung thí nghiệm fine-tune trên train GOV.UK.
  - Tập test chỉ 389 mẫu → dùng bootstrap để tính khoảng tin cậy ở GĐ7.
- **Đánh đổi:** tóm tắt chỉ một câu và miền hẹp (thông tin chính phủ Anh); `description` do biên tập viên viết cho trang web, chưa được người kiểm định chất lượng nhãn (sẽ chấm tay mẫu nhỏ ở GĐ7).

## D4. Mô hình chính: BART-base (so sánh với T5-small, Seq2Seq+Attention)
- **Lựa chọn:** Seq2Seq+Attention (từ đầu), BART, T5, PEGASUS.
- **Quyết định (giả thuyết):** BART-base.
- **Lý do:** tiền huấn luyện dạng denoising phù hợp bài toán sinh; ~140M tham số, vừa GPU T4 khi dùng fp16; kết quả mạnh trên CNN/DM.
- **Vì sao không chọn khác:**
  - Seq2Seq+Attention từ đầu: cần nhiều dữ liệu, hội tụ chậm, kết quả thường thấp hơn.
  - T5-small: nhẹ nhưng dung lượng nhỏ → dự kiến thấp hơn BART-base.
  - PEGASUS: mạnh nhưng nặng hơn, vượt tài nguyên miễn phí.
- **Bằng chứng:** _(điền sau GĐ6–7)_

## D5. Chỉ số đánh giá: ROUGE-1/2/L + BERTScore (+ độ dài, tốc độ)
- **Lý do:** ROUGE là chuẩn của tóm tắt nhưng chỉ đo trùng từ; BERTScore bổ sung độ tương đồng ngữ nghĩa; thêm phân tích định tính cho hallucination.
- **Đánh đổi:** cả hai không đo được tính đúng sự thật → cần chấm tay mẫu nhỏ.

## D6. Hạ tầng và phạm vi
- Colab/Kaggle GPU T4, tập con ~20–30k mẫu, fp16.
- Seed cố định, mỗi thí nghiệm một file config trong `configs/`.
- Checkpoint lưu ngoài GitHub (Drive/HuggingFace Hub).
## D7. Tiền xử lý và cấu hình độ dài
- **Làm sạch CNN/DM:** bỏ tiền tố Daily Mail (byline, PUBLISHED/UPDATED), dateline CNN (kể cả địa danh viết hoa/thường và kiểu "(CNN)Nội dung" không có dấu "--"), cụm rác giữa bài (Scroll down for video, E-mail to a friend, Share this article) và URL; chuẩn hóa khoảng trắng. Nội dung bị thay đổi ở 14.084/20.000 bài train (70,4%), 333/1.000 val (33,3%) và 356/1.000 test (35,6%).
- **Kết quả kiểm tra (120 ký tự đầu, train):** "By ." giảm từ 34,16% xuống 0,02%; PUBLISHED/UPDATED từ 21,96% xuống 0,14%; "(CNN)" từ 25,89% xuống 0,03%. Còn sót 33/19.895 bài train (0,17%), 0 bài ở val và test. Quy tắc làm sạch được hiệu chỉnh hai vòng sau khi kiểm tra các bài còn sót (vòng 1 còn sót 8,5% ở train do dateline viết thường và byline có từ nối).
- **Khác biệt định dạng giữa các tập:** trước khi làm sạch, val/test không có tiền tố Daily Mail (0%) nhưng train có "By ." ở 34,16% và PUBLISHED/UPDATED ở 21,96%. Nếu không làm sạch, mô hình sẽ học các tiền tố không bao giờ xuất hiện ở tập test.
- **Lọc ngoại lệ:** chỉ ở train CNN/DM (bài < 50 từ, tóm tắt < 10 hoặc > 150 từ, > 10 câu): loại 105/20.000 bài (0,5%), còn 19.895. Val (1.000) và test (1.000) giữ nguyên để so sánh công bằng với công trình khác. GOV.UK đã lọc ở GĐ2, chỉ thống nhất khoảng trắng (3.102 / 387 / 389).
- **Rò rỉ dữ liệu** (trùng 300 ký tự đầu đã chuẩn hóa với train, tính sau làm sạch): CNN/DM validation 0, test 0; GOV.UK validation 1/387 (0,26%), test 0. Bài GOV.UK trùng ở validation được giữ lại vì tập này chỉ dùng chọn mô hình và tác động không đáng kể.
- **Đầu vào mô hình:** BART nhận văn bản thuần, T5 thêm tiền tố "summarize: "; tóm tắt nhiều câu nối bằng khoảng trắng khi huấn luyện, giữ xuống dòng để tính ROUGE-Lsum; đệm động (DataCollatorForSeq2Seq, nhãn đệm -100).
- **Độ dài (đo bằng tokenizer thật, train, sau làm sạch):**
  - CNN/DM (BART): trung vị 788, p95 1.690, p99 2.104 token. Ở 512 token: 79,0% bài bị cắt, mất 43,7% token; ở 1.024 token: 29,1% bị cắt, mất 12,4%. T5 (có tiền tố) cắt nhiều hơn: 84,8% (mất 49,4%) ở 512 và 38,4% (mất 17,1%) ở 1.024.
  - GOV.UK (BART): trung vị 673, p95 1.737 token; ở 512 token: 64,3% bị cắt, mất 43,1% token; ở 1.024 token: 26,3%, mất 14,4%.
  - Tóm tắt: CNN/DM trung vị 60, p95 109 token; 2,0% dài hơn 128 token (BART; T5: 4,9%), mất 0,5% token. GOV.UK trung vị 27, p95 44 token; 0,7% dài hơn 64 token (BART), 0% dài hơn 128.
  - **Quyết định:** `max_input = 512` làm cấu hình mặc định cho mọi mô hình (huấn luyện nhanh, vừa GPU T4, công bằng khi so sánh); `max_target = 128` (CNN/DM) và `64` (GOV.UK). BART-base chạy thêm `max_input = 1024` như ablation ở GĐ6.
- **Đánh đổi:** cắt ở 512 token làm mất thông tin cuối bài (79% bài CNN/DM bị cắt). Với CNN/DM, 64% câu tóm tắt có nguồn nằm trong 10 câu đầu nên việc cắt chấp nhận được nhưng không miễn phí; ablation 512 vs 1024 ở GĐ6 sẽ đo tác động thực tế lên ROUGE. Các regex làm sạch được thiết kế theo định dạng của CNN/DM nên không dùng được nguyên cho nguồn khác.
## D8. Giao thức đánh giá baseline (GĐ4)
- **Hệ thống:** Lead-k, Random-k, TextRank-k, Oracle-k (k = 1, 2, 3); test CNN/DM (1.000) và GOV.UK (389).
- **Chỉ số:** ROUGE-1/2/L/Lsum (có stemming), BERTScore F1 (roberta-large, rescale), độ dài, ms/doc; bootstrap 95% (1.000 lần, seed 42).
- **Dự đoán trước khi chạy:**
  1. CNN/DM: Lead-3 > TextRank-3 về ROUGE-1 và ROUGE-2.
  2. GOV.UK: Lead-1 là baseline extractive tốt nhất; Lead-3 thấp hơn Lead-1 vì dài hơn tham chiếu (~21 từ).
  3. Cả hai bộ: Lead-k > Random-k, và Oracle chọn câu ở 3 câu đầu nhiều hơn mức phân bố đều.
- **Kết quả / kết luận:** _(điền sau khi chạy)_