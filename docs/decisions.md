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
- **Hệ quả cho cấu hình:**
  - `max_target = 128` token: tóm tắt p95 = 86 từ (khoảng 110 token) nên 128 bao phủ gần như toàn bộ.
  - `max_input = 512` token (≈ 390 từ): nhỏ hơn cả p25 độ dài bài (446 từ), nên phần lớn bài bị cắt cuối và có thể mất tới ~36% nội dung nguồn (các câu từ vị trí 11 trở đi). Chọn 512 để cân bằng tốc độ, bộ nhớ trên T4 và công bằng giữa các mô hình. Sẽ kiểm chứng bằng đo tỉ lệ bị cắt bằng tokenizer (GĐ3) và ablation 512 vs 1024 (GĐ6); việc làm sạch tiền tố rác ở GĐ3 cũng giải phóng thêm token hữu ích.
- **Đánh đổi:** CNN/DM chủ yếu là tin tức tiếng Anh nên kết quả chưa chắc tổng quát sang miền khác → bổ sung dữ liệu tự thu thập ở GĐ2 để kiểm tra.

## D3. Có thu thập dữ liệu riêng
- **Quyết định:** tự crawl bộ dữ liệu nhỏ có tóm tắt do người viết.
- **Lý do:** điểm cộng theo yêu cầu giảng viên; kiểm tra khả năng tổng quát hóa ngoài CNN/DM.
- **Bằng chứng:** _(điền sau GĐ2)_

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
