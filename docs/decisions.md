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
- **Quyết định (giả thuyết):** CNN/DailyMail.
- **Lý do:** benchmark phổ biến nhất, dễ so sánh với công trình khác, đủ lớn để lấy tập con huấn luyện trên GPU miễn phí.
- **Đánh đổi:** thiên về extractive (ý chính thường nằm ở đầu bài) → cần ghi nhận khi đánh giá.
- **Bằng chứng:** _(điền sau GĐ1: EDA mức độ trùng lặp từ bài–tóm tắt)_

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
