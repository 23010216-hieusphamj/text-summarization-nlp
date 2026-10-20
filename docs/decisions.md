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
- **Bằng chứng (GĐ4):** mức trần của extractive không thấp (Oracle-3 CNN/DM: R1 56,7, R2 33,5; Oracle-1 GOV.UK: R1 43,1, R2 23,6), nên không thể nói extractive "bị giới hạn" bởi chính cách chọn câu. Điều số liệu cho thấy là extractive thực tế còn rất xa mức trần: Lead-3 chỉ đạt 71% R1 và 52% R2 của Oracle-3, vì không biết trước câu nào quan trọng (chỉ 14,2% bài CNN/DM có câu tốt nhất là câu đầu tiên). Việc abstractive có thực sự tốt hơn sẽ được kiểm chứng bằng BART-base ở GĐ6.
- **Bằng chứng (GĐ7):** _(điền sau: ROUGE, BERTScore, phân tích lỗi)_

## D2. Dữ liệu huấn luyện chính: CNN/DailyMail
- **Lựa chọn:** CNN/DailyMail, XSum, Gigaword, Multi-News.
- **Quyết định:** CNN/DailyMail (tập con 20.000 train / 1.000 val / 1.000 test, seed 42).
- **Lý do:** benchmark phổ biến nhất, dễ so sánh với công trình khác, đủ lớn để lấy tập con huấn luyện trên GPU miễn phí của Colab.
- **Bằng chứng (EDA, GĐ1, trên tập con train 20.000 mẫu):**
  - **Độ dài:** bài trung vị 637 từ (p95 = 1.353), tóm tắt trung vị 46 từ (p95 = 86), trung bình 3,8 câu tóm tắt; tỉ lệ nén trung vị 13,6 lần.
  - **Mức độ abstractive:** 13,0% unigram, 52,2% bigram và 71,7% trigram của tóm tắt không xuất hiện trong bài. Tóm tắt dùng lại phần lớn từ vựng nhưng sắp xếp, rút gọn, ghép cụm khác đi. Hơn một nửa bigram là mới nên tóm tắt tham chiếu mang tính diễn đạt lại, không chỉ chép câu gốc. Ở GĐ4, mức trần extractive (Oracle) vẫn cao (R2 33,5), nên lý do chọn abstractive không phải là "extractive bị chặn ở ROUGE-2" mà là extractive thực tế còn rất xa mức trần đó (xem D1, D8).
  - **Lead bias:** 25,6% câu tóm tắt có câu nguồn nằm trong 3 câu đầu và 64,0% nằm trong 10 câu đầu. Dữ liệu thiên về đầu bài (so với mức phân bố đều ~12,7% cho 3 câu đầu, tính theo số câu của từng bài test), nhưng khoảng 36% nội dung tóm tắt nằm từ câu 11 trở đi. Đã kiểm chứng ở GĐ4 (xem D8): Lead-3 là baseline extractive mạnh nhất (R1 40,3; R2 17,6), vượt TextRank-3 và Random-3 có ý nghĩa thống kê. Phép đo bằng Oracle theo ROUGE cho thấy câu tốt nhất nằm trong 3 câu đầu ở 39,8% bài (gấp 3,1 lần mức phân bố đều 12,7%) và trong 10 câu đầu ở 77,2%; nhưng chỉ 14,2% bài có câu tốt nhất là câu đầu tiên, nên Lead-1 yếu (R1 27,9) còn Lead-3 mới mạnh.
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
  - **Hiệu chỉnh ở GĐ4:** phép đo lead bias bằng độ trùng từ cho 67,1% câu tóm tắt có nguồn ở 3 câu đầu, cao hơn nhiều so với phép đo bằng Oracle theo ROUGE (câu tốt nhất nằm trong 3 câu đầu ở 50,4% bài GOV.UK và 39,8% bài CNN/DM). So với mức phân bố đều, lead bias của GOV.UK (2,6×) không mạnh hơn CNN/DM (3,1×); mức phân bố đều của GOV.UK cao hơn (19,7% so với 12,7%), nghĩa là bài ít câu hơn. Tuy nhiên câu đầu tiên là câu tốt nhất ở 34,5% bài GOV.UK so với 14,2% ở CNN/DM. Lead-1 mạnh trên GOV.UK nhiều khả năng do tóm tắt chỉ ~1 câu và câu đầu bài hay chứa ý chính; đây là suy luận, chưa được kiểm tra riêng.
- **Hệ quả cho các giai đoạn sau:**
  - Đã kiểm chứng ở GĐ4 (xem D8): Lead-1 là baseline extractive mạnh nhất trên bộ này; Lead-2/3 thấp hơn vì dài hơn tham chiếu (~1 câu).
  - `max_target = 64` cho GOV.UK (tóm tắt dài nhất 63 từ). Cắt đầu vào 512 token ít gây mất thông tin hơn ở CNN/DM.
  - Mô hình huấn luyện trên CNN/DM sinh tóm tắt dài hơn nhiều so với tham chiếu GOV.UK → báo cáo kèm độ dài tóm tắt và bổ sung thí nghiệm fine-tune trên train GOV.UK.
  - Tập test chỉ 389 mẫu → dùng bootstrap để tính khoảng tin cậy (đã áp dụng từ GĐ4).
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
- **Cài đặt cố định (GĐ4):** BERTScore F1, roberta-large, rescale_with_baseline; ROUGE có stemming, báo cáo R1/R2/L/Lsum; CI 95% bootstrap 1.000 lần (kiểm định ghép cặp 2.000 lần), seed 42. Giữ nguyên cài đặt này ở GĐ6–7. Với cùng k, BERTScore và ROUGE xếp hạng Lead/TextRank/Random giống nhau, trừ CNN/DM k=1 (Lead-1 vs TextRank-1, BERTScore bằng nhau). Thứ hạng giữa các cấu hình khác k hoặc các hệ thống gần ngang nhau (Random-k, Oracle-2 vs Oracle-3) có thể đảo chiều giữa hai chỉ số.
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
- **Kết quả (test CNN/DM n=1.000, GOV.UK n=389; CI 95% bootstrap, kiểm định ghép cặp trên ROUGE):**
  1. CNN/DM: Lead-3 vượt TextRank-3 ở mọi chỉ số (R1 40,3 vs 35,2; R2 17,6 vs 13,8; BERTScore 26,0 vs 21,6; hiệu R1 +5,1, CI [4,4; 5,9]). Lead-3 xấp xỉ mức công bố của văn liệu (~40/17/37) phù hợp với việc pipeline hoạt động đúng.
  2. GOV.UK: Lead-1 là baseline extractive tốt nhất (R1 33,1; R2 15,0; BERTScore 30,0), vượt TextRank-1 (24,8; 8,1; 22,6) và vượt Lead-3 (R1 +9,1); điểm giảm dần khi tăng k vì tham chiếu chỉ ~1 câu.
  3. Lead-k vượt Random-k ở cả 6 cấu hình. Oracle-1 nằm trong 3 câu đầu ở 39,8% (CNN/DM) và 50,4% (GOV.UK), cao hơn mức phân bố đều 12,7% và 19,7%.
  4. Ngoại lệ: CNN/DM k=1, TextRank-1 nhỉnh hơn Lead-1 (R1 +1,4, CI [0,5; 2,3]) nhưng BERTScore bằng nhau và TextRank-1 dài hơn 17%; chênh lệch nhỏ, chưa hiệu chỉnh nhiều phép so sánh.
  5. Khoảng cách tới Oracle còn lớn: Lead-3 đạt 71% R1 và 52% R2 của Oracle-3 (CNN/DM); Lead-1 đạt 77% R1 của Oracle-1 (GOV.UK).
- **Kết luận:** giả thuyết lead bias được xác nhận ở cả hai bộ. Lead-3 (CNN/DM) và Lead-1 (GOV.UK) là mốc extractive thực tế cần vượt ở GĐ6.
- **Hạn chế:** Oracle là xấp xỉ tham lam dùng tham chiếu nên không phải hệ thống thực tế. Lead-1 GOV.UK dài hơn tham chiếu 72% (36,6 vs 21,3 từ). Bộ lọc overlap ≥ 0,9 ở GĐ2 có thể làm lệch test GOV.UK. Test GOV.UK chỉ 389 mẫu (CI rộng ~±1,7 điểm R1). Kiểm định ghép cặp mới làm cho ROUGE, chưa làm cho BERTScore, và chưa hiệu chỉnh đa so sánh.

## D9. Mô hình Seq2Seq + Attention huấn luyện từ đầu (GĐ5)
- **Bối cảnh:** GĐ4 cho mốc extractive rất mạnh (Lead-3: R1 40,3 / R2 17,6 trên CNN/DM). Cần một mô hình neural **không tiền huấn luyện** để (i) có đường cong hội tụ "từ đầu" đối chiếu với BART/T5 ở GĐ6, (ii) biện minh vì sao cần Transformer tiền huấn luyện.
- **Lựa chọn:**
  - Kiến trúc: (a) GRU/LSTM + attention, (b) thêm pointer-generator/coverage, (c) Transformer nhỏ từ đầu.
  - Từ vựng: (a) word-level, (b) SentencePiece BPE huấn luyện riêng, (c) dùng lại tokenizer BART (không dùng trọng số).
- **Quyết định (giả thuyết):**
  - Encoder BiGRU 2 tầng, decoder GRU 1 tầng, attention cộng (Bahdanau); embedding 256, hidden 256, dropout 0,3; output chia sẻ trọng số với embedding (~7–8 triệu tham số, ước lượng).
  - SentencePiece BPE 16k, huấn luyện **chỉ trên train CNN/DM sau làm sạch** (tránh rò rỉ), dùng chung cho bài và tóm tắt. Id đặc biệt: `<pad>=0, <unk>=1, <s>=2, </s>=3`.
  - `max_input = 512`, `max_target = 128`, **giống BART/T5 (D7)** để so sánh công bằng giữa Seq2Seq và các mô hình tiền huấn luyện.
  - Huấn luyện: Adam lr 1e-3, batch 32, clip gradient 1,0, teacher forcing, early stopping theo val loss, seed 42; lưu loss mỗi epoch vào `results/logs/`.
  - Giải mã: beam search 4 beam, chặn lặp trigram; so sánh với greedy trong ablation nhỏ.
  - Huấn luyện trên train CNN/DM; GOV.UK (3.102 mẫu train) quá ít để huấn luyện từ đầu nên chỉ **đánh giá chéo miền** (train CNN/DM → test GOV.UK).
- **Lý do:** GRU nhẹ hơn LSTM, vừa GPU T4 của Colab; BPE tránh từ ngoài từ vựng (tên riêng nhiều trong tin tức) mà không cần cơ chế copy; cùng bộ chỉ số và cài đặt đánh giá với D5/D8 nên so sánh được trực tiếp với baseline.
- **Chốt `max_input = 512`:** thử 256 / 400 / 512 bằng tokenizer SentencePiece (xem bằng chứng). Chọn 512 vì (i) tỉ lệ token bị mất (46,5%) gần với BART ở 512 (43,7%, D7), nên điều kiện đầu vào giữa Seq2Seq và BART/T5 tương đương; (ii) 400 mất nhiều hơn khoảng 10 điểm phần trăm token (57,1%) trong khi tiết kiệm thời gian huấn luyện không nhiều (độ phức tạp GRU gần tuyến tính theo độ dài, ước lượng ~20%).
- **Không chọn:** pointer-generator/coverage (giảm lặp và ảo giác nhưng làm phạm vi GĐ5 phình ra; nêu như hướng mở rộng); Transformer từ đầu (tốn tài nguyên, trùng vai trò với BART ở GĐ6).
- **Dự đoán trước khi chạy** (chưa có số liệu, chỉ là giả thuyết):
  1. Seq2Seq **thấp hơn Lead-3** về ROUGE-1 trên CNN/DM, với CI không chồng lấn (dự kiến R1 khoảng 25–33, R2 khoảng 5–12; khoảng này là ước đoán, không phải kết quả).
  2. Không chặn lặp trigram thì tóm tắt lặp câu/cụm nhiều hơn rõ rệt, ROUGE thấp hơn bản có chặn.
  3. Val loss ngừng giảm sau một số epoch trong khi train loss vẫn giảm (quá khớp do chỉ ~20k mẫu).
  4. Chéo miền sang GOV.UK kém hơn Lead-1 (R1 33,1) và sinh dài hơn tham chiếu (~21 từ).
  5. Nếu Seq2Seq **vượt** Lead-3, coi là kết quả bất thường và kiểm tra rò rỉ dữ liệu, lỗi đánh giá trước khi ghi nhận.
- **Bằng chứng (từ vựng, đã đo):**
  - SentencePiece BPE 16k cho 1,33 token/từ (CNN/DM train); tỉ lệ `<unk>` 0,18% (CNN/DM) và 0,33–0,39% (GOV.UK).
  - CNN/DM train, đầu vào: ở 256 / 400 / 512 token, bị cắt 98,0% / 90,8% / 81,9% bài, mất 72,1% / 57,1% / 46,5% token (tương đương khoảng 193 / 301 / 385 từ giữ lại, xấp xỉ). Số đo trên val và test gần với train.
  - Tóm tắt: `max_target = 128` cắt 2,8% mẫu train CNN/DM, mất 0,8% token; GOV.UK ở 128 là 0%.
  - Chi tiết: `results/tables/spm_truncation_report.csv`, `spm_summary.csv`.
- **Kết quả (test CNN/DM n=1.000, GOV.UK n=389; checkpoint epoch 13; CI 95% bootstrap; cấu hình chính beam 4 + chặn lặp trigram được chốt ở D9 trước khi chạy):**
  1. Huấn luyện: 15 epoch (không dừng sớm), tốt nhất ở epoch 13 (val loss 3,674; ppl 39,4); 7.463.040 tham số; 4.384 giây (~73 phút) trên Tesla T4. Val loss gần như phẳng từ epoch ~10 trong khi train loss vẫn giảm (3,43 so với 3,68 ở epoch 15).
  2. CNN/DM: Seq2Seq (beam 4, chặn lặp) đạt R1 33,3 / R2 13,0 / Lsum 30,9 / BERTScore 20,8, thấp hơn Lead-3 (40,3 / 17,6 / 36,5 / 26,0): hiệu R1 −7,0 CI [−7,8; −6,3], R2 −4,6 [−5,3; −3,9]. Cao hơn Random-3 (R1 28,5) và thấp hơn TextRank-3 (35,2) theo R1 (CI không chồng lấn: [32,6; 34,0] so với [34,5; 35,9]; so sánh này chưa ghép cặp).
  3. Ablation giải mã (CNN/DM, R1/R2): greedy 27,6/8,5; beam 4 không chặn lặp 31,5/12,1; beam 4 + chặn lặp 33,3/13,0. Chặn lặp tăng R1 +1,8 (CI [1,4; 2,1]) so với beam 4 không chặn và +5,7 ([5,1; 6,3]) so với greedy. Tỉ lệ dự đoán có trigram từ lặp: 70,2% / 56,5% / 2,2%.
  4. GOV.UK (chéo miền, không huấn luyện trên GOV.UK): R1 24,9 / R2 8,0 / Lsum 20,3 / BERTScore 14,3, thấp hơn Lead-1 (33,1 / 15,0 / 25,5 / 30,0): hiệu R1 −8,2 CI [−9,7; −6,6]. So với Lead-3 (R1 23,9) khác biệt R1 không có ý nghĩa (+1,0, CI [−0,1; 2,0]), R2 thấp hơn (−2,7). Độ dài sinh ra 37,4 từ so với tham chiếu 21,3 (+76%). Chặn lặp: R1 +1,3 (CI [0,8; 1,9]); tỉ lệ lặp 60,7% / 45,8% / 2,6%.
  5. Thời gian giải mã khoảng 100 ms/bài (beam 4) và 81–83 ms/bài (greedy).
- **Đối chiếu dự đoán:** (1) đúng hướng (thấp hơn Lead-3, CI không chồng lấn) nhưng khoảng ước đoán trượt nhẹ: R1 33,3 (dự kiến 25–33), R2 13,0 (dự kiến 5–12). (2) đúng, mạnh hơn kỳ vọng. (3) đúng ở mức vừa (chênh train/val 0,25). (4) đúng cả hai vế. (5) không xảy ra.
- **Kết luận:** Seq2Seq huấn luyện từ đầu trên 20k mẫu chưa vượt baseline extractive mạnh nhất (Lead-3 ở CNN/DM, Lead-1 ở GOV.UK), nên Lead-k vẫn là mốc cần vượt ở GĐ6. Trong cấu hình giải mã, chặn lặp là yếu tố quan trọng nhất (ROUGE tăng rõ và lỗi lặp giảm từ 56–70% xuống 2%). Beam 4 + chặn lặp trigram là mốc "neural từ đầu" dùng ở GĐ6–7.
- **Hạn chế:** một seed và một lần huấn luyện (CI chỉ phản ánh mẫu test, không phản ánh phương sai huấn luyện); siêu tham số mô hình và `length_penalty` (=1,0) chưa tinh chỉnh (nếu tinh chỉnh phải làm trên val, không phải test); chặn lặp ở mức token SentencePiece nên vẫn còn 2,2% dự đoán lặp trigram từ; ablation không có BERTScore; beam 4 so với greedy chưa kiểm định ghép cặp; chưa phân tích định tính (để GĐ7); độ dài sinh ra ngắn hơn tham chiếu 29% ở CNN/DM (37,5 so với 52,6 từ) và dài hơn 76% ở GOV.UK; kết quả không tách được ảnh hưởng của kiến trúc, lượng dữ liệu và tiền huấn luyện, cần đối chiếu với BART/T5 ở GĐ6.
