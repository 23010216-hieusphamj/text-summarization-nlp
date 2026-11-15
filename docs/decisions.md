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
- **Bằng chứng:** 
  - GĐ6/7: A (BART-base) được chọn làm mô hình chính; hơn Lead-3 có ý nghĩa ở cả bốn chỉ số trên test (R1 +0,70 [0,06; 1,34], BERTScore +4,61 [3,89; 5,34]) và hơn Seq2Seq từ đầu rất rõ (R1 +7,75); T5-small không khác biệt có ý nghĩa với BART-base (R1 −0,19 [−0,77; 0,42] khi tính A − T5). Xem D10.

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

## D10. Tinh chỉnh BART-base và T5-small, ablation, mốc zero-shot (GĐ6)
- **Bối cảnh:** GĐ4–5 cho hai mốc: Lead-3 (CNN/DM) / Lead-1 (GOV.UK) và Seq2Seq từ đầu (thấp hơn Lead, D9). GĐ6 trả lời: (i) mô hình tiền huấn luyện được tinh chỉnh có vượt mốc extractive không; (ii) phần cải thiện đến từ tinh chỉnh hay đã có sẵn từ tiền huấn luyện (mốc zero-shot, nhận xét 1 của giảng viên); (iii) cắt đầu vào 512 token và cấu hình giải mã ảnh hưởng thế nào (nhận xét 2, 3, 5).
- **Hệ thống và thí nghiệm** (mỗi thí nghiệm một file `configs/*.yaml`, seed 42, ghi thời gian huấn luyện):
  - Z1, Z2: BART-base và T5-small (tiền tố `summarize: `) **chưa tinh chỉnh**, chỉ giải mã.
  - A: BART-base tinh chỉnh trên train CNN/DM, `max_input = 512`, `max_target = 128` (mô hình chính, D4).
  - B: T5-small tinh chỉnh trên train CNN/DM, cùng độ dài.
  - C: như A nhưng `max_input = 1.024` (ablation D7).
  - D: BART-base tinh chỉnh trực tiếp từ `facebook/bart-base` trên train GOV.UK (3.102 mẫu), `max_target = 64`. Không khởi tạo từ checkpoint A để tách ảnh hưởng của dữ liệu miền.
  - E: quét learning rate (số mức và số epoch chốt sau khi đo giây/epoch bằng smoke test; chưa có số đo nên chưa ước lượng thời gian).
  - Tùy chọn nếu còn thời gian: T5-small trên GOV.UK; chọn câu bằng TextRank rồi đưa vào 512 token; seed thứ hai cho A.
- **Quy tắc chọn cấu hình (khai trước khi chạy):**
  - Mọi lựa chọn (k của Lead-k, learning rate, tham số giải mã, epoch tốt nhất) thực hiện trên **validation**, theo điểm S = trung bình (ROUGE-1, ROUGE-2, ROUGE-Lsum) trên validation. Hai cấu hình chênh S < 0,2 điểm thì chọn cấu hình gần mặc định/đơn giản hơn.
  - Test chạy **một lần** cho mỗi cấu hình đã chốt (kèm BERTScore); ablation trên validation không dùng BERTScore.
  - Lead-k chọn lại trên validation với k = 1..5 (nhận xét 5). Nếu k chọn khác mốc cũ (Lead-3, Lead-1) thì mốc cần vượt được cập nhật và ghi rõ trong D10.
  - Ablation giải mã làm một yếu tố một lần quanh cấu hình gốc (không lưới đầy đủ), riêng CNN/DM và GOV.UK: `num_beams`, `max_length`, `min_length`, `length_penalty`, `no_repeat_ngram_size`.
  - Chọn epoch tốt nhất: epoch sau chỉ thay epoch hiện tại nếu S trên tập con validation cao hơn ít nhất 0,2 điểm (select_tol); chênh nhỏ hơn coi là hòa và giữ epoch sớm hơn. Áp dụng từ thí nghiệm B trở đi; A chạy với quy tắc "S cao nhất" nhưng kết quả trùng (epoch 2).
- **Giá trị mặc định ban đầu (đề xuất, chưa kiểm chứng; chỉnh sau smoke test):** BART: AdamW, lr 5e-5, batch hiệu dụng 16, khoảng 3 epoch, fp16. T5-small: lr 1e-4, fp32 (fp16 với T5 hay gây NaN theo kinh nghiệm chung). Giải mã ban đầu: beam 4, chặn lặp trigram, `length_penalty` 1,0, `max_length` 128 (CNN/DM) và 64 (GOV.UK).
- **Lý do:** cùng cài đặt đánh giá D5 và cùng `max_input` 512 với Seq2Seq để so sánh trực tiếp; mốc zero-shot tách phần "tiền huấn luyện" khỏi phần "tinh chỉnh".
- **Lưu ý về mốc zero-shot:**
  - BART-base là mô hình khử nhiễu, không có tác vụ tóm tắt trong tiền huấn luyện, nên zero-shot không phản ánh một "mô hình tóm tắt".
  - Model card `t5-small` ghi checkpoint được tiền huấn luyện trên hỗn hợp tác vụ không giám sát và có giám sát; HF docs dùng tiền tố `summarize:` cho tóm tắt. Việc hỗn hợp có gồm CNN/DM là theo mình nhớ từ bài báo T5, các trang đã tra không liệt kê tên bộ dữ liệu. Nếu đúng, zero-shot T5 không "thuần" và có thể đã thấy dữ liệu train CNN/DM. Cần xác nhận từ bài báo gốc trước khi viết vào báo cáo.
- **Dự đoán trước khi chạy** (giả thuyết, chưa có số liệu):
  1. Z1 (BART zero-shot) chủ yếu chép hoặc gần chép đầu bài: tỉ lệ trigram dự đoán không xuất hiện trong bài < 15% (tham chiếu: 71,7%), độ dài trung bình sát `max_length`, R1 khoảng 25–38, thấp hơn A rõ rệt về ROUGE-2 và BERTScore.
  2. Z2 (T5-small zero-shot) sinh ra tóm tắt thực sự (không chỉ chép), R1 khoảng 33–40, cao hơn Z1 về R1.
  3. A (BART-base, CNN/DM, 512): R1 khoảng 38–42, R2 khoảng 16–20, Lsum khoảng 35–39; **cùng mức hoặc nhỉnh hơn Lead-3** về R1 (không dự đoán vượt có ý nghĩa thống kê), BERTScore cao hơn Lead-3 (26,0), và R1 cao hơn Seq2Seq từ đầu (33,3) ít nhất 4 điểm.
  4. B (T5-small) thấp hơn A về R1 khoảng 0,5–3 điểm.
  5. C (1.024) hơn A về R1 không quá 1,0 điểm (có thể không có ý nghĩa thống kê), thời gian mỗi epoch tăng ít nhất 1,7 lần.
  6. GOV.UK: BART chỉ huấn luyện trên CNN/DM (chéo miền) sinh dài hơn tham chiếu (≥ 1,5 lần 21,3 từ) và R1 thấp hơn Lead-1 (33,1). D (tinh chỉnh trên GOV.UK) rút ngắn độ dài về gần tham chiếu (≤ 1,3 lần), R1 khoảng 35–40; không chắc vượt Lead-1 và nếu vượt thì không coi là bất thường.
  7. Ablation giải mã: `length_penalty`/`min_length`/`max_length` đổi độ dài sinh ra nhiều hơn `num_beams`; beam 1 → 4 tăng R1 khoảng 0,3–1,0; chặn lặp trigram ít tác dụng với BART đã tinh chỉnh (tỉ lệ lặp trigram < 10% ngay khi không chặn), khác với Seq2Seq ở D9.
  8. Learning rate trong khoảng 3e-5–1e-4 cho BART chênh nhau dưới 1 điểm R1.
  9. Đường cong: val loss của BART đạt cực tiểu trong 1–3 epoch và có thể tăng nếu huấn luyện lâu hơn, hội tụ nhanh hơn Seq2Seq (epoch 13). Không so sánh trực tiếp giá trị loss giữa BART/T5 và Seq2Seq vì từ vựng khác nhau; so sánh bằng ROUGE.
  10. Theo độ dài bài (BART, token): nhóm > 512 token thấp hơn nhóm ≤ 512 về R1 nhưng chênh dưới 3 điểm (do lead bias, D2).
  11. Nếu A thấp hơn Lead-3 hơn 2 điểm R1 (CI không chứa 0) thì coi là bất thường, kiểm tra tokenization nhãn, chuẩn hóa tách câu và cấu hình giải mã trước khi ghi nhận.
- **Bằng chứng (đã đo):**
  - **Môi trường và quy ước:** Colab, GPU T4; BART-base fp16, T5-small fp32; seed 42; batch hiệu dụng 16; AdamW, warmup 6%, giảm tuyến tính; 3 epoch (D: tối đa 8, dừng sớm khi 3 epoch liền không cải thiện). Giải mã mặc định: beam 4, chặn lặp trigram 3, length_penalty 1,0, max_length 128 (GOV.UK: 64). Epoch chọn theo S trên 300 mẫu validation cố định; epoch sau chỉ thay epoch hiện tại nếu S cao hơn ít nhất 0,2 điểm (select_tol; A chạy với "S cao nhất" nhưng kết quả trùng). Bart-base mang sẵn mặc định giải mã ẩn (num_beams 4, no_repeat_ngram_size 3, early_stopping true, forced_bos_token_id 0, forced_eos_token_id 2); mọi tham số khác được truyền tường minh; forced_bos/eos không bị ghi đè, khớp định dạng nhãn lúc huấn luyện. T5-small không mang mặc định giải mã ẩn.
  - **Huấn luyện (S trên 300 mẫu validation theo epoch; epoch được chọn in đậm):** A (BART 512, lr 5e-5): 33,65 / **34,38** / 34,24; E lr 3e-5: 33,90 / **34,11** / 34,07; E lr 1e-4: 33,08 / 33,90 / **34,13**; C (1.024): 33,78 / **34,50** / 34,50; B (T5-small): 33,92 / 33,89 / **34,20**; D (GOV.UK): 30,28 / **33,28** / 32,93 / 32,25 / 31,55 (dừng sớm sau epoch 5; val loss tăng từ epoch 3: 1,853 → 1,874 → 1,908 → 1,926).
  - **Thời gian trên T4 (mỗi epoch, huấn luyện / đánh giá):** A 484–496 giây / 54 giây (tổng huấn luyện 1.473 giây); C 1.025–1.050 giây / khoảng 109 giây (gấp 2,1 lần A; tổng 3.124 giây); B 974–981 giây / khoảng 85 giây (tổng 2.935 giây); E lr 3e-5 và 1e-4 khoảng 561–583 giây; D khoảng 76–82 giây / 26 giây. Lưu last.pt của BART trung bình 15,9 giây (khoảng 1,7 GB). Dung lượng checkpoint trên Drive khoảng 12 GB.
  - **Toàn bộ validation, giải mã mặc định, không BERTScore (S; CNN/DM n=1.000):** A 34,276; E lr 3e-5 34,242; E lr 1e-4 34,162; C 34,288; B 34,320 (khoảng cách lớn nhất 0,16 điểm, nằm trong ngưỡng hòa 0,2). Lead-3 trên cùng 1.000 bài: 33,30. GOV.UK (n=387): D 32,965; Lead-1: 24,69. Chọn A (lr 5e-5, max_input 512) làm mô hình chính theo quy tắc hòa.
  - **Ablation giải mã trên toàn bộ validation, đổi từng tham số một quanh cấu hình gốc, không BERTScore.** CNN/DM, mô hình A (S gốc 34,276): max_length 96 → 34,479 (63,7 từ); beam 2 → 34,470; min_length 56 → 34,292; lp 0,6 → 34,249; lp 1,5 → 34,169; beam 8 → 34,168; lp 2,0 → 33,992; max_length 64 → 33,919; beam 1 → 33,650; bỏ chặn lặp trigram → 33,498 (lặp trigram 42,2% so với 3,3%). Chọn max_length 96 (hơn gốc 0,203 điểm, sát ngưỡng hòa 0,2). GOV.UK, mô hình D (S gốc 32,965): tốt nhất max_length 48 → 33,003 (+0,038, hòa) nên giữ cấu hình gốc; bỏ chặn lặp → 32,797 (lặp trigram 5,2% so với 1,8%). Không chạy ablation giải mã cho T5-small và cấu hình ghép các giá trị thắng (--combine) vì thời gian: T5-small và C được đánh giá với giải mã mặc định (C dùng cùng max_length 96 với A).
  - **Zero-shot (test, n=1.000, giải mã mặc định):** Z1 BART-base R1/R2/Lsum/BERTScore = 39,25 / 17,71 / 36,04 / 24,54, 101,9 từ, trigram mới 5,8%, lặp trigram 26,2%; Z2 T5-small = 38,94 / 17,39 / 33,49 / 25,25, 50,7 từ, trigram mới 20,1%. Lsum của Z2 thấp một phần vì đầu ra chữ thường không tách được câu (1,08 câu/bài).
  - **Test, một lần chạy cho mỗi cấu hình đã chốt (R1 / R2 / Lsum / BERTScore; từ trung bình; ms/bài ở batch 1 trên T4).** CNN/DM (n=1.000; tham chiếu 52,6 từ): A (max_length 96) 41,04 / 18,41 / 37,93 / 30,65; 62,9 từ; 894 ms. A (giải mã mặc định) 40,76 / 18,42 / 37,74 / 30,59; 69,3 từ; 1.025 ms. C (1.024, max_length 96) 41,31 / 18,47 / 38,00 / 30,44; 64,3 từ; 982 ms. B (T5-small) 41,23 / 18,66 / 38,02 / 31,11; 56,7 từ; 1.163 ms. Mốc: Lead-3 40,3 / 17,6 / 36,5 / 26,0; TextRank-3 35,2 / 13,8 / 31,5 / 21,6; Seq2Seq (GĐ5, sinh lại từ best.pt epoch 13, khớp số GĐ5) 33,3 / 13,0 / 30,9 / 20,8; 111 ms. GOV.UK (n=389; tham chiếu 21,3 từ): D 40,72 / 20,90 / 33,48 / 40,66; 21,7 từ; 360 ms. A chéo miền (huấn luyện CNN/DM, max_length 64) 27,44 / 10,05 / 22,53 / 20,04; 48,8 từ. Mốc: Lead-1 33,1 / 15,0 / 25,5 / 30,0; Seq2Seq chéo miền 24,9 / 8,0 / 20,3 / 14,3.
  - **Kiểm định ghép cặp trên test (bootstrap ghép cặp 2.000 lần, CI 95%, KHÔNG hiệu chỉnh đa so sánh; hiệu A − B, thang 100).** CNN/DM: A − Lead-3: R1 +0,70 [0,06; 1,34], R2 +0,83 [0,21; 1,42], Lsum +1,40 [0,76; 2,01], BERTScore +4,61 [3,89; 5,34] (cả bốn có ý nghĩa; R1 sát ngưỡng). B − Lead-3: R1 +0,90 [0,16; 1,58], R2 +1,08, Lsum +1,49, BERTScore +5,06 (có ý nghĩa). A − Seq2Seq: R1 +7,75 [6,93; 8,49], BERTScore +9,81 [8,86; 10,72]. A − Z1: R1 +1,79 [1,22; 2,38], R2 +0,69 [0,16; 1,23], Lsum +1,89, BERTScore +6,11 [5,44; 6,76]; B − Z2: R1 +2,29 [1,73; 2,86], R2 +1,27, BERTScore +5,86. A − B: R1 −0,19 [−0,77; 0,42], BERTScore −0,46 [−1,17; 0,24] (không có ý nghĩa). C − A: R1 +0,27 [−0,19; 0,72], BERTScore −0,22 [−0,74; 0,30] (không có ý nghĩa). A − A mặc định: R1 +0,28 [0,11; 0,46], Lsum +0,18 [0,01; 0,37] (có ý nghĩa), R2 và BERTScore không. Z1 − Lead-3: R1 −1,09 [−1,48; −0,70], BERTScore −1,50 (thấp hơn có ý nghĩa), R2 +0,13 (không); Z2 − Lead-3: R1 −1,39 [−2,11; −0,69]. GOV.UK: D − Lead-1: R1 +7,66 [5,99; 9,41], R2 +5,94, Lsum +8,02, BERTScore +10,72 [8,99; 12,59]; D − A chéo miền: R1 +13,28; A chéo miền − Lead-1: R1 −5,62 [−7,21; −4,00], BERTScore −9,91; A chéo miền − Seq2Seq: R1 +2,55 [1,31; 3,74], BERTScore +5,70 (tất cả có ý nghĩa).
  - **Điểm theo độ dài bài (số token BART; CNN/DM: ≤512 có 262 bài, >512 có 738 bài; GOV.UK: ≤512 có 126 bài, >512 có 263 bài).** R1 theo (≤512 / >512): Lead-3 42,69 / 39,50; A 43,57 / 40,14; C 42,81 / 40,78; B 43,69 / 40,36. C − A trong nhóm >512: R1 +0,64 [0,13; 1,14] (có ý nghĩa), R2 +0,46, Lsum +0,43, BERTScore +0,05 (không); trong nhóm ≤512 (đầu vào của C và A giống hệt nhau): R1 −0,76 [−1,72; 0,20], R2 −1,05 [−1,89; −0,22], Lsum −0,92 [−1,84; −0,04], tức chênh khoảng 1 điểm giữa hai lần huấn luyện riêng biệt khi đầu vào y hệt. A − Lead-3 trong từng nhóm: BERTScore +4,58 (≤512) và +4,62 (>512), cả hai có ý nghĩa; R1 +0,88 và +0,64 (không có ý nghĩa riêng từng nhóm). GOV.UK, R1 của D / Lead-1: ≤512: 48,40 / 36,21 (+12,19); >512: 37,03 / 31,54 (+5,49), cả hai có ý nghĩa.
- **Kết quả:**
  - **CNN/DM:** BART-base tinh chỉnh (A) hơn Lead-3 có ý nghĩa ở cả bốn chỉ số, nhưng mức hơn về ROUGE nhỏ (R1 +0,7, R2 +0,8, Lsum +1,4) và R1 chỉ vừa qua ngưỡng (CI dưới 0,06); mức hơn về BERTScore lớn hơn nhiều (+4,6). T5-small tinh chỉnh ngang A (không khác biệt có ý nghĩa). Tinh chỉnh có tác dụng rõ so với zero-shot (BART +1,8 R1, +6,1 BERTScore; T5-small +2,3 R1, +5,9 BERTScore), và cả hai mô hình zero-shot thấp hơn Lead-3 một chút có ý nghĩa ở R1 và BERTScore.
  - **max_input 1.024 so với 512 (C so với A):** không có ý nghĩa ở mọi chỉ số trên toàn test, với thời gian huấn luyện gấp 2,1 lần. Trong nhóm bài dài (>512 token) C hơn A về R1 +0,64 (có ý nghĩa), không hơn về BERTScore. Nhóm bài ngắn (≤512), nơi hai mô hình thấy cùng đầu vào, cho thấy hai lần huấn luyện khác nhau tới khoảng 1 điểm R2/Lsum.
  - **Giải mã:** max_length 96 hơn mặc định 0,28 điểm R1 trên test (có ý nghĩa), hầu như không đổi R2 và BERTScore. Chặn lặp trigram quan trọng trên CNN/DM (bỏ ra: S giảm 0,78 và lặp trigram tăng lên 42,2%) nhưng ít ảnh hưởng trên GOV.UK.
  - **GOV.UK:** BART tinh chỉnh trên GOV.UK (D) hơn Lead-1 rất rõ (R1 +7,7, BERTScore +10,7). Mô hình huấn luyện trên CNN/DM chuyển sang GOV.UK kém Lead-1 (R1 −5,6) và sinh dài gấp 2,3 lần tham chiếu, nhưng vẫn hơn Seq2Seq chéo miền. Mức hơn Lead-1 của D lớn ở bài ngắn (+12,2 R1) và nhỏ hơn ở bài dài (+5,5 R1; 68% bài GOV.UK dài hơn 512 token).
- **Đối chiếu dự đoán:**
  1. **Z1:** trigram mới dưới 15% ✓ (5,8%); độ dài sát max_length ✓ (101,9 từ; chưa đếm số token); R1 khoảng 25–38 ✗ (39,25, vượt cận trên 1,25 điểm); thấp hơn A về R2 và BERTScore ✓ (R2 −0,69 [0,16; 1,23] nên chênh nhỏ nhưng có ý nghĩa; BERTScore −6,11).
  2. **Z2:** không chỉ chép ✓ (trigram mới 20,1% so với 5,8% của Z1); R1 33–40 ✓ (38,94); "cao hơn Z1 về R1" ✗ (thấp hơn 0,31 điểm; chênh nhỏ, chưa kiểm định riêng).
  3. **A:** R1 38–42 ✓ (41,04), R2 16–20 ✓ (18,41), Lsum 35–39 ✓ (37,93); "cùng mức hoặc nhỉnh hơn Lead-3, không dự đoán vượt có ý nghĩa" ✓ về hướng, nhưng thực tế VƯỢT có ý nghĩa ở cả bốn chỉ số (R1 sát ngưỡng); BERTScore cao hơn Lead-3 ✓ (+4,61); R1 cao hơn Seq2Seq ít nhất 4 điểm ✓ (+7,75).
  4. **T5-small thấp hơn A khoảng 0,5–3 điểm R1:** ✗. T5-small cao hơn A 0,19 điểm R1 (không có ý nghĩa, CI [−0,42; 0,77] khi tính A − B là −0,19); BART và T5-small không phân biệt được. Lưu ý A đã được chỉnh max_length trên validation còn T5-small dùng giải mã mặc định.
  5. **C:** R1 hơn A không quá 1,0 điểm ✓ (+0,27, không có ý nghĩa); thời gian mỗi epoch tăng ít nhất 1,7 lần ✓ (2,1 lần).
  6. **GOV.UK:** BART chỉ huấn luyện trên CNN/DM sinh dài hơn tham chiếu ít nhất 1,5 lần ✓ (2,3 lần) và R1 thấp hơn Lead-1 ✓ (27,44 so với 33,06); D sinh dài tối đa 1,3 lần tham chiếu ✓ (1,02 lần); R1 của D khoảng 35–40 ✗ (40,72, vượt cận trên 0,72 điểm); "không chắc vượt Lead-1": thực tế vượt rõ (+7,66).
  7. **Ablation giải mã:** beam 1 → 4 tăng R1 khoảng 0,3–1,0 ✓ (+0,69 trên validation); chặn lặp trigram ít tác dụng, lặp dưới 10% khi không chặn: ✗ trên CNN/DM (42,2%; S −0,78), ✓ trên GOV.UK (5,2%); length_penalty/min_length/max_length đổi độ dài nhiều hơn num_beams: không khớp rõ (beam 1 giảm 15,1 từ so với beam 4, tương đương lp 2,0 làm tăng 13,9 từ; max_length 64 giảm 22,0 từ; min_length 56 chỉ tăng 2,3 từ).
  8. **Learning rate 3e-5–1e-4 chênh dưới 1 điểm R1:** ✓ (R1 trên toàn validation chênh 0,07 điểm).
  9. **Đường cong:** val loss của D đạt cực tiểu ở epoch 2 rồi tăng ✓; trên CNN/DM val loss giảm và phẳng dần đến epoch 3 (cực tiểu ở epoch cuối, không thấy tăng trong 3 epoch nên chưa xác nhận được vế "có thể tăng"); BART hội tụ nhanh hơn Seq2Seq ✓ (epoch tốt nhất 2–3 so với 13). Loss và S không luôn cùng chiều (C có val loss thấp nhất nhưng S không hơn A).
  10. **Nhóm >512 thấp hơn nhóm ≤512 về R1 nhưng chênh dưới 3 điểm:** hướng ✓, độ lớn ✗ (A chênh 3,43 điểm; T5 3,33; Lead-3 cũng chênh 3,19 nên phần lớn do bản thân bài dài hơn khó hơn; C chênh 2,03).
  11. **Điều kiện bất thường (A thấp hơn Lead-3 hơn 2 điểm R1):** không xảy ra.
- **Kết luận:** Chọn A (BART-base, max_input 512, lr 5e-5, giải mã beam 4 / chặn lặp trigram 3 / max_length 96) làm mô hình chính cho CNN/DM, và D (BART-base tinh chỉnh trên GOV.UK, giải mã mặc định) cho GOV.UK, theo D4 và quy tắc hòa 0,2 điểm của D10 (không có lựa chọn nào khác trên validation khác biệt đủ ngưỡng). Trên CNN/DM tinh chỉnh chỉ hơn Lead-3 một khoảng nhỏ về ROUGE (R1 +0,7) nhưng rõ về BERTScore (+4,6); trên GOV.UK hơn Lead-1 rất rõ. T5-small cùng mức với BART-base trên CNN/DM (không khác biệt có ý nghĩa; chưa kiểm chứng T5-small có thấy CNN/DM khi tiền huấn luyện hay không). max_input 1.024 không đáng chi phí (gấp 2,1 lần thời gian, không hơn có ý nghĩa trên toàn test).
- **Hạn chế:** 
  - T5-small và C chạy test với giải mã mặc định (không ablation riêng) trong khi A được chỉnh max_length trên validation (chênh tối đa khoảng 0,28 điểm R1).
  - Với cùng 512 token, T5 thấy ít chữ hơn BART (cắt 84,9% bài so với 79,0%) và nhãn T5 bị cắt nhiều hơn (5,1% so với 2,1% nhãn chạm trần 128 token); nhãn CNN/DM giữ định dạng gốc có " ." (ROUGE không bị ảnh hưởng).
  - ROUGE-Lsum của Z2 không so sánh được (đầu ra chữ thường, hàm tách câu cần chữ hoa).
  - Một seed, một lần huấn luyện mỗi cấu hình: khi đầu vào giống hệt, hai lần huấn luyện (C và A trên nhóm ≤512) chênh tới khoảng 1 điểm R2/Lsum; các chênh lệch dưới ngưỡng đó giữa các mô hình không nên diễn giải.
  - Chưa hiệu chỉnh đa so sánh (nhiều cặp, nhiều chỉ số); kết quả R1 của A so với Lead-3 sát ngưỡng ý nghĩa.
  - S dùng để chọn epoch tính trên 300 mẫu validation; lr và max_input chọn lại trên toàn validation (1.000 mẫu).
  - Số bài GOV.UK trên test chỉ 389 (CI rộng); 68% bài GOV.UK dài hơn 512 token và chưa thử max_input 1.024 trên GOV.UK.
  - ms/bài không so trực tiếp giữa họ (Lead và TextRank chạy trên CPU; Seq2Seq và BART/T5 trên GPU T4, batch 1).