# Kết quả phân tích lỗi định tính trên 16 mẫu (D11)

| Mẫu | Nhãn đã chốt | Nhận xét đối chiếu |
|---|---|---|
| E01 | `none` | Dự đoán gần như trùng tham chiếu, nêu đúng người quay phim, địa điểm, hoạt động chống đẩy và chi tiết lần chống đẩy thứ 12. Không thấy thông tin không có căn cứ. |
| E02 | `missing_information` | Dự đoán nêu tỷ lệ 40%, thời gian chờ hơn 30 phút và ngân sách giảm 1,2 tỷ USD, nhưng bỏ chi tiết hơn 8 triệu người bị ngắt cuộc gọi và việc Ủy viên IRS quy nguyên nhân cho các khoản cắt ngân sách do Quốc hội phê duyệt. Đây là thiếu chi tiết, không phải thông tin bịa đặt. |
| E03 | `missing_information`, `other` | Dự đoán giữ được ý về đợt hạn hán và lệnh cắt giảm 25%, nhưng đầu ra kết thúc giữa câu ở cụm “State reservoirs have a”. Đây là dấu hiệu đầu ra bị cắt cụt; câu cuối không hoàn chỉnh và thông tin về hồ chứa không được truyền đạt. |
| E04 | `none` | Dự đoán tóm tắt đúng chức năng trang: có bản ghi chương trình, hỗ trợ đọc hiểu/từ vựng và cách đăng ký xuất hiện trong CNN Student News Roll Call. Không thấy lỗi đáng kể. |
| E05 | `hallucination` | Có lỗi ghép mốc thời gian: dự đoán nói Neymar ghi bàn thứ hai “in the quarter-hour”, trong khi bài nêu bàn mở tỷ số ở phút 14 và bàn thứ hai ở phút 34. Các chi tiết David Luiz bị phạt thẻ và PSG yếu ở trung lộ được bài hỗ trợ. |
| E06 | `missing_information` | Dự đoán nhắc Westley Capper đã thừa nhận lái xe đưa Agnese rời hộp đêm và một người bị triệu tập từ chối trả lời, nhưng chưa nêu rõ trọng tâm vụ việc là Agnese mất tích, thời điểm và bối cảnh cô biến mất. Chưa thấy đủ căn cứ để gắn nhãn ảo giác cho các chi tiết được nêu. |
| E07 | `none` | Các chi tiết về Glenna Kohl, việc nhuộm da trong nhà/ngoài trời, chẩn đoán melanoma năm 2005 và qua đời ba năm sau đều được bài hỗ trợ. Dự đoán truyền đạt được thông điệp cảnh báo chính. |
| E08 | `missing_information` | Dự đoán nêu Klopp rời Dortmund vào mùa hè và muốn tìm thử thách mới, nhưng bỏ bối cảnh Dortmund có mùa giải khó khăn/đứng thứ 10 Bundesliga và lời khuyến khích cầu thủ tiếp tục mỉm cười. Không thấy chi tiết rõ ràng là bịa đặt trong phần đã nêu. |
| E09 | `missing_information` | Dự đoán lấy hai ý ở phần sau bài (liên minh giữa người bảo thủ tôn giáo và giới kinh doanh; sức hấp dẫn với khách du lịch LGBT/khách hàng trẻ) nhưng bỏ luận điểm trung tâm ở phần mở đầu: phản ứng dữ dội và tẩy chay sau các luật “religious freedom” tại Indiana và Arkansas. Các ý được nêu có trong bài, nhưng lựa chọn thông tin lệch trọng tâm. |
| E10 | `missing_information` | Dự đoán tập trung vào bữa tiệc gây quỹ của Tony Blair và Chuka Umunna, trong khi tham chiếu là phần đính chính về ảnh của ứng viên Anna Firth, địa điểm thực tế là Bexley chứ không phải Bromley. Không nhất thiết là bịa đặt; đây là trường hợp chọn một mẩu tin khác trong bài nhiều chủ đề và bỏ sót thông tin mà tham chiếu nhấn mạnh. |
| E11 | `missing_information` | Dự đoán chỉ tập trung vào nhu cầu tuyển chuyên gia của 77 Brigade, bỏ các thông tin nổi bật: quy mô dự kiến 2.000 người nhưng thực tế 454 quân nhân chính quy/dự bị, cùng mục tiêu đối phó sự thống trị tuyên truyền trực tuyến của Islamic State. |
| E12 | `missing_information` | Dự đoán nói về các loạt phim Marvel khác và dàn diễn viên, nhưng bỏ thông tin chính ở đầu bài: Daredevil được xác nhận có mùa 2 và nhân vật Matt Murdock. Các chi tiết về các loạt phim và diễn viên có trong bài. |
| E13 | `none` | Dự đoán trùng với tham chiếu và được câu mở đầu của bài hỗ trợ trực tiếp. |
| E14 | `none` | Dự đoán trùng với tham chiếu và được câu mở đầu của bài hỗ trợ trực tiếp. |
| E15 | `hallucination`, `missing_information` | Dự đoán ghép nhầm hai số liệu: bài nêu GDP tăng 0,2% trong tháng Hai, còn mức tăng trong ba tháng tính đến tháng Hai là 0,4%. Dự đoán gắn 0,2% với “the previous three month period”, làm sai quan hệ thời gian/số liệu. Đồng thời bỏ thông điệp của Bộ trưởng về triển vọng tăng trưởng, lạm phát và nợ. |
| E16 | `missing_information` | Dự đoán nêu khoản đầu tư 3,2 triệu bảng và mục tiêu mở rộng chương trình giảm danh sách chờ MSK 20%, nhưng bỏ nội dung hỗ trợ việc làm đi kèm điều trị và mục tiêu giúp bệnh nhân quay lại làm việc. Chi tiết được nêu có căn cứ trong bài; thiếu thông tin là lỗi nổi bật hơn ảo giác. |

## Tổng hợp kết quả

- Thiếu thông tin hoặc lựa chọn lệch trọng tâm: 10/16 mẫu (E02, E03, E06, E08, E09, E10, E11, E12, E15, E16).
- Sai sự thật hoặc ghép sai dữ kiện: 2/16 mẫu (E05, E15).
- Đầu ra không hoàn chỉnh/bị cắt cụt: 1/16 mẫu (E03).
- Không phát hiện lặp trigram trong 16 dự đoán theo cờ tự động; không có mẫu nào được gán nhãn `repetition`.
- Không phát hiện lỗi đáng kể (`none`): 5/16 mẫu (E01, E04, E07, E13, E14).


Lưu ý: Một mẫu có thể nhận nhiều nhãn, vì vậy các nhóm lỗi có thể chồng lấp và tổng số lượt gán nhãn không bằng 16. Đây là phân tích định tính trên 16 mẫu được chọn có chủ đích, không dùng để suy rộng tỷ lệ lỗi cho toàn bộ tập test.
