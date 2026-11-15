# Phiếu hướng dẫn chấm tay – GĐ7 (D11)

## Phạm vi

- Lượt chấm chính: 40 mẫu, gồm 30 mẫu CNN/DM (hệ thống A) và 10 mẫu GOV.UK (hệ thống D).
- Người chấm: một người chấm toàn bộ 40 mẫu.
- Lượt chấm lại: 8 mẫu có mã R01–R08, thực hiện sau ít nhất 3 ngày kể từ lượt đầu; không mở lại hoặc xem điểm cũ khi chấm lại.
- Phiếu không hiển thị nguồn dữ liệu, tên hệ thống, điểm ROUGE, BERTScore hoặc điểm tự động.

## Nguyên tắc chung

1. Đọc bài gốc trước, sau đó đọc tóm tắt.
2. Chấm đúng sự thật bằng cách đối chiếu trực tiếp với bài gốc, không đối chiếu với tóm tắt tham chiếu.
3. Chấm độc lập ba tiêu chí; không để điểm ở tiêu chí này tự động quyết định điểm ở tiêu chí khác.
4. Chấp nhận cách diễn đạt lại nếu ý nghĩa được bài gốc hỗ trợ.
5. Nếu tóm tắt nêu chi tiết không được bài gốc hỗ trợ hoặc mâu thuẫn với bài, ghi rõ trong `notes` và cân nhắc giảm điểm đúng sự thật.
6. `error_type` có thể ghi nhiều nhãn, phân cách bằng dấu chấm phẩy: `repetition`, `missing_information`, `hallucination`, `other`; ghi `none` nếu không phát hiện lỗi đáng kể.

## Thang điểm 1–5

### 1. Trôi chảy (Fluency)

| Điểm | Neo đánh giá |
|---:|---|
| 1 | Khó hiểu; nhiều lỗi ngữ pháp/cấu trúc; câu rời rạc hoặc bị vỡ nghiêm trọng. |
| 2 | Có thể hiểu một phần nhưng diễn đạt vụng, nhiều lỗi hoặc liên kết kém. |
| 3 | Nhìn chung hiểu được; còn một số chỗ gượng, lỗi diễn đạt hoặc liên kết chưa tự nhiên. |
| 4 | Rõ ràng, tự nhiên; chỉ có lỗi nhỏ không ảnh hưởng đáng kể đến việc đọc hiểu. |
| 5 | Rất tự nhiên, mạch lạc, súc tích; không có lỗi ngôn ngữ đáng kể. |

### 2. Đầy đủ (Completeness)

| Điểm | Neo đánh giá |
|---:|---|
| 1 | Bỏ sót gần như toàn bộ ý chính hoặc chỉ giữ chi tiết vụn vặt. |
| 2 | Bỏ sót nhiều ý chính; người đọc khó nắm được nội dung trọng tâm. |
| 3 | Giữ được một phần ý chính nhưng còn thiếu một ý quan trọng hoặc bối cảnh cần thiết. |
| 4 | Bao quát hầu hết ý chính; chỉ thiếu chi tiết phụ hoặc bối cảnh ít quan trọng. |
| 5 | Truyền đạt đầy đủ các ý chính và quan hệ quan trọng trong giới hạn độ dài; không bỏ sót trọng tâm. |

### 3. Đúng sự thật (Factuality)

| Điểm | Neo đánh giá |
|---:|---|
| 1 | Có nhiều chi tiết bịa đặt, sai hoặc mâu thuẫn nghiêm trọng với bài gốc. |
| 2 | Có ít nhất một lỗi sự thật nghiêm trọng hoặc nhiều chi tiết không được bài gốc hỗ trợ. |
| 3 | Phần lớn đúng nhưng có một sai lệch đáng kể, quan hệ dữ kiện sai hoặc một vài chi tiết thiếu căn cứ. |
| 4 | Hầu hết chi tiết được bài gốc hỗ trợ; nếu có sai lệch thì nhỏ, không làm đổi ý nghĩa chính. |
| 5 | Mọi khẳng định đều được bài gốc hỗ trợ; không phát hiện thông tin sai, mâu thuẫn hoặc suy diễn không có căn cứ. |

## Ghi kết quả

- Điền điểm số nguyên 1–5 vào ba cột `fluency_1_5`, `completeness_1_5`, `factuality_1_5`.
- Trong `error_type`, ghi một hoặc nhiều nhãn theo quy ước phía trên.
- Trong `notes`, ghi ngắn gọn bằng chứng cho điểm thấp hoặc lỗi đáng chú ý; đặc biệt ghi rõ chi tiết sai/không có căn cứ và đoạn bài gốc hỗ trợ đối chiếu.
- Không sửa `item_id`, `article`, `summary`.

## Hai file cần sử dụng

- `human_rating_form.csv`: lượt chấm đầu tiên, 40 mẫu H01–H40.
- `human_rerate_form.csv`: lượt chấm lại, 8 mẫu R01–R08. Chỉ mở file này sau ít nhất 3 ngày.

Có thể mở trực tiếp bằng Excel hoặc tải lên Google Sheets. Lưu một bản sao sau khi chấm; không ghi đè lên bản phiếu trống ban đầu.
