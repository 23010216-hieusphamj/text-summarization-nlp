# Thẻ dữ liệu: GOV.UK News Summaries (bộ dữ liệu tự thu thập)

## Nguồn và giấy phép
- **Nguồn:** bài tin (`news_story`), thông cáo báo chí (`press_release`), phản hồi của chính phủ (`government_response`) trên [GOV.UK](https://www.gov.uk).
- **Giấy phép:** Open Government Licence v3.0 (OGL v3.0). Theo hướng dẫn tái sử dụng của GOV.UK, nội dung được tái sử dụng với điều kiện ghi nguồn.
- **Ghi nguồn:** *Contains public sector information licensed under the Open Government Licence v3.0.*
- **Mục đích sử dụng:** học tập (bài tập lớn học phần Xử lý ngôn ngữ tự nhiên), đánh giá khả năng tổng quát hóa của mô hình tóm tắt.

## Cách thu thập
- Công cụ: GOV.UK Search API (liệt kê bài) và Content API (lấy nội dung), không scrape HTML của trang.
- Khoảng thời gian: 2023-01 → 2026-09; lấy tối đa 60 bài cho mỗi (loại bài × tháng) để cân bằng theo thời gian, rồi lấy ngẫu nhiên tối đa 6.000 bài (seed 42).
- Lịch sự với máy chủ: User-Agent định danh, nghỉ 0,5 giây giữa các request, thử lại có chờ khi gặp 429/5xx, có kiểm tra `robots.txt`.
- Mã: `src/crawl/govuk_crawler.py`, cấu hình `configs/data_govuk.yaml`.

## Cấu trúc bản ghi (JSONL)
| Trường | Ý nghĩa |
|---|---|
| `id` | đường dẫn bài trên GOV.UK |
| `title` | tiêu đề |
| `article` | nội dung bài đã làm sạch (đầu vào của mô hình) |
| `highlights` | trường `description` do biên tập viên viết (tóm tắt tham chiếu) |
| `doc_type`, `date` | loại bài, ngày cập nhật công khai |

## Làm sạch và lọc (`src/crawl/clean.py`)
1. HTML → văn bản: bỏ bảng, hình, tiêu đề mục; cắt từ mục "Notes to editors", "Further information", "Media enquiries" trở đi; bỏ URL và email; chuẩn hóa Unicode/khoảng trắng.
2. Lọc: có đủ nội dung và tóm tắt; tiếng Anh (`langdetect`); bài 80–2500 từ; tóm tắt 8–80 từ; tỉ lệ nén ≥ 3; tóm tắt không gần như trùng tiêu đề; tóm tắt không phải đoạn chép nguyên văn từ bài.
3. Khử trùng lặp theo 600 ký tự đầu của bài và theo nội dung tóm tắt.
4. Chia ngẫu nhiên 80/10/10 (seed 42).
- Số bài còn lại sau từng bước: `results/tables/govuk_cleaning_funnel.csv`.

## Hạn chế đã biết
- Tóm tắt thường chỉ một câu, ngắn hơn nhiều so với CNN/DailyMail; phong cách gần với XSum hơn.
- Miền hẹp (thông tin chính phủ Anh), văn phong hành chính.
- `description` do biên tập viên viết cho trang web, không phải để làm tóm tắt chuẩn, nên có thể chứa thông tin không có trong bài hoặc bỏ sót ý chính.
- Chưa kiểm định chất lượng nhãn bằng người; sẽ chấm tay một mẫu nhỏ ở Giai đoạn 7.

## Kết quả thống kê
| Chỉ số | Giá trị |
|---|---|
| Bài thô tải về | 5.477 |
| Bài sau làm sạch | 3.878 (70,8%) |
| Train / Val / Test | 3.102 / 387 / 389 |
| Loại bài (train) | news_story 1.636; press_release 1.400; government_response 66 |
| Số từ bài (trung vị / p95) | 560,5 / 1.453,9 |
| Số từ tóm tắt (trung vị / p95 / max) | 21 / 34 / 63 |
| Số câu tóm tắt (trung bình) | 1,04 |
| Tỉ lệ nén (trung vị) | 26,9 lần |
| % unigram / bigram / trigram mới | 19,0 / 56,0 / 74,1 |
| % câu tóm tắt có nguồn ở 3 / 10 câu đầu | 67,1 / 87,6 |

Chi tiết từng bước lọc: `results/tables/govuk_cleaning_funnel.csv`.
