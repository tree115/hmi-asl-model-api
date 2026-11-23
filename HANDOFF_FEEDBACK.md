# Bàn giao API Feedback (thử nghiệm)

`POST /api/v1/ai/feedback` nhận landmark của **một lượt thử chữ cái tĩnh**, phát hiện
thành phần sai (H hình dạng, O hướng, L vị trí) và trả phản hồi Cấp 1 theo lịch sử lỗi.

AI Server **không lưu lịch sử**. BE lưu trường `history` của lượt trước rồi gửi lại ở lượt sau.

## Cấu hình

Cần file mẫu chuẩn. Tạo bằng `python -m scripts.build_error_templates --data <thư mục ASL Alphabet>`,
rồi trỏ biến môi trường tới file đó trước khi chạy server:

```powershell
$env:AI_ERROR_TEMPLATE_DATA = "$PWD\data\error_templates.npz"
./start-ai.ps1
```

Thiếu file mẫu chuẩn thì endpoint trả HTTP 503 `template_data_unavailable`.

## Request

Giống `/predict/landmarks`, thêm 3 trường:

| Trường | Ý nghĩa |
| --- | --- |
| `mode` | `"adaptive"` (cấp tăng/giảm theo lịch sử) hoặc `"fixed"` (luôn Cấp 1, dùng cho điều kiện baseline) |
| `history` | `history` mà lượt trước của **cùng người học, cùng chữ, cùng phiên** trả về. Lượt đầu gửi `{}` |
| `max_feedbacks` | Tùy chọn, 1–4. Giới hạn số thành phần sai được trả về (sai nặng nhất trước) |

Ví dụ: [integration/feedback_example_request.json](integration/feedback_example_request.json).

## Response

| Trường | Ý nghĩa |
| --- | --- |
| `status` | `correct`, `has_errors`, hoặc `not_evaluated` |
| `reason` | Khi `not_evaluated`: `no_hand`, `invalid_landmarks`, `unsupported_target` (J, Z, từ) |
| `results` | Đúng/sai từng thành phần, ví dụ `{"H": "wrong", "O": "correct"}` |
| `feedbacks` | Danh sách phản hồi cho các thành phần sai: `component`, `level`, `message`, `hint`, `details` |
| `needs_review` | Thành phần sai kéo dài ở cấp cao nhất (chỉ ở `adaptive`) |
| `history` | **BE lưu lại** và gửi ở lượt sau |
| `detection` | JSON gốc của error detection, để log/phân tích |

Ví dụ: [integration/feedback_example_response.json](integration/feedback_example_response.json).

## Quy tắc cho BE

- Khóa lưu `history`: (người học, chữ mục tiêu, phiên). Bắt đầu phiên mới thì gửi `{}` (báo cáo mục 4.2.3).
- `not_evaluated`: **không** tính là lượt sai, không ghi lịch sử lỗi người học. Response trả nguyên `history` cũ.
- `mode` lấy theo điều kiện thí nghiệm của bộ bài, không đổi giữa chừng.
- Lưu mỗi lượt: `request_id`, `mode`, `status`, `results`, cấp của từng feedback, để phân tích thí nghiệm.

## Gợi ý hiển thị cho FE

- `message` là dòng chính, `hint` là câu hướng dẫn sửa ngay dưới.
- `level` 2 và 3 hiện **chưa có** nội dung trực quan (`visual`, `steps` là `null`). Hiển thị như Cấp 1.
- `not_evaluated`: hiện thông báo điều chỉnh camera/tay như `/predict/landmarks`, không hiện là "sai".

## Giới hạn hiện tại

- Phát hiện lỗi còn **báo nhầm nhiều** với tay của người khác người làm mẫu chuẩn. Chưa dùng
  kết quả này làm điểm kỹ năng. Cần chỉnh quy tắc phát hiện lỗi và mẫu chuẩn trước thí nghiệm.
- Hướng dẫn trái/phải giả định landmark gửi lên **chưa lật gương** (`mirrored: false`).
- Đặc trưng vị trí (L) đang tính sau khi chuẩn hóa theo khung bao bàn tay, chưa phản ánh vị trí trong khung hình.
