# Bàn giao word_model cho bài luyện từ ASL

Chạy AI Server bằng `./setup.ps1` và `./start-ai.ps1`. Model GestureSign được nạp từ `models/gesturesign/` cùng lúc với model chữ cái. Swagger ở `http://127.0.0.1:8001/docs`.

## Backend gọi API

- `GET /api/v1/ai/words/capabilities` trả danh sách 250 nhãn và phiên bản model.
- `POST /api/v1/ai/predict/word` nhận một lượt đã tách. Backend chuyển nguyên payload sang AI Server và ghép kết quả bằng `client_attempt_id`. Nếu lưu lượt, dùng ID này để chống ghi trùng.
- `POST /api/v1/ai/predict/attempt` tiếp tục dùng cho bài chữ cái và J/Z.

Chạy thử từ thư mục repo sau khi bật server:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/api/v1/ai/predict/word -Method Post `
  -ContentType 'application/json' -InFile integration/word_example_request.json
```

Request gồm `client_attempt_id`, `target_label`, `fps`, `frames`. Mặc định `coordinate_space` là `mediapipe_holistic_image_normalized_v1` và `mirrored` là `false`. Mỗi frame có bốn trường:

| Trường | Dữ liệu |
| --- | --- |
| `left_hand`, `right_hand` | Mỗi bên 21 bộ `[x, y, z]`, theo đúng bên cơ thể từ Holistic |
| `pose` | 33 điểm ảnh chuẩn hóa, gồm vai, khuỷu tay, cổ tay và các điểm trên mặt |
| `face` | 468 điểm đầu tiên theo thứ tự face mesh, dùng các vị trí môi cần cho model |

Truyền `null` cho phần hoặc điểm mất dấu. Giữ thứ tự frame và giữ khoảng trống khi tay tạm khuất. Dùng FPS thực tế của chuỗi, có thể tính bằng `(số frame - 1) * 1000 / (timestamp cuối - timestamp đầu)`. Mỗi lượt tối đa khoảng 5 giây, server cho phép sai số 100 ms. Gửi một yêu cầu mỗi lượt. Model xử lý tọa độ ảnh chuẩn hóa của MediaPipe Holistic trên server.

## Frontend thu và hiển thị

1. Hiển thị **từ mục tiêu** và video hướng dẫn lấy từ nội dung bài học. Bật webcam, giữ mặt, hai vai và vùng hoạt động của hai tay trong khung. Vẽ **xương tay từ MediaPipe Holistic** trực tiếp trên camera. Có thể lật gương màn hình và lớp xương cùng nhau, gửi tọa độ gốc lên backend.
2. Hiện các trạng thái **Sẵn sàng → Đang ghi → Đang nhận diện**. Tự bắt đầu khi thấy người học thực hiện, tự kết thúc khi tay dừng, rời khung hoặc đạt khoảng 5 giây. Chỉ gửi lượt vừa kết thúc. Giữ mỗi lần ghi cho một người trong khung.
3. Hiện kết quả theo bảng. Dùng `matches_target` để phản hồi khớp đề khi lượt được nhận diện.

| API trả về | Giao diện hiển thị |
| --- | --- |
| `recognized`, `matches_target: true` | Đúng ký hiệu mục tiêu |
| `recognized`, `matches_target: false` | AI nhận là **predicted_label**. Cho xem lại mẫu và thử lại |
| `uncertain` | Chưa nhận diện rõ. Đưa mặt, vai và tay vào khung rồi thử lại |
| `no_hand` | Chưa thấy tay. Đưa tay vào khung |
| `unsupported_target` | Bài này chưa hỗ trợ nhận diện |
| HTTP 422 hoặc lỗi kết nối | Báo không gửi được lượt, cho thử lại |

`score` và `model_confidence` là độ tin cậy của nhãn đã được chấp nhận. `top3` dành cho kiểm thử. `diagnosis_supported` là `false`. Tránh gắn nhãn lỗi hình tay, vị trí hay chuyển động từ dự đoán này.
