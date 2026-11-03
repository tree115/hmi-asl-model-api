# ASL Model API

Gói bàn giao cho backend: **model ONNX + nhận diện landmark + FastAPI**.
Không có frontend/demo, dataset, mã train, SVM hoặc model Hugging Face cũ.

## Chạy

Windows PowerShell, cần uv và Python 3.11:

```powershell
./setup.ps1
./start-ai.ps1
```

Setup tạo môi trường riêng `.cache/asl-practice-ai/venv-api` ngoài source,
cài dependencies và kiểm tra/tải ONNX nếu thiếu. Không tải dataset hoặc MediaPipe.

Không dùng uv: tạo venv Python 3.11 (ưu tiên đường dẫn không dấu), cài
`pip install -r requirements.lock.txt`, chạy `python scripts/prepare_onnx_alphabet.py`,
rồi `python -m uvicorn api.main:app --host 127.0.0.1 --port 8001`.

Swagger: http://127.0.0.1:8001/docs. Chi tiết: [HANDOFF_BE.md](HANDOFF_BE.md).

## API

- `GET /health`
- `GET /api/v1/ai/capabilities`
- `POST /api/v1/ai/predict/landmarks`: 21 điểm của một khung hình.
- `POST /api/v1/ai/predict/attempt`: một chuỗi 21 điểm theo thời gian.

API v0.4.0 **đã bỏ `/predict/image`, `/demo/`, `/demo-model.task`**.
MediaPipe chạy ở FE; FE/BE gửi landmark gốc, không gửi ảnh/video vào gói này.

24 chữ tĩnh dùng ONNX. J cần hình tay I/J + quỹ đạo ngón út.
Z cần hình tay D/Z từ ONNX + quỹ đạo ngón trỏ đã lọc, cả hai bắt buộc đạt.
Chưa có WLASL, điểm kỹ năng, chẩn đoán lỗi thành phần hoặc adaptive feedback.

## Kiểm tra

```powershell
$aiPython = Join-Path $env:USERPROFILE '.cache\asl-practice-ai\venv-api\Scripts\python.exe'
& $aiPython -m scripts.check_handoff
& $aiPython -m scripts.check_z_index
```

Không cần dataset; fixture chuyển động là mô phỏng, không chứng minh accuracy webcam.
Báo cáo sinh trong reports/ và được Git bỏ qua.

Giữ attribution tại models/onnx_alphabet/LICENSE và README.md. License MIT của code
upstream không bao phủ trọng số; không gán MIT cho toàn bộ model/dataset.
