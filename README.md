# hmi-asl-model-api

AI Server cho **Signora**, hệ thống luyện tập ngôn ngữ ký hiệu Mỹ (ASL) có phản hồi
thích ứng dựa trên lịch sử lỗi (BTL môn Tương tác người máy, INT2041, UET – VNU).

Service nhận **landmark bàn tay** do FE trích bằng MediaPipe. Service không nhận ảnh
hay video. Từ landmark, service trả kết quả nhận diện, thành phần làm sai và phản hồi
sửa lỗi theo cấp.

| Khả năng | Phạm vi | Endpoint |
| --- | --- | --- |
| Nhận diện chữ cái tĩnh | 24 chữ (A–Y, trừ J, Z) | `/predict/landmarks`, `/predict/attempt` |
| Nhận diện chữ cái động | J, Z (hình tay + quỹ đạo) | `/predict/attempt` |
| Nhận diện từ | 250 từ ASL | `/predict/word` |
| Phát hiện thành phần sai | Hình dạng, hướng, vị trí bàn tay (chữ tĩnh) | `/errors/detect` |
| Phản hồi thích ứng | Feedback cấp 1, tăng/giảm cấp theo lịch sử lỗi | `/feedback` |

> Trạng thái: **thử nghiệm**. Xem [Giới hạn](#giới-hạn) trước khi dùng kết quả cho thí nghiệm.

## Kiến trúc

```
Webcam ─► FE (MediaPipe, trình duyệt) ── landmark ──► AI Server (repo này, :8001)
                │                                        │ nhận diện + lỗi + feedback
                │◄───────────────────────────────────────┘
                └── kết quả lượt thử ──► BE Signora (:8002) ── DB
```

- **FE**: [`chauanh-ui/hmi-asl-app`](https://github.com/chauanh-ui/hmi-asl-app) (:5173)
- **BE**: [`HoaiiThuongg/hci_backend`](https://github.com/HoaiiThuongg/hci_backend) (:8002)
- AI Server **không lưu trạng thái**. Lịch sử lỗi do bên gọi giữ và gửi kèm request.

## Cài đặt và chạy

Yêu cầu: Windows PowerShell, **Python 3.11**. Nên đặt repo ở đường dẫn không dấu.

### Cách 1: dùng script (cần [uv](https://docs.astral.sh/uv/))

```powershell
./setup.ps1      # tạo venv ở %USERPROFILE%\.cache\asl-practice-ai\venv-api, cài thư viện, kiểm tra model
./start-ai.ps1   # chạy ở http://127.0.0.1:8001
```

### Cách 2: thủ công

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.lock.txt
python scripts/prepare_onnx_alphabet.py
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Kiểm tra: http://127.0.0.1:8001/health · Swagger: http://127.0.0.1:8001/docs

### Cấu hình (biến môi trường, tùy chọn)

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `AI_CONFIDENCE_THRESHOLD` | `0.5` | Ngưỡng tin cậy của classifier chữ cái |
| `AI_CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173` | Origin FE được gọi trực tiếp |
| `AI_ERROR_TEMPLATE_DATA` | `data/error_templates.npz` | File mẫu chuẩn cho phát hiện lỗi |

Xem [.env.example](.env.example).

## API

Mọi endpoint nghiệp vụ có tiền tố `/api/v1/ai`. Schema đầy đủ ở
[integration/openapi.json](integration/openapi.json). Request/response mẫu ở [integration/](integration/).

| Method | Endpoint | Mô tả |
| --- | --- | --- |
| GET | `/health` | Trạng thái service và phiên bản model |
| GET | `/api/v1/ai/capabilities` | Ký hiệu, ngưỡng và tính năng đang hỗ trợ |
| POST | `/api/v1/ai/predict/landmarks` | Nhận diện chữ tĩnh từ **một** khung hình (21 điểm) |
| POST | `/api/v1/ai/predict/attempt` | Nhận diện một lượt nhiều khung hình (4–180), gồm cả J/Z |
| POST | `/api/v1/ai/errors/detect` | So đặc trưng một khung hình với mẫu chuẩn, trả sai lệch theo thành phần |
| POST | `/api/v1/ai/feedback` | Nhận diện + thành phần sai + Feedback cấp 1 theo lịch sử, trả `errors` đúng schema BE |
| GET | `/api/v1/ai/words/capabilities` | Danh sách 250 từ hỗ trợ |
| POST | `/api/v1/ai/predict/word` | Nhận diện từ từ chuỗi landmark MediaPipe Holistic (≤ 5 giây) |

Trạng thái nhận diện chung: `recognized`, `uncertain`, `no_hand`, `invalid_landmarks`,
`unsupported_target`. Chỉ `recognized` mới được tính là lượt đúng hoặc sai của người học.

### Ví dụ: phản hồi cho một lượt thử

```powershell
Invoke-RestMethod http://127.0.0.1:8001/api/v1/ai/feedback -Method Post `
  -ContentType 'application/json' -InFile integration/feedback_example_request.json
```

Response gồm `status` (`correct` / `has_errors` / `not_evaluated`), `feedbacks[]`
(`message`, `hint`, `level`), `errors[]` để lưu vào BE, và `history` để gửi lại ở lượt sau.

## Cách hoạt động

**Nhận diện chữ cái.** Landmark được chuẩn hóa và vẽ lại thành ảnh khung xương bàn tay,
rồi đưa vào CNN (ONNX). Một lượt nhiều khung hình được chấm theo đa số phiếu. Với J/Z,
hệ thống kết hợp hình tay từ CNN (I/J, D/Z) với luật quỹ đạo của ngón út hoặc ngón trỏ.
Cả hai điều kiện đều phải đạt.

**Phát hiện thành phần sai.** Mỗi khung hình được biến thành vector 46 đặc trưng: 25 góc
khớp ngón, 15 khoảng cách, 4 đặc trưng hướng lòng bàn tay và 2 tọa độ cổ tay. Vector này
được so với trung bình và độ lệch chuẩn của chữ mục tiêu (z-score). Nhóm hình dạng, hướng
hoặc vị trí nào có đặc trưng lệch quá ngưỡng thì bị coi là sai.

**Phản hồi thích ứng** (`/feedback`):
1. Classifier quyết định lượt thử đúng hay sai.
2. Nếu sai, so sánh đặc trưng (lấy trung vị qua các khung hình) để chỉ ra thành phần sai.
3. Sinh câu phản hồi Cấp 1 và gợi ý sửa theo quy tắc, ví dụ "Hãy duỗi ngón trỏ thẳng hơn".
   Không dùng LLM.
4. Cấp phản hồi tăng hoặc giảm theo chuỗi đúng/sai liên tiếp của từng thành phần. Hai chế
   độ `adaptive` và `fixed` (luôn Cấp 1) dùng cho thí nghiệm so sánh.

## Cấu trúc thư mục

```text
api/            FastAPI: schema, endpoint, test API
ai/             Adapter ONNX chữ cái, luật quỹ đạo J/Z, model từ GestureSign
features/       Trích 46 đặc trưng ngữ âm từ landmark
errors/         So sánh với mẫu chuẩn (z-score), tên và đơn vị đặc trưng, tạo mẫu chuẩn
feedback/       Sinh Feedback cấp 1, cập nhật lịch sử và chọn cấp
models/         ONNX chữ cái, GestureSign (kèm manifest, checksum, license)
data/           error_templates.npz: mẫu chuẩn 24 chữ tĩnh
integration/    Proxy cho BE, openapi.json, request/response mẫu
scripts/        Chuẩn bị model, tạo mẫu chuẩn, script kiểm tra
```

## Kiểm thử

```powershell
python -m unittest api.testing.test_feedback_api feedback.testing.test_feedback_level1 errors.testing.test_detection features.testing.test_feature_extraction
python -m scripts.check_handoff     # fixture A/J/Z qua API thật, ghi lại openapi.json
python -m scripts.check_z_index
python -m scripts.check_word
```

Dùng Python của môi trường đã cài: `.venv` hoặc `%USERPROFILE%\.cache\asl-practice-ai\venv-api\Scripts\python.exe`.

## Tạo lại mẫu chuẩn

```powershell
python -m scripts.build_error_templates --data <thư mục asl_alphabet_train> --out data/error_templates.npz --per-class 500
```

Dataset: [ASL Alphabet (Kaggle, grassknoted)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
Script này cần thêm `mediapipe` (chỉ dùng để chạy script, server không cần).

## Tài liệu bàn giao

- [HANDOFF_BE.md](HANDOFF_BE.md): nối BE, input và output của nhận diện chữ cái
- [HANDOFF_FEEDBACK.md](HANDOFF_FEEDBACK.md): luồng FE → AI → BE cho feedback, việc FE/BE cần sửa khi nối
- [HANDOFF_WORD.md](HANDOFF_WORD.md): nhận diện 250 từ (Holistic)

## Giới hạn

- **Phát hiện lỗi còn báo nhầm nhiều.** Mẫu chuẩn được tạo gần như từ một người. Khi thử
  trên tay người khác, tỷ lệ báo nhầm cao. Vì vậy `/feedback` chỉ dùng kết quả này khi
  classifier đã xác định lượt thử là sai, và `capabilities.error_diagnosis` vẫn là `false`.
- **Feedback Cấp 2/3** chưa có nội dung từ AI (`visual` và `steps` là `null`). FE tự hiển thị phần trực quan.
- **Đặc trưng vị trí** được tính sau khi chuẩn hóa theo khung bao bàn tay, chưa phản ánh vị trí tay trong khung hình.
- **Landmark gửi lên phải chưa lật gương** (`mirrored: false`). Các hướng dẫn trái/phải dựa trên giả định này.
- **Độ chính xác trên webcam thật chưa được đánh giá đầy đủ.** Số liệu chính thức của các model
  là do tác giả công bố trên dữ liệu của họ.

## Nguồn model và dữ liệu

| Thành phần | Nguồn | Giấy phép |
| --- | --- | --- |
| CNN chữ cái (ONNX) | [punpuniacitizen/MediaPipe-ASL-sign-language-recognition](https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition), ghim commit `45ff2c0` | MIT |
| Model từ GestureSign | Shashank Yadav, huấn luyện trên dataset `asl-signs` (top-1 ≈ 72% trên tập test theo `model-meta.json`) | MIT |
| Mẫu chuẩn phát hiện lỗi | Trích từ ASL Alphabet (Kaggle, grassknoted) | Theo điều khoản dataset |

Các file model được kiểm tra bằng sha256 trong `manifest.json` khi khởi động. **Không sửa
hay format lại** các file trong `models/`, nếu không service sẽ không khởi động.
