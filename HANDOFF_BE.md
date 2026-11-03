# Bàn giao cho BE FastAPI — gói API-only

## Cấu trúc giữ lại

```text
ai/                         ONNX adapter, quỹ đạo J/Z, đường dẫn runtime
api/main.py                 FastAPI + schema input/output
models/onnx_alphabet/        ONNX, nhãn, renderer, manifest, nguồn và license
integration/                adapter BE, OpenAPI và request/response mẫu
scripts/                    chuẩn bị model + 2 bộ kiểm tra không cần dataset
requirements*.txt           dependencies runtime
setup.ps1 / start-ai.ps1     cài và chạy service
```

Gửi cả gói này, không chỉ file ONNX. `.venv`, cache, secrets và báo cáo sinh ra
được Git bỏ qua. Nghiên cứu/demo/model cũ đã được chuyển ra ngoài thư mục dự án.
Repo FE hmi-asl-app vẫn độc lập và không bị gộp vào đây.

## Khởi chạy và nối BE

Chạy `./setup.ps1`, `./start-ai.ps1`; mặc định 127.0.0.1:8001.
Health phải trả provider onnx_alphabet và version +motion-v4.
Chưa có user/DB/auth trong AI service.

BE có thể gọi HTTP trực tiếp hoặc copy `integration/backend_proxy.py` vào code BE:

```python
from fastapi import FastAPI
from integration.backend_proxy import create_ai_router
app = FastAPI()
app.include_router(create_ai_router())
```

Sửa import theo thư mục BE. Đặt AI_SERVICE_URL trỏ tới service AI. Nếu chạy hai
container, dùng hostname service AI thay vì localhost. Backend thêm xác thực,
lưu DB và idempotency theo (user_id, client_attempt_id). request_id không phải
khóa idempotency. Giữ model_version trong bản ghi.

## Input

Endpoint chính: POST /api/v1/ai/predict/attempt.
Request mẫu và response trong integration/, schema integration/openapi.json.

- client_attempt_id, target_label, image_width/image_height thực tế.
- coordinate_space=mediapipe_image_normalized_v1, mirrored=false.
- frames: timestamp_ms, landmarks gồm 21 bộ [x,y,z] hoặc null.
- handedness Left/Right: bắt buộc nhất quán cho J/Z.
- 4..180 frame, timestamp tăng; span 750..5000ms.
- J/Z tối thiểu 15 frame, không mất tay giữa đoạn, gap <=250ms.

Không center/scale từng frame tại FE, không gửi điểm của đường hướng dẫn.
FE có thể tự ghi tối đa 5 giây, kết thúc khi tay rời khung >=0,6s; cắt null
ở ranh giới cuối, giữ lỗi tracking ở giữa để AI kiểm tra. Sau lượt chờ rút tay
để tránh gửi trùng. BE nhận đúng một request cho một đoạn.

POST /predict/landmarks chỉ nhận một khung hình và chỉ hỗ trợ 24 chữ tĩnh.
API v0.4.0 không còn nhận ảnh hoặc cung cấp demo; MediaPipe nằm ở FE.

## Phạm vi và output

- 24 chữ tĩnh: skeleton ONNX, trả confidence classifier.
- J: ONNX I/J + quỹ đạo ngón út, luật thử nghiệm.
- Z: ONNX D/Z với confidence >= threshold (mặc định 0,5) ở >=50% tổng frame,
  VÀ quỹ đạo ngón trỏ sau lọc khớp ba nét. Không có nhánh bỏ qua ONNX.
- Không nhận từ WLASL; HELLO... trả unsupported_target.

Fields chính: status, predicted_label/predicted_sign, matches_target,
model_confidence, request_id, model_version, recognition_method, motion_details.
J/Z: model_confidence/score/stability/votes là null; không bịa xác suất cho luật.
shape_family_agreement.Z và shape_gate_passed.Z là kiểm tra classifier;
index_path_normalized là đường đầu ngón trỏ đã lọc. handshape_assessed=true
chỉ xác nhận đã chạy classifier gate, không có nghĩa đã chấm kỹ năng chuẩn.

score của API là alias confidence cũ, không lưu vào điểm kỹ năng. BE để score
nghiệp vụ=null, errors=[]; error_type=null, diagnosis_supported=false,
update_error_history=false. Không tự suy ra lỗi ngón tay từ nhãn khác mục tiêu.

uncertain/no_hand/invalid_landmarks/unsupported_target: matches_target=null.
422: request sai; 503/504: service lỗi. Không lưu lỗi hệ thống thành người học sai.

## Kiểm tra và giới hạn

`python -m scripts.check_handoff`: fixture A/J/Z, schema, router BE; không dataset.
`python -m scripts.check_z_index`: ONNX thực sự được gọi, từ chối khi chỉ hình tay
hoặc chỉ quỹ đạo đạt. Fixture mô phỏng; chưa có accuracy webcam nhiều người.
Môi trường runtime rút gọn được kiểm tra riêng, không dựa trên thư viện train.

Model ~1,18 MiB: asl_cnn_model.onnx.
SHA256 c3843cabc57032080a81e5995b578139c90631080af1e6d070df1c6dfdc1172b.
Giữ nguyên model README/LICENSE/manifest. MIT upstream chỉ áp dụng code,
không bao phủ trọng số.
