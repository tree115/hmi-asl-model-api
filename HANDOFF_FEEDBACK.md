# Bàn giao API Feedback (FE ↔ AI ↔ BE)

`POST /api/v1/ai/feedback` chấm **một lượt thử chữ cái** và trả luôn: kết quả nhận diện,
thành phần sai (H hình dạng, O hướng, L vị trí, M chuyển động cho J/Z), câu phản hồi
theo cấp, và danh sách lỗi **đúng schema BE** để lưu.

Endpoint này thay cho cặp `predict/attempt` + phân tích lỗi giả (`pickMockErrors`) ở FE.

## Luồng nối (theo phương án B hiện tại của FE/BE)

```
FE ──(frames + mode + history)──► AI  POST /api/v1/ai/feedback
FE ◄──(recognition, feedbacks, errors, feedback_level, history)──┘
FE lưu history theo (user, lesson_item, phiên), hiển thị feedbacks
FE ──► BE  POST /api/v1/attempts  { ..., errors, client_feedback_level }
```

- AI Server **không lưu trạng thái**. Bên gọi giữ `history` của lượt trước rồi gửi lại.
- BE **không cần sửa**: `errors[]` và `feedback_level` đã khớp `AttemptCreate` của BE.

## Request

Giống hệt body `/api/v1/ai/predict/attempt` (FE đang gửi), thêm:

| Trường | Ý nghĩa |
| --- | --- |
| `mode` | `"adaptive"` (cấp tăng/giảm theo lịch sử) hoặc `"fixed"` (luôn Cấp 1, điều kiện baseline) |
| `history` | `history` của lượt trước, **cùng người học + cùng ký hiệu + cùng phiên**. Lượt đầu/phiên mới gửi `{}` |
| `max_feedbacks` | Tùy chọn, 1–4: số thành phần sai tối đa trả về (sai nặng nhất trước) |

Ví dụ: [integration/feedback_example_request.json](integration/feedback_example_request.json).

## Response

| Trường | Dùng để |
| --- | --- |
| `status` | `correct` / `has_errors` / `not_evaluated` |
| `reason` | Khi `not_evaluated`: `uncertain`, `no_hand`, `invalid_landmarks`, `unsupported_target` |
| `is_correct` | Gửi BE `is_correct` (`null` khi `not_evaluated`) |
| `recognition` | `status`, `predicted_label`, `model_confidence`, `stability`… như `/predict/attempt`. Gửi BE `status`, `predicted_sign`, `model_confidence` |
| `feedbacks[]` | Hiển thị: `message` (dòng chính), `hint` (cách sửa), `level`, `error_type`, `details` |
| `feedback_level` | Cấp của feedback đầu tiên → BE `client_feedback_level` |
| `errors[]` | `{type, severity, details}` → gửi nguyên vào BE `errors` |
| `history` | **Lưu lại**, gửi ở lượt sau |
| `diagnosis_source` | `feature_templates` (so đặc trưng với mẫu chuẩn) hoặc `classifier_only` (J/Z, thiếu mẫu chuẩn: chỉ biết classifier nhận ra chữ khác) |
| `detection` | JSON gốc của error detection, để log/phân tích |
| `feedback_version` | Ghi kèm khi lưu để phân tích thí nghiệm |

Ví dụ: [integration/feedback_example_response.json](integration/feedback_example_response.json).

### Quy tắc chấm

1. Classifier (giống `/predict/attempt`) quyết định đúng/sai. Không chấm được → `not_evaluated`,
   `history` trả lại nguyên vẹn, **không** tính là lượt sai.
2. Đúng chữ mục tiêu → mọi thành phần `correct` (bỏ qua sai lệch đặc trưng vì phần phát hiện
   lỗi còn báo nhầm).
3. Sai chữ → thành phần sai lấy từ so sánh đặc trưng (trung vị trên các frame có tay). Nếu không
   chỉ ra được thành phần nào → mặc định `handshape` (J/Z: `movement` nếu hình tay đã qua).

## Ví dụ gọi từ FE

```ts
const res = await fetch(`${AI_URL}/api/v1/ai/feedback`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ ...attemptBody, mode: feedbackMode, history: historyFor(itemId) }),
})
const fb = await res.json()
saveHistory(itemId, fb.history)
await api.createAttempt({
  user_id, lesson_id, lesson_item_id: itemId,
  status: fb.recognition.status,
  predicted_sign: fb.recognition.predicted_label,
  model_confidence: fb.recognition.model_confidence,
  is_correct: fb.is_correct,
  feedback_mode: feedbackMode,
  errors: fb.errors,
  client_feedback_level: fb.feedback_level,
  duration_ms,
})
```

## Lịch sử lỗi và đánh giá tiến bộ

Mỗi response `/feedback` có `history_summary`: bảng 4 dòng H, L, O, M, dùng để hiển thị
màn lịch sử lỗi. Ví dụ:

| Thành phần | Cấp hiện tại | Sai liên tiếp | Đúng liên tiếp | Đánh giá tiến bộ |
| --- | --- | --- | --- | --- |
| H – Hình dạng bàn tay | 3 | 2 | 0 | Chưa cải thiện |
| L – Vị trí bàn tay | 1 | 1 | 0 | Lỗi mới |
| O – Hướng bàn tay | 2 | 0 | 2 | Đang tiến bộ |
| M – Chuyển động | – | 0 | 0 | Chưa có dữ liệu |

Mỗi dòng gồm: `name`, `level`, `wrong_streak`, `correct_streak`, `total_wrong`, `total_correct`,
`attempts`, `recent` (tối đa 10 kết quả gần nhất, cũ → mới), `feedbacks_shown`, `corrections`,
`needs_review`, `progress`, `progress_label`.

Quy tắc đánh giá tiến bộ (`progress`):

| Mã | Nhãn | Khi nào |
| --- | --- | --- |
| `no_data` | Chưa có dữ liệu | Thành phần chưa được đánh giá lượt nào (M ở chữ tĩnh) |
| `new_error` | Lỗi mới | Lượt gần nhất sai, và đây là lần sai đầu tiên |
| `not_improving` | Chưa cải thiện | Lượt gần nhất sai, đã sai từ 2 lần trở lên |
| `improving` | Đang tiến bộ | Lượt gần nhất đúng, trước đó từng sai |
| `stable` | Ổn định | Đúng từ đầu, chưa sai lần nào |

Tiêu chí đánh giá **sau khi cung cấp feedback** (lưu trong `history`, dùng cho thí nghiệm):
- `feedbacks_shown`: số lần đã hiển thị feedback cho thành phần đó.
- `corrections`: số lần sửa được lỗi (lượt sai có feedback, lượt kế tiếp đúng).
- `fixed_after_level`: `corrections` tách theo cấp feedback ngay trước khi sửa được,
  ví dụ `{"1": 2, "2": 1}`. Dùng để so sánh hiệu quả từng cấp giữa `fixed` và `adaptive`.

Màn lịch sử không cần chấm lượt mới: gọi `POST /api/v1/ai/feedback/history-summary` với
`{"history": <history đã lưu>}` để lấy lại bảng.

**Lưu trữ:** `history` mới có thêm các trường trên. History cũ (không có các trường này)
vẫn được nhận. Bên giữ history (FE `sessionStorage` hoặc cột JSON ở BE) lưu nguyên object.

## Việc cần sửa khi nối (gửi BE/FE)

AI đã sẵn sàng. Không sửa gì thì FE/BE vẫn chạy như cũ (`/predict/attempt` không đổi), nhưng
người học **vẫn thấy lỗi giả**. Để dùng feedback thật cần sửa như sau (đối chiếu code ngày 23/11/2026).

### FE — repo `chauanh-ui/hmi-asl-app`

1. `src/api/ai.ts`: thêm hàm `getFeedback(...)` gọi `POST ${AI_URL}/api/v1/ai/feedback`, body =
   body hiện tại của `predictAttempt` + `mode` + `history`. Thêm kiểu cho response (xem bảng trên
   hoặc `integration/openapi.json` → `FeedbackResponse`).
2. `src/pages/S3Practice.tsx`, hàm chấm điểm:
   - Thay `predictAttempt(...)` bằng `getFeedback(...)`; lấy `predicted`, `confidence`, `stability`,
     `aiStatus` từ `fb.recognition`.
   - Bỏ lỗi giả: thay `compareErrors = pickMockErrors(...)` bằng `fb.errors` (giữ `computeDeviations`
     nếu vẫn cần cho ComparePanel).
   - `isCorrect` = `fb.is_correct`.
   - Gọi `api.createAttempt` với `errors: fb.errors` và `client_feedback_level: fb.feedback_level`
     (hiện đang gửi `null`).
   - Hiển thị `fb.feedbacks[0].message` + `fb.feedbacks[0].hint` cho Cấp 1.
3. Lưu `history`: giữ `fb.history` theo key `(user_id, lesson_item_id)` trong state hoặc
   `sessionStorage`, gửi lại ở lượt sau của cùng ký hiệu; xóa khi bắt đầu phiên mới.
4. `src/api/real.ts` → `createAttempt`: đang tự tính cấp bằng `computeAdaptiveLevel` và tự tạo
   câu bằng `buildFeedback` (`src/lib/feedback.ts`). Khi dùng AI thì lấy cấp/câu từ response AI
   thay cho hai hàm này, để FE và dữ liệu thí nghiệm dùng chung một quy tắc.
5. Lỗi HTTP: `detail` của AI có thể là object `{code, message}` → hiển thị `detail.message`
   (hiện FE `String(detail)` sẽ ra `[object Object]`).
6. Chữ không hỗ trợ (từ V2): AI trả `status = not_evaluated`, `reason = unsupported_target`;
   FE giữ nhánh mô phỏng hiện có cho trường hợp này.

### BE — repo `HoaiiThuongg/hci_backend`

- **Không bắt buộc sửa.** `fb.errors` khớp `AttemptErrorItem` (`type`, `severity`, `details`),
  `fb.feedback_level` khớp `client_feedback_level` của `POST /api/v1/attempts`.
- Tùy chọn (để phân tích thí nghiệm đầy đủ hơn):
  - Thêm vào `AttemptCreate` các trường `feedback_message`, `feedback_version` và lưu một dòng
    bảng `feedbacks` (đã có cột `mode`, `level`, `target_error_type`, `message`, `suggested_action`)
    — hiện `create_attempt` chưa ghi bảng này.
  - Nếu muốn BE giữ lịch sử thay FE: thêm cột JSON `feedback_history` theo
    (user, lesson_item, phiên), trả về cho FE để gửi kèm request AI.
- `README.md` của FE đang bảo clone BE từ fork `chauanh-ui/hci_backend` nhánh
  `fix/week3-integration`; nhánh này đã merge vào `HoaiiThuongg/hci_backend` (commit `e2b46af`).

### Cần nhóm chốt trước khi nối

- Quy tắc tăng cấp: module Tuyền = sai 3 lần liên tiếp, báo cáo = 2 lần, FE mock = 1→2→3.
- Ai giữ `history`: FE (không sửa BE) hay BE (thêm cột JSON).

## Chạy

`./start-ai.ps1` hoặc `python -m uvicorn api.main:app --port 8001`. Mẫu chuẩn mặc định là
`data/error_templates.npz` trong repo; đặt `AI_ERROR_TEMPLATE_DATA` để dùng file khác. Thiếu
mẫu chuẩn thì endpoint vẫn chạy với `diagnosis_source = classifier_only`.

Kiểm tra: `python -m unittest api.testing.test_feedback_api`.

## Giới hạn hiện tại

- Cấp tăng sau **3** lần sai liên tiếp cùng thành phần, giảm sau 2 lần đúng (theo
  `feedback/feedback_adaptation.py`). Báo cáo ghi 2 lần, FE mock đang dùng 1→2→3: cần nhóm chốt.
- Cấp 2/3 chưa có nội dung từ AI (`visual`, `steps` = `null`). FE dùng ComparePanel/Level3Guide hiện có.
- So sánh đặc trưng còn báo nhầm nhiều với tay người khác người làm mẫu chuẩn; vì vậy chỉ dùng
  khi classifier đã xác định là sai.
- Hướng dẫn trái/phải giả định landmark **chưa lật gương** (`mirrored: false`), đúng như FE đang gửi.
- Đặc trưng vị trí (L) tính sau khi chuẩn hóa theo khung bao bàn tay, chưa phản ánh vị trí trong khung hình.
