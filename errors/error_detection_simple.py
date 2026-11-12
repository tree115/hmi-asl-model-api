import numpy as np

# bỏ
IDX = np.arange(
    0, 44
)  # chưa cố định lấy 44 đặc trưng - à bỏ do làm kí hiệu tĩnh
STD_MIN = 6.0  # sàn độ lệch chuẩn, tránh chia cho ~0 (đổi nếu dùng radian)


# xây dựng mẫu chuẩn
""" 
1. lấy tất cả mẫu thuộc cùng 1 chữ cái 
2. chỉ giữ 44 đặc trưng đầu tiên (là cái gì???)
3. loại bỏ mẫu có NaN
4. Tính Mean và Standrad Deviation 
5. Loại mẫu bất thường 
6. Tính lại mean và std sau lọc 
"""


def build_templates(X, y):
    """X: (N,120), y: nhãn chữ cái. Trả về {chữ: (mean, std)}."""
    templates = {}
    for c in np.unique(y):
        # lấy các dòng thuộc chữ c và chỉ giữ 44 cột trong IDX
        Xc = X[y == c][:, IDX]
        Xc = Xc[~np.isnan(Xc).any(axis=1)]  # bỏ NaN
        mean, std = Xc.mean(0), Xc.std(0)  # tính mean + std
        # lọc nhiễu: bỏ mẫu lệch > 3 std ở quá nhiều đặc trưng
        z = np.abs(Xc - mean) / np.maximum(
            std, STD_MIN
        )  # tính độ leehcj từng mẫu so với trung bình
        Xc = Xc[(z > 3).mean(1) < 0.1]
        templates[c] = (Xc.mean(0), np.maximum(Xc.std(0), STD_MIN))
    return templates


"""
1. lấy 44 đặc trưng của người học 
2. so sánh với mẫu chuẩn 
3. tính Z-score 
4. Tính điểm tổng 
5. Tìm tối đa 3 đặc trưng sai lệch lớn nhất 
6. Return điểm + danh sách lỗi 
"""


# x đặc trưng người học, target chữ cái muốn thực hiện, templates bộ mẫu chuẩn
def detect_errors(x, target, templates, names=None, top_k=3):
    """names: dict {index: 'ngón trỏ'} để in cho dễ đọc (tùy chọn)."""
    x = np.asarray(x, float)[IDX]  # 44 đặc trưng của frame hiện tại
    if np.isnan(x).any():  # trùng
        return {"score": None, "errors": ["Không thấy tay"]}
    # lấy mẫu chuẩn
    mean, std = templates[target]
    z = (
        x - mean
    ) / std  # Dấu của z cho biết hướng lệch, độ lớn cho biết lệch bao nhiêu.

    score = 100 * (1 - min(np.sqrt(np.mean(np.clip(np.abs(z), 0, 4) ** 2)), 4) / 4)
    # sắp xếp indx lớn bé + giữ đặc trưng thật sự lệch + lấy top3
    bad = [i for i in np.argsort(-np.abs(z)) if abs(z[i]) > 2][:top_k]
    errors = []

    for i in bad:
        n = (names or {}).get(int(i), f"đặc trưng {i}")  # name là tên lỗi
        errors.append(
            f"{n}: {'cao' if z[i] > 0 else 'thấp'} hơn chuẩn {abs(x[i]-mean[i]):.1f}"  # đơn giản
        )
    return {"score": round(float(score), 1), "errors": errors}
