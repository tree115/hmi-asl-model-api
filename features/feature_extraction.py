import numpy as np
import math

# Chỉ số landmark MediaPipe Hands (cố định theo tài liệu MediaPipe).
# Không import mediapipe ở server: MediaPipe chạy ở FE, server chỉ nhận landmark.
WRIST = 0
MIDDLE_FINGER_MCP = 9


def cdist(a, b):
    """Khoảng cách Euclid giữa mọi cặp điểm (thay scipy.spatial.distance.cdist)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))

from models.onnx_alphabet import preprocessing as pre

# biến tọa độ bàn tay thành đặc trưng số -> mô hình nhận dạng
# đặc trưng gồm: hình dạng bàn tay, góc giữa các ngón tay, vị trí của cổ tay và ngón tay, vận tốc của cổ tay


# Get the euclidean distance between two points (x1, y1) and (x2, y2)
# handshape


def preprocess_hand(hand_landmarks, frame_width, frame_height):
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")

    if hasattr(hand_landmarks, "landmark"):
        points_px = pre.landmarks_to_pixels(
            hand_landmarks, frame_width, frame_height
        )
    else:
        landmarks = np.asarray(hand_landmarks, dtype=float)
        if landmarks.shape not in ((21, 2), (21, 3)):
            raise ValueError("landmarks must have shape (21, 2) or (21, 3)")
        if not np.isfinite(landmarks).all():
            raise ValueError("landmarks must contain only finite values")
        points_px = landmarks[:, :2] * np.array(
            [frame_width, frame_height], dtype=float
        )

    if not np.isfinite(points_px).all():
        raise ValueError("landmarks must contain only finite values")

    hand_extent = np.ptp(points_px, axis=0).max()
    palm_length = np.linalg.norm(points_px[9] - points_px[0])
    if hand_extent <= 0 or palm_length < 0.05 * hand_extent:
        raise ValueError("Palm too foreshortened")

    norm_points = pre.normalize_landmarks(points_px)
    return norm_points


def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# https://github.com/nicknochnack/MediaPipePoseEstimation/blob/main/Media%20Pipe%20Pose%20Tutorial.ipynb
def angle_between(start, mid, end):
    start, mid, end = np.array(start), np.array(mid), np.array(end)
    # Compute radians, then convert to angle
    radians = np.arctan2(end[:, 1] - mid[:, 1], end[:, 0] - mid[:, 0]) - np.arctan2(
        start[:, 1] - mid[:, 1], start[:, 0] - mid[:, 0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    # Rectify angles that are large than 180
    too_large = np.where(angle > 180.0)
    angle[too_large] = 360 - angle[too_large]
    return angle / 180


# https://stackoverflow.com/questions/8905501/extract-upper-or-lower-triangular-part-of-a-numpy-matrix
# https://stackoverflow.com/questions/47877530/calculate-distance-of-2-list-of-points-in-numpy
# input: tay
def calc_hand_distance(hand_data, selection=True):
    # khoang cách giữa các điểm trên bàn tay, biến đổi vị trí thành đặc trưng số
    # Getting the wrist, palm indices from the mediapipe pose model
    wrist_ind = WRIST
    palm_ind = MIDDLE_FINGER_MCP
    x1, y1 = hand_data[wrist_ind]
    x2, y2 = hand_data[palm_ind]
    palm_size = get_distance(x1, y1, x2, y2)

    # Then, we select only the hand landmarks we actually want to calculate the distances between
    # These are the landmarks of the handpalm, wrist, and fingertips
    if selection:
        hand_selection = [0, 4, 8, 12, 16, 20]  # [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
        hand_data = hand_data[hand_selection, :]

    # Computes sqrt((x1-x2)**2 + (y1-y2)**2) for all point pairs, result is N x N matrix
    # We only want the upper triangle because the matrix is symmetrical and diagonal sim. is self-similarity (=1)
    # We normalize by the palm size to make this robust to scaling
    dist_matrix = cdist(hand_data, hand_data) / palm_size
    N = dist_matrix.shape[0]
    return dist_matrix[np.triu_indices(N, k=1)]


# orientation


# def _unit(v, eps=1e-8):
#     return v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)


def calc_hand_orientation_2d(hand_data, is_left=False, eps=1e-8):
    """
    hand_data: (frames, 21, 2). Trả về (frames, 4):
    sin/cos roll và hướng trục ngang đã chuẩn hóa.
    """
    del is_left  # Retained for compatibility; these features are side-agnostic.
    wrist = hand_data[:, 0]
    idx = hand_data[:, 5]
    mid = hand_data[:, 9]
    pinky = hand_data[:, 17]

    v = mid - wrist  # trục dọc
    h = pinky - idx  # trục ngang
    palm_wid = np.linalg.norm(h, axis=1) + eps

    roll = np.arctan2(v[:, 0], -v[:, 1])
    roll_feat = np.stack([np.sin(roll), np.cos(roll)], axis=1)
    h_unit = h / palm_wid[:, None]

    return np.concatenate([roll_feat, h_unit], axis=1)


def slope(x1, y1, x2, y2):  # Line slope given two points:
    # To avoid division by 0, add a small constant
    return (y2 - y1) / (x2 - x1 + 0.000001)


# Given two list of slopes (for each frame)
def angle_lists(slopes1, slopes2):
    ang = abs(np.degrees(np.arctan((slopes2 - slopes1) / (1 + (slopes2 * slopes1)))))
    # Angles bigger than 180 are fixed
    too_large = np.where(ang > 180)
    ang[too_large] = 360 - ang[too_large]
    return ang / 180


def get_slope(arr):
    return slope(*arr.flatten())


# Compute the angle of a midpoint between two points
# góc giữa các ngón tay và cổ tay
# input: cần landmark tay và pose
def calc_hand_angle_midpoint(hand_data):
    selection = [[1, 2, 4], [5, 6, 8], [9, 10, 12], [13, 14, 16], [17, 18, 20]]
    angles = []
    for s_1 in selection:
        begin, mid, end = s_1
        s_2 = [0, begin, end]
        angles.append(angle_between(*hand_data[:, s_1].transpose(1, 0, 2)))
        angles.append(angle_between(*hand_data[:, s_2].transpose(1, 0, 2)))

    specific_angles = [
        [2, 3, 4],
        [5, 6, 7],
        [6, 7, 8],
        [9, 10, 11],
        [10, 11, 12],
        [13, 14, 15],
        [14, 15, 16],
        [17, 18, 19],
        [18, 19, 20],
        [4, 0, 8],
        [8, 0, 20],
        [16, 17, 20],
        [8, 5, 12],
        [4, 5, 20],
        [8, 13, 20],
    ]
    for s in specific_angles:
        angles.append(angle_between(*hand_data[:, s].transpose(1, 0, 2)))

    return np.array(angles).T  # (frames, 25)


def calc_angles_per_frame(hand_data):
    return calc_hand_angle_midpoint(np.array(hand_data))


def calc_distance_per_frame(hand_data):
    # Loop over the frames in list comprehension, compute distance features for each frame
    distance_per_frame = [calc_hand_distance(np.array(d)) for d in hand_data]
    # hand_data = np.array(hand_data)
    # distance_per_frame = []
    # for frame in range(hand_data.shape[0]):
    #     d = calc_hand_distance(hand_data[frame])
    #     distance_per_frame.append(d)
    return np.array(distance_per_frame)


# tính vận tốc của cổ tay dựa vào mô hình Pose của Mediapipe (chính xác hơn so với mô hình Hand)
# Compute the velocities of the wrists based on the Mediapipe Pose model (more accurate than Hand model)
def compute_wrist_velocities_from_pose(pose, t1=None, t2=None):
    # Get the wrists from the Pose (more accurate than the Hands wrist landmark)
    l_wrist, r_wrist = pose[:, 15, :], pose[:, 16, :]
    # Compute wrist velocity, NOT removing any NaN-frames (need velocity for each frame for our features)
    wrist_l_vel = compute_wrist_velocity(l_wrist, t1, t2, remove_nan=False)
    wrist_r_vel = compute_wrist_velocity(r_wrist, t1, t2, remove_nan=False)
    return wrist_l_vel, wrist_r_vel


# chuyển động cổ tay dựa vào vị trí của cổ tay trong mỗi khung hình
# input: tay
def compute_wrist_velocity(
    wrist, t1=None, t2=None, remove_nan=False, pad=False, fused_only=True
):
    if t1 and t2:  # Crop to specific timestamp if specified
        wrist = wrist[t1:t2, :]
    # If remove_nan = True, find where the wrist is detected (not NaN) and use only those frames
    if remove_nan:
        frames_present = np.unique(np.where(~np.isnan(wrist))[0])
        wrist = wrist[frames_present, :]
        if (
            frames_present.shape[0] <= 1
        ):  # No frames or max. 1 frame detected -> 0 velocity
            return 0

    wrist_velocity = np.diff(wrist, axis=0)
    # For x frames, we will get x-1 velocities, we pad to x velocities
    if pad:
        wrist_velocity = np.insert(wrist_velocity, 0, [0, 0], axis=0)

    # We compute a fused velocity: sqrt((x1-x2)**2 + (y1-y2)**2)
    wrist_velocity_fused = np.sqrt(np.sum(wrist_velocity**2, axis=1))

    # Either only return the velocity that's fused, or return fused and non-fused
    if fused_only:
        return wrist_velocity_fused
    else:
        wrist_velocity_fused = wrist_velocity_fused.reshape(-1, 1)
        # print(wrist_velocity_fused.shape)
        # print(wrist_velocity.shape)
        wrist_velocities = np.append(wrist_velocity_fused, wrist_velocity, axis=1)
        return wrist_velocities


# vận tốc trung bình cho 2 tay
# input: tay
# Computing the average wrist velocity based on the euclidean distance between each time point t and t+1
# This used to be called compute_wrist_velocity
def compute_both_wrist_velocities(landmarks, t1=None, t2=None, remove_nan=False):
    # Compute velocity of both hands
    l_hand = np.array(landmarks["l_hand"])
    r_hand = np.array(landmarks["r_hand"])
    # Pass the wrist of both hands to get their velocities
    wrist_l_vel = compute_wrist_velocity(l_hand[:, 0, :], t1, t2, remove_nan)
    wrist_r_vel = compute_wrist_velocity(r_hand[:, 0, :], t1, t2, remove_nan)

    # Sum the velocity for each frame, average at the end
    wrist_l_vel = np.sum(wrist_l_vel, axis=0) / l_hand.shape[0]
    wrist_r_vel = np.sum(wrist_r_vel, axis=0) / r_hand.shape[0]

    # Also return which hand has more velocity (aka the dominant hand)
    return wrist_l_vel, wrist_r_vel, wrist_l_vel <= wrist_r_vel


# tính biến đổi vị trí cổ tay trong một cửa sổ không hình
# input: tay
# Compute how the x, y coordinates change over the next few frames
def calc_loc_change(hand_data, diff_length=5):
    wrist_data = hand_data[:, 0, :]
    wrist_data = np.nan_to_num(wrist_data)
    wrist_x = wrist_data[:, 0]
    wrist_y = wrist_data[:, 1]
    # Find difference between all frames
    diff_x = np.subtract.outer(wrist_x, wrist_x)  # normalizing by wrist size???
    diff_y = np.subtract.outer(wrist_y, wrist_y)
    # abs_diff_x = np.abs(diff_x)
    # abs_diff_y = np.abs(diff_y)
    sum_x, sum_y = [], []

    for i in range(diff_x.shape[0]):
        to_sum_x = diff_x[max(0, i - diff_length) : i + 1]
        to_sum_y = diff_y[max(0, i - diff_length) : i + 1]
        sum_x.append(np.sum(to_sum_x))
        sum_y.append(np.sum(to_sum_y))

    return np.array([sum_x, sum_y]).T


# trích xuất đặc trưng riel


# lấy dữ liệu landmark của bàn tay từ đầu vào (payload), chuyển thành mảng NumPy và chỉ giữ lại tọa độ X, Y
def _get_hand(payload):
    """
    Trả về:
      lmrk: (21, 2), chỉ chứa tọa độ x, y
    """
    if payload.landmarks is None:
        raise ValueError("Landmarks không được để trống")

    lmrk = np.array(payload.landmarks)[:, :2]

    # return lmrk
    return lmrk, "l_hand"


def get_handshape_features(payload):
    """Handshape: 25 góc + 15 khoảng cách = 40 features."""
    lmrk, _ = _get_hand(payload)

    angle = calc_angles_per_frame(lmrk)  # (frames, 25)
    dist = calc_distance_per_frame(lmrk)  # (frames, 15)

    return np.concatenate([angle, dist], axis=1)


def get_orientation_features(payload):
    """Orientation: sin/cos roll và hướng trục ngang = 4 features."""
    lmrk, is_left = _get_hand(payload)
    return calc_hand_orientation_2d(lmrk, is_left)


def get_location_features(payload):  # bỏ
    """Location: wrist_x, wrist_y = 2."""
    lmrk, _ = _get_hand(payload)
    return lmrk[:, 0, :]


def extract_static_phonological_features(
    landmarks, frame_width=None, frame_height=None, is_left=False
):
    """Extract 46 features from preprocessed (21, 2) or raw (21, 3) landmarks.

    Raw landmarks require frame dimensions. Preprocessed landmarks are already
    centred and scaled, so their coordinates are used directly.
    """
    landmarks = np.asarray(landmarks, dtype=float)
    if not np.isfinite(landmarks).all():
        raise ValueError("landmarks must contain only finite values")

    if landmarks.shape == (21, 2):
        if frame_width is not None or frame_height is not None:
            raise ValueError(
                "frame dimensions must be omitted for preprocessed landmarks"
            )
        normalized_xy = landmarks
        geometry_xy = landmarks
    elif landmarks.shape == (21, 3):
        if frame_width is None or frame_height is None:
            raise ValueError("frame dimensions are required for raw landmarks")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        normalized_xy = landmarks[:, :2]
        geometry_xy = normalized_xy * np.array(
            [frame_width, frame_height], dtype=float
        )
    else:
        raise ValueError(
            "landmarks must be a finite array with shape (21, 2) or (21, 3)"
        )

    frame = geometry_xy[np.newaxis, :, :]

    angles = calc_angles_per_frame(frame)[0]
    distances = calc_distance_per_frame(frame)[0]
    wrist = normalized_xy[0]
    orientation = calc_hand_orientation_2d(frame, is_left=is_left)[0]

    features = np.concatenate([angles, distances, orientation, wrist])
    if features.shape != (46,):
        raise RuntimeError(
            f"Expected 46 static phonological features, got {len(features)}"
        )
    return features


def combine_features(*feature_blocks):
    """Ghép theo cột: (frames, tổng số chiều)."""
    return np.concatenate(feature_blocks, axis=1)


# Static feature order: 25 angles + 15 distances + 4 orientation + 2 wrist location.


def get_static_features(lmrks, key):
    lmrk = np.array(lmrks[key])[:, :, :2]  # chỉ x, y
    is_left = key == "l_hand"

    angle = calc_angles_per_frame(lmrk)  # 25 - handshape
    dist = calc_distance_per_frame(lmrk)  # 15 - handshape
    wrist = lmrk[:, 0, :]  # 2 - location
    orient = calc_hand_orientation_2d(lmrk, is_left)  # 4 - orientation

    return np.concatenate([angle, dist, orient, wrist], axis=1)  # 46


# hàm xử lí chính - bao hàm cả location???
# input: tay và pose
def get_wrist_angle_distance(lmrks, key, pose=None):
    lmrk = lmrks[key]  # Get the specified hand

    # Replace the Hand model's fingertip coords if possible with the Pose model
    # Because the Pose model of Mediapipe is more accurate
    # The Pose model only keeps track of the wrist, thumb, index and pinky (not middle and ring finger)
    # if pose is not None:
    #     # From the Pose model, we either grab index 15 (left) or 16 (right hand)
    #     lmrk[:,0,:] = pose[:, 15, :] if left else pose[:, 16, :]
    #     # Grab the indices of the pinky, index finger and thumb (indices depend on which hand we are using)
    #     fingertips = pose[:, [17,19,21], :] if left else pose[:, [18,20,22], :]
    #     for f in range(fingertips.shape[1]):
    #         finger = fingertips[:, f, :]
    #         # Get the base index (pinky, index, or thumb) indices from the Hand model of Mediapipe
    #         lmrk_equivalent = [17,5,1][f]
    #         lmrk[:, lmrk_equivalent, :] = finger

    # Compute distances and angles between hand's landmarks
    angle = calc_angles_per_frame(lmrk, pose)
    dist = np.array(calc_distance_per_frame(lmrk))

    wrist = lmrk[:, 0, :]  # wrist of the landmarks

    # Putting them together for each frame as (angles, distances, wrist)
    features = np.append(angle, dist, axis=1)
    features = np.append(features, wrist, axis=1)

    # Get the velocity and append it as a feature for each frame
    velocity = compute_wrist_velocity(
        wrist, pad=True, fused_only=False
    )  # .reshape(-1,1)
    features = np.append(features, velocity, axis=1)
    # print('{} dist, {} angle, {} wrist, {} velocity'.format(dist.shape, angle.shape, wrist.shape, velocity.shape))

    # Getting the centroid of all the landmarks of the hand
    # centroid = np.mean(lmrk, axis = 1)
    # features = np.append(features, centroid, axis = 1)

    # if pose is not None:
    #     shoulder = pose[:, 11, :] if left else pose[:, 12, :]
    #     features = np.append(features, shoulder, axis = 1)

    # loc_change = calc_loc_change(lmrk)
    # features = np.append(features, loc_change, axis = 1)

    # print('Distance shape', dist.shape)
    # print('Angle shape', angle.shape)
    # print('Wrist loc shape', wrist.shape)
    # print('Velocity shape', velocity.shape)
    # print('Centroid shape', centroid.shape)
    # print('Shoulder shape', shoulder.shape)
    return features
