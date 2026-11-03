"""Experimental clip rules inspired by upstream motion.py; no learned motion model.

Use one median pixel hand scale for the clip to preserve translation, real time
resampling, and strict continuity. Thresholds require real signer validation.
"""
import numpy as np

VERSION = "trajectory_rules_v4"


def fit_z(track):
    """Fit the whole path to horizontal / diagonal-down / horizontal strokes.

    Displacement-based segmentation tolerates variable speed, holds and small jitter.
    Both horizontal directions are accepted for camera/handedness conventions;
    this is recognition, not certification of ASL orientation.
    """
    best = float("inf")
    for i in range(3, len(track)-6):
        for j in range(i+3, len(track)-3):
            vertices = track[[0, i, j, -1]]
            delta = np.diff(vertices, axis=0)
            dx, dy = delta[:, 0], delta[:, 1]
            if not (min(abs(dx)) >= .10 and dx[0]*dx[1] < 0 and dx[1]*dx[2] < 0):
                continue
            if dy[1] < .10 or vertices[-1, 1] - vertices[0, 1] < .10:
                continue
            if abs(dy[0]) > .06 + .45*abs(dx[0]) or abs(dy[2]) > .06 + .45*abs(dx[2]):
                continue
            if max(abs(dx[0]), abs(dx[2])) < .22:
                continue
            errors = []
            for segment, start, end in [(track[:i+1], vertices[0], vertices[1]), (track[i:j+1], vertices[1], vertices[2]), (track[j:], vertices[2], vertices[3])]:
                vector = end-start
                t = np.clip((segment-start)@vector / (vector@vector), 0, 1)
                errors.extend(np.linalg.norm(segment-(start+t[:, None]*vector), axis=1))
            error = float(np.quantile(errors, .95))
            best = min(best, error)
    return {"matched": bool(best <= .085), "p95_deviation": None if not np.isfinite(best) else best, "max_deviation": .085}


def trajectory_candidates(points, timestamps):
    pixels = np.asarray(points, dtype=float)
    scale = float(np.median(np.ptp(pixels, axis=1).max(axis=1)))
    if scale <= 1e-6:
        raise ValueError("degenerate_hand")
    times = np.asarray(timestamps)
    grid = np.linspace(times[0], times[-1], 45)
    found, measurements = [], {}
    for label, tip in [("J", 20), ("Z", 8)]:
        track = np.column_stack([np.interp(grid, times, pixels[:, tip, k]) for k in range(2)]) / scale
        track = np.column_stack([np.convolve(np.pad(track[:, k], (2, 2), mode="edge"), np.ones(5)/5, mode="valid") for k in range(2)])
        track -= track[0]
        path = float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())
        low = int(track[:, 1].argmax())
        descent = float(track[:, 1].max())
        hook = float(abs(track[-1, 0] - track[low, 0]))
        z_fit = fit_z(track) if label == "Z" else None
        match = (path >= .55 and descent >= .20 and hook >= .12 and low >= 8) if label == "J" else z_fit["matched"]
        measurements[label] = {"path_hand_widths": path, "descent": descent, "hook": hook, "trajectory_match": bool(match)}
        if z_fit is not None:
            measurements[label]["three_strokes"] = z_fit
        if match:
            found.append(label)
    return found, measurements


def filtered_index_path(points, timestamps):
    scale = float(np.median(np.linalg.norm(points[:, 9]-points[:, 0], axis=1)))
    if scale < 1e-6:
        raise ValueError("Degenerate palm scale")
    grid = np.linspace(timestamps[0], timestamps[-1], 60)
    tip = points[:, 8]
    padded = np.pad(tip, ((1, 1), (0, 0)), mode="edge")
    tip = np.stack([np.median(padded[i:i+3], axis=0) for i in range(len(tip))])
    track = np.column_stack([np.interp(grid, timestamps, tip[:, k]) for k in range(2)]) / scale
    track = np.column_stack([np.convolve(np.pad(track[:, k], (1, 1), mode="edge"), np.ones(3)/3, mode="valid") for k in range(2)])
    track -= track[0]
    return track, fit_z(track)


def recognize_motion(payload, classifier, threshold):
    frames = payload.frames
    details = {"version": VERSION, "experimental": True, "reason": None}
    base = dict(status="uncertain", recognition_method=VERSION, motion_details=details)
    if all(f.landmarks is None for f in frames):
        return dict(base, status="no_hand")
    reason = None
    if len(frames) < 15:
        reason = "insufficient_frames"
    elif any(f.landmarks is None for f in frames):
        reason = "tracking_lost_retry"
    elif any(f.handedness is None for f in frames):
        reason = "handedness_required"
    elif len({f.handedness for f in frames}) != 1:
        reason = "hand_changed_retry"
    elif max(np.diff([f.timestamp_ms for f in frames])) > 250:
        reason = "capture_gap_retry"
    if reason:
        details["reason"] = reason
        return base
    try:
        features = [classifier.features(f.landmarks, payload.image_width, payload.image_height) for f in frames]
        points = np.asarray([f.landmarks for f in frames])[:, :, :2] * [payload.image_width, payload.image_height]
        candidates, measurements = trajectory_candidates(points, [f.timestamp_ms for f in frames])
        track, fitted = filtered_index_path(points, [f.timestamp_ms for f in frames])
        candidates = [letter for letter in candidates if letter != "Z"]
        if fitted["matched"]:
            candidates.append("Z")
        measurements["Z"].update(trajectory_match=fitted["matched"], three_strokes=fitted)
        details.update(index_path_normalized=track.tolist(), landmark_index=8,
                       handshape_assessed=True, handshape_assessment_scope="classifier_gate_not_skill_diagnosis")
    except ValueError:
        return dict(base, status="invalid_landmarks")
    probabilities = classifier.predict_proba(np.asarray(features))
    labels = np.asarray(classifier.labels)[probabilities.argmax(axis=1)]
    mean_scores = probabilities.mean(axis=0)
    details["onnx_frame_summary"] = {
        "frames": len(frames),
        "label_counts": {str(label): int(np.sum(labels == label)) for label in np.unique(labels)},
        "top3_mean_frame_scores": [{"label": str(classifier.labels[i]), "score": float(mean_scores[i])} for i in np.argsort(mean_scores)[::-1][:3]],
        "semantics": "Static handshape scores only; not motion confidence"
    }
    confident = probabilities.max(axis=1) >= threshold
    # Shape family and motion must agree; the requested target never picks a label.
    agreements = {letter: float(np.mean(confident & np.isin(labels, family))) for letter, family in [("J", ["I", "J"]), ("Z", ["D", "Z"])]}
    # Both checks are mandatory; geometry and the requested label cannot bypass ONNX.
    shape_pass = {"J": agreements["J"] >= .5, "Z": agreements["Z"] >= .5}
    details["shape_gate_passed"] = shape_pass
    details["shape_family_threshold"] = .5
    details["classifier_frame_threshold"] = threshold
    details["z_shape_source"] = "classifier" if shape_pass["Z"] else "unconfirmed"
    accepted = [letter for letter in candidates if shape_pass[letter]]
    details.update(measurements=measurements, shape_family_agreement=agreements)
    if len(accepted) != 1:
        details["reason"] = "motion_or_handshape_not_confirmed"
        if payload.target_label.strip().upper() == "Z":
            details["reason"] = "z_handshape_not_confirmed" if fitted["matched"] and not shape_pass["Z"] else "index_path_not_confirmed" if shape_pass["Z"] and not fitted["matched"] else "motion_or_handshape_not_confirmed"
        return dict(base, usable=len(frames))
    label = accepted[0]
    details["reason"] = "trajectory_and_handshape_matched"
    # No fabricated probability: classifier confidence is not confidence in a rule.
    return dict(base, status="recognized", usable=len(frames), predicted_label=label,
                valid=True, matches_target=label == payload.target_label.strip().upper())
