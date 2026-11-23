"""Names and units for the handshape features from feature_extraction.py."""

# 25 góc khớp ngón tay/ ngón tay
_ANGLE_FEATURES = (
    ("thumb_mcp_angle", "thumb"),
    ("thumb_base_angle", "thumb"),
    ("index_pip_angle", "index"),
    ("index_mcp_angle", "index"),
    ("middle_pip_angle", "middle"),
    ("middle_mcp_angle", "middle"),
    ("ring_pip_angle", "ring"),
    ("ring_mcp_angle", "ring"),
    ("pinky_pip_angle", "pinky"),
    ("pinky_mcp_angle", "pinky"),
    ("thumb_ip_angle", "thumb"),
    ("index_pip_inner_angle", "index"),
    ("index_dip_angle", "index"),
    ("middle_pip_inner_angle", "middle"),
    ("middle_dip_angle", "middle"),
    ("ring_pip_inner_angle", "ring"),
    ("ring_dip_angle", "ring"),
    ("pinky_pip_inner_angle", "pinky"),
    ("pinky_dip_angle", "pinky"),
    ("thumb_index_spread_angle", "thumb_index"),
    ("index_pinky_spread_angle", "index_pinky"),
    ("ring_pinky_angle", "ring_pinky"),
    ("index_middle_angle", "index_middle"),
    ("thumb_pinky_angle", "thumb_pinky"),
    ("index_pinky_angle", "index_pinky"),
)

# khoảng cách cổ tay + đầu ngón + giữa đầu ngón
_DISTANCE_PAIRS = (
    ("wrist", "thumb"),
    ("wrist", "index"),
    ("wrist", "middle"),
    ("wrist", "ring"),
    ("wrist", "pinky"),
    ("thumb", "index"),
    ("thumb", "middle"),
    ("thumb", "ring"),
    ("thumb", "pinky"),
    ("index", "middle"),
    ("index", "ring"),
    ("index", "pinky"),
    ("middle", "ring"),
    ("middle", "pinky"),
    ("ring", "pinky"),
)

HANDSHAPE_FEATURES = {}
for index, (feature, finger) in enumerate(_ANGLE_FEATURES):
    HANDSHAPE_FEATURES[index] = {
        "feature": feature,
        "finger": finger,
        "unit": "degree",
        "scale": 180.0,
    }

for index, (first, second) in enumerate(_DISTANCE_PAIRS, start=25):
    HANDSHAPE_FEATURES[index] = {
        "feature": f"{first}_{second}_distance",
        "finger": second if first == "wrist" else f"{first}_{second}",
        "unit": "palm_ratio",
        "scale": 1.0,
    }

if len(HANDSHAPE_FEATURES) != 40:
    raise RuntimeError("Handshape feature metadata must describe all 40 features")
