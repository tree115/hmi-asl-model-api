"""Names and units for the orientation features from feature_extraction.py."""

ORIENTATION_FEATURES = {
    0: {
        "feature": "palm_roll_sin",
        "unit": "unitless",
        "scale": 1.0,
    },
    1: {
        "feature": "palm_roll_cos",
        "unit": "unitless",
        "scale": 1.0,
    },
    2: {
        "feature": "palm_horizontal_axis_x",
        "unit": "unitless",
        "scale": 1.0,
    },
    3: {
        "feature": "palm_horizontal_axis_y",
        "unit": "unitless",
        "scale": 1.0,
    },
}

if tuple(ORIENTATION_FEATURES) != tuple(range(4)):
    raise RuntimeError("Orientation feature metadata must describe indices 0..3")