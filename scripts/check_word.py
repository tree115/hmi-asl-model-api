"""Smoke-test the word route using a published, de-identified GestureSign replay."""
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from ai.gesturesign_features import landmarks as lm


def main():
    replay = json.loads((Path(__file__).resolve().parents[1] / "integration" /
                         "gesturesign_hello_replay.json").read_text(encoding="utf-8"))
    clip = replay["clips"][0]
    frames = []
    for sample in clip["frames"]:
        points = [None] * lm.N_POINTS
        for index, x, y in zip(replay["points"], sample[::2], sample[1::2]):
            if x is not None and y is not None:
                points[index] = [x, y, 0]
        frames.append({
            part: points[offset:offset + lm.TYPE_SIZES[part]]
            for part, offset in lm.TYPE_OFFSETS.items()
        })
    payload = {"client_attempt_id": "replay-hello", "target_label": "hello",
               "fps": clip["fps"], "frames": frames}
    with TestClient(app) as client:
        assert len(client.get("/api/v1/ai/words/capabilities").json()["supported_targets"]) == 250
        route = "/api/v1/ai/predict/word"
        result = client.post(route, json=payload)
        assert result.status_code == 200, result.text
        data = result.json()
        assert (data["status"], data["predicted_label"], data["matches_target"]) == (
            "recognized", "hello", True
        ), data
        wrong = client.post(route, json={**payload, "target_label": "water"}).json()
        assert wrong["predicted_label"] == "hello" and wrong["matches_target"] is False
        no_hand = client.post(route, json={**payload, "frames": [{}]}).json()
        assert no_hand["status"] == "no_hand" and no_hand["matches_target"] is None
        uncertain = client.post(route, json={
            **payload, "frames": [{**frames[0], "pose": None}]
        }).json()
        assert uncertain["status"] == "uncertain" and uncertain["matches_target"] is None
        unsupported = client.post(route, json={**payload, "target_label": "sorry"}).json()
        assert unsupported["status"] == "unsupported_target"
        assert client.post(route, json={**payload, "mirrored": True}).status_code == 422
        assert client.post(route, json={**payload, "fps": 1}).status_code == 422
        if "--write-example" in sys.argv:
            example = Path(__file__).resolve().parents[1] / "integration" / "word_example_request.json"
            example.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print("GestureSign word API passed replay, mismatch, no-hand, uncertainty and validation checks.")


if __name__ == "__main__":
    main()
