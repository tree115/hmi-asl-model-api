"""Fetch a second frozen ASL candidate that renders the 21-point hand skeleton."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "punpuniacitizen/MediaPipe-ASL-sign-language-recognition"
REVISION = "45ff2c0e26925c50f4546073bd573fe3ca7a5e6e"


def main():
    directory = ROOT / "models/onnx_alphabet"
    directory.mkdir(parents=True, exist_ok=True)
    existing = directory / "manifest.json"
    if existing.exists():
        frozen = json.loads(existing.read_text(encoding="utf-8"))
        names = {entry["path"] for entry in frozen.get("files", [])}
        required = {"asl_cnn_model.onnx", "class_names.txt", "preprocessing.py", "README.md", "LICENSE"}
        if frozen.get("revision") == REVISION and names == required and all(
            (directory / entry["path"]).exists() and hashlib.sha256((directory / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
            for entry in frozen["files"]
        ):
            print("Frozen ONNX assets verified; no download needed")
            return
    manifest = {"repo": REPO, "revision": REVISION, "files": []}
    for name in ["asl_cnn_model.onnx", "class_names.txt", "preprocessing.py", "README.md", "LICENSE"]:
        url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        (directory / name).write_bytes(content)
        manifest["files"].append({"path": name, "source": url, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        print(name, len(content), flush=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
