"""Inference: detect with the trained model. --source camera | video.mp4 | image.jpg."""
import argparse
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.prompt import ask_existing_dir, ask_text


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--source", default=None, help="camera | <video/image path>")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    name, _ = ask_existing_dir("어떤 모델로 추론할까요?", args.name, Path("model"))
    source = ask_text("추론 대상 (camera | 영상/이미지 경로)", args.source, "camera")

    weights = args.weights or f"model/{name}/train/weights/best.pt"
    if not Path(weights).exists():
        raise SystemExit(f"weights missing: {weights} (run 06_train_yolo first)")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("ultralytics required: pip install -U ultralytics") from e

    src = 0 if source == "camera" else source
    yolo = YOLO(weights)
    results = yolo.predict(source=src, show=True, stream=(src == 0), device=0)
    if src == 0:
        for _ in results:  # consume the live stream to keep the camera window open
            pass
    print("done")


if __name__ == "__main__":
    main()
