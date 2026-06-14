"""Inference: detect with the trained model. --source camera | video.mp4 | image.jpg."""
import argparse
from pathlib import Path

from utils.config import load_config


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=cfg["name"])
    ap.add_argument("--source", default="camera", help="camera | <video/image path>")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    weights = args.weights or f"model/{args.name}/train/weights/best.pt"
    if not Path(weights).exists():
        raise SystemExit(f"weights missing: {weights} (run 06_train_yolo first)")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("ultralytics required: pip install -U ultralytics") from e

    source = 0 if args.source == "camera" else args.source
    yolo = YOLO(weights)
    results = yolo.predict(source=source, show=True, stream=(source == 0))
    if source == 0:
        for _ in results:  # consume the live stream to keep the camera window open
            pass
    print("done")


if __name__ == "__main__":
    main()
