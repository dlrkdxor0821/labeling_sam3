"""Interactive YOLO training: pick model + settings, train, save under model/<name>/."""
import argparse
from pathlib import Path

from utils.config import load_config
from utils.paths import DATASETS_ROOT

AVAILABLE_MODELS = [
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
]


def ask(label, default):
    val = input(f"{label} [{default}]: ").strip()
    return val or str(default)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=cfg["name"])
    ap.add_argument("--yes", action="store_true", help="use config defaults, no prompts")
    args = ap.parse_args()

    dataset_dir = DATASETS_ROOT / args.name
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml missing: {data_yaml} (run 05_export_yolo first)")

    t = cfg["train"]
    if args.yes:
        model, epochs, imgsz, batch = t["yolo_model"], t["epochs"], t["imgsz"], t["batch"]
    else:
        print("Available models:", " / ".join(AVAILABLE_MODELS))
        model = ask("YOLO model", t["yolo_model"])
        epochs = int(ask("epochs", t["epochs"]))
        imgsz = int(ask("imgsz", t["imgsz"]))
        batch = int(ask("batch", t["batch"]))

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("ultralytics required: pip install -U ultralytics") from e

    yolo = YOLO(f"{model}.pt")
    # absolute project path: a relative one gets nested under runs/detect/ by Ultralytics
    yolo.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=0,
        project=str((Path("model") / args.name).resolve()), name="train", exist_ok=True,
    )
    print(f"done -> model/{args.name}/train/weights/best.pt + results.png")


if __name__ == "__main__":
    main()
