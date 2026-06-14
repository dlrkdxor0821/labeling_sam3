"""Interactive YOLO training: pick model + settings, train, save under model/<name>/."""
import argparse
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT
from utils.prompt import ask_existing_dir, ask_choice, ask_int

AVAILABLE_MODELS = [
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--yes", action="store_true", help="use config defaults, no prompts")
    args = ap.parse_args()

    name, dataset_dir = ask_existing_dir("어떤 데이터셋으로 학습할까요?", args.name, DATASETS_ROOT)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml missing: {data_yaml} (run 05_export_yolo first)")

    t = cfg["train"]
    if args.yes:
        model, epochs, imgsz, batch = t["yolo_model"], t["epochs"], t["imgsz"], t["batch"]
    else:
        default_model = t["yolo_model"] if t["yolo_model"] in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        print("선택 가능 모델:", " / ".join(AVAILABLE_MODELS))
        model = ask_choice("YOLO model", None, AVAILABLE_MODELS, default_model)
        epochs = ask_int("epochs", None, t["epochs"])
        imgsz = ask_int("imgsz", None, t["imgsz"])
        batch = ask_int("batch", None, t["batch"])

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("ultralytics required: pip install -U ultralytics") from e

    yolo = YOLO(f"{model}.pt")
    # absolute project path: a relative one gets nested under runs/detect/ by Ultralytics
    yolo.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=0,
        project=str((Path("model") / name).resolve()), name="train", exist_ok=True,
    )
    print(f"done -> model/{name}/train/weights/best.pt + results.png")


if __name__ == "__main__":
    main()
