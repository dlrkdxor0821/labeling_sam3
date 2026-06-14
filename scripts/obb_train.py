"""OBB pipeline (2/2): train an oriented-box YOLO model.

Trains on an OBB dataset produced by obb_label.py (datasets/<name>_obb/). The
result detects rotated boxes — `results[0].obb.xywhr` gives center, size and the
rotation angle that a robot gripper can use to grasp.

    python scripts/obb_train.py
        1) 어떤 OBB 데이터셋으로 학습할까요?
        2~) 모델 / epochs / imgsz / batch
"""
import argparse
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT
from utils.prompt import ask_existing_dir, ask_choice, ask_int

OBB_MODELS = [
    "yolo11n-obb", "yolo11s-obb", "yolo11m-obb", "yolo11l-obb", "yolo11x-obb",
    "yolov8n-obb", "yolov8s-obb", "yolov8m-obb", "yolov8l-obb", "yolov8x-obb",
]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--yes", action="store_true", help="config 기본값으로 바로 학습")
    args = ap.parse_args()

    print("=== OBB 모델 학습 ===")
    name, dataset_dir = ask_existing_dir("1) 어떤 OBB 데이터셋으로 학습할까요?", args.name, DATASETS_ROOT)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml missing: {data_yaml} (먼저 obb_label.py 실행)")

    t = cfg["train"]
    if args.yes:
        model, epochs, imgsz, batch = "yolov8n-obb", t["epochs"], t["imgsz"], t["batch"]
    else:
        print("선택 가능 OBB 모델:", " / ".join(OBB_MODELS))
        model = ask_choice("YOLO-OBB model", None, OBB_MODELS, "yolov8n-obb")
        epochs = ask_int("epochs", None, t["epochs"])
        imgsz = ask_int("imgsz", None, t["imgsz"])
        batch = ask_int("batch", None, t["batch"])

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("ultralytics required: pip install -U ultralytics") from e

    yolo = YOLO(f"{model}.pt")  # obb pretrained weights
    yolo.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=0,
        project=str((Path("model") / name).resolve()), name="train", exist_ok=True,
    )
    print(f"done -> model/{name}/train/weights/best.pt (OBB)")
    print(f"  변환: python scripts/export_ncnn.py --name {name}")


if __name__ == "__main__":
    main()
