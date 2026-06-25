"""Interactive YOLO training: pick model + settings, train, save under model/<name>/."""
import argparse
import time
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT
from utils.prompt import ask_existing_dir, ask_choice, ask_int
from utils.metrics import top_epochs, format_top_table, save_ranking_png

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

    # ----- 전체 학습 진행률 + 예상 남은시간 (epoch 끝날 때마다 1줄) -----
    start = time.time()

    def _fmt(sec):
        sec = int(sec)
        h, sec = divmod(sec, 3600)
        m, sec = divmod(sec, 60)
        return f"{h}h {m}m" if h else f"{m}m {sec}s"

    def _on_epoch_end(trainer):
        total = getattr(trainer, "epochs", epochs)
        done = getattr(trainer, "epoch", 0) + 1
        elapsed = time.time() - start
        eta = elapsed / done * (total - done) if done else 0
        print(f"  ⏱  전체 진행률 {done}/{total} ({100 * done / total:.0f}%) | "
              f"경과 {_fmt(elapsed)} | 남은시간 ≈ {_fmt(eta)}")

    yolo.add_callback("on_fit_epoch_end", _on_epoch_end)

    # absolute project path: a relative one gets nested under runs/detect/ by Ultralytics
    yolo.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=0,
        project=str((Path("model") / name).resolve()), name="train", exist_ok=True,
    )
    print(f"done -> model/{name}/train/weights/best.pt + results.png")

    # fitness 상위 10 epoch 표 — best.pt 가 어느 epoch에서 나왔는지 바로 확인
    results_csv = Path("model") / name / "train" / "results.csv"
    if results_csv.exists():
        rows = top_epochs(results_csv, 10)
        print("\n=== fitness 상위 10 epoch (best.pt = 1위) ===")
        print(format_top_table(rows))
        print(f"best.pt = epoch {rows[0]['epoch']}")
        png = save_ranking_png(results_csv, results_csv.parent / "metrics_report.png", name=name)
        print(f"리포트 이미지 저장됨 -> {png}")


if __name__ == "__main__":
    main()
