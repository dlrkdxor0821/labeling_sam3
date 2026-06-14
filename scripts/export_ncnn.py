"""Export a trained YOLO model for fast on-device inference (NCNN by default).

Converts model/<name>/train/weights/best.pt to an ARM-optimized format so it
runs fast on a Raspberry Pi. NCNN is the usual pick for Pi CPU; onnx/tflite are
also available (tflite INT8 is what a Coral Edge TPU needs).

    python scripts/export_ncnn.py
        1) 어떤 모델을 변환할까요?   (model/<name>)
        2) 포맷?                    (ncnn / onnx / tflite)
        3) imgsz?                   (Pi 는 320 권장)
"""
import argparse

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.prompt import ask_existing_dir, ask_choice, ask_int

MODEL_ROOT = pathlib.Path("model")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--weights", default=None, help="직접 .pt 경로 지정(이름 대신)")
    ap.add_argument("--format", default=None, choices=["ncnn", "onnx", "tflite"])
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--int8", action="store_true", help="INT8 양자화(더 빠름, 정확도 약간↓)")
    args = ap.parse_args()

    print("=== 모델 변환 (배포용) ===")
    if args.weights:
        weights = pathlib.Path(args.weights)
    else:
        name, _ = ask_existing_dir("1) 어떤 모델을 변환할까요?", args.name, MODEL_ROOT)
        weights = MODEL_ROOT / name / "train" / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"가중치 없음: {weights} (먼저 06_train_yolo 로 학습)")

    fmt = ask_choice("2) 어떤 포맷으로?", args.format, ["ncnn", "onnx", "tflite"], "ncnn")
    imgsz = ask_int("3) imgsz (Pi 는 320 권장)", args.imgsz, cfg["train"]["imgsz"])

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("ultralytics required: pip install -U ultralytics") from e

    print(f"\n  {weights}  ->  {fmt} (imgsz={imgsz}{', int8' if args.int8 else ''})")
    print("  변환 중... (NCNN 은 첫 실행 시 pnnx 도구를 자동 다운로드합니다)")
    out = YOLO(str(weights)).export(format=fmt, imgsz=imgsz, int8=args.int8)
    print(f"\n완료 ✅ -> {out}")
    print("  Pi 로 이 파일을 복사해 추론에 사용하세요 (YOLO('<출력>').predict(...)).")


if __name__ == "__main__":
    main()
