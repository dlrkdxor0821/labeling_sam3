"""학습 지표 보기: results.csv -> fitness 상위 N epoch 표 (best.pt = 1위).

    python scripts/show_metrics.py                      # 모델 선택 후 상위 10
    python scripts/show_metrics.py --name person        # 바로 person 모델
    python scripts/show_metrics.py --name person --top 20
"""
import argparse
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.prompt import ask_existing_dir
from utils.metrics import top_epochs, format_top_table, save_ranking_png

MODEL_ROOT = Path("model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    name, model_dir = ask_existing_dir("어떤 모델의 지표를 볼까요?", args.name, MODEL_ROOT)
    results_csv = model_dir / "train" / "results.csv"
    if not results_csv.exists():
        raise SystemExit(f"results.csv 없음: {results_csv} (먼저 06_train_yolo 로 학습)")

    rows = top_epochs(results_csv, args.top)
    print(f"\n=== {name}: fitness 상위 {len(rows)} epoch (best.pt = 1위) ===")
    print(format_top_table(rows))
    best = rows[0]
    print(f"\nbest.pt = epoch {best['epoch']}  "
          f"(mAP50 {best['mAP50']:.3f} / mAP50-95 {best['mAP50-95']:.3f} / "
          f"P {best['precision']:.3f} / R {best['recall']:.3f})")

    png = save_ranking_png(results_csv, results_csv.parent / "metrics_report.png",
                           name=name, top=args.top)
    print(f"\n리포트 이미지 저장됨 -> {png}")


if __name__ == "__main__":
    main()
