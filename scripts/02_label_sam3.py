"""SAM3 concept labeling: images -> YOLO .txt labels + classes.txt + conf sidecar."""
import argparse
import json
from pathlib import Path

import cv2

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT, SPLITS, split_subdirs
from utils.boxes import yolo_label_lines
from utils.prompt import ask_existing_dir, ask_text


def load_predictor(model_path, conf, half):
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as e:
        raise SystemExit(
            "ultralytics is required:\n"
            "  pip install -U ultralytics\n"
            "  pip install git+https://github.com/ultralytics/CLIP.git"
        ) from e
    if not Path(model_path).exists():
        raise SystemExit(
            f"SAM3 weights missing: {model_path}\n"
            "Request access on Hugging Face, then place sam3.pt under model/."
        )
    overrides = dict(conf=conf, task="segment", mode="predict", device=0,
                     model=str(model_path), half=half, save=False, verbose=False)
    return SAM3SemanticPredictor(overrides=overrides)


def label_one(predictor, image_path, prompt):
    """Run SAM3 on one image -> (yolo_lines, max_conf or None)."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    predictor.set_image(str(image_path))
    results = predictor(text=[prompt])
    res = results[0] if isinstance(results, (list, tuple)) else results
    boxes_xyxy, confs = [], []
    b = getattr(res, "boxes", None)
    if b is not None:
        xyxy = b.xyxy
        xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
        boxes_xyxy = [tuple(map(float, row)) for row in xyxy]
        if getattr(b, "conf", None) is not None:
            c = b.conf
            c = c.cpu().numpy() if hasattr(c, "cpu") else c
            confs = [float(x) for x in c]
    lines = yolo_label_lines(boxes_xyxy, w, h, class_id=0)
    return lines, (max(confs) if confs else None)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    name, dataset_dir = ask_existing_dir("어떤 데이터셋을 라벨링할까요?", args.name, DATASETS_ROOT)
    prompt = ask_text("라벨링할 객체(프롬프트)?", args.prompt, cfg["prompt"])
    subs = split_subdirs(dataset_dir)
    (dataset_dir / "classes.txt").write_text(prompt + "\n")

    predictor = load_predictor(cfg["sam3"]["model"], cfg["sam3"]["conf"], cfg["sam3"]["half"])
    print(f"[label] SAM3 prompt='{prompt}' -> {dataset_dir}")

    for split in SPLITS:
        img_dir, lbl_dir = subs[split]["images"], subs[split]["labels"]
        if not img_dir.exists():
            print(f"  ! {split}: no images - skipped")
            continue
        lbl_dir.mkdir(parents=True, exist_ok=True)
        confs, total = {}, 0
        imgs = sorted(img_dir.glob("*.jpg"))
        for img in imgs:
            lines, mc = label_one(predictor, img, prompt)
            (lbl_dir / f"{img.stem}.txt").write_text("\n".join(lines))
            if mc is not None:
                confs[img.stem] = mc
            total += len(lines)
        (lbl_dir / "_conf.json").write_text(json.dumps(confs))
        print(f"  done {split}: {len(imgs)} imgs -> {total} boxes")


if __name__ == "__main__":
    main()
