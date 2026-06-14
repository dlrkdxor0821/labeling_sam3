"""OBB pipeline (1/2): build an oriented-box dataset from SAM3 masks.

The main pipeline produces *axis-aligned* boxes — no angle, useless for grasp
orientation. This is a SEPARATE track: it re-runs SAM3 in segment mode on an
already-extracted dataset, fits a rotated rectangle to each mask, and writes
YOLO-OBB labels (4 corner points) into a new datasets/<name>_obb/ dataset.

    python scripts/obb_label.py
        1) 어떤 데이터셋(프레임)을 OBB 라벨링할까요?   (datasets/<name>, 이미지 필요)
        2) 라벨링할 객체(프롬프트)?

Then train with obb_train.py.
"""
import argparse
import json
import shutil

import cv2
import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT, SPLITS, resolve_versioned_dir, split_subdirs
from utils.dataset import make_data_yaml
from utils.prompt import ask_existing_dir, ask_text, confirm


def load_predictor(model_path, conf, half):
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as e:
        raise SystemExit(
            "ultralytics required:\n  pip install -U ultralytics\n"
            "  pip install git+https://github.com/ultralytics/CLIP.git"
        ) from e
    if not pathlib.Path(model_path).exists():
        raise SystemExit(f"SAM3 weights missing: {model_path} (model/sam3.pt)")
    overrides = dict(conf=conf, task="segment", mode="predict", device=0,
                     model=str(model_path), half=half, save=False, verbose=False)
    return SAM3SemanticPredictor(overrides=overrides)


def polygon_to_obb_line(points, img_w, img_h, class_id=0):
    """Fit a rotated rect to a mask polygon -> normalized YOLO-OBB line (4 corners)."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return None
    rect = cv2.minAreaRect(pts)              # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)                # 4 corner points (pixel)
    coords = []
    for x, y in box:
        coords.append(max(0.0, min(1.0, x / img_w)))
        coords.append(max(0.0, min(1.0, y / img_h)))
    return f"{class_id} " + " ".join(f"{c:.6f}" for c in coords)


def label_image(predictor, image_path, prompt):
    """Run SAM3 segment -> list of OBB lines (one per mask), max conf."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    predictor.set_image(str(image_path))
    results = predictor(text=[prompt])
    res = results[0] if isinstance(results, (list, tuple)) else results
    lines, confs = [], []
    masks = getattr(res, "masks", None)
    if masks is not None and getattr(masks, "xy", None) is not None:
        for poly in masks.xy:                # poly = Nx2 pixel coords
            line = polygon_to_obb_line(poly, w, h)
            if line:
                lines.append(line)
        b = getattr(res, "boxes", None)
        if b is not None and getattr(b, "conf", None) is not None:
            c = b.conf
            c = c.cpu().numpy() if hasattr(c, "cpu") else c
            confs = [float(x) for x in c]
    return lines, (max(confs) if confs else None)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    print("=== OBB 라벨 생성 (SAM3 마스크 -> 회전박스) ===")
    name, src_dir = ask_existing_dir("1) 어떤 데이터셋을 OBB 라벨링할까요?", args.name, DATASETS_ROOT)
    prompt = ask_text("2) 라벨링할 객체(프롬프트)?", args.prompt, cfg["prompt"])
    src_subs = split_subdirs(src_dir)

    out_dir = resolve_versioned_dir(DATASETS_ROOT, f"{name}_obb")
    print(f"  출력: {out_dir}")
    if not confirm("진행할까요? (SAM3 재실행, 시간이 걸립니다)", preset_yes=args.yes):
        raise SystemExit("취소됨")

    predictor = load_predictor(cfg["sam3"]["model"], cfg["sam3"]["conf"], cfg["sam3"]["half"])
    out_subs = split_subdirs(out_dir)
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text(prompt + "\n")

    for split in SPLITS:
        img_dir = src_subs[split]["images"]
        if not img_dir.exists():
            print(f"  ! {split}: 이미지 없음 - 건너뜀")
            continue
        o_img, o_lbl = out_subs[split]["images"], out_subs[split]["labels"]
        o_img.mkdir(parents=True, exist_ok=True)
        o_lbl.mkdir(parents=True, exist_ok=True)
        confs, total = {}, 0
        imgs = sorted(img_dir.glob("*.jpg"))
        for ip in imgs:
            lines, mc = label_image(predictor, ip, prompt)
            shutil.copy(ip, o_img / ip.name)
            (o_lbl / f"{ip.stem}.txt").write_text("\n".join(lines))
            if mc is not None:
                confs[ip.stem] = mc
            total += len(lines)
        (o_lbl / "_conf.json").write_text(json.dumps(confs))
        print(f"  done {split}: {len(imgs)} imgs -> {total} OBB boxes")

    yaml = make_data_yaml(out_dir, [prompt])
    print(f"\n완료 -> {out_dir}  ({yaml})")
    print(f"  학습: python scripts/obb_train.py --name {out_dir.name}")


if __name__ == "__main__":
    main()
