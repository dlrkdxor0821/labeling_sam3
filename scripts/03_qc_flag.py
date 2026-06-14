"""QC: flag suspicious frames into _needs_review/ (image + labelme JSON to edit)."""
import argparse
import json
import shutil

import cv2

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT, SPLITS, split_subdirs
from utils.qc import flagged_stems, frame_suspicion, read_yolo_boxes
from utils.labelme_io import yolo_to_labelme
from utils.prompt import ask_existing_dir, ask_float


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    name, dataset_dir = ask_existing_dir("어떤 데이터셋을 QC 선별할까요?", args.name, DATASETS_ROOT)
    threshold = ask_float("의심 기준 threshold (높일수록 적게 선별)", args.threshold, 1.0)
    subs = split_subdirs(dataset_dir)
    classes_file = dataset_dir / "classes.txt"
    class_names = classes_file.read_text().split() if classes_file.exists() else ["object"]

    for split in SPLITS:
        img_dir, lbl_dir, rev_dir = (
            subs[split]["images"], subs[split]["labels"], subs[split]["needs_review"],
        )
        if not lbl_dir.exists():
            print(f"  ! {split}: no labels - skipped")
            continue

        conf_map = {}
        conf_file = lbl_dir / "_conf.json"
        if conf_file.exists():
            conf_map = json.loads(conf_file.read_text())

        frames = []
        for img in sorted(img_dir.glob("*.jpg")):
            boxes = read_yolo_boxes(lbl_dir / f"{img.stem}.txt")
            frames.append({"stem": img.stem, "boxes": boxes, "conf": conf_map.get(img.stem)})

        scores = frame_suspicion(frames, conf_threshold=cfg["qc"]["conf_threshold"])
        flagged = set(flagged_stems(scores, threshold=threshold))

        rev_dir.mkdir(parents=True, exist_ok=True)
        for img in sorted(img_dir.glob("*.jpg")):
            if img.stem not in flagged:
                continue
            shutil.copy(img, rev_dir / img.name)
            im = cv2.imread(str(img))
            h, w = im.shape[:2]
            lines = (lbl_dir / f"{img.stem}.txt").read_text().splitlines()
            lj = yolo_to_labelme(lines, w, h, class_names, img.name)
            (rev_dir / f"{img.stem}.json").write_text(json.dumps(lj, indent=2))
        print(f"  {split}: {len(flagged)}/{len(frames)} flagged -> {rev_dir}")


if __name__ == "__main__":
    main()
