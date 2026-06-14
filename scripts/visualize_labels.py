"""Draw YOLO labels onto images so you can eyeball the whole dataset.

For each frame it renders the boxes (with class name + SAM3 confidence if
available) and writes annotated copies to datasets/<name>/<split>/_viz/.
Open that folder in any image viewer to flip through them one by one.

Usage:
    python scripts/visualize_labels.py --name bluebook            # train + test
    python scripts/visualize_labels.py --name bluebook --split train
    python scripts/visualize_labels.py --name bluebook --grid     # + contact sheet
"""
import argparse
import json

import cv2
import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT, SPLITS, split_subdirs
from utils.qc import read_yolo_boxes
from utils.prompt import ask_existing_dir, ask_choice, confirm


def render(img, boxes, class_names, header):
    """Draw boxes (cx,cy,w,h normalized) + header text onto a copy of img."""
    out = img.copy()
    h, w = out.shape[:2]
    for cls, (cx, cy, bw, bh) in boxes:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        name = class_names[cls] if cls < len(class_names) else str(cls)
        cv2.putText(out, name, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(out, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def read_boxes_with_cls(txt_path):
    """YOLO .txt -> list of (class_id, (cx,cy,w,h))."""
    out = []
    p = pathlib.Path(txt_path)
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                out.append((int(parts[0]), tuple(float(x) for x in parts[1:])))
    return out


def visualize_split(subs, split, class_names, make_grid):
    img_dir, lbl_dir = subs[split]["images"], subs[split]["labels"]
    if not img_dir.exists():
        print(f"  [{split}] images 없음 - 건너뜀")
        return
    conf_map = {}
    conf_file = lbl_dir / "_conf.json"
    if conf_file.exists():
        conf_map = json.loads(conf_file.read_text())

    viz_dir = img_dir.parent / "_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for old in viz_dir.glob("*.jpg"):
        old.unlink()

    tiles, n_box, n_empty = [], 0, 0
    for ip in sorted(img_dir.glob("*.jpg")):
        img = cv2.imread(str(ip))
        boxes = read_boxes_with_cls(lbl_dir / f"{ip.stem}.txt")
        n_box += len(boxes)
        if not boxes:
            n_empty += 1
        c = conf_map.get(ip.stem)
        header = f"{ip.stem}" + (f"  c={c:.2f}" if c is not None else "  (no box)")
        out = render(img, boxes, class_names, header)
        cv2.imwrite(str(viz_dir / ip.name), out)
        if make_grid:
            tile = cv2.resize(out, (320, 180))
            # 축소돼도 프레임 번호가 또렷하게 보이도록 고정 크기 라벨바를 덧그림
            cv2.rectangle(tile, (0, 0), (320, 26), (0, 0, 0), -1)
            cv2.putText(tile, ip.stem, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)

    if make_grid and tiles:
        cols = 6
        rows = (len(tiles) + cols - 1) // cols
        grid = np.full((rows * 180, cols * 320, 3), 40, np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            grid[r * 180:r * 180 + 180, c * 320:c * 320 + 320] = t
        cv2.imwrite(str(viz_dir / "_grid.jpg"), grid)

    print(f"  [{split}] {len(list(img_dir.glob('*.jpg')))}장 -> {viz_dir}  "
          f"(박스 {n_box}개, 빈라벨 {n_empty}장)"
          + ("  + _grid.jpg" if make_grid and tiles else ""))


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", default=None, choices=["all", "train", "test"])
    ap.add_argument("--grid", action="store_true", help="also write a _grid.jpg contact sheet")
    args = ap.parse_args()

    name, dataset_dir = ask_existing_dir("어떤 데이터셋을 시각화할까요?", args.name, DATASETS_ROOT)
    split_sel = ask_choice("어느 split을 볼까요?", args.split, ["all", "train", "test"], "all")
    grid = args.grid or confirm("그리드(_grid.jpg)도 만들까요?")
    subs = split_subdirs(dataset_dir)
    classes_file = dataset_dir / "classes.txt"
    class_names = classes_file.read_text().split() if classes_file.exists() else ["object"]

    print(f"[viz] dataset: {dataset_dir}  classes={class_names}")
    splits = SPLITS if split_sel == "all" else (split_sel,)
    for split in splits:
        visualize_split(subs, split, class_names, grid)
    print("이미지 뷰어로 _viz 폴더를 열어 한 장씩 확인하세요 (예: xdg-open <_viz>)")


if __name__ == "__main__":
    main()
