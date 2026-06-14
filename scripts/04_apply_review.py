"""Apply manual review, then clean up and verify the dataset.

After step 3 (QC) copies suspicious frames into _needs_review/, you open them in
labelme and either (a) fix the boxes and save the JSON, or (b) delete the JSON
for frames you judge useless. This script turns those decisions into the real
dataset:

  _needs_review/<f>.json present  -> merge into labels/<f>.txt (keep frame)
  _needs_review/<f>.jpg, no .json -> DELETE images/<f>.jpg + labels/<f>.txt
  (frame not in _needs_review)     -> left untouched (original SAM3 label kept)

It then removes the _needs_review/ folder (leftover jpgs included) and runs an
integrity check over the whole dataset. Deletions are previewed first and require
confirmation (or pass --yes).

Usage:
    python scripts/04_apply_review.py --name bluebook
    python scripts/04_apply_review.py --name bluebook --yes   # no prompt
"""
import argparse
import json
import shutil

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT, SPLITS, split_subdirs
from utils.labelme_io import labelme_to_yolo


def plan_split(subs, split):
    """Inspect _needs_review and return (merges, rejects) stems for one split."""
    img_dir, lbl_dir, rev_dir = (
        subs[split]["images"], subs[split]["labels"], subs[split]["needs_review"],
    )
    merges, rejects = [], []
    if rev_dir.exists():
        json_stems = {p.stem for p in rev_dir.glob("*.json")}
        jpg_stems = {p.stem for p in rev_dir.glob("*.jpg")}
        merges = sorted(json_stems)
        rejects = sorted(jpg_stems - json_stems)  # jpg kept but JSON removed = rejected
    return img_dir, lbl_dir, rev_dir, merges, rejects


def integrity_check(subs):
    """Cross-check images vs labels per split. Returns list of problem strings."""
    problems = []
    for split in SPLITS:
        img_dir, lbl_dir = subs[split]["images"], subs[split]["labels"]
        if not img_dir.exists():
            continue
        imgs = {p.stem for p in img_dir.glob("*.jpg")}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()
        missing_lbl = sorted(imgs - lbls)      # image with no label file
        orphan_lbl = sorted(lbls - imgs)       # label with no image
        empty = total_boxes = 0
        bad_lines = []
        for stem in imgs & lbls:
            txt = (lbl_dir / f"{stem}.txt").read_text().strip()
            lines = [ln for ln in txt.splitlines() if ln.strip()]
            if not lines:
                empty += 1
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    bad_lines.append(f"{split}/{stem}: '{ln}'")
                else:
                    total_boxes += 1
        print(f"  [{split}] images {len(imgs)} / labels {len(lbls)} | "
              f"박스 {total_boxes}개 | 빈라벨(배경) {empty}장")
        if missing_lbl:
            problems.append(f"{split}: 라벨 없는 이미지 {len(missing_lbl)}개 -> {missing_lbl}")
        if orphan_lbl:
            problems.append(f"{split}: 이미지 없는 라벨(고아) {len(orphan_lbl)}개 -> {orphan_lbl}")
        for b in bad_lines:
            problems.append(f"형식 오류 {b}")
    return problems


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=cfg["name"])
    ap.add_argument("--yes", action="store_true", help="skip confirmation prompt")
    args = ap.parse_args()

    dataset_dir = DATASETS_ROOT / args.name
    if not dataset_dir.exists():
        raise SystemExit(f"dataset missing: {dataset_dir}")
    subs = split_subdirs(dataset_dir)
    classes_file = dataset_dir / "classes.txt"
    class_names = classes_file.read_text().split() if classes_file.exists() else ["object"]
    class_to_id = {n: i for i, n in enumerate(class_names)}

    # 1) build & show the plan
    plans = {split: plan_split(subs, split) for split in SPLITS}
    n_merge = sum(len(p[3]) for p in plans.values())
    n_reject = sum(len(p[4]) for p in plans.values())
    print(f"[review] dataset: {dataset_dir}  classes={class_names}")
    for split in SPLITS:
        _, _, rev_dir, merges, rejects = plans[split]
        if not rev_dir.exists():
            print(f"  [{split}] _needs_review 없음 - 건너뜀")
            continue
        print(f"  [{split}] 병합 {len(merges)}개, 삭제 {len(rejects)}개")
        if merges:
            print(f"      merge : {merges}")
        if rejects:
            print(f"      DELETE: {rejects}")

    if n_merge == 0 and n_reject == 0:
        print("  적용할 검수 결과 없음 (무결성 점검만 수행)")
    elif not args.yes:
        ans = input(f"\n병합 {n_merge}개 / 삭제 {n_reject}개를 적용할까요? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            raise SystemExit("취소됨 (변경사항 없음)")

    # 2) apply
    for split in SPLITS:
        img_dir, lbl_dir, rev_dir, merges, rejects = plans[split]
        if not rev_dir.exists():
            continue
        for stem in merges:
            data = json.loads((rev_dir / f"{stem}.json").read_text())
            lines = labelme_to_yolo(data, class_to_id)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        for stem in rejects:
            (img_dir / f"{stem}.jpg").unlink(missing_ok=True)
            (lbl_dir / f"{stem}.txt").unlink(missing_ok=True)
        shutil.rmtree(rev_dir)  # remove review folder incl. leftover jpgs
        print(f"  [{split}] 적용 완료: 병합 {len(merges)}, 삭제 {len(rejects)}, _needs_review 정리됨")

    # 3) integrity check
    print("\n[verify] 전체 무결성 점검")
    problems = integrity_check(subs)
    if problems:
        print("  ⚠️ 문제 발견:")
        for p in problems:
            print(f"    - {p}")
        raise SystemExit(1)
    print("  ✅ 이상 없음 (images ↔ labels 1:1, 형식 정상)")


if __name__ == "__main__":
    main()
