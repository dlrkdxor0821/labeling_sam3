"""Send specific frames back into _needs_review/ for a manual fix (interactive).

QC (03) only stages *suspicious* frames. If you spot a wrong label in a frame
that already passed (it's only in labels/, not in _needs_review/), use this to
push that exact frame back into the review queue — image + labelme JSON of its
current box — so labelme shows the box and you can correct it. Then apply with
04_apply_review as usual.

Run with no arguments for interactive prompts:
    python scripts/send_to_review.py
        1) 데이터셋 이름?  (예: lecture_book)
        2) train / test ?
        3) frame 번호?     (예: 1,2,3  ->  frame_00001, frame_00002, frame_00003)

Or pass any of them to skip that prompt:
    python scripts/send_to_review.py --name bluebook --split train --frames 1,2,3
"""
import argparse
import json

import cv2

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT, split_subdirs
from utils.labelme_io import yolo_to_labelme
from utils.prompt import ask_existing_dir, ask_choice, confirm, _input


def to_stem(token):
    """'5' -> 'frame_00005'; 'frame_00005' (or any non-numeric) -> as-is."""
    token = token.strip()
    return f"frame_{int(token):05d}" if token.isdigit() else token


def ask_frames(preset, img_dir):
    """Parse comma/space-separated numbers into existing frame stems, validating each."""
    while True:
        raw = preset if preset is not None else _input(
            "3) 바꿀 frame 번호를 적어주세요 (예: 1,2,3)\n   > ")
        tokens = [t for t in raw.replace(",", " ").split() if t]
        stems = [to_stem(t) for t in tokens]
        existing = [s for s in stems if (img_dir / f"{s}.jpg").exists()]
        missing = [s for s in stems if not (img_dir / f"{s}.jpg").exists()]
        if missing:
            print(f"   ! 다음 프레임을 찾을 수 없습니다: {missing}")
        if existing:
            return existing
        print("   유효한 프레임이 없습니다. 다시 입력하세요.")
        preset = None


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", default=None, choices=["train", "test"])
    ap.add_argument("--frames", default=None, help="예: 1,2,3 또는 frame_00012")
    ap.add_argument("--yes", action="store_true", help="확인 프롬프트 생략")
    args = ap.parse_args()

    print("=== 검수 큐에 프레임 올리기 ===")
    name, dataset_dir = ask_existing_dir("1) 어떤 데이터셋을 바꾸고 싶습니까?", args.name, DATASETS_ROOT)
    subs = split_subdirs(dataset_dir)
    split = ask_choice("2) train / test 중 어디입니까?", args.split, ["train", "test"])
    img_dir, lbl_dir, rev_dir = (
        subs[split]["images"], subs[split]["labels"], subs[split]["needs_review"],
    )
    if not img_dir.exists():
        raise SystemExit(f"이미지 폴더 없음: {img_dir}")
    print(f"   -> datasets/{name}/{split} 확인됨 (이미지 {len(list(img_dir.glob('*.jpg')))}장)")

    stems = ask_frames(args.frames, img_dir)
    classes_file = dataset_dir / "classes.txt"
    class_names = classes_file.read_text().split() if classes_file.exists() else ["object"]

    print(f"\n다음 {len(stems)}개를 검수 큐에 올립니다: {stems}")
    if not confirm("진행할까요?", preset_yes=args.yes):
        raise SystemExit("취소됨 (변경사항 없음)")

    rev_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        ip = img_dir / f"{stem}.jpg"
        im = cv2.imread(str(ip))
        h, w = im.shape[:2]
        lp = lbl_dir / f"{stem}.txt"
        lines = lp.read_text().splitlines() if lp.exists() else []
        lj = yolo_to_labelme(lines, w, h, class_names, ip.name)
        cv2.imwrite(str(rev_dir / ip.name), im)
        (rev_dir / f"{stem}.json").write_text(json.dumps(lj, indent=2))
        print(f"  staged {stem}: 박스 {len(lj['shapes'])}개")

    print(f"\n{len(stems)}개 올림 완료. 다음 단계:")
    print(f"  python scripts/run_labelme.py {rev_dir}/        # 박스 수정 후 저장")
    print(f"  python scripts/04_apply_review.py --name {name}   # 수정 반영")


if __name__ == "__main__":
    main()
