"""Video (video/<name>/{train,test}/) -> frames (datasets/<name>/{train,test}/images/)."""
import argparse
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT, SPLITS, resolve_versioned_dir, split_subdirs
from utils.frames import extract_video_frames
from utils.prompt import ask_existing_dir, ask_float

VIDEO_ROOT = Path("video")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()

    name, _ = ask_existing_dir("어떤 영상 세트를 추출할까요?", args.name, VIDEO_ROOT)
    fps = ask_float("초당 추출 프레임 수(fps)", args.fps, cfg["extract"]["fps"])

    dataset_dir = resolve_versioned_dir(DATASETS_ROOT, name)
    subs = split_subdirs(dataset_dir)
    print(f"[extract] output dataset: {dataset_dir}")

    for split in SPLITS:
        src = VIDEO_ROOT / name / split
        videos = [p for p in sorted(src.glob("*")) if p.suffix.lower() in VIDEO_EXTS] if src.exists() else []
        if not videos:
            print(f"  ! {split}: no videos at {src} - skipped")
            continue
        total = 0
        for v in videos:
            total += extract_video_frames(
                v, subs[split]["images"], fps,
                dedup=cfg["extract"]["dedup"],
                dedup_threshold=cfg["extract"]["dedup_threshold"],
            )
        print(f"  done {split}: {len(videos)} video(s) -> {total} frames")


if __name__ == "__main__":
    main()
