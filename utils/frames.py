from pathlib import Path

import cv2
import numpy as np


def frame_indices(total_frames: int, video_fps: float, target_fps: float) -> list:
    """Indices to sample so output ~= target_fps. target>=source -> keep all."""
    if total_frames <= 0:
        return []
    if target_fps <= 0 or target_fps >= video_fps:
        return list(range(total_frames))
    step = video_fps / target_fps
    count = int(total_frames / step)
    return [int(round(i * step)) for i in range(count)]


def is_duplicate(prev, curr, threshold: float) -> bool:
    """Mean absolute pixel diff below threshold -> near-duplicate."""
    if prev is None:
        return False
    diff = np.abs(curr.astype("int16") - prev.astype("int16")).mean()
    return bool(diff < threshold)


def extract_video_frames(video_path, out_dir, target_fps,
                         dedup=True, dedup_threshold=3.0, prefix="frame") -> int:
    """Sample (and dedup) frames from a video into out_dir. Returns count saved."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keep = set(frame_indices(total, video_fps, target_fps))
    saved, prev, i = 0, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in keep and not (dedup and is_duplicate(prev, frame, dedup_threshold)):
            cv2.imwrite(str(out_dir / f"{prefix}_{saved:05d}.jpg"), frame)
            prev, saved = frame, saved + 1
        i += 1
    cap.release()
    return saved
