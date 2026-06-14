"""Pure QC scoring: flag suspicious frames from per-frame YOLO boxes."""
import math
from pathlib import Path
from statistics import median


def read_yolo_boxes(txt_path):
    """Read a YOLO .txt -> list of (cx, cy, w, h)."""
    boxes = []
    p = Path(txt_path)
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append(tuple(float(x) for x in parts[1:]))
    return boxes


def _primary(boxes):
    """Largest box by area, or None."""
    return max(boxes, key=lambda b: b[2] * b[3]) if boxes else None


def frame_suspicion(frames, conf_threshold=0.40, jump_thresh=0.15):
    """frames: ordered list of dict(stem, boxes=[(cx,cy,w,h)], conf=float|None).

    Returns list of dict(stem, score, reasons) — higher score = more suspicious.
    """
    counts = [len(f["boxes"]) for f in frames]
    typical = median(counts) if counts else 0
    areas = [b[2] * b[3] for f in frames for b in f["boxes"]]
    area_med = median(areas) if areas else 0
    results = []
    for i, f in enumerate(frames):
        score, reasons = 0.0, []
        n = len(f["boxes"])
        if n == 0:
            score += 2.0
            reasons.append("empty")
        elif typical and abs(n - typical) >= 1:
            score += 1.0
            reasons.append("count")
        if f.get("conf") is not None and f["conf"] < conf_threshold:
            score += 1.0
            reasons.append("low_conf")
        prim = _primary(f["boxes"])
        if prim is not None:
            neigh = [
                p
                for j in (i - 1, i + 1)
                if 0 <= j < len(frames)
                for p in [_primary(frames[j]["boxes"])]
                if p
            ]
            if neigh and min(math.hypot(prim[0] - p[0], prim[1] - p[1]) for p in neigh) > jump_thresh:
                score += 1.0
                reasons.append("jump")
            if area_med:
                a = prim[2] * prim[3]
                if a > area_med * 3 or a < area_med / 3:
                    score += 1.0
                    reasons.append("geom")
        results.append({"stem": f["stem"], "score": score, "reasons": reasons})
    return results


def flagged_stems(scores, threshold=1.0):
    """Stems whose suspicion score >= threshold (the review queue)."""
    return [s["stem"] for s in scores if s["score"] >= threshold]
