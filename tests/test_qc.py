from utils.qc import flagged_stems, frame_suspicion


def _f(stem, boxes, conf=None):
    return {"stem": stem, "boxes": boxes, "conf": conf}


def test_empty_frame_is_flagged():
    frames = [_f("a", [(0.5, 0.5, 0.2, 0.2)]), _f("b", []), _f("c", [(0.5, 0.5, 0.2, 0.2)])]
    flagged = flagged_stems(frame_suspicion(frames))
    assert "b" in flagged
    assert "a" not in flagged


def test_temporal_jump_flagged():
    frames = [
        _f("a", [(0.1, 0.1, 0.2, 0.2)]),
        _f("b", [(0.9, 0.9, 0.2, 0.2)]),
        _f("c", [(0.1, 0.1, 0.2, 0.2)]),
    ]
    by = {s["stem"]: s for s in frame_suspicion(frames)}
    assert "jump" in by["b"]["reasons"]


def test_geometric_outlier_flagged():
    frames = [_f(str(i), [(0.5, 0.5, 0.2, 0.2)]) for i in range(5)]
    frames.append(_f("big", [(0.5, 0.5, 0.9, 0.9)]))
    by = {s["stem"]: s for s in frame_suspicion(frames)}
    assert "geom" in by["big"]["reasons"]


def test_low_conf_flagged():
    frames = [
        _f("a", [(0.5, 0.5, 0.2, 0.2)], conf=0.9),
        _f("b", [(0.5, 0.5, 0.2, 0.2)], conf=0.1),
    ]
    by = {s["stem"]: s for s in frame_suspicion(frames, conf_threshold=0.4)}
    assert "low_conf" in by["b"]["reasons"]
    assert "low_conf" not in by["a"]["reasons"]
