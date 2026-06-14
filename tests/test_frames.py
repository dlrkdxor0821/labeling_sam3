import cv2
import numpy as np

from utils.frames import extract_video_frames, frame_indices, is_duplicate


def test_frame_indices_keeps_all_when_target_ge_video():
    assert frame_indices(10, 30, 60) == list(range(10))


def test_frame_indices_downsamples_30fps_to_2fps():
    assert frame_indices(30, 30, 2) == [0, 15]


def test_frame_indices_empty_when_no_frames():
    assert frame_indices(0, 30, 2) == []


def test_is_duplicate_true_for_identical():
    a = np.zeros((4, 4, 3), dtype="uint8")
    assert is_duplicate(a, a.copy(), threshold=1.0) is True


def test_is_duplicate_false_for_different():
    a = np.zeros((4, 4, 3), dtype="uint8")
    b = np.full((4, 4, 3), 100, dtype="uint8")
    assert is_duplicate(a, b, threshold=1.0) is False


def test_is_duplicate_false_when_no_prev():
    a = np.zeros((4, 4, 3), dtype="uint8")
    assert is_duplicate(None, a, threshold=1.0) is False


def test_extract_video_frames_writes_images(tmp_path):
    vid = tmp_path / "clip.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(vid), fourcc, 30.0, (32, 32))
    for k in range(30):
        frame = np.full((32, 32, 3), k * 8, dtype="uint8")
        writer.write(frame)
    writer.release()

    out = tmp_path / "images"
    n = extract_video_frames(vid, out, target_fps=2, dedup=False)
    assert n == 2
    assert len(list(out.glob("*.jpg"))) == 2
