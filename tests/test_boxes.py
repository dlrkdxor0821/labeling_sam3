from utils.boxes import xyxy_to_yolo, yolo_label_lines, yolo_to_xyxy


def test_xyxy_to_yolo_center_box():
    # 640x480, box [160,120,480,360] -> center (0.5,0.5), w 0.5, h 0.5
    cx, cy, w, h = xyxy_to_yolo([160, 120, 480, 360], 640, 480)
    assert (round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)) == (0.5, 0.5, 0.5, 0.5)


def test_roundtrip_xyxy_yolo():
    yolo = (0.5, 0.4, 0.2, 0.3)
    xyxy = yolo_to_xyxy(yolo, 640, 480)
    back = xyxy_to_yolo(xyxy, 640, 480)
    assert all(abs(a - b) < 1e-9 for a, b in zip(yolo, back))


def test_label_lines_format():
    lines = yolo_label_lines([[160, 120, 480, 360]], 640, 480, class_id=0)
    assert lines == ["0 0.500000 0.500000 0.500000 0.500000"]
