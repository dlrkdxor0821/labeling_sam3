from utils.labelme_io import labelme_to_yolo, yolo_to_labelme


def test_yolo_to_labelme_rectangle():
    d = yolo_to_labelme(["0 0.5 0.5 0.5 0.5"], 640, 480, ["book"], "frame.jpg")
    assert d["imageWidth"] == 640 and d["imageHeight"] == 480
    assert len(d["shapes"]) == 1
    s = d["shapes"][0]
    assert s["label"] == "book"
    assert s["shape_type"] == "rectangle"
    assert s["points"] == [[160.0, 120.0], [480.0, 360.0]]


def test_labelme_to_yolo_roundtrip():
    d = yolo_to_labelme(["0 0.500000 0.500000 0.500000 0.500000"], 640, 480, ["book"], "f.jpg")
    lines = labelme_to_yolo(d, {"book": 0})
    assert lines == ["0 0.500000 0.500000 0.500000 0.500000"]
