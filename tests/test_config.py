from utils.config import load_config


def test_missing_file_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg["extract"]["fps"] == 2
    assert cfg["train"]["yolo_model"] == "yolo11s"


def test_user_values_override_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("extract:\n  fps: 5\ntrain:\n  epochs: 50\n")
    cfg = load_config(p)
    assert cfg["extract"]["fps"] == 5
    assert cfg["extract"]["dedup"] is True
    assert cfg["train"]["epochs"] == 50
    assert cfg["train"]["yolo_model"] == "yolo11s"
