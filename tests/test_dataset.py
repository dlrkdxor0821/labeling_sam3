from utils.dataset import make_data_yaml


def test_make_data_yaml(tmp_path):
    out = make_data_yaml(tmp_path / "ds", ["book"])
    text = out.read_text()
    assert "train:" in text and "train/images" in text
    assert "val:" in text and "test/images" in text
    assert "0: book" in text
