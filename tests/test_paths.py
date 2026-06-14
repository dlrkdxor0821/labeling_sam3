from utils.paths import resolve_versioned_dir, split_subdirs


def test_resolve_returns_plain_name_when_absent(tmp_path):
    assert resolve_versioned_dir(tmp_path, "book") == tmp_path / "book"


def test_resolve_bumps_to_v2_when_name_exists(tmp_path):
    (tmp_path / "book").mkdir()
    assert resolve_versioned_dir(tmp_path, "book") == tmp_path / "book_v2"


def test_resolve_bumps_to_v3_when_v2_exists(tmp_path):
    (tmp_path / "book").mkdir()
    (tmp_path / "book_v2").mkdir()
    assert resolve_versioned_dir(tmp_path, "book") == tmp_path / "book_v3"


def test_split_subdirs_structure(tmp_path):
    subs = split_subdirs(tmp_path / "book")
    assert subs["train"]["images"] == tmp_path / "book" / "train" / "images"
    assert subs["test"]["needs_review"] == tmp_path / "book" / "test" / "_needs_review"
