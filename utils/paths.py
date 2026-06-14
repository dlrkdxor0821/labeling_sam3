from pathlib import Path

DATASETS_ROOT = Path("datasets")
SPLITS = ("train", "test")


def resolve_versioned_dir(base_dir, name: str) -> Path:
    """base_dir/name, already exists -> name_v2, _v3 ... (no overwrite)."""
    base_dir = Path(base_dir)
    candidate = base_dir / name
    if not candidate.exists():
        return candidate
    version = 2
    while (base_dir / f"{name}_v{version}").exists():
        version += 1
    return base_dir / f"{name}_v{version}"


def split_subdirs(dataset_dir) -> dict:
    """For each split (train/test): images/labels/_needs_review paths."""
    dataset_dir = Path(dataset_dir)
    return {
        split: {
            "images": dataset_dir / split / "images",
            "labels": dataset_dir / split / "labels",
            "needs_review": dataset_dir / split / "_needs_review",
        }
        for split in SPLITS
    }
