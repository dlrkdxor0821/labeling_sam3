"""Build the YOLO data.yaml (train = train images, val = independent test video)."""
from pathlib import Path


def make_data_yaml(dataset_dir, class_names):
    """Write data.yaml mapping train->train/images, val->test/images. Returns path."""
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    content = (
        f"train: {dataset_dir.resolve()}/train/images\n"
        f"val: {dataset_dir.resolve()}/test/images\n"
        f"names:\n{names}\n"
    )
    out = dataset_dir / "data.yaml"
    out.write_text(content)
    return out
