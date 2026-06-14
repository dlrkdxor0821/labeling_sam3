"""Export: merge labelme fixes back into YOLO labels, then write data.yaml."""
import argparse
import json

from utils.config import load_config
from utils.paths import DATASETS_ROOT, SPLITS, split_subdirs
from utils.labelme_io import labelme_to_yolo
from utils.dataset import make_data_yaml


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=cfg["name"])
    args = ap.parse_args()

    dataset_dir = DATASETS_ROOT / args.name
    subs = split_subdirs(dataset_dir)
    classes_file = dataset_dir / "classes.txt"
    class_names = classes_file.read_text().split() if classes_file.exists() else ["object"]
    class_to_id = {n: i for i, n in enumerate(class_names)}

    for split in SPLITS:
        lbl_dir, rev_dir = subs[split]["labels"], subs[split]["needs_review"]
        if not rev_dir.exists():
            continue
        merged = 0
        for js in sorted(rev_dir.glob("*.json")):
            data = json.loads(js.read_text())
            lines = labelme_to_yolo(data, class_to_id)
            (lbl_dir / f"{js.stem}.txt").write_text("\n".join(lines))
            merged += 1
        print(f"  {split}: merged {merged} corrected label(s)")

    out = make_data_yaml(dataset_dir, class_names)
    print(f"  data.yaml -> {out}")


if __name__ == "__main__":
    main()
