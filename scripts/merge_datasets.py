"""Merge several datasets into one (for training on combined data).

Combines datasets/<a>, datasets/<b>, ... into a single datasets/<out> with the
usual train/ + test/ layout, so 06_train_yolo can train on all of them at once.

Two things are handled automatically:
  - filename collisions: every frame is copied as `<dataset>__<stem>.jpg` (and
    its `.txt`), so identical frame names across datasets don't overwrite.
  - class IDs: classes.txt of all inputs are unioned (first-seen order) and each
    label's class id is remapped to the combined id (e.g. redbook's 0=apple ->
    combined 1 when bluebook already took 0=book).

Interactive (or pass --names / --out to skip prompts):
    python scripts/merge_datasets.py
        1) 합칠 데이터셋들?  (예: bluebook,redbook)
        2) 합친 결과 이름?   (예: books_all)
"""
import argparse
import shutil

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.config import load_config
from utils.paths import DATASETS_ROOT, SPLITS, resolve_versioned_dir, split_subdirs
from utils.dataset import make_data_yaml
from utils.prompt import ask_text, confirm, _input


def read_classes(dataset_dir):
    f = dataset_dir / "classes.txt"
    return f.read_text().split() if f.exists() else ["object"]


def ask_datasets(preset):
    """Prompt for a comma/space list of existing dataset names; validate all exist."""
    while True:
        if preset is not None:
            raw = preset
        else:
            available = sorted(p.name for p in DATASETS_ROOT.glob("*") if p.is_dir())
            hint = f" (있는 것: {', '.join(available)})" if available else " (없음)"
            raw = _input(f"1) 합칠 데이터셋들을 쉼표로 적어주세요 (예: bluebook,redbook){hint}\n   > ")
        names = [t for t in raw.replace(",", " ").split() if t]
        missing = [n for n in names if not (DATASETS_ROOT / n).exists()]
        if missing:
            print(f"   ! 찾을 수 없는 데이터셋: {missing}")
        elif len(names) < 2:
            print("   ! 2개 이상 입력하세요.")
        else:
            return names
        preset = None


def remap_label(text, remap):
    """Rewrite each YOLO line's class id via `remap` (dict local_id -> new_id)."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5:
            parts[0] = str(remap.get(int(parts[0]), int(parts[0])))
            out.append(" ".join(parts))
    return "\n".join(out)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=None, help="예: bluebook,redbook")
    ap.add_argument("--out", default=None, help="합친 결과 데이터셋 이름")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    print("=== 데이터셋 합치기 ===")
    names = ask_datasets(args.names)
    out_name = ask_text("2) 합친 결과 이름?", args.out, "merged")
    if out_name in names:
        raise SystemExit(f"출력 이름 '{out_name}' 이 입력 데이터셋과 겹칩니다. 다른 이름을 쓰세요.")

    # union of classes (first-seen order) + per-dataset id remap
    combined, index = [], {}
    for n in names:
        for c in read_classes(DATASETS_ROOT / n):
            if c not in index:
                index[c] = len(combined)
                combined.append(c)
    print(f"  입력: {names}")
    print(f"  합쳐진 클래스: {combined}")

    out_dir = resolve_versioned_dir(DATASETS_ROOT, out_name)
    print(f"  출력: {out_dir}")
    if not confirm("진행할까요?", preset_yes=args.yes):
        raise SystemExit("취소됨")

    out_subs = split_subdirs(out_dir)
    totals = {s: 0 for s in SPLITS}
    for n in names:
        ds = DATASETS_ROOT / n
        subs = split_subdirs(ds)
        remap = {i: index[c] for i, c in enumerate(read_classes(ds))}
        for split in SPLITS:
            img_dir, lbl_dir = subs[split]["images"], subs[split]["labels"]
            if not img_dir.exists():
                continue
            o_img, o_lbl = out_subs[split]["images"], out_subs[split]["labels"]
            o_img.mkdir(parents=True, exist_ok=True)
            o_lbl.mkdir(parents=True, exist_ok=True)
            for ip in sorted(img_dir.glob("*.jpg")):
                stem = f"{n}__{ip.stem}"
                shutil.copy(ip, o_img / f"{stem}.jpg")
                lp = lbl_dir / f"{ip.stem}.txt"
                text = lp.read_text() if lp.exists() else ""
                (o_lbl / f"{stem}.txt").write_text(remap_label(text, remap))
                totals[split] += 1
            print(f"  [{n}/{split}] {len(list(img_dir.glob('*.jpg')))}장 복사")

    (out_dir / "classes.txt").write_text("\n".join(combined) + "\n")
    yaml = make_data_yaml(out_dir, combined)
    print(f"\n완료 -> {out_dir}  (train {totals['train']}장 / test {totals['test']}장)")
    print(f"  classes.txt + {yaml}")
    print(f"  학습: python scripts/06_train_yolo.py --name {out_dir.name}")


if __name__ == "__main__":
    main()
