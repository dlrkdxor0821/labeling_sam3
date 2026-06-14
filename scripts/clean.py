"""Empty the datasets/ , model/ , and/or video/ folders (interactive, safe).

Removes the *contents* of the chosen folders (the folders themselves stay).
Shows a size preview and asks for confirmation before deleting anything.

Safety: when emptying `model/`, top-level weight files (`sam3.pt`, `*.pt`) are
KEPT by default — re-downloading the gated SAM3 weight is painful. Pass
--include-weights (or answer the prompt) to wipe those too.

    python scripts/clean.py
        1) 무엇을 비울까요?  (datasets/model/video/all, 쉼표로 여러 개)
"""
import argparse
import shutil

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.prompt import confirm, _input

TARGETS = {
    "datasets": pathlib.Path("datasets"),
    "model": pathlib.Path("model"),
    "video": pathlib.Path("video"),
}


def dir_size(p):
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def ask_targets(preset):
    valid = set(TARGETS) | {"all"}
    while True:
        raw = preset if preset is not None else _input(
            "1) 무엇을 비울까요? (datasets/model/video/all, 쉼표로 여러 개)\n   > ")
        picked = [t for t in raw.replace(",", " ").split() if t]
        if "all" in picked:
            return list(TARGETS)
        bad = [t for t in picked if t not in valid]
        if bad:
            print(f"   ! 알 수 없는 항목: {bad} (가능: datasets/model/video/all)")
        elif picked:
            return picked
        else:
            print("   유효한 항목을 입력하세요.")
        preset = None


def entries_to_delete(name, root, keep_weights):
    """List child paths to remove. For model/, optionally keep top-level *.pt."""
    children = []
    for c in sorted(root.glob("*")):
        if name == "model" and keep_weights and c.is_file() and c.suffix == ".pt":
            continue  # preserve base weights (sam3.pt 등)
        children.append(c)
    return children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=None, help="예: datasets,video 또는 all")
    ap.add_argument("--include-weights", dest="include_weights", action="store_true",
                    help="model/ 비울 때 sam3.pt 등 base 가중치도 삭제")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    print("=== 폴더 비우기 ===")
    names = ask_targets(args.targets)
    keep_weights = not args.include_weights

    plan = {}
    for name in names:
        root = TARGETS[name]
        if not root.exists():
            print(f"  [{name}] 폴더 없음 - 건너뜀")
            continue
        children = entries_to_delete(name, root, keep_weights)
        if not children:
            print(f"  [{name}] 비울 것 없음")
            continue
        size = sum(dir_size(c) if c.is_dir() else c.stat().st_size for c in children)
        plan[name] = children
        kept = "  (sam3.pt 등 가중치는 보존)" if name == "model" and keep_weights else ""
        print(f"  [{name}] {len(children)}개 항목, {human(size)} 삭제 예정{kept}")

    if not plan:
        raise SystemExit("삭제할 것이 없습니다.")
    if not confirm("\n정말 삭제할까요? (되돌릴 수 없음)", preset_yes=args.yes):
        raise SystemExit("취소됨 (변경사항 없음)")

    for name, children in plan.items():
        for c in children:
            if c.is_dir():
                shutil.rmtree(c)
            else:
                c.unlink()
        print(f"  [{name}] 비움 완료")


if __name__ == "__main__":
    main()
