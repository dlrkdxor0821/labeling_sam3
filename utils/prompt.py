"""Small interactive-prompt helpers shared by the pipeline scripts.

Every helper takes a `preset` (usually the CLI arg value): if it is not None the
prompt is skipped and the preset is used (so `--name x` still works headlessly);
otherwise the user is asked, with validation loops. EOF/blank cancels cleanly.
"""
from pathlib import Path


def _input(msg):
    try:
        return input(msg)
    except EOFError:
        raise SystemExit("\n취소됨 (입력 종료)")


def ask_existing_dir(label, preset, root, example="lecture_book"):
    """Prompt for a sub-directory name under `root`, validate it exists.

    Returns (name, root/name). Blank input cancels.
    """
    root = Path(root)
    while True:
        if preset is not None:
            name = str(preset).strip()
        else:
            available = sorted(p.name for p in root.glob("*") if p.is_dir()) if root.exists() else []
            hint = f" (있는 것: {', '.join(available)})" if available else " (없음)"
            name = _input(f"{label} (예: {example}){hint}\n   > ").strip()
        if not name:
            raise SystemExit("취소됨")
        d = root / name
        if d.exists():
            return name, d
        print(f"   ! '{name}' 을(를) {root}/ 에서 찾을 수 없습니다.")
        preset = None  # drop a bad preset and fall back to prompting


def ask_text(label, preset, default=None):
    if preset is not None:
        return str(preset)
    suffix = f" [{default}]" if default is not None else ""
    val = _input(f"{label}{suffix}\n   > ").strip()
    return val or (default if default is not None else "")


def ask_int(label, preset, default):
    if preset is not None:
        return int(preset)
    while True:
        val = _input(f"{label} [{default}]\n   > ").strip()
        if not val:
            return int(default)
        try:
            return int(val)
        except ValueError:
            print("   ! 정수를 입력하세요.")


def ask_float(label, preset, default):
    if preset is not None:
        return float(preset)
    while True:
        val = _input(f"{label} [{default}]\n   > ").strip()
        if not val:
            return float(default)
        try:
            return float(val)
        except ValueError:
            print("   ! 숫자를 입력하세요.")


def ask_choice(label, preset, choices, default=None):
    if preset is not None:
        return str(preset)
    opts = "/".join(choices)
    suffix = f" [{default}]" if default is not None else ""
    while True:
        val = _input(f"{label} ({opts}){suffix}\n   > ").strip()
        if not val and default is not None:
            return default
        if val in choices:
            return val
        print(f"   ! 다음 중 하나를 입력하세요: {opts}")


def confirm(label, preset_yes=False):
    if preset_yes:
        return True
    return _input(f"{label} [y/N]: ").strip().lower() in ("y", "yes")
