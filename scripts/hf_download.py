"""Download a model or dataset from the Hugging Face Hub into this project.

Counterpart of hf_upload.py. Pulls a whole repo into model/<name>/ (model) or
datasets/<name>/ (dataset). Public repos need no login; private/gated repos
need `hf auth login` first.

    python scripts/hf_download.py
        1) model / dataset ?
        2) HF username?      (사용자/조직)
        3) repo 이름?
        4) 로컬 저장 이름?    (model/<name> 또는 datasets/<name>, 기본 = repo 이름)
"""
import argparse

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.paths import DATASETS_ROOT, resolve_versioned_dir
from utils.prompt import ask_choice, ask_text, confirm, _input

MODEL_ROOT = pathlib.Path("model")


def ask_required(label, preset, default=None):
    """ask_text but never returns blank (re-prompts). `default` is offered, not auto-used."""
    while True:
        val = ask_text(label, preset, default).strip()
        if val:
            return val
        print("   ! 값을 입력하세요.")
        preset = None
        default = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None, choices=["model", "dataset"])
    ap.add_argument("--username", default=None, help="HF 사용자/조직 이름")
    ap.add_argument("--repo-name", dest="repo_name", default=None, help="repo 이름")
    ap.add_argument("--name", default=None, help="로컬 저장 이름")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError as e:
        raise SystemExit("huggingface_hub 필요: pip install -U huggingface_hub") from e

    user = None
    try:
        user = HfApi().whoami().get("name")
    except Exception:
        pass  # not logged in — fine for public repos

    print("=== Hugging Face 다운로드 ===" + (f" (로그인: {user})" if user else " (비로그인)"))
    kind = ask_choice("1) 무엇을 받을까요?", args.kind, ["model", "dataset"], "model")
    username = ask_required("2) HF username (사용자/조직)?", args.username, default=user)
    repo_name = ask_required("3) repo 이름?", args.repo_name)
    repo = f"{username}/{repo_name}"
    repo_type = "model" if kind == "model" else "dataset"
    root = MODEL_ROOT if kind == "model" else DATASETS_ROOT

    local_name = ask_text("4) 로컬 저장 이름?", args.name, repo_name)
    target = resolve_versioned_dir(root, local_name)

    url = f"https://huggingface.co/{'datasets/' if repo_type=='dataset' else ''}{repo}"
    print(f"\n  {url}  ->  {target}")
    if not confirm("받을까요?", preset_yes=args.yes):
        raise SystemExit("취소됨")

    target.mkdir(parents=True, exist_ok=True)
    print("  다운로드 중... (용량에 따라 시간이 걸립니다)")
    snapshot_download(repo_id=repo, repo_type=repo_type, local_dir=str(target))
    print(f"\n완료 ✅ -> {target}")

    if repo_type == "model":
        print(f"  추론:  python scripts/07_predict.py --name {target.name}")
    else:
        print(f"  ⚠️ data.yaml 경로는 업로더 기준이라, 학습 전 재생성하세요:")
        print(f"     python scripts/05_export_yolo.py --name {target.name}")
        print(f"     python scripts/06_train_yolo.py  --name {target.name}")


if __name__ == "__main__":
    main()
