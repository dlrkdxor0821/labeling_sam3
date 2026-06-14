"""Upload a trained model or a dataset to the Hugging Face Hub (interactive).

Requires a one-time login first:  hf auth login   (Read/Write token)

    python scripts/hf_upload.py
        1) model / dataset ?
        2) 올릴 이름?        (model/<name> 또는 datasets/<name>)
        3) HF username?      (사용자/조직 이름)
        4) repo 이름?
        5) private?          [y/N]
    -> repo 는 <username>/<repo-name> 으로 만들어집니다.
"""
import argparse

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.paths import DATASETS_ROOT
from utils.prompt import ask_existing_dir, ask_choice, ask_text, confirm

MODEL_ROOT = pathlib.Path("model")
# preview / scratch artifacts that should never be uploaded
IGNORE = ["**/_viz/**", "**/_needs_review/**", "**/__pycache__/**", "**/*.pyc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None, choices=["model", "dataset"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--username", default=None, help="HF 사용자/조직 이름")
    ap.add_argument("--repo-name", dest="repo_name", default=None, help="repo 이름")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("huggingface_hub 필요: pip install -U huggingface_hub") from e

    api = HfApi()
    try:
        user = api.whoami().get("name")
    except Exception:
        raise SystemExit("HF 로그인이 필요합니다: hf auth login (Write 토큰)")
    print(f"=== Hugging Face 업로드 (로그인: {user}) ===")

    kind = ask_choice("1) 무엇을 올릴까요?", args.kind, ["model", "dataset"], "model")
    root = MODEL_ROOT if kind == "model" else DATASETS_ROOT
    label = "올릴 모델 이름?" if kind == "model" else "올릴 데이터셋 이름?"
    name, folder = ask_existing_dir(f"2) {label}", args.name, root)

    username = ask_text("3) HF username (사용자/조직)?", args.username, user)
    repo_name = ask_text("4) repo 이름?", args.repo_name, name)
    repo = f"{username}/{repo_name}"
    repo_type = "model" if kind == "model" else "dataset"
    private = args.private or confirm("5) private 로 만들까요?")

    print(f"\n  {kind}: {folder}  ->  https://huggingface.co/{'datasets/' if repo_type=='dataset' else ''}{repo}"
          f"  ({'private' if private else 'public'})")
    if not confirm("진행할까요?", preset_yes=args.yes):
        raise SystemExit("취소됨")

    api.create_repo(repo_id=repo, repo_type=repo_type, private=private, exist_ok=True)
    print("  업로드 중... (용량에 따라 시간이 걸립니다)")
    api.upload_folder(
        folder_path=str(folder), repo_id=repo, repo_type=repo_type, ignore_patterns=IGNORE,
    )
    url = f"https://huggingface.co/{'datasets/' if repo_type=='dataset' else ''}{repo}"
    print(f"\n완료 ✅ -> {url}")


if __name__ == "__main__":
    main()
