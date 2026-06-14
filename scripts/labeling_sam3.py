"""Interactive controller for the whole labeling_sam3 pipeline.

Run from the project root:
    python scripts/labeling_sam3.py

Pick a number and it launches that step (each step is itself interactive and
prompts for its own dataset/options). Returns to this menu when the step ends.
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

# (key, label, script, needs_path?)  needs_path = ask for a folder to pass as arg
# a row with key=None is a section header (label holds the heading text)
MENU = [
    (None, "── 파이프라인 ──", None, False),
    ("1", "프레임 추출",            "01_extract_frames.py", False),
    ("2", "SAM3 라벨링",            "02_label_sam3.py",     False),
    ("3", "QC 의심분 선별",          "03_qc_flag.py",        False),
    ("4", "검수 적용/정리/점검",      "04_apply_review.py",   False),
    ("5", "데이터셋 확정(data.yaml)", "05_export_yolo.py",    False),
    ("6", "모델 학습",              "06_train_yolo.py",     False),
    ("7", "탐지/추론",              "07_predict.py",        False),
    (None, "── 보조 도구 ──", None, False),
    ("8", "라벨 검수 (labelme)",     "run_labelme.py",       True),
    ("9", "라벨 시각화",            "visualize_labels.py",  False),
    ("10", "통과 프레임 재검수",      "send_to_review.py",    False),
    ("11", "데이터셋 합치기",        "merge_datasets.py",    False),
    ("12", "Hugging Face 업로드",    "hf_upload.py",         False),
    ("13", "Hugging Face 다운로드",  "hf_download.py",       False),
]
ACTIONS = {key: (script, needs_path) for key, _, script, needs_path in MENU if key}


def show_menu():
    print("\n" + "=" * 40)
    print("  labeling_sam3 파이프라인")
    print("=" * 40)
    for key, label, script, _ in MENU:
        if key is None:
            print(f"\n  {label}")
        else:
            print(f"   {key:>2}) {label}")
    print("\n    q) 종료")


def run(script, needs_path):
    cmd = [sys.executable, str(SCRIPTS / script)]
    if needs_path:
        path = input("   열 폴더 경로 (예: datasets/bluebook/train/_needs_review/): ").strip()
        if not path:
            print("   경로가 없어 취소했습니다.")
            return
        cmd.append(path)
    try:
        # run from project root so relative paths (datasets/, model/) resolve
        subprocess.run(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\n   (중단됨 - 메뉴로 돌아갑니다)")


def main():
    while True:
        show_menu()
        try:
            choice = input("\n번호 선택> ").strip().lower()
        except EOFError:
            break
        if choice in ("q", "quit", "exit"):
            break
        if choice not in ACTIONS:
            print("   ! 메뉴에 있는 번호를 입력하세요.")
            continue
        script, needs_path = ACTIONS[choice]
        run(script, needs_path)
    print("종료합니다.")


if __name__ == "__main__":
    main()
