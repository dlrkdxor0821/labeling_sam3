# Phase 1: Foundation & Frame Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 프로젝트 뼈대를 세우고, 테스트로 검증된 "영상 → 프레임" 추출 단계(버전관리 포함)를 완성한다.

**Architecture:** 순수 헬퍼 함수는 `utils/`(config 로딩, 버전 폴더 해석, 프레임 샘플링/중복제거)에 두고 단위테스트로 덮는다. `01_extract_frames.py`는 이를 OpenCV 영상 읽기와 연결하는 얇은 CLI로, 프레임을 `datasets/<name>/{train,test}/images/`에 저장한다.

**Tech Stack:** Python 3.12, OpenCV(opencv-python), NumPy, PyYAML, pytest.

**참고 spec:** `docs/superpowers/specs/2026-06-14-sam3-yolo-labeling-pipeline-design.md` (§3 폴더구조, §4 config, §5.01 추출, §2 버전관리)

---

## File Structure (Phase 1)

```
laveling_sam3/
├── requirements.txt          # Create — opencv/numpy/pyyaml/pytest
├── .gitignore                # Create — datasets·model·video·venv 제외
├── config.yaml               # Create — 기본 파라미터
├── utils/
│   ├── __init__.py           # Create — 빈 패키지
│   ├── config.py             # Create — load_config()
│   ├── paths.py              # Create — 버전 폴더/스플릿 경로
│   └── frames.py             # Create — frame_indices/is_duplicate/extract_video_frames
├── 01_extract_frames.py      # Create — 얇은 CLI
└── tests/
    ├── __init__.py           # Create
    ├── test_paths.py         # Create
    ├── test_config.py        # Create
    └── test_frames.py        # Create
```

> 순수 로직을 `utils/frames.py`에 두는 이유: 파일명이 숫자로 시작(`01_...`)하면 `import` 불가 → 테스트하려면 로직을 import 가능한 모듈에 둬야 함.

---

## Task 1: 프로젝트 스캐폴딩

**Files:**
- Create: `requirements.txt`, `.gitignore`, `config.yaml`, `utils/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: `requirements.txt` 작성**

```
# Phase 1 (frame extraction)
opencv-python>=4.9
numpy>=1.26
pyyaml>=6.0
pytest>=8.0
# (ML 의존성 ultralytics·labelme 는 Phase 2+ 에서 추가)
```

- [ ] **Step 2: `.gitignore` 작성**

```
__pycache__/
*.pyc
.pytest_cache/
venv/
.venv/
# 대용량 산출물·가중치는 추적 제외
datasets/
model/*.pt
model/**/weights/
video/
runs/
```

- [ ] **Step 3: `config.yaml` 작성** (spec §4 기준)

```yaml
name: lecture_book          # 프로젝트/데이터셋 이름
prompt: "book"              # SAM3 라벨링 대상 (= 클래스명)
extract:
  fps: 2                    # 초당 추출 프레임 수
  dedup: true               # 거의 동일한 프레임 스킵
  dedup_threshold: 3.0      # 평균 픽셀차가 이 값 미만이면 중복
sam3:
  model: model/sam3.pt
  conf: 0.25
  half: true
qc:
  conf_threshold: 0.40
  topk_percent: null
train:
  yolo_model: yolo11s
  epochs: 100
  imgsz: 640
  batch: 16
```

- [ ] **Step 4: 빈 패키지 파일 생성**

`utils/__init__.py` 와 `tests/__init__.py` 를 빈 파일로 생성.

- [ ] **Step 5: 의존성 설치 + pytest 동작 확인**

Run: `pip install -r requirements.txt && pytest -q`
Expected: `no tests ran` (수집된 테스트 0개, 에러 없음)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore config.yaml utils/__init__.py tests/__init__.py
git commit -m "build: scaffold project (requirements, gitignore, config, packages)"
```

---

## Task 2: `utils/paths.py` — 버전 폴더 & 스플릿 경로 (TDD)

**Files:**
- Create: `utils/paths.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: 실패하는 테스트 작성** → `tests/test_paths.py`

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.paths'`

- [ ] **Step 3: 최소 구현** → `utils/paths.py`

```python
from pathlib import Path

DATASETS_ROOT = Path("datasets")
SPLITS = ("train", "test")


def resolve_versioned_dir(base_dir, name: str) -> Path:
    """base_dir/name, 이미 있으면 name_v2, _v3 … (덮어쓰기 방지)."""
    base_dir = Path(base_dir)
    candidate = base_dir / name
    if not candidate.exists():
        return candidate
    version = 2
    while (base_dir / f"{name}_v{version}").exists():
        version += 1
    return base_dir / f"{name}_v{version}"


def split_subdirs(dataset_dir) -> dict:
    """각 split(train/test)의 images/labels/_needs_review 경로."""
    dataset_dir = Path(dataset_dir)
    return {
        split: {
            "images": dataset_dir / split / "images",
            "labels": dataset_dir / split / "labels",
            "needs_review": dataset_dir / split / "_needs_review",
        }
        for split in SPLITS
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_paths.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add utils/paths.py tests/test_paths.py
git commit -m "feat: add versioned dataset path helpers"
```

---

## Task 3: `utils/config.py` — 기본값 병합 로딩 (TDD)

**Files:**
- Create: `utils/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 실패하는 테스트 작성** → `tests/test_config.py`

```python
from utils.config import load_config


def test_missing_file_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg["extract"]["fps"] == 2
    assert cfg["train"]["yolo_model"] == "yolo11s"


def test_user_values_override_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("extract:\n  fps: 5\ntrain:\n  epochs: 50\n")
    cfg = load_config(p)
    assert cfg["extract"]["fps"] == 5          # 덮어씀
    assert cfg["extract"]["dedup"] is True     # 기본값 유지
    assert cfg["train"]["epochs"] == 50        # 덮어씀
    assert cfg["train"]["yolo_model"] == "yolo11s"  # 기본값 유지
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.config'`

- [ ] **Step 3: 최소 구현** → `utils/config.py`

```python
import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "extract": {"fps": 2, "dedup": True, "dedup_threshold": 3.0},
    "sam3": {"model": "model/sam3.pt", "conf": 0.25, "half": True},
    "qc": {"conf_threshold": 0.40, "topk_percent": None},
    "train": {"yolo_model": "yolo11s", "epochs": 100, "imgsz": 640, "batch": 16},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path="config.yaml") -> dict:
    """config.yaml 을 DEFAULTS 위에 깊은 병합. 파일 없으면 기본값."""
    path = Path(path)
    user = {}
    if path.exists():
        user = yaml.safe_load(path.read_text()) or {}
    return _deep_merge(DEFAULTS, user)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add utils/config.py tests/test_config.py
git commit -m "feat: add config loader with default merge"
```

---

## Task 4: `utils/frames.py` + `01_extract_frames.py` — 프레임 추출 (TDD)

**Files:**
- Create: `utils/frames.py`, `01_extract_frames.py`
- Test: `tests/test_frames.py`

- [ ] **Step 1: 순수 함수 실패 테스트 작성** → `tests/test_frames.py`

```python
import numpy as np

from utils.frames import frame_indices, is_duplicate


def test_frame_indices_keeps_all_when_target_ge_video():
    assert frame_indices(10, 30, 60) == list(range(10))


def test_frame_indices_downsamples_30fps_to_2fps():
    # step = 30/2 = 15 → 0, 15
    assert frame_indices(30, 30, 2) == [0, 15]


def test_frame_indices_empty_when_no_frames():
    assert frame_indices(0, 30, 2) == []


def test_is_duplicate_true_for_identical():
    a = np.zeros((4, 4, 3), dtype="uint8")
    assert is_duplicate(a, a.copy(), threshold=1.0) is True


def test_is_duplicate_false_for_different():
    a = np.zeros((4, 4, 3), dtype="uint8")
    b = np.full((4, 4, 3), 100, dtype="uint8")
    assert is_duplicate(a, b, threshold=1.0) is False


def test_is_duplicate_false_when_no_prev():
    a = np.zeros((4, 4, 3), dtype="uint8")
    assert is_duplicate(None, a, threshold=1.0) is False
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_frames.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.frames'`

- [ ] **Step 3: 순수 함수 + 추출 함수 구현** → `utils/frames.py`

```python
from pathlib import Path

import cv2
import numpy as np


def frame_indices(total_frames: int, video_fps: float, target_fps: float) -> list:
    """출력이 ≈ target_fps 가 되도록 샘플링할 프레임 인덱스. target>=원본이면 전부."""
    if total_frames <= 0:
        return []
    if target_fps <= 0 or target_fps >= video_fps:
        return list(range(total_frames))
    step = video_fps / target_fps
    count = int(total_frames / step)
    return [int(round(i * step)) for i in range(count)]


def is_duplicate(prev, curr, threshold: float) -> bool:
    """직전 프레임과 평균 픽셀차가 threshold 미만이면 중복으로 간주."""
    if prev is None:
        return False
    diff = np.abs(curr.astype("int16") - prev.astype("int16")).mean()
    return bool(diff < threshold)


def extract_video_frames(video_path, out_dir, target_fps,
                         dedup=True, dedup_threshold=3.0, prefix="frame") -> int:
    """영상에서 프레임을 샘플링·(중복제거)하여 out_dir 에 저장. 저장한 장수 반환."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keep = set(frame_indices(total, video_fps, target_fps))
    saved, prev, i = 0, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in keep and not (dedup and is_duplicate(prev, frame, dedup_threshold)):
            cv2.imwrite(str(out_dir / f"{prefix}_{saved:05d}.jpg"), frame)
            prev, saved = frame, saved + 1
        i += 1
    cap.release()
    return saved
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_frames.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: 합성 영상 통합 테스트 추가** → `tests/test_frames.py` 끝에 append

```python
def test_extract_video_frames_writes_images(tmp_path):
    import cv2
    # 30프레임 합성 영상(.avi, MJPG) 생성
    vid = tmp_path / "clip.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(vid), fourcc, 30.0, (32, 32))
    for k in range(30):
        frame = np.full((32, 32, 3), k * 8, dtype="uint8")  # 매 프레임 다른 밝기
        writer.write(frame)
    writer.release()

    from utils.frames import extract_video_frames
    out = tmp_path / "images"
    n = extract_video_frames(vid, out, target_fps=2, dedup=False)
    assert n == 2                                   # 30fps→2fps → 2장
    assert len(list(out.glob("*.jpg"))) == 2
```

- [ ] **Step 6: 통합 테스트 통과 확인**

Run: `pytest tests/test_frames.py -v`
Expected: PASS (7 passed)
> 만약 환경에 MJPG 코덱이 없어 영상 생성이 실패하면, 이 테스트만 `@pytest.mark.skipif`로 감싸고 수동 스모크(Step 8)로 대체.

- [ ] **Step 7: CLI 작성** → `01_extract_frames.py`

```python
"""영상(video/<name>/{train,test}/) → 프레임(datasets/<name>/{train,test}/images/)."""
import argparse
from pathlib import Path

from utils.config import load_config
from utils.paths import DATASETS_ROOT, SPLITS, resolve_versioned_dir, split_subdirs
from utils.frames import extract_video_frames

VIDEO_ROOT = Path("video")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=cfg["name"])
    ap.add_argument("--fps", type=float, default=cfg["extract"]["fps"])
    args = ap.parse_args()

    dataset_dir = resolve_versioned_dir(DATASETS_ROOT, args.name)
    subs = split_subdirs(dataset_dir)
    print(f"[extract] 출력 데이터셋: {dataset_dir}")

    for split in SPLITS:
        src = VIDEO_ROOT / args.name / split
        videos = [p for p in sorted(src.glob("*")) if p.suffix.lower() in VIDEO_EXTS] if src.exists() else []
        if not videos:
            print(f"  ! {split}: 영상 없음 ({src}) — 건너뜀")
            continue
        total = 0
        for v in videos:
            total += extract_video_frames(
                v, subs[split]["images"], args.fps,
                dedup=cfg["extract"]["dedup"],
                dedup_threshold=cfg["extract"]["dedup_threshold"],
            )
        print(f"  ✓ {split}: {len(videos)}개 영상 → {total}장 추출")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: 수동 스모크 (선택)**

`video/lecture_book/train/` 에 짧은 영상을 넣고:
Run: `python 01_extract_frames.py --name lecture_book --fps 2`
Expected: `datasets/lecture_book/train/images/` 에 `frame_00000.jpg …` 생성, 콘솔에 추출 장수 출력

- [ ] **Step 9: Commit**

```bash
git add utils/frames.py 01_extract_frames.py tests/test_frames.py
git commit -m "feat: add video frame extraction with sampling and dedup"
```

---

## Phase 1 완료 기준 (Acceptance)

- [ ] `pytest -q` 전부 통과 (paths 4 + config 2 + frames 6~7)
- [ ] `python 01_extract_frames.py --name <n>` 가 train/test 영상을 `datasets/<n>/{train,test}/images/` 로 추출
- [ ] 같은 이름 재실행 시 `<n>_v2` 데이터셋 생성 (덮어쓰기 없음)
- [ ] `config.yaml` 값이 CLI 기본값으로 반영됨

> **다음 단계 (Phase 2):** SAM3 자동 라벨링. 시작 전 `SAM3SemanticPredictor` 실제 API 를 context7·실측으로 검증 후 계획 작성.
