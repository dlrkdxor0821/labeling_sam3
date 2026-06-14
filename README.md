# labeling_sam3

> 직접 촬영한 영상에서 **SAM3로 객체를 자동 라벨링** → 의심되는 것만 **labelme로 검수** → **YOLO 모델 학습**까지 한 번에 처리하는 로컬 파이프라인.

🚧 **상태: 구현 예정** — 아래 명령어는 *설계 기준*이며, 스크립트는 곧 작성됩니다.
📄 설계 문서: [`docs/superpowers/specs/2026-06-14-sam3-yolo-labeling-pipeline-design.md`](docs/superpowers/specs/2026-06-14-sam3-yolo-labeling-pipeline-design.md)

---

## ✨ 특징

- 🤖 **텍스트 프롬프트 기반 자동 라벨링** — `"book"` 처럼 단어만 주면 SAM3가 프레임마다 박스 생성
- 🎯 **스마트 QC** — 이상한 라벨만 자동 선별, 사람은 그것만 labelme로 수정
- 📐 **믿을 수 있는 평가** — 별도 시험영상으로 정직한 mAP
- 🧩 **모델 선택** — `yolo11n/s/m…` 인터랙티브 선택 후 학습
- 📊 **결과 그래프 자동 생성** — `results.png` (loss·precision·recall·mAP)
- 📹 **실시간 카메라 확인** — 웹캠으로 모델 동작 즉시 검증

## 📦 환경

- Ubuntu 24.04 · Python 3.12 · NVIDIA RTX 4060 (VRAM 8GB)
- **SAM3 가중치** `sam3.pt` 는 [Hugging Face `facebook/sam3`](https://huggingface.co/facebook/sam3)에서 접근 승인 후 수동 다운로드 → `model/sam3.pt` 에 배치 (자세한 절차: [0. 준비](#0-준비-최초-1회))

---

## 📁 폴더 구조

```
laveling_sam3/
├── video/                          # 원본 영상 (사용자가 직접 넣음)
│   └── <name>/                     # 예: lecture_book
│       ├── train/  ← 학습용 영상.mp4   (객체의 모든 모습)
│       └── test/   ← 시험용 영상.mp4   (다른 세션에 짧게, 독립 평가용)
│
├── datasets/                       # 파이프라인 산출물
│   └── <name>/                     # 같은 이름 충돌 시 <name>_v2, _v3…
│       ├── train/
│       │   ├── images/             # ① 프레임 추출
│       │   ├── labels/             # ② SAM3 라벨 (YOLO .txt)
│       │   └── _needs_review/      # ③ 의심분 → labelme로 수정
│       ├── test/                   #   (train과 동일 구조)
│       ├── classes.txt
│       └── data.yaml               # ⑤ YOLO 학습 설정
│
├── model/                          # 모델 저장
│   ├── sam3.pt                     # SAM3 베이스 (HF 수동 다운로드)
│   └── <name>/
│       ├── weights/best.pt         # ⭐ 학습된 탐지 모델
│       └── results.png …           # 📊 학습 결과 그래프 (자동 생성)
│
├── config.yaml                     # 기본 파라미터
├── utils/                          # 공용 모듈 (스크립트가 import)
└── scripts/                        # 파이프라인 스크립트 (루트에서 실행)
    ├── 01_extract_frames.py
    ├── 02_label_sam3.py
    ├── 03_qc_flag.py
    ├── 05_export_yolo.py
    ├── 06_train_yolo.py
    ├── 07_predict.py
    └── run_labelme.py               # labelme 실행 런처 (Qt 충돌 우회)
```

> ⚠️ 스크립트는 **프로젝트 루트에서** 실행하세요: `python scripts/01_extract_frames.py …`
> (상대경로 `datasets/`·`model/`·`config.yaml` 가 루트 기준입니다)

> ※ 04(라벨 수정)은 별도 스크립트가 아니라 **외부 툴 labelme** 를 사용합니다.

---

## 🔄 전체 워크플로우

```
👤 영상 2개 촬영        🤖 ①프레임   🤖 ②SAM3      🤖 ③QC선별
   (train/test)    →    추출      →   라벨링    →   (의심분만)
                                                      ↓
   🤖 ⑥YOLO학습   ←  🤖 ⑤data.yaml  ←  👤 ④labelme  ←──┘
   +결과그래프         +YOLO변환·병합     의심분 수정
        ↓
   🤖 ⑦탐지 (영상/이미지/실시간 카메라)
```

| 단계 | 주체 | 행동 | 산출물 |
|------|:----:|------|--------|
| 0. 셋업 | 👤 | 환경 설치 + `sam3.pt` 다운로드 | 실행 환경 |
| 1. 촬영 | 👤 | 학습/시험 영상 2개 배치 | `video/<name>/{train,test}/` |
| 2. 추출 | 🤖 | 영상 → 프레임 | `images/*.jpg` |
| 3. 라벨링 | 🤖 | SAM3 자동 박스 | `labels/*.txt` |
| 4. QC | 🤖 | 의심분만 선별 | `_needs_review/` |
| 5. 검수 | 👤 | labelme로 수정 | 수정된 JSON |
| 6. 확정 | 🤖 | JSON→YOLO + `data.yaml` (자동 병합) | 학습용 데이터셋 |
| 7. 학습 | 👤+🤖 | 모델·설정 선택 → 학습 | `best.pt` + `results.png` |
| 8. 탐지 | 🤖 | 새 영상/카메라에 적용 | 탐지 결과 |

---

## ⚙️ 사용법 (단계별)

> 예시: 객체 `"science book"`, 프로젝트 이름 `lecture_book`

### 0. 준비 (최초 1회)

이 저장소에는 이미 `.venv` 가상환경이 구성되어 있고 의존성도 설치되어 있습니다.
**매 작업 시작 시 가상환경만 활성화하면 됩니다.**

```bash
source .venv/bin/activate        # 가상환경 활성화 (이후 모든 스크립트는 여기서 실행)
```

> 처음부터 새로 구성하거나 패키지가 빠졌을 때만:
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> # GPU(RTX 4060)는 ultralytics 전에 CUDA torch 먼저:
> #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> # SAM3용 Ultralytics CLIP 포크:
> #   pip install git+https://github.com/ultralytics/CLIP.git
> ```

**설치 확인** — 아래가 `cuda? True` 면 GPU 학습 준비 완료:
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
# → torch 2.6.0+cu124 cuda? True
```

#### SAM3 가중치 받기 (`model/sam3.pt`)

SAM3는 Meta 라이선스 게이트 모델이라 **자동 다운로드가 안 됩니다.** 접근 승인 후 직접 받아야 해요.
(`scripts/02_label_sam3.py` 실행 전 필수)

**1) Hugging Face 접근 권한 신청**
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) → 약관 동의 → **Request access** 클릭 → Meta 승인 대기 (수 분~수 시간)

**2) HF 토큰 발급 & 로그인** (승인되면 인증 필요)
```bash
source .venv/bin/activate
pip install -U huggingface_hub
hf auth login        # huggingface.co/settings/tokens 에서 Read 토큰 발급 후 붙여넣기
```

**3) 가중치 다운로드 → `model/sam3.pt`**
```bash
hf download facebook/sam3 sam3.pt --local-dir model/
ls -la model/sam3.pt          # 파일 존재 확인 (수 GB)
```
> `config.yaml`의 `sam3.model: model/sam3.pt` 경로와 일치합니다.

**4) CLIP 포크** (텍스트 프롬프트용 — 보통 이미 설치됨)
```bash
pip install git+https://github.com/ultralytics/CLIP.git
```

> ⚠️ 비공식 미러(`1038lab/sam3` 등) 말고 **공식 `facebook/sam3`** 를 쓰세요.
> 라벨링 시 `bpe_simple_vocab` 관련 에러가 나면 BPE vocab도 별도로 받아야 합니다.

### 1. 영상 배치 👤
- 학습용 영상 → `video/lecture_book/train/`  (책을 여러 각도·거리·조명으로)
- 시험용 영상 → `video/lecture_book/test/`   (다른 세션에 짧게)

### 2. 프레임 추출 🤖
```bash
python scripts/01_extract_frames.py --name lecture_book --fps 2
```

### 3. SAM3 자동 라벨링 🤖
```bash
python scripts/02_label_sam3.py --name lecture_book --prompt "book"
# 같은 이름이 있으면 lecture_book_v2 로 자동 생성
```

### 4. QC 자동 선별 🤖
```bash
python scripts/03_qc_flag.py --name lecture_book
# → 의심 프레임만 _needs_review/ 로 모음 (예: "300장 중 27장 검수 필요")
```

### 5. 수동 검수 (labelme) 👤
```bash
python scripts/run_labelme.py datasets/lecture_book/train/_needs_review/
python scripts/run_labelme.py datasets/lecture_book/test/_needs_review/
# SAM3 박스가 미리 그려진 채로 열림 → 틀린 것만 수정 후 저장
```

> ⚠️ **`labelme` 를 직접 실행하지 말고 `scripts/run_labelme.py` 런처를 쓰세요.**
> `opencv-python` 이 import 시 Qt 플러그인 경로를 가로채서, 그냥 `labelme` 로 켜면
> `Could not load the Qt platform plugin "xcb"` 에러로 죽습니다. 런처는 cv2를 먼저
> 로드시킨 뒤 Qt 경로를 PyQt5의 정상 플러그인으로 되돌려 이 충돌을 피합니다.
> (인자는 `labelme` 와 동일하게 받습니다)

### 6. 데이터셋 확정 🤖
```bash
python scripts/05_export_yolo.py --name lecture_book
# 수정본을 자동 병합(JSON→YOLO) + data.yaml 생성 (train=train, val=test)
```

### 7. 모델 선택 & 학습 👤+🤖
```bash
python scripts/06_train_yolo.py --name lecture_book
# 인터랙티브로 모델(yolo11n/s/m…)·epochs·imgsz·batch 를 차례로 질문
# → model/lecture_book/weights/best.pt + results.png
```

### 8. 탐지 / 실시간 카메라 확인 🤖
```bash
python scripts/07_predict.py --name lecture_book --source camera      # 웹캠 실시간
python scripts/07_predict.py --name lecture_book --source 새영상.mp4   # 영상 파일
```

---

## 💡 참고

- **버전 관리**: 같은 이름으로 다시 라벨링하면 `<name>_v2`, `_v3` 자동 생성 (덮어쓰기 방지)
- **평가 전략**: 학습영상은 100% 학습에 사용, 독립 시험영상으로 평가 → 프레임 누수 없이 믿을 수 있는 mAP
- **수정 후 흐름**: labelme로 고친 뒤 `scripts/05_export_yolo.py` 실행만 하면 **자동 병합** (SAM3 재실행·수동 이동 불필요)
- **8GB GPU**: SAM3는 fp16(`half=True`) + 한 장씩 추론, YOLO는 `n/s` 권장
