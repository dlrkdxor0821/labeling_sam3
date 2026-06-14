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
    ├── 04_apply_review.py           # 검수 반영(병합/삭제) + 정리 + 무결성 점검
    ├── 05_export_yolo.py
    ├── 06_train_yolo.py
    ├── 07_predict.py
    ├── run_labelme.py               # labelme 실행 런처 (Qt 충돌 우회)
    ├── visualize_labels.py          # 라벨 박스 시각화 (_viz/ 에 저장)
    ├── send_to_review.py            # 통과 프레임을 검수 큐로 재전송 (스팟 수정)
    └── merge_datasets.py            # 여러 데이터셋을 하나로 합치기 (학습용)
```

> ⚠️ 스크립트는 **프로젝트 루트에서** 실행하세요: `python scripts/01_extract_frames.py …`
> (상대경로 `datasets/`·`model/`·`config.yaml` 가 루트 기준입니다)

> ※ 04(라벨 수정)은 별도 스크립트가 아니라 **외부 툴 labelme** 를 사용합니다.

---

## 🔄 전체 워크플로우

```
👤 영상 2개 촬영        🤖 프레임     🤖 SAM3       🤖 QC선별
   (train/test)    →    추출      →   라벨링    →   (의심분만)
                                                      ↓
                                                  👤 labelme 검수
                                                 (수정 / 불필요분 삭제)
                                                      ↓
   🤖 탐지     ←   🤖 학습    ←  🤖 data.yaml  ←  🤖 검수적용
 (영상/카메라)     +결과그래프                     병합·삭제·정리·점검
```

| 단계 | 주체 | 행동 | 산출물 |
|------|:----:|------|--------|
| 0. 셋업 | 👤 | 환경 설치 + `sam3.pt` 다운로드 | 실행 환경 |
| 1. 촬영 | 👤 | 학습/시험 영상 2개 배치 | `video/<name>/{train,test}/` |
| 2. 추출 | 🤖 | 영상 → 프레임 | `images/*.jpg` |
| 3. 라벨링 | 🤖 | SAM3 자동 박스 | `labels/*.txt` |
| 4. QC | 🤖 | 의심분만 선별 | `_needs_review/` |
| 5. 검수 | 👤 | labelme로 수정 / 불필요분 삭제 | 수정된 JSON |
| 6. 적용 | 🤖 | 검수 반영(병합/삭제) + 정리 + 점검 | 정리된 `labels/` |
| 7. 확정 | 🤖 | `data.yaml` 생성 | 학습용 데이터셋 |
| 8. 학습 | 👤+🤖 | 모델·설정 선택 → 학습 | `best.pt` + `results.png` |
| 9. 탐지 | 🤖 | 새 영상/카메라에 적용 | 탐지 결과 |

---

## ⚙️ 사용법 (단계별)

> 예시: 객체 `"science book"`, 프로젝트 이름 `lecture_book`

> 💬 **모든 파이프라인 스크립트는 대화형입니다.** 인자 없이 실행하면 데이터셋 이름·옵션을
> 차례로 물어봐요(없는 이름이면 다시 입력). 아래 예시의 `--name … --fps …` 같은 인자는
> **생략 가능**하며, 주면 그 값은 묻지 않고 바로 사용합니다(자동화용).
> ```bash
> python scripts/01_extract_frames.py        # 대화형: 영상 세트·fps 를 물어봄
> python scripts/01_extract_frames.py --name lecture_book --fps 2   # 인자로 바로 지정
> ```

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
>
> 검수 시 두 가지 행동을 합니다: **틀린 박스는 수정 후 저장**(→ JSON 생성),
> **필요없는 프레임은 해당 `.json` 을 지우면**(jpg만 남김) 다음 단계에서 데이터셋에서 제거됩니다.

### 6. 검수 적용 + 정리 + 점검 🤖
```bash
python scripts/04_apply_review.py --name lecture_book
# _needs_review/ 의 검수 결과를 실제 데이터셋에 반영:
#   .json 있음     -> labels/ 에 병합(수정 반영)
#   .jpg 만 남김    -> images/ + labels/ 에서 삭제(불필요분 제거)
# 그 뒤 _needs_review/ 정리 + images↔labels 무결성 점검
# (삭제가 포함되어 적용 전 계획을 보여주고 y/N 확인을 받음, --yes 로 생략)
```

> 👀🔁 **확정 전에 라벨을 전체적으로 다시 검수하세요.** labelme 수정은 6번(검수 적용)을
> 거쳐야 비로소 `labels/` 에 반영되므로, **재검수는 반드시 적용(6번) 후**에 합니다.
>
> **① 눈으로 확인** — `visualize_labels.py` 로 박스를 그려 `_viz/` 에 뽑아 훑어보기
> ```bash
> python scripts/visualize_labels.py --name lecture_book --grid
> xdg-open datasets/lecture_book/train/_viz/
> ```
> **② 다시 QC** — 개선된 `labels/` 기준으로 의심분 재선별 (보통 더 적게 flagged)
> ```bash
> python scripts/03_qc_flag.py --name lecture_book   # _needs_review/ 재생성
> ```
> 이상이 있으면 **5번(labelme) → 6번(적용)** 을 의심분이 충분히 줄 때까지 반복하고,
> 깨끗하면 아래 7번으로 넘어갑니다.

### 7. 데이터셋 확정 🤖
```bash
python scripts/05_export_yolo.py --name lecture_book
# data.yaml 생성 (train=train, val=test)
```

### 8. 모델 선택 & 학습 👤+🤖
```bash
python scripts/06_train_yolo.py --name lecture_book
# 인터랙티브로 모델(yolo11n/s/m…)·epochs·imgsz·batch 를 차례로 질문
# → model/lecture_book/weights/best.pt + results.png
```

### 9. 탐지 / 실시간 카메라 확인 🤖
```bash
python scripts/07_predict.py --name lecture_book --source camera      # 웹캠 실시간
python scripts/07_predict.py --name lecture_book --source 새영상.mp4   # 영상 파일
```

---

## 🔍 라벨 전체 눈으로 확인하기 (보조 도구)

라벨링/검수 결과를 박스가 그려진 이미지로 한 장씩 보고 싶을 때 (아무 단계 후나 실행 가능):
```bash
python scripts/visualize_labels.py --name lecture_book          # train+test 전체
python scripts/visualize_labels.py --name lecture_book --split train
python scripts/visualize_labels.py --name lecture_book --grid   # + 한 장짜리 컨택트시트(_grid.jpg)
# → datasets/<name>/<split>/_viz/ 에 박스+클래스+신뢰도가 그려진 이미지 저장
xdg-open datasets/lecture_book/train/_viz/        # 이미지 뷰어로 폴더 열어 넘겨보기
```

---

## 🩹 이미 통과한 프레임 하나만 다시 고치기 (보조 도구)

검수를 끝내고(4·5·6단계) `_viz` 로 전체를 보다가 **이미 통과한 특정 프레임의 라벨이
틀린 걸** 발견했을 때 씁니다. QC(`03`)는 의심분만 올리므로, 통과한 프레임은 직접 검수
큐로 보내야 labelme 에서 박스째로 고칠 수 있어요. `send_to_review.py` 가 그 프레임의
이미지 + 현재 박스(JSON)를 `_needs_review/` 로 다시 올려줍니다.

```bash
python scripts/send_to_review.py
#  1) 데이터셋 이름?  (예: lecture_book)
#  2) train / test ?
#  3) frame 번호?     (예: 1,2,3  ->  frame_00001, frame_00002, frame_00003)
# 이후:
python scripts/run_labelme.py datasets/lecture_book/train/_needs_review/   # 박스 수정 후 저장
python scripts/04_apply_review.py --name lecture_book                      # 수정 반영
```
> frame 번호는 `_viz` 이미지 좌상단에 찍힌 파일명(`frame_00012`)으로 확인합니다.

---

## 🧩 여러 데이터셋 합쳐서 학습하기 (보조 도구)

`06_train_yolo` 는 데이터셋 1개만 받습니다. 여러 개(예: `bluebook` + `redbook`)를 합쳐
학습하려면 `merge_datasets.py` 로 하나로 합친 뒤 그 결과로 학습하세요. 폴더 이름 기준으로
합치며, **파일명 충돌**(둘 다 `frame_00000`)은 `<데이터셋>__` prefix 로, **클래스 ID** 는
모든 `classes.txt` 를 합쳐(union) 자동 재매핑합니다. `data.yaml` 까지 만들어줘 바로 학습 가능.

```bash
python scripts/merge_datasets.py
#  1) 합칠 데이터셋들?  (예: bluebook,redbook)
#  2) 합친 결과 이름?   (예: books_all)
# -> datasets/books_all/ 생성 (train/test 병합 + classes.txt union + data.yaml)

python scripts/06_train_yolo.py --name books_all     # 합친 데이터셋으로 학습
```
> 클래스가 다르면(book + apple) 합친 모델은 **여러 클래스를 동시에** 탐지하게 됩니다.

---

## 💡 참고

- **버전 관리**: 같은 이름으로 다시 라벨링하면 `<name>_v2`, `_v3` 자동 생성 (덮어쓰기 방지)
- **평가 전략**: 학습영상은 100% 학습에 사용, 독립 시험영상으로 평가 → 프레임 누수 없이 믿을 수 있는 mAP
- **검수 후 흐름**: labelme로 고치고(불필요분은 `.json` 삭제) → `scripts/04_apply_review.py` 가 병합·삭제·정리·점검을 한 번에 (SAM3 재실행·수동 이동 불필요)
- **라벨 확인**: `scripts/visualize_labels.py` 로 박스 그려진 이미지를 `_viz/` 에 뽑아 한 장씩 확인
- **8GB GPU**: SAM3는 fp16(`half=True`) + 한 장씩 추론, YOLO는 `n/s` 권장
