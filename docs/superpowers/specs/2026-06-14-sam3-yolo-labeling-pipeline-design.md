# SAM3 자동 라벨링 → YOLO 학습 파이프라인 — 설계 문서 (Design Spec)

- **작성일**: 2026-06-14
- **프로젝트 경로**: `/home/asd/laveling_sam3`
- **상태**: 설계 확정, 구현 계획 대기

---

## 1. 개요 & 목표

사용자가 **텍스트 프롬프트(concept)** 로 지정한 객체를, 직접 촬영한 영상으로부터 **SAM3로 자동 라벨링**하고, 품질이 의심되는 라벨만 골라 **labelme로 직접 수정**한 뒤, **선택한 YOLO 모델을 학습**시키는 **로컬 파이프라인**.

**핵심 가치**
1. 수작업 라벨링 최소화 (SAM3가 1차 라벨 생성)
2. "이상한 라벨만" 자동 선별 → 사람은 그것만 검수
3. 믿을 수 있는 평가 (독립 시험영상)
4. YOLO 모델 선택·학습·저장·**결과 리포트 자동 생성**

**비목표 (YAGNI)**: 실시간 영상 추적, 클라우드/웹 UI, 다중 사용자, SAM3D, 멀티-GPU 분산학습.

---

## 2. 환경 & 제약

| 항목 | 값 |
|------|-----|
| OS / Python | Ubuntu 24.04.4 LTS / Python 3.12 |
| GPU | NVIDIA RTX 4060, **VRAM 8GB** |
| SAM3 | Ultralytics ≥ 8.3.237, `SAM3SemanticPredictor`, 전용 CLIP 패키지 |
| SAM3 가중치 | `sam3.pt` (3.45GB, **HF gated → 수동 다운로드**) |
| 추론 정밀도 | fp16 (`half=True`) |
| 라벨 수정 툴 | **labelme** (오픈소스, JSON 출력 → SAM3↔YOLO 변환기 필요) |

**8GB 제약 대응**: SAM3는 한 장씩 추론 + fp16 + 필요시 입력 리사이즈/CPU 폴백. 라벨링은 offline 배치라 속도는 비핵심. YOLO 학습은 `yolo11n/s` 기준 8GB에서 충분.

**labelme**: 오픈소스 데스크톱 라벨링 툴(`pip install labelme`). 폴리곤/사각형 지원, 유지보수 활발(Python 3.12 호환 양호). **JSON으로 저장**하므로 변환기 2개 필요:
- (1) **SAM3 결과 → labelme JSON** : 검수 시 박스가 미리 그려진 채로 열리게 (직접 작성)
- (2) **labelme JSON → YOLO `.txt`** : 학습용 변환 (`labelme2yolo` 등 활용; 폴리곤은 자동으로 박스 변환)

---

## 3. 아키텍처

### 3.1 폴더 구조

```
laveling_sam3/
├── video/                              # 원본 영상 (사용자가 직접 넣음)
│   └── <name>/                         # 예: lecture_book
│       ├── train/  → 학습용영상.mp4     # 객체의 모든 모습
│       └── test/   → 시험용영상.mp4     # 다른 세션에 짧게 (독립 평가용)
│
├── datasets/                           # 파이프라인 산출물
│   └── <name>/                         # 같은 이름, 충돌 시 <name>_v2, _v3…
│       ├── train/
│       │   ├── images/                 # ① 프레임 추출
│       │   ├── labels/                 # ② SAM3 라벨 (YOLO .txt)
│       │   └── _needs_review/          # ③ 의심 프레임 (이미지+labelme JSON) — labelme로 수정
│       ├── test/
│       │   ├── images/
│       │   ├── labels/
│       │   └── _needs_review/
│       ├── classes.txt
│       └── data.yaml                   # ⑤ YOLO 학습 설정
│
├── model/                              # 모델 저장
│   ├── sam3.pt                         # SAM3 베이스 (HF 수동 다운로드)
│   └── <name>/                         # 학습 결과 (이름별)
│       ├── weights/
│       │   ├── best.pt                 #   ⭐ 최종 모델
│       │   └── last.pt
│       └── results.png · confusion_matrix.png · PR_curve.png · results.csv   # 자동 생성 리포트
│
├── config.yaml                         # 기본 파라미터
├── utils/                              # 공용: 박스변환, IO, labelme JSON 변환, 버전네이밍
├── 01_extract_frames.py
├── 02_label_sam3.py
├── 03_qc_flag.py
├── 05_export_yolo.py
├── 06_train_yolo.py
├── 07_predict.py                       # (옵션) 학습된 모델로 추론
├── setup.sh / requirements.txt
└── README.md
```

> ※ 04(라벨 수정)은 별도 스크립트가 아니라 **외부 툴 labelme** 사용. 03이 검수 대상을 (SAM3 결과 → labelme JSON 변환하여) `_needs_review/` 에 staging 하고, 05가 수정된 labelme JSON을 YOLO로 변환·병합한다.

### 3.2 데이터 흐름

```
사용자: video/<name>/{train,test}/ 에 영상 2개 넣기
  └─[01]→ datasets/<name>/{train,test}/images/        (프레임 추출)
       └─[02]→ datasets/<name>/{train,test}/labels/   (SAM3 텍스트 라벨, YOLO .txt)
            └─[03]→ _needs_review/ (이미지 + labelme JSON)   (의심 라벨 선별)
                 └─ 사용자: labelme로 _needs_review/ 수정 (박스 미리보기)
                      └─[05]→ data.yaml (+ JSON→YOLO 변환·병합)   (train=train, val=test)
                           └─[06]→ model/<name>/  (best.pt + results.png 등)  (선택 모델 학습)
                                └─[07]→ 새 영상/이미지 탐지 (옵션)
```

---

## 4. 설정 (Config)

`config.yaml` + CLI 인자 오버라이드. 주요 키:

```yaml
name: lecture_book          # 프로젝트/데이터셋 이름
prompt: "book"              # SAM3 라벨링 대상 (= 클래스명)
extract:
  fps: 2                    # 초당 추출 프레임 수
  dedup: true              # 거의 동일한 프레임 스킵
sam3:
  model: model/sam3.pt
  conf: 0.25
  half: true               # fp16
qc:
  conf_threshold: 0.40     # 이 미만이면 의심
  topk_percent: null       # 상위 N%만 검수 (null이면 임계값 방식)
train:
  yolo_model: yolo11s       # 선택 가능 (n/s/m…)
  epochs: 100
  imgsz: 640
  batch: 16
```

---

## 5. 컴포넌트 상세

### 01_extract_frames.py — 영상 → 프레임
- **입력**: `--name <name>` (→ `video/<name>/{train,test}/` 의 영상 자동 탐색)
- **동작**: 각 영상에서 `fps` 간격으로 프레임 추출, `dedup` 시 인접 유사 프레임(프레임 차분/해시) 스킵
- **출력**: `datasets/<name>/{train,test}/images/*.jpg`
- **엣지**: 영상 없음 → 안내 / 프레임 0 → 경고

### 02_label_sam3.py — SAM3 자동 라벨링
- **입력**: `--name <name> --prompt "book"` (또는 `--images <경로>`)
- **동작**: `SAM3SemanticPredictor(half=True)` 로드 → 각 이미지에 `text=[prompt]` 추론 → 마스크/박스 획득 → YOLO 포맷(`class cx cy w h`, 0~1 정규화) `.txt` 저장 + `classes.txt` 생성
- **출력**: `datasets/<name>/{train,test}/labels/*.txt`, `classes.txt`
- **버전관리**: `datasets/<name>/` 이미 존재 시 `<name>_v2`, `_v3`… 자동 증가 (덮어쓰기 금지)
- **엣지**: `sam3.pt` 없음/HF 권한 없음 → 다운로드 안내 / OOM → fp16·리사이즈·CPU 폴백 안내 / 0개 검출 프레임 → 빈 라벨 + 자동 검수 대상 표시

### 03_qc_flag.py — 의심 라벨 자동 선별 ★
- **목적**: 전체 박스를 서로 비교하여 "잘 된 것" vs "잘 안 된 것" 분리
- **의심 신호 (합산하여 점수화)**:
  1. **누락/과검출** — 시간상 이웃 프레임 대비 박스 개수 급변 (예: 이웃 1개인데 0/2개)
  2. **낮은 신뢰도** — SAM3 conf < `conf_threshold`
  3. **시간적 튐(temporal jump)** — 박스 중심/크기가 앞뒤 프레임 대비 급격히 점프
  4. **기하 이상치** — 박스 넓이·종횡비가 전체 분포에서 크게 벗어남 (median ± k·MAD)
- **출력**: 의심 점수 초과 프레임의 **이미지 + labelme JSON**(SAM3 박스를 미리 그려둠)을 `_needs_review/` 로 staging
- **구현 순서**: 1·2번(누락·신뢰도)부터, 3·4번(temporal·기하)은 점진 추가
- **엣지**: 검수 대상 0개 → "전부 통과" 안내 / 전부 의심 → 임계값 재조정 제안

### 04 — labelme 연동 (수동 검수)
- 사용자가 `labelme datasets/<name>/<split>/_needs_review/` 실행
- SAM3 박스가 **labelme JSON으로 미리 그려진 채로** 열림 → 틀린 것만 수정/삭제/추가 → 저장
- 03이 `SAM3 결과 → labelme JSON` 변환 담당, 05가 `labelme JSON → YOLO` 변환·병합
- (labelme는 유지보수되어 Python 3.12 venv에서 패치 없이 동작 예상)

### 05_export_yolo.py — 데이터셋 확정 & data.yaml
- `_needs_review/` 의 수정된 **labelme JSON → YOLO `.txt`** 변환 후 `labels/` 에 **자동 병합** (SAM3 재실행·수동 파일이동 불필요 — 명령 1개로 처리)
- `data.yaml` 생성:
  ```yaml
  train: datasets/<name>/train/images
  val:   datasets/<name>/test/images      # 독립 시험영상 = 정직한 평가셋
  names: { 0: book }
  ```
- 라벨 무결성 검증 (형식, 좌표 범위 0~1, 이미지↔라벨 매칭)

### 06_train_yolo.py — 모델 선택 & 학습 (완전 인터랙티브)
- 학습 시작 전 **모든 설정을 하나씩 질문** (`config.yaml` 값을 기본으로 제시, 엔터 시 기본값):
  1. 모델 선택 — 사용 가능 목록(`yolo11n/s/m/l/x`) 표시 후 선택
  2. epochs · imgsz · batch — 각각 차례로 질문
- **train/test "비율"은 묻지 않음** — 전략 1(별도 시험영상)이라 폴더로 이미 분리됨
- 베이스 가중치 미설치 시 자동 다운로드 안내/수행
- `YOLO(<선택모델>).train(data=data.yaml, epochs, imgsz, batch, project="model/<name>")`
- **출력 (자동 생성 리포트 포함)**:
  - `model/<name>/weights/best.pt`, `last.pt`
  - **`results.png`** — train/val 의 box·cls·dfl loss + precision·recall·**mAP50·mAP50-95** (10 패널)
  - `confusion_matrix.png`, `PR_curve.png`, `results.csv`
  - 학습 종료 후 결과 그래프 경로 안내(또는 자동 열기)
- **엣지**: OOM → batch/imgsz 축소 제안 / 데이터 부족 → 경고

### 07_predict.py — 추론 / 실시간 카메라 확인
- `model/<name>/weights/best.pt` 로 탐지. `--source` 로 입력 선택:
  - `--source 영상.mp4` (영상 파일) · `--source image.jpg` (이미지)
  - **`--source camera` (웹캠 실시간)** — 카메라 창을 띄우고 탐지 박스+라벨+신뢰도를 **실시간 표시**하여 모델 동작을 눈으로 확인
- 구현: Ultralytics `model.predict(source=0, show=True, stream=True)` 기반 (옵션: FPS 표시·스크린샷)

---

## 6. 데이터 분할 & 평가 전략

- **전략 1 채택 — 별도 시험영상**: 학습영상(`train/`)은 100% 학습, 독립 촬영한 시험영상(`test/`)으로 평가.
- `data.yaml` 에서 `train=train/images`, `val=test/images` 매핑.
- **근거**: 영상 프레임은 인접 프레임이 거의 동일 → 한 영상을 무작위 분할하면 train/val 누수로 점수 뻥튀기. 독립 영상은 누수 0 + 학습 데이터 안 버림.
- (더 엄격히 하려면 train 내부에서 작은 val을 추가 분리 가능 — 현재는 2폴더 단순 구조 채택)

---

## 7. 에러 처리 & 엣지 케이스 (요약)

| 상황 | 처리 |
|------|------|
| `sam3.pt` 없음 / HF 권한 없음 | HF 접근신청 + 다운로드 경로 안내 |
| GPU OOM (SAM3/YOLO) | fp16·리사이즈·batch↓·CPU 폴백 안내 |
| 0개 검출 프레임 | 빈 라벨 + 자동 검수 대상 |
| 폴더명 충돌 | `_v2`, `_v3` 자동 증가 |
| 영상/프레임/데이터 부족 | 경고 후 중단 |
| labelme Qt 플러그인 오류 | `libxcb-cursor0` 등 설치 안내 |
| labelme JSON↔YOLO 변환 실패 | 클래스명·좌표 검증 후 명확한 에러 |

---

## 8. 테스트 전략

- **단위 테스트(순수 함수)**: 마스크→YOLO박스 변환, **SAM3→labelme JSON 변환**, **labelme JSON→YOLO 변환**, 의심점수 계산, 버전 네이밍(`_v2`), `_needs_review` 병합
- **스모크 테스트**: 각 스크립트를 소수 프레임으로 1회 실행 (합성 이미지 가능)
- **데이터셋 무결성**: 라벨 형식·좌표 범위·이미지↔라벨 매칭·train↔val 누수 없음 확인
- ML 학습 자체는 단위테스트 대상 아님 (대신 데이터/설정 검증으로 대체)

---

## 9. 가정 & 미해결

- 사용자가 **train/test 영상을 각각 촬영**한다 (전략 1).
- 기본은 **단일 클래스**(프롬프트 1개). 다중 프롬프트→다중 클래스 확장 가능하게 구조만 열어둠.
- SAM3가 fine-grained 구분("과학책" vs 일반책)은 약할 수 있으나, **촬영 맥락상 프레임 내 대상이 그 객체뿐**이므로 coarse 프롬프트("book")로 충분.
- **구현 직전, `SAM3SemanticPredictor`/YOLO 학습 API/`labelme2yolo` 사용법을 context7·실측으로 재확인**하여 코드에 반영.

---

## 10. Acceptance Criteria (검증 체크리스트)

- [ ] `video/<name>/{train,test}/` 영상 → 프레임 추출, train/test 구분 유지
- [ ] SAM3가 `--prompt` 객체를 라벨링하여 YOLO `.txt` + `classes.txt` 생성
- [ ] 같은 이름 재실행 시 `<name>_v2` 폴더 생성 (덮어쓰기 없음)
- [ ] QC가 의심 프레임을 `_needs_review/` 로 분리 (이미지 + labelme JSON)
- [ ] labelme가 `_needs_review/` 를 박스 미리보기와 함께 열고, 수정본이 YOLO로 변환·병합됨
- [ ] `data.yaml` 생성: `train=train/images`, `val=test/images`
- [ ] 사용 가능 YOLO 모델 목록 표시 → 선택 → 학습 → `model/<name>/weights/best.pt` 저장
- [ ] 학습 후 **`results.png`(loss·precision·recall·mAP 그래프)** 등 리포트가 `model/<name>/` 에 자동 생성
- [ ] **웹캠 실시간 모드(`--source camera`)** 에서 카메라 창에 탐지 박스가 실시간 표시됨
- [ ] 수동 검수 후 `05_export_yolo.py` 실행 시 수정본이 **자동 병합**(수동 이동 불필요)
- [ ] RTX 4060(8GB)에서 SAM3 라벨링(fp16) 및 YOLO 학습이 OOM 없이 동작
- [ ] labelme가 Python 3.12 venv에서 정상 실행
