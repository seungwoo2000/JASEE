# 🪑 자세히봐 (Fit Me Up)

> **RULA + VDT 고시 제2020-17호** 기반의 AI 실시간 자세 & 작업환경 측정 서비스

---

## 📌 프로젝트 개요

사무직 근로자의 근골격계 질환 예방을 목적으로,  
웹캠을 통해 **실시간으로 자세와 작업환경을 분석**하고 개선 피드백을 제공하는 AI 서비스입니다.

---

## 🗂️ 프로젝트 구조

```
📁 JASEE/
├── 📁 Yolo_env/                        ← 작업환경 인식 모델 (2번 작업)
│   └── 📁 images_data/
│       ├── 📁 new_data_set/            ← 원본 데이터 (new)
│       ├── 📁 old_data_set/            ← 원본 데이터 (old)
│       ├── 📁 dataset/                 ← split 결과 (train/val/test)
│       ├── 📁 augmented/               ← 증강 결과
│       └── 📁 runs/                    ← 학습 결과 (best.pt)
│
├── 📁 Yolo_pose/                       ← 자세 분류 모델 (1번 작업)
│   ├── 📁 01_data_preprocessing/
│   ├── 📁 02_model_comparison/
│   ├── 📁 03_model(attetion MLP)_improve/
│   ├── 📁 04_final_model/
│   │   └── 📁 output/
│   │       └── final_attention_mlp.pt  ← 최종 자세 분류 모델
│   └── 📁 05_documentation/
│
├── 📄 jasee_core.py                    ← 핵심 로직 (모델 로드, 각도 계산, 판정)
├── 📄 app_desktop.py                   ← 데스크탑용 Streamlit 앱
├── 📄 app_mobile.py                    ← 모바일용 Streamlit 앱
├── 📄 step1_split.py                   ← 데이터 분할 스크립트
├── 📄 step2_augmentation.py            ← 데이터 증강 스크립트
├── 📄 step3_train.py                   ← YOLOv8 학습 스크립트
├── 📄 logo_transparent.png             ← 앱 로고
├── 📄 users.json                       ← 사용자 계정 (gitignore)
├── 📄 user_history.json                ← 측정 이력 (gitignore)
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🧠 모델 구조

### 1번 작업 — 자세 분류 (Yolo_pose)
| 단계 | 내용 |
|------|------|
| 입력 | 웹캠 이미지 |
| 키포인트 추출 | YOLOv8n-pose (17개 키포인트) |
| 각도 계산 | CVA, TIA (RULA + VDT 기준서 3.1절) |
| 분류 | Attention MLP → GOOD / BAD |

### 2번 작업 — 작업환경 인식 (Yolo_env)
| 단계 | 내용 |
|------|------|
| 학습 데이터 | 708장 (new + old) → 증강 후 약 1,698장 |
| 분할 비율 | train 80% / val 10% / test 10% |
| 클래스 | chair_back, chair_seat, desk_surface, monitor |
| 모델 | YOLOv8n (Detection) |

---

## 📐 측정 기준 (RULA + VDT 고시 제2020-17호)

| 지표 | 정상 범위 | 기준 |
|------|----------|------|
| CVA (목굴곡각) | 0° ~ 20° | RULA Neck Zone |
| TIA (몸통굴곡각) | 0° ~ 20° | RULA Trunk Zone |
| 무릎 각도 | 85° ~ 100° | VDT 고시 |
| 모니터 시선각 | 하방 10° ~ 15° | VDT 고시 제6조 |
| 작업대 높이 | 팔꿈치 기준 ±10% | VDT 고시 |
| 의자 등받이 | 골반너비 20% 이내 | VDT 고시 |

---

## ⚙️ 설치 및 실행

### 1. 가상환경 생성
```bash
conda create -n Yolo_env python=3.10
conda activate Yolo_env
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 실행
```bash
cd E:\python\Jasee

# 데스크탑
python -m streamlit run app_desktop.py

# 모바일
python -m streamlit run app_mobile.py
```

---

## 📊 데이터 전처리 순서

```bash
python step1_split.py        # train/val/test 분할 (8:1:1)
python step2_augmentation.py # train 데이터 증강 (약 3배)
python step3_train.py        # YOLOv8 학습
```

---

## 🖥️ 주요 기능

- **실시간 자세 측정** — GOOD 자세 5초 유지 시 다음 단계로 자동 진행
- **작업환경 인식** — 의자/책상/모니터 실시간 탐지
- **음성 안내** — 자세 상태 변화 시 음성 피드백
- **점수 및 위험도** — 10점 만점 / 안전·주의·위험 3단계
- **측정 이력** — 사용자별 측정 기록 및 점수 추이 그래프
- **바른자세 챌린지** — 팀원 간 포인트 레이스
- **로그인/회원가입** — 사용자별 데이터 분리 관리

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.10 |
| 딥러닝 | PyTorch, YOLOv8 (Ultralytics) |
| 자세 분류 | Attention MLP |
| 웹앱 | Streamlit |
| 컴퓨터 비전 | OpenCV, YOLOv8-pose |
| 데이터 증강 | Albumentations |
| 음성 | pyttsx3 |
| GPU | NVIDIA RTX 4060 (CUDA 12.1) |

---

## ⚠️ 주의사항

- 본 서비스는 **의료 진단을 대체하지 않습니다.**
- 자세 분석 결과는 **참고용**으로만 활용하세요.
- 모델 가중치 파일(`.pt`)은 용량 문제로 GitHub에 포함되지 않습니다.
- 실행 전 `jasee_core.py`의 모델 경로를 본인 환경에 맞게 확인하세요.

---

## 👥 팀 정보

**4팀 — 척추처척추**  
아시아경제교육센터 | 2026년 5월
