# 🪑 자세히봐 (Fit Me Up)

> **RULA + VDT 고시 제2020-17호** 기반 AI 실시간 자세 & 작업환경 측정 서비스

---

## 📌 프로젝트 개요

사무직 근로자의 근골격계 질환 예방을 목적으로,  
카메라를 통해 **실시간으로 자세와 작업환경을 분석**하고 개선 피드백을 제공하는 AI 서비스입니다.

---

## 🗂️ 프로젝트 구조

```
📁 JASEE/
├── 📁 Yolo_env/                        ← 작업환경 인식 모델
│   ├── 📄 step1_split.py               ← 데이터 분할 (8:1:1)
│   ├── 📄 step2_augmentation.py        ← 데이터 증강
│   ├── 📄 step3_train.py               ← YOLOv8 학습
│   └── 📁 images_data/
│       └── 📁 runs/
│           └── 📁 posture_v1/
│               └── 📁 weights/
│                   └── 📄 best.pt      ← ⭐ 환경인식 모델 (여기에 넣기)
│
├── 📁 Yolo_pose/                       ← 자세 분류 모델
│   ├── 📁 01_data_preprocessing/
│   ├── 📁 02_model_comparison/
│   ├── 📁 03_model(attetion MLP)_improve/
│   ├── 📁 04_final_model/
│   │   └── 📁 output/
│   │       └── 📄 final_attention_mlp.pt  ← ⭐ 자세 분류 모델 (여기에 넣기)
│   └── 📁 05_documentation/
│
├── 📄 jasee_core.py                    ← 핵심 로직 (모델 로드, 각도 계산, 판정)
├── 📄 app_mobile.py                    ← 모바일용 Streamlit 앱 (메인)
├── 📄 app_desktop.py                   ← 데스크탑용 Streamlit 앱
├── 📄 logo_transparent.png             ← 앱 로고
├── 📄 yolov8n-pose.pt                  ← ⭐ YOLOv8 포즈 모델 (여기에 넣기)
├── 📄 requirements.txt
└── 📄 README.md
```

> ⭐ 표시된 모델 파일 3개를 위 경로에 맞게 넣으면 바로 실행됩니다.

---

## ⭐ 모델 파일 설치 (중요)

GitHub에서 클론 후 아래 3개 파일을 직접 경로에 넣어주세요:

| 파일명 | 넣을 경로 |
|--------|---------|
| `yolov8n-pose.pt` | `JASEE/yolov8n-pose.pt` |
| `final_attention_mlp.pt` | `JASEE/Yolo_pose/04_final_model/output/` |
| `best.pt` | `JASEE/Yolo_env/images_data/runs/posture_v1/weights/` |

---

## 📐 측정 기준 (RULA + VDT 고시 제2020-17호)

| 지표 | 정상 범위 | 기준 |
|------|----------|------|
| CVA (목굴곡각) | 0° ~ 20° | RULA Neck Zone |
| TIA (몸통굴곡각) | 0° ~ 20° | RULA Trunk Zone |
| 무릎 각도 | 85° ~ 100° | VDT 고시 |
| 손목 각도 | ±15° 이내 | VDT 고시 |
| 모니터 시선각 | 하방 10° ~ 15° | VDT 고시 제6조 |
| 작업대 높이 | 팔꿈치 기준 ±10% | VDT 고시 |
| 의자 등받이 | 골반너비 20% 이내 | VDT 고시 |

---

## ⚙️ 설치 및 실행

### 1. 클론
```bash
git clone https://github.com/seungwoo2000/JASEE.git
cd JASEE
```

### 2. 가상환경 생성
```bash
conda create -n Yolo_env python=3.10
conda activate Yolo_env
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. 모델 파일 넣기
위 ⭐ 경로에 모델 파일 3개 배치

### 5. 실행
```bash
# 모바일 (권장)
python -m streamlit run app_mobile.py

# 데스크탑
python -m streamlit run app_desktop.py
```

---

## 📊 데이터 전처리 순서 (재학습 시)

```bash
python Yolo_env/step1_split.py        # train/val/test 분할 (8:1:1)
python Yolo_env/step2_augmentation.py # 증강 (약 3배)
python Yolo_env/step3_train.py        # YOLOv8 학습
```

---

## 🖥️ 주요 기능

| 기능 | 설명 |
|------|------|
| 실시간 자세 측정 | 20초 안에 GOOD 5초 유지 → 자동으로 환경 측정 전환 |
| 이미지 자세 분석 | 측면 사진 업로드로 즉시 분석 |
| 작업환경 인식 | 의자/책상/모니터 4개 동시 감지 완료 시 기록 |
| AI 교정 코멘트 | BAD 부위별 맞춤 교정 안내 |
| 측정 이력 | 사용자별 점수 추이 기록 |
| 바른자세 챌린지 | 팀원 간 포인트 레이스 |
| 근골격계 리포트 | 위험도 기반 리포트 생성 |
| 운동 추천 | BAD 부위 기반 스트레칭 추천 |
| 제품 추천 | 교정 필요 부위 기반 제품 추천 |

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.10 |
| 딥러닝 | PyTorch, YOLOv8 (Ultralytics) |
| 자세 분류 | Attention MLP (input_dim=16) |
| 웹앱 | Streamlit |
| 컴퓨터 비전 | OpenCV, YOLOv8-pose (17 keypoints) |
| 데이터 증강 | Albumentations |
| 음성 안내 | pyttsx3 |
| GPU | NVIDIA RTX 4060 (CUDA 12.1) |

---

## ⚠️ 주의사항

- 본 서비스는 **의료 진단을 대체하지 않습니다.**
- 측정 결과는 **참고용**으로만 활용하세요.
- 실시간 측정은 **PC 웹캠** 기반으로 동작합니다.

---

## 👥 팀 정보

**4팀 — 척추처척추**  
아시아경제교육센터 | 2026년 5월
