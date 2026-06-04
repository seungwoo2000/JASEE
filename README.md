<!--
████████████████████████████████████████████████████████████████████
  seungwoo2000 · JASEE — K-디지털 트레이닝 포트폴리오 README
████████████████████████████████████████████████████████████████████
-->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:1d4ed8,100:38bdf8&height=220&section=header&text=🪑%20JASEE&fontSize=64&fontColor=ffffff&fontAlignY=40&desc=자세히봐%20|%20AI%20실시간%20자세%20·%20작업환경%20측정%20서비스&descAlignY=62&descColor=bae6fd&animation=fadeIn" alt="header" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python_3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO_Pose-관절추출-00CFDD?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-환경탐지-FF6B35?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-챗봇-8B5CF6?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-멀티플랫폼-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA_12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

<br/>

> **RULA + VDT 고시 제2020-17호 기반 AI 실시간 자세 & 작업환경 측정 서비스**  
> [Fit\_me\_up](https://github.com/seungwoo2000/Fit_me_up) 에서 한 단계 발전한 고도화 버전

</div>

---

## 📋 훈련 과정 정보

| 항목 | 내용 |
|:---|:---|
| 🏫 **훈련기관** | 아시아경제 교육센터 |
| 📚 **훈련과정명** | 융합\_데이터 기반 차세대 디지털 헬스케어 AI 솔루션 5회차 |
| 🏷️ **훈련유형** | K-디지털 트레이닝 (고용노동부) |
| 📅 **훈련기간** | 2026-02-03 ~ 2026-07-30 (6개월) |
| 💡 **프로젝트 분류** | 컴퓨터 비전 · 헬스케어 AI · RAG 챗봇 · 실시간 자세 분석 |

---

## 🪑 프로젝트 소개

```
"사무직 근로자의 근골격계 질환 예방을 위해,
카메라로 실시간 자세와 작업환경을 분석하고 개선 피드백을 제공합니다."
```

**JASEE**는 [Fit\_me\_up](https://github.com/seungwoo2000/Fit_me_up) 프로젝트를 고도화한 버전으로,  
다음 세 가지 핵심 기능이 새롭게 추가·개선됐습니다.

| 구분 | Fit\_me\_up | JASEE (업그레이드) |
|:---:|:---:|:---:|
| **분석 방식** | 이미지 분석 | ✅ **실시간 + 이미지 모두** |
| **관절 추출** | MediaPipe | ✅ **YOLO Pose** (17 keypoints) |
| **챗봇** | ❌ | ✅ **RAG 챗봇** 추가 |
| **플랫폼** | 단일 앱 | ✅ **모바일 · 데스크탑 · 웹** |
| **추가 기능** | 기본 피드백 | ✅ 운동 추천 · 제품 추천 · 챌린지 |

---

## 🗂️ 목차

1. [분석 파이프라인](#-분석-파이프라인)
2. [주요 기능](#-주요-기능)
3. [측정 기준](#-측정-기준-rula--vdt-고시)
4. [기술 스택](#-기술-스택)
5. [파일 구조](#-파일-구조)
6. [배운 점 · 성장 포인트](#-배운-점--성장-포인트)
7. [실행 방법](#-실행-방법)

---

## 🔄 분석 파이프라인

```
웹캠 (실시간) / 이미지 업로드
        ↓
┌──────────────────────────────────────┐
│  ① YOLO Pose       관절 17개 좌표 추출 │
│  ② Attention MLP   자세 Good/Bad 분류  │
│  ③ YOLO Env        환경 객체 탐지       │
│  ④ RAG 챗봇         맞춤 교정 답변 생성  │
└──────────────────────────────────────┘
        ↓
⑤ RULA + VDT 기준으로 각도 측정 및 점수 산출
        ↓
⑥ 교정 코멘트 · 운동 추천 · 제품 추천 출력
        ↓
⑦ 측정 이력 저장 · 근골격계 리포트 생성
```

---

## ✨ 주요 기능

| 기능 | 설명 |
|:---|:---|
| 🎥 **실시간 자세 측정** | 20초 안에 GOOD 5초 유지 → 자동으로 환경 측정 전환 |
| 🖼️ **이미지 자세 분석** | 측면 사진 업로드로 즉시 분석 |
| 🪑 **작업환경 인식** | 의자·책상·모니터 4개 동시 감지 완료 시 기록 |
| 💬 **AI 교정 코멘트** | BAD 부위별 맞춤 교정 안내 |
| 🤖 **RAG 챗봇** | 자세·인간공학 지식 기반 질의응답 챗봇 |
| 🏆 **바른자세 챌린지** | 포인트 레이스 기반 자세 개선 동기 부여 |
| 📊 **근골격계 리포트** | 위험도 기반 측정 결과 리포트 생성 |
| 🏃 **운동 추천** | BAD 부위 기반 맞춤 스트레칭 추천 |
| 🛒 **제품 추천** | 교정 필요 부위 기반 보조 제품 추천 |
| 📈 **측정 이력** | 사용자별 점수 추이 기록 및 시각화 |

---

## 📐 측정 기준 (RULA + VDT 고시)

> 💡 **RULA**: 작업자 자세 위험도를 평가하는 국제 표준 인간공학 지표  
> 💡 **VDT 고시 제2020-17호**: 국내 영상표시단말기 작업 안전보건 기준

| 측정 항목 | 정상 범위 | 기준 |
|:---:|:---:|:---:|
| CVA (목굴곡각) | 0° ~ 20° | RULA Neck Zone |
| TIA (몸통굴곡각) | 0° ~ 20° | RULA Trunk Zone |
| 무릎 각도 | 85° ~ 100° | VDT 고시 |
| 손목 각도 | ±15° 이내 | VDT 고시 |
| 모니터 시선각 | 하방 10° ~ 15° | VDT 고시 제6조 |
| 작업대 높이 | 팔꿈치 기준 ±10% | VDT 고시 |
| 의자 등받이 | 골반너비 20% 이내 | VDT 고시 |

---

## 🛠️ 기술 스택

> 비전공자도 이해할 수 있도록, 각 기술이 **어떤 역할**을 하는지 함께 설명합니다.

| 분류 | 기술 | 한 줄 설명 |
|:---:|:---:|:---|
| **언어** | Python 3.10 | AI 개발의 표준 언어 |
| **관절 추출** | YOLOv8-Pose | 사람의 관절 17개를 실시간으로 인식 |
| **자세 분류** | Attention MLP | 관절 좌표로 Good/Bad 자세를 판정하는 딥러닝 모델 |
| **환경 탐지** | YOLOv8 | 책상·의자·모니터 위치를 실시간 인식 |
| **RAG 챗봇** | RAG + Vector DB | 자세 관련 문서를 검색해 정확한 답변 생성 |
| **웹앱** | Streamlit | 모바일·데스크탑·웹 3가지 버전 제공 |
| **컴퓨터 비전** | OpenCV | 영상 처리 및 오버레이 시각화 |
| **데이터 증강** | Albumentations | 학습 데이터를 다양하게 변형해 모델 성능 향상 |
| **음성 안내** | pyttsx3 | 교정 피드백을 텍스트 → 음성으로 변환 |
| **GPU** | NVIDIA RTX 4060 (CUDA 12.1) | 딥러닝 학습·추론 가속 |

---

## 📁 파일 구조

```
JASEE/
│
├── 📂 Yolo_env/                       # 작업환경 인식 모델
│   ├── step1_split.py                 # 데이터 분할 (8:1:1)
│   ├── step2_augmentation.py          # 데이터 증강
│   ├── step3_train.py                 # YOLOv8 학습
│   └── images_data/runs/posture_v1/
│       └── weights/best.pt            # ⭐ 환경인식 모델
│
├── 📂 Yolo_pose/                      # 자세 분류 모델
│   ├── 01_data_preprocessing/
│   ├── 02_model_comparison/
│   ├── 03_model(attention MLP)_improve/
│   ├── 04_final_model/
│   │   └── output/
│   │       └── final_attention_mlp.pt # ⭐ 자세 분류 모델
│   └── 05_documentation/
│
├── 📂 RAG/                            # RAG 챗봇 관련 파일
├── 📂 vector_db/                      # 챗봇 지식 벡터 데이터베이스
├── 📂 자세히봐_RAG/                    # RAG 통합 모듈
├── 📂 processed_data/                 # 전처리된 학습 데이터
├── 📂 assets/                         # 이미지·아이콘 리소스
│
├── jasee_core.py                      # 🧠 핵심 로직 (모델 로드·각도 계산·판정)
├── app_mobile.py                      # 📱 모바일용 Streamlit 앱 (권장)
├── app_desktop.py                     # 🖥️ 데스크탑용 Streamlit 앱
├── app_web.py                         # 🌐 웹용 Streamlit 앱
├── chatbot.py                         # 🤖 RAG 챗봇
├── build_vectordb.py                  # 벡터 DB 구축 스크립트
├── preprocess_jasee.py                # 데이터 전처리
├── yolov8n-pose.pt                    # ⭐ YOLO Pose 모델
└── requirements.txt
```

> ⭐ 표시된 모델 파일 3개를 위 경로에 넣으면 바로 실행됩니다.

---

## 📈 배운 점 · 성장 포인트

| 분야 | 배운 것 | 이걸 배워서 뭘 할 수 있게 됐나? |
|:---|:---|:---|
| 🎥 **실시간 처리** | OpenCV + YOLO 스트리밍 파이프라인 | 웹캠 영상을 프레임 단위로 실시간 분석 |
| 🦴 **YOLO Pose** | 17개 관절 keypoint 추출 | MediaPipe 대비 더 빠르고 정확한 자세 분석 |
| 🧠 **Attention MLP** | 관절 좌표 → Good/Bad 분류 모델 설계 | 어텐션 메커니즘으로 중요 관절에 가중치 부여 |
| 🤖 **RAG 챗봇** | 벡터 DB 구축 + 문서 검색 + 답변 생성 | 자세 관련 전문 지식을 AI 챗봇으로 서비스화 |
| 📱 **멀티플랫폼** | 모바일·데스크탑·웹 3버전 동시 개발 | 하나의 로직으로 다양한 환경에 대응 |
| 📐 **인간공학 기준** | RULA · VDT 고시 제2020-17호 적용 | 국제·국내 표준 지표를 AI 판정 기준으로 구현 |
| 🔊 **음성 안내** | pyttsx3 TTS 연동 | 화면을 보지 않아도 교정 피드백을 전달 |
| 📊 **데이터 증강** | Albumentations 3배 증강 | 적은 데이터로도 모델 일반화 성능 향상 |

---

## ⚙️ 실행 방법

**1️⃣ 클론 및 가상환경 설정**
```bash
git clone https://github.com/seungwoo2000/JASEE.git
cd JASEE

# conda 가상환경 생성 (Python 3.10)
conda create -n Yolo_env python=3.10
conda activate Yolo_env
```

**2️⃣ 패키지 설치**
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3️⃣ 모델 파일 배치** (⭐ 중요)

| 파일명 | 넣을 경로 |
|:---|:---|
| `yolov8n-pose.pt` | `JASEE/yolov8n-pose.pt` |
| `final_attention_mlp.pt` | `JASEE/Yolo_pose/04_final_model/output/` |
| `best.pt` | `JASEE/Yolo_env/images_data/runs/posture_v1/weights/` |

**4️⃣ 앱 실행**
```bash
# 📱 모바일 (권장)
python -m streamlit run app_mobile.py

# 🖥️ 데스크탑
python -m streamlit run app_desktop.py

# 🌐 웹
python -m streamlit run app_web.py
```

> ⚠️ 본 서비스는 **의료 진단을 대체하지 않습니다.** 측정 결과는 참고용으로 활용하세요.

---

<div align="center">

<br/>

*"실시간으로 자세를 잡고, AI와 대화하며 환경까지 개선합니다."* 🪑🤖

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:38bdf8,50:1d4ed8,100:0f172a&height=130&section=footer&text=K-디지털%20트레이닝%20|%20아시아경제%20교육센터&fontSize=15&fontColor=ffffff&fontAlignY=65" width="100%"/>

**📅 2026.02 ~ 2026.07** &nbsp;|&nbsp; Made with 🪑 during K-Digital Training

</div>
