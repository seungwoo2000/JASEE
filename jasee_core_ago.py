# -*- coding: utf-8 -*-
"""
jasee_core.py  —  자세히봐 핵심 로직 (통합 정제 버전)
위치: E:\\python\\Jasee\\jasee_core.py

포함 내용
  - YOLOv8-pose 17개 키포인트 기반 각도 계산
  - AttentionMLP 자세 판정
  - 환경 YOLO (chair_back / chair_seat / desk_surface / monitor)
  - 오버레이 그리기 (posture / env)

변경 이력 (리팩터링)
  - 색상 상수 단일 블록으로 통합 (BGR)
    · COL_GOOD / COL_BAD / COL_ARROW / COL_NA
    · COLOR_GOOD / COLOR_BAD / COLOR_NA 는 COL_* 별칭으로 통일
  - _render_realtime_feedback_card 미사용 함수 제거 → app_mobile로 이관
  - calc_gaze_angle 인자명 통일 (monitor_bbox)
  - 모든 docstring 한국어로 통일
"""

import math
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import pyttsx3
import cv2
from ultralytics import YOLO
from pathlib import Path

# ───────────────────────────────────────────────────────────────
# 경로 설정
# ───────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"C:\python\Jasee")
POSE_MLP_PATH = BASE_DIR / "Yolo_pose" / "04_final_model" / "output" / "final_attention_mlp.pt"
ENV_YOLO_PATH = BASE_DIR / "Yolo_env" / "images_data" / "runs" / "posture_v1" / "weights" / "best.pt"

ENV_CLASSES = {0: "chair_back", 1: "chair_seat", 2: "desk_surface", 3: "monitor"}
ENV_COLORS  = {0: (0, 200, 255), 1: (0, 165, 255), 2: (255, 165, 0), 3: (255, 200, 0)}


# ───────────────────────────────────────────────────────────────
# 측정 기준 (RULA + VDT 고시 제2020-17호)
# ───────────────────────────────────────────────────────────────
CRITERIA = {
    "CVA"        : (0,    20),    # 목굴곡각 0~20° Good
    "TIA"        : (0,    20),    # 몸통굴곡각 0~20° Good
    "knee_angle" : (85,  100),    # 무릎각도 85~100° Good
    "elbow_angle": (90,  120),    # 팔꿈치 각도 90~120° Good
    "wrist_angle": (0,    15),    # 손목 편차 ±15° 이내
    "gaze_angle" : (10,   15),    # 시선각 10~15° 하방 Good
    "desk_diff"  : (0,  0.10),    # 작업대 높이 ±10% 이내
    "chair_gap"  : (0,  0.20),    # 등받이 거리 골반너비 20% 이내
}


def is_good(key: str, value: float) -> bool:
    """CRITERIA 기준으로 해당 키/값이 정상 범위인지 반환합니다."""
    if key not in CRITERIA or value is None:
        return False
    lo, hi = CRITERIA[key]
    return lo <= value <= hi


# ───────────────────────────────────────────────────────────────
# 피드백 메시지
# ───────────────────────────────────────────────────────────────
FEEDBACK = {
    "CVA": {
        "no": "01", "label": "목굴곡각", "eng": "CVA", "cat": "posture",
        "range": "정상 0°~20° · 위험 20°초과",
        "good": "머리·경추 수직 정렬 유지\n경추 부담 최소화 상태",
        "bad":  "전방두부자세(FHP) 의심\n모니터를 눈높이로 올리세요\n1시간마다 목 스트레칭 시행",
    },
    "TIA": {
        "no": "02", "label": "몸통굴곡각", "eng": "TIA", "cat": "posture",
        "range": "정상 0°~20° · 위험 20°초과",
        "good": "척추 수직 정렬 양호\n요추 압박 최소화 상태",
        "bad":  "과도한 몸통 전굴 감지\n등받이에 허리 완전 밀착\n의자 깊숙이 앉으세요",
    },
    "무릎": {
        "no": "04", "label": "무릎 각도", "eng": "Knee", "cat": "posture",
        "range": "정상 85°~100° · 위험 85° 미만 또는 100° 초과",
        "good": "하지 혈액순환 원활\n하체 부담 최소화 상태",
        "bad":  "무릎 각도 기준 이탈\n의자 높이 조절 필요\n발받침대 사용 권장",
    },
    "손목": {
        "no": "05", "label": "손목 각도", "eng": "Wrist", "cat": "posture",
        "range": "정상 ±15° 이내 손목 중립 자세 유지",
        "good": "손목 중립 자세 유지\n손목 터널 부담 최소화",
        "bad":  "손목 과굴곡 감지\n손목 받침대 설치 필요\n키보드 앞 15cm 확보",
    },
    "시선각": {
        "no": "06", "label": "모니터 시선각", "eng": "Gaze", "cat": "env",
        "range": "정상 하방 10°~15° · 위험 10° 미만 또는 15° 초과",
        "good": "시선각 기준 충족\n경추 부담 최소화",
        "bad":  "시선각 기준 이탈\n모니터 상단을 눈높이에 맞추세요\n화면 거리 40cm 이상 권장",
    },
    "책상높이": {
        "no": "07", "label": "작업대 높이", "eng": "Desk", "cat": "env",
        "range": "정상 ±5% 이내 · 위험 ±5% 초과",
        "good": "작업대·팔꿈치 정렬 양호\n상지 부담 최소화",
        "bad":  "작업대 높이 불일치\n책상 높이 또는 의자 높이 조정 필요",
    },
    "등받이": {
        "no": "08", "label": "의자 등받이", "eng": "Chair", "cat": "env",
        "range": "정상 20% 이내 · 위험 20% 초과",
        "good": "등받이 지지 충분\n요추 안정성 확보",
        "bad":  "등받이 지지 부족\n의자 깊숙이 착석\n허리 완전 밀착 필요",
    },
}

INDICATOR_NAMES = {
    "CVA":      "CVA 목굴곡각",
    "TIA":      "TIA 몸통굴곡각",
    "무릎":     "무릎 각도",
    "손목":     "손목 각도",
    "시선각":   "모니터 시선각",
    "책상높이": "작업대 높이",
    "등받이":   "의자 등받이",
}

IND_UNITS = {
    "CVA": "°", "TIA": "°",
    "무릎": "°", "손목": "°", "시선각": "°",
    "책상높이": "", "등받이": "",
}

DISPLAY_ORDER = ["CVA", "TIA", "무릎", "손목", "시선각", "책상높이", "등받이"]


# ───────────────────────────────────────────────────────────────
# 오버레이 색상 (BGR) — 단일 블록으로 통합
#
#  · COL_GOOD / COL_BAD / COL_ARROW / COL_NA  ← 실시간 오버레이용 (진한 계열)
#  · COLOR_GOOD / COLOR_BAD / COLOR_NA         ← 이미지 분석 오버레이용 (밝은 계열)
#    (app_mobile.py 하위호환을 위해 별칭으로 유지)
# ───────────────────────────────────────────────────────────────
COL_GOOD  = (100, 220, 100)   # 초록  — STEP 1 실시간
COL_BAD   = (50,   50, 220)   # 빨강  — STEP 1 실시간
COL_ARROW = (255, 100,   0)   # 파랑 화살표
COL_NA    = (160, 160, 160)   # 회색

COLOR_GOOD = (29,  158, 117)  # 민트 초록 — 이미지 분석
COLOR_BAD  = (74,   50, 230)  # 파랑-보라 — 이미지 분석
COLOR_NA   = (180, 180, 180)  # 회색

# STEP 1 오버레이 관절 인덱스 (CVA/TIA 관련만 표시)
CVA_KP = [3, 4, 5, 6]        # 왼귀, 오른귀, 왼어깨, 오른어깨
TIA_KP = [5, 6, 11, 12]      # 어깨(양), 골반(양)


# ───────────────────────────────────────────────────────────────
# 음성 안내
# ───────────────────────────────────────────────────────────────
_tts_lock = threading.Lock()


def speak(text: str):
    """TTS 안내를 별도 스레드에서 재생합니다."""
    def _run():
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()


# ───────────────────────────────────────────────────────────────
# AttentionMLP 모델 정의
# ───────────────────────────────────────────────────────────────
class AttentionMLP(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 256, num_classes: int = 1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=1, batch_first=True
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, x):
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        return self.net(attn_out.squeeze(1))


# ───────────────────────────────────────────────────────────────
# 모델 로드
# ───────────────────────────────────────────────────────────────
def load_models():
    """pose_yolo, mlp, env_yolo, device 를 반환합니다."""
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    pose_yolo = YOLO("yolov8n-pose.pt")

    mlp = AttentionMLP()
    checkpoint = torch.load(POSE_MLP_PATH, map_location=device)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    mlp.load_state_dict(state_dict)
    mlp.eval().to(device)

    env_yolo = YOLO(str(ENV_YOLO_PATH))
    print(f"[모델 로드 완료] device={device}")
    return pose_yolo, mlp, env_yolo, device


# ───────────────────────────────────────────────────────────────
# 각도 계산 (YOLOv8-pose 17개 인덱스 기준)
# ───────────────────────────────────────────────────────────────
def calc_angle_3pt(A, B, C) -> float:
    """세 점의 내각을 계산합니다 (B가 꼭짓점)."""
    v1 = np.array(A) - np.array(B)
    v2 = np.array(C) - np.array(B)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def calc_vertical_angle(p1, p2) -> float:
    """수직축 기준 두 점의 기울기 각도를 반환합니다."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


def calc_cva(lm) -> float:
    """CVA 목굴곡각 — conf 합산이 높은 쪽 귀→어깨 vs 수직."""
    r_conf = lm[4][2] + lm[6][2]
    l_conf = lm[3][2] + lm[5][2]
    if r_conf >= l_conf:
        ear      = np.array([lm[4][0], lm[4][1]])
        shoulder = np.array([lm[6][0], lm[6][1]])
    else:
        ear      = np.array([lm[3][0], lm[3][1]])
        shoulder = np.array([lm[5][0], lm[5][1]])
    return calc_vertical_angle(ear, shoulder)


def calc_tia(lm) -> float:
    """TIA 몸통굴곡각 — 어깨 중점 → 골반 중점 vs 수직."""
    sh_mid  = np.array([(lm[5][0] + lm[6][0]) / 2,  (lm[5][1] + lm[6][1]) / 2])
    hip_mid = np.array([(lm[11][0] + lm[12][0]) / 2, (lm[11][1] + lm[12][1]) / 2])
    return calc_vertical_angle(sh_mid, hip_mid)


def calc_knee_angle(lm) -> float:
    """무릎 각도 — conf 높은 쪽 골반-무릎-발목."""
    r_conf = lm[12][2] + lm[14][2] + lm[16][2]
    l_conf = lm[11][2] + lm[13][2] + lm[15][2]
    if r_conf >= l_conf:
        return calc_angle_3pt(lm[12][:2], lm[14][:2], lm[16][:2])
    return calc_angle_3pt(lm[11][:2], lm[13][:2], lm[15][:2])


def calc_elbow_angle(lm) -> float | None:
    """팔꿈치 각도 — conf 높은 쪽 어깨-팔꿈치-손목."""
    try:
        r_conf = lm[6][2] + lm[8][2] + lm[10][2]
        l_conf = lm[5][2] + lm[7][2] + lm[9][2]
        if r_conf >= l_conf:
            return calc_angle_3pt(lm[6][:2], lm[8][:2], lm[10][:2])
        return calc_angle_3pt(lm[5][:2], lm[7][:2], lm[9][:2])
    except Exception:
        return None


def calc_wrist_angle(lm) -> float | None:
    """손목 편차 — 어깨-팔꿈치-손목 내각에서 180° 차이."""
    try:
        r_conf = lm[6][2] + lm[8][2] + lm[10][2]
        l_conf = lm[5][2] + lm[7][2] + lm[9][2]
        if r_conf >= l_conf:
            inner = calc_angle_3pt(lm[6][:2], lm[8][:2], lm[10][:2])
        else:
            inner = calc_angle_3pt(lm[5][:2], lm[7][:2], lm[9][:2])
        return float(abs(inner - 180.0))
    except Exception:
        return None


def calc_gaze_angle(lm, monitor_bbox) -> float | None:
    """모니터 시선각 — 눈 중점 → 모니터 중심."""
    if monitor_bbox is None:
        return None
    eye = np.array([(lm[1][0] + lm[2][0]) / 2, (lm[1][1] + lm[2][1]) / 2])
    mx  = (monitor_bbox[0] + monitor_bbox[2]) / 2
    my  = (monitor_bbox[1] + monitor_bbox[3]) / 2
    return float(np.degrees(np.arctan2(my - eye[1], mx - eye[0])))


def calc_desk_diff(lm, desk_bbox) -> float | None:
    """작업대 높이 비율 — conf 높은 쪽 팔꿈치 vs 책상 상면."""
    if desk_bbox is None:
        return None
    elbow_y   = lm[8][1] if lm[8][2] >= lm[7][2] else lm[7][1]
    desk_y    = desk_bbox[1]
    sh_mid_y  = (lm[5][1] + lm[6][1]) / 2
    hip_mid_y = (lm[11][1] + lm[12][1]) / 2
    ref       = abs(hip_mid_y - sh_mid_y)
    return float(abs(desk_y - elbow_y) / ref) if ref > 1e-4 else None


def calc_chair_gap(lm, chair_back_bbox) -> float | None:
    """등받이 거리 비율 — 골반 중점 vs chair_back 가까운 끝."""
    if chair_back_bbox is None:
        return None
    hip_x      = (lm[11][0] + lm[12][0]) / 2
    back_left  = chair_back_bbox[0]
    back_right = chair_back_bbox[2]
    back_x     = back_left if abs(hip_x - back_left) < abs(hip_x - back_right) else back_right
    hip_w      = abs(lm[11][0] - lm[12][0])
    if hip_w < 5:
        hip_w = abs(lm[5][0] - lm[6][0])
    gap = abs(hip_x - back_x)
    return float(gap / hip_w) if hip_w > 1e-4 else None


# ───────────────────────────────────────────────────────────────
# 자세 MLP 판정
# ───────────────────────────────────────────────────────────────
def predict_posture(mlp, cva: float, tia: float, device: str) -> str:
    """CVA/TIA를 AttentionMLP에 넣어 GOOD/BAD 문자열을 반환합니다."""
    cva_good = 1 if is_good("CVA", cva) else 0
    tia_good = 1 if is_good("TIA", tia) else 0
    feat = torch.tensor(
        [[cva, tia, cva_good, tia_good] + [0.0] * 12],
        dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        out  = mlp(feat)
        pred = (torch.sigmoid(out) > 0.5).item()
    result = "GOOD" if pred else "BAD"
    print(f"[판정] CVA={cva:.1f}° ({'OK' if cva_good else 'NG'}), "
          f"TIA={tia:.1f}° ({'OK' if tia_good else 'NG'}) → {result}")
    return result


# ───────────────────────────────────────────────────────────────
# 자세 측정 (실시간 웹캠 프레임)
# ───────────────────────────────────────────────────────────────
def run_posture(frame, pose_yolo, mlp, device, env_bboxes: dict) -> dict:
    """
    프레임 1장을 받아 자세 측정 결과 dict를 반환합니다.

    반환 dict 키:
      result        : 'GOOD' | 'BAD' | None
      metrics       : {CVA, TIA, knee_angle, elbow_angle, wrist_angle,
                       gaze_angle, desk_diff, chair_gap}
      keypoints     : {귀, 어깨, 골반, 무릎, 발목, 팔꿈치, 손목} 픽셀 좌표
      keypoints_raw : np.ndarray shape (17, 3) — 환경 필터링용
      gate_pass     : bool (CVA + TIA 모두 GOOD)
    """
    out = {"result": None, "metrics": {}, "keypoints": {}, "gate_pass": False}
    pose_results = pose_yolo(frame, verbose=False)

    for r in pose_results:
        if r.keypoints is None:
            continue
        for kp in r.keypoints.data:
            if kp.shape[0] < 17:
                continue
            try:
                lm = kp.cpu().numpy()

                cva       = calc_cva(lm)
                tia       = calc_tia(lm)
                knee      = calc_knee_angle(lm)
                elbow     = calc_elbow_angle(lm)
                wrist     = calc_wrist_angle(lm)
                gaze      = calc_gaze_angle(lm, env_bboxes.get("monitor"))
                desk_diff = calc_desk_diff(lm, env_bboxes.get("desk_surface"))
                chair_gap = calc_chair_gap(lm, env_bboxes.get("chair_back"))

                out["metrics"] = {
                    "CVA": cva, "TIA": tia,
                    "knee_angle": knee, "elbow_angle": elbow,
                    "wrist_angle": wrist, "gaze_angle": gaze,
                    "desk_diff": desk_diff, "chair_gap": chair_gap,
                }
                out["result"]       = predict_posture(mlp, cva, tia, device)
                out["gate_pass"]    = is_good("CVA", cva) and is_good("TIA", tia)
                out["keypoints_raw"] = lm
                out["keypoints"] = {
                    "귀":    (int(lm[4][0]),  int(lm[4][1])),
                    "어깨":  (int(lm[6][0]),  int(lm[6][1])),
                    "골반":  (int(lm[12][0]), int(lm[12][1])),
                    "무릎":  (int(lm[14][0]), int(lm[14][1])),
                    "발목":  (int(lm[16][0]), int(lm[16][1])),
                    "팔꿈치":(int(lm[8][0]),  int(lm[8][1])),
                    "손목":  (int(lm[10][0]), int(lm[10][1])),
                }
            except Exception:
                pass
            break
        break

    return out


# ───────────────────────────────────────────────────────────────
# 환경 인식
# ───────────────────────────────────────────────────────────────
def run_environment(frame, env_yolo, pose_keypoints=None) -> dict:
    """
    프레임에서 작업환경 객체(의자/책상/모니터)를 인식합니다.

    반환 dict 키:
      detected : {label: conf}
      bboxes   : {label: [x1, y1, x2, y2]}

    필터링 전략:
      - conf=0.5, iou=0.3 으로 중복 박스 제거
      - 클래스당 conf 가장 높은 1개만 유지
      - pose_keypoints 있으면 관절 근처 객체만 선택
          chair_back/chair_seat → 골반(11, 12) 근처
          desk_surface          → 팔꿈치(7~10) 근처
          monitor               → 얼굴 바라보는 방향 앞쪽
    """
    detected, bboxes = {}, {}
    env_results = env_yolo(frame, verbose=False, conf=0.5, iou=0.3)

    # ── 클래스별 최고 conf 후보 수집 ───────────────────────────
    candidates = {}
    for r in env_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = ENV_CLASSES.get(cls, str(cls))
            if label not in candidates or conf > candidates[label]["conf"]:
                candidates[label] = {"conf": conf, "bbox": [x1, y1, x2, y2]}

    # pose_keypoints 없으면 필터 없이 반환
    if pose_keypoints is None:
        for label, v in candidates.items():
            detected[label] = v["conf"]
            bboxes[label]   = v["bbox"]
        return {"detected": detected, "bboxes": bboxes}

    kp  = pose_keypoints
    h, w = frame.shape[:2]

    def bbox_center(bbox):
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def pt_dist(p1, p2):
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    def kp_valid(idx):
        return kp[idx][2] > 0.3

    for label, v in candidates.items():
        bbox    = v["bbox"]
        cx, cy  = bbox_center(bbox)
        accept  = False

        if label in ("chair_back", "chair_seat"):
            hip_pts = [
                (kp[idx][0], kp[idx][1])
                for idx in [11, 12] if kp_valid(idx)
            ]
            if hip_pts:
                hip_cx = sum(p[0] for p in hip_pts) / len(hip_pts)
                hip_cy = sum(p[1] for p in hip_pts) / len(hip_pts)
                if pt_dist((cx, cy), (hip_cx, hip_cy)) < w * 0.6:
                    accept = True
            else:
                accept = True

        elif label == "desk_surface":
            # 손목(9=왼손목, 10=오른손목) 근처 책상만 인식
            best_wrist, best_conf = None, 0
            for idx in [9, 10]:
                if kp_valid(idx) and kp[idx][2] > best_conf:
                    best_conf  = kp[idx][2]
                    best_wrist = (kp[idx][0], kp[idx][1])
            if best_wrist:
                if pt_dist((cx, cy), best_wrist) < w * 0.6:
                    accept = True
            else:
                accept = True

        elif label == "monitor":
            # 어깨 X 중점 → 손목 X 방향으로 모니터가 있는지 판별
            # 측면 촬영 기준: 손목이 모니터 쪽(어깨 앞쪽)에 있음
            sh_pts = [kp[i][:2] for i in [5, 6] if kp_valid(i)]
            wrist_pts = [kp[i][:2] for i in [9, 10] if kp_valid(i)]
            if sh_pts and wrist_pts:
                sh_cx    = sum(p[0] for p in sh_pts) / len(sh_pts)
                wrist_cx = sum(p[0] for p in wrist_pts) / len(wrist_pts)
                # 손목→어깨 방향 벡터 (모니터가 있어야 할 X 방향)
                view_dx  = wrist_cx - sh_cx
                mon_dx   = cx - sh_cx
                # 모니터가 손목 방향에 있고, 사람과 가장 가까운 것 선택
                if view_dx * mon_dx > 0:
                    accept = True
            else:
                accept = True

        if accept:
            detected[label] = v["conf"]
            bboxes[label]   = v["bbox"]

    return {"detected": detected, "bboxes": bboxes}


# ───────────────────────────────────────────────────────────────
# 오버레이 그리기 — STEP 1 자세 측정 (실시간)
# ───────────────────────────────────────────────────────────────
def draw_posture_overlay(frame: np.ndarray, lm: np.ndarray, metrics: dict) -> np.ndarray:
    """
    CVA/TIA 관련 관절만 실시간 오버레이합니다.
      - GOOD: 초록(COL_GOOD) 점/선
      - BAD:  빨강(COL_BAD) 점/선 + 파란(COL_ARROW) 화살표

    Args:
        frame   : BGR 프레임
        lm      : shape (17, 3) numpy array  (x, y, conf)
        metrics : {"CVA": float|None, "TIA": float|None, ...}

    Returns:
        오버레이가 적용된 BGR 프레임 복사본
    """
    out = frame.copy()

    cva_raw = metrics.get("CVA")
    tia_raw = metrics.get("TIA")
    cva_ok  = is_good("CVA", cva_raw) if cva_raw is not None else None
    tia_ok  = is_good("TIA", tia_raw) if tia_raw is not None else None

    def kp_color(ok):
        if ok is None:
            return COL_NA
        return COL_GOOD if ok else COL_BAD

    cva_col = kp_color(cva_ok)
    tia_col = kp_color(tia_ok)

    KP_COL = {
        3: cva_col, 4: cva_col,      # 귀
        5: tia_col, 6: tia_col,      # 어깨
        11: tia_col, 12: tia_col,    # 골반
    }

    LINES = [
        (3, 5, cva_col), (4, 6, cva_col),          # 귀-어깨 (CVA)
        (5, 6, tia_col),                             # 어깨-어깨
        (5, 11, tia_col), (6, 12, tia_col),          # 어깨-골반 (TIA)
        (11, 12, tia_col),                           # 골반-골반
    ]
    for a, b, col in LINES:
        if lm[a][2] > 0.3 and lm[b][2] > 0.3:
            pa = (int(lm[a][0]), int(lm[a][1]))
            pb = (int(lm[b][0]), int(lm[b][1]))
            cv2.line(out, pa, pb, (0, 0, 0), 1)
            cv2.line(out, pa, pb, col, 1)

    for idx, col in KP_COL.items():
        if lm[idx][2] < 0.3:
            continue
        px, py = int(lm[idx][0]), int(lm[idx][1])
        if col == COL_BAD:
            cv2.circle(out, (px, py), 1, col, -1)
            cv2.circle(out, (px, py), 2, (255, 255, 255), 1)
            cv2.circle(out, (px, py), 3, col, 1)
        else:
            cv2.circle(out, (px, py), 1, col, -1)
            cv2.circle(out, (px, py), 2, (255, 255, 255), 1)

    # CVA BAD → 귀에서 뒤쪽(후방) 화살표
    if cva_ok is False:
        ear_idx = 4 if lm[4][2] >= lm[3][2] else 3
        if lm[ear_idx][2] > 0.3:
            ep  = (int(lm[ear_idx][0]), int(lm[ear_idx][1]))
            tgt = (ep[0] + 60, ep[1] - 15)
            cv2.arrowedLine(out, ep, tgt, COL_ARROW, 1, tipLength=0.35)
            cv2.circle(out, tgt, 1, COL_ARROW, -1)

    # TIA BAD → 어깨에서 위쪽 화살표
    if tia_ok is False:
        sh_idx = 6 if lm[6][2] >= lm[5][2] else 5
        if lm[sh_idx][2] > 0.3:
            sp  = (int(lm[sh_idx][0]), int(lm[sh_idx][1]))
            tgt = (sp[0], sp[1] - 70)
            cv2.arrowedLine(out, sp, tgt, COL_ARROW, 1, tipLength=0.35)
            cv2.circle(out, tgt, 1, COL_ARROW, -1)

    return out


# ───────────────────────────────────────────────────────────────
# 오버레이 그리기 — STEP 2 환경 측정 (실시간)
# ───────────────────────────────────────────────────────────────
def draw_env_overlay(
    frame: np.ndarray,
    bboxes: dict,
    detected: dict,
    lm: np.ndarray = None,
    metrics: dict = None,
) -> np.ndarray:
    """
    감지된 환경 객체에 굵은 박스 + 한글 라벨을 표시합니다.
    자세 GOOD 상태일 때만 파란 조정 화살표를 추가로 표시합니다.
    (BAD 상태에서는 환경 박스만 표시하고 화살표는 생략)

    Args:
        frame    : BGR 프레임
        bboxes   : {label: [x1,y1,x2,y2]}
        detected : {label: conf}
        lm       : shape (17, 3) numpy array (선택)
        metrics  : {"CVA": float, "TIA": float, ...} (선택)
    """
    out  = frame.copy()
    h, w = out.shape[:2]

    LABEL_KR = {
        "chair_back":   "등받이",
        "chair_seat":   "의자시트",
        "desk_surface": "책상",
        "monitor":      "모니터",
    }
    BOX_COLORS = {
        "chair_back":   (255, 180,   0),
        "chair_seat":   (255, 140,   0),
        "desk_surface": (0,   180, 255),
        "monitor":      (180, 255,   0),
    }

    posture_good = False
    if metrics:
        cva_ok = is_good("CVA", metrics.get("CVA")) if metrics.get("CVA") is not None else False
        tia_ok = is_good("TIA", metrics.get("TIA")) if metrics.get("TIA") is not None else False
        posture_good = cva_ok and tia_ok

    # 환경 박스 + 라벨
    for label, bbox in bboxes.items():
        x1, y1, x2, y2 = bbox
        col = BOX_COLORS.get(label, (200, 200, 200))
        lbl = LABEL_KR.get(label, label)

        cv2.rectangle(out, (x1, y1), (x2, y2), col, 5)

        font_scale = 1.1
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(out, (x1, y1 - th - 20), (x1 + tw + 12, y1), col, -1)
        cv2.putText(out, lbl, (x1 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    # 체크리스트 (좌측)
    required = ["chair_back", "desk_surface", "monitor"]
    for i, item in enumerate(required):
        ok  = item in detected
        col = (100, 220, 100) if ok else (80, 80, 80)
        lbl = LABEL_KR.get(item, item)
        mark = "v" if ok else "o"
        cv2.putText(out, f"{mark} {lbl}",
                    (15, 80 + i * 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)

    # 자세 GOOD 일 때만 조정 화살표 표시
    if posture_good and lm is not None:
        # 모니터: 눈높이 → 모니터 중심
        if "monitor" in bboxes and lm[1][2] > 0.3 and lm[2][2] > 0.3:
            eye_y  = int((lm[1][1] + lm[2][1]) / 2)
            eye_x  = int((lm[1][0] + lm[2][0]) / 2)
            mon    = bboxes["monitor"]
            mon_cy = (mon[1] + mon[3]) // 2
            if abs(eye_y - mon_cy) > 30:
                cv2.arrowedLine(out, (eye_x, eye_y), (eye_x, mon_cy),
                                COL_ARROW, 1, tipLength=0.2)
                cv2.circle(out, (eye_x, mon_cy), 1, COL_ARROW, -1)

        # 책상: 팔꿈치 높이 → 책상면
        if "desk_surface" in bboxes and lm is not None:
            el_idx = 8 if lm[8][2] >= lm[7][2] else 7
            if lm[el_idx][2] > 0.3:
                ep     = (int(lm[el_idx][0]), int(lm[el_idx][1]))
                desk_y = bboxes["desk_surface"][1]
                if abs(ep[1] - desk_y) > 20:
                    tgt = (ep[0], desk_y)
                    cv2.arrowedLine(out, ep, tgt, COL_ARROW, 1, tipLength=0.25)
                    cv2.circle(out, tgt, 1, COL_ARROW, -1)

    return out



