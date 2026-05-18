"""
jasee_core.py - 자세히봐 핵심 로직 (통합 버전)
위치: E:\python\Jasee\jasee_core.py

integrate_team.py + jasee_core.py 통합
- 키포인트: YOLOv8-pose 17개 (실시간 웹캠)
- 자세 분류: PyTorch AttentionMLP
- 지표: 6개 (CVA, TIA, knee, gaze, desk, chair)
- YOLO: chair_back / chair_seat / desk_surface / monitor
"""

import math
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import pyttsx3
from ultralytics import YOLO
from pathlib import Path

# ───────────────────────────────────────────
# 경로 설정
# ───────────────────────────────────────────
BASE_DIR      = Path(r"E:\python\Jasee")
POSE_MLP_PATH = BASE_DIR / "Yolo_pose" / "04_final_model" / "output" / "final_attention_mlp.pt"
ENV_YOLO_PATH = BASE_DIR / "Yolo_env" / "images_data" / "runs" / "posture_v1" / "weights" / "best.pt"

ENV_CLASSES = {0: "chair_back", 1: "chair_seat", 2: "desk_surface", 3: "monitor"}
ENV_COLORS  = {0: (0,200,255),  1: (0,165,255),  2: (255,165,0),   3: (255,200,0)}

# ───────────────────────────────────────────
# 측정 기준 (RULA + VDT 고시 제2020-17호)
# integrate_team.py THRESHOLD 값 기준 통합
# ───────────────────────────────────────────
CRITERIA = {
    "CVA"        : (0,    20),    # 목굴곡각 0~20° Good
    "TIA"        : (0,    20),    # 몸통굴곡각 0~20° Good (기준서 기준)
    "knee_angle" : (85,   100),   # 무릎각도 85~100° Good
    "elbow_angle": (90,   120),   # 팔꿈치 각도 90~120° Good (integrate 추가)
    "wrist_angle": (0,    15),    # 손목 편차 ±15° 이내 (integrate 추가)
    "gaze_angle" : (10,   15),    # 시선각 10~15° 하방 Good
    "desk_diff"  : (0,    0.10),  # 작업대 높이 ±10% 이내
    "chair_gap"  : (0,    0.20),  # 등받이 거리 골반너비 20% 이내
}

def is_good(key: str, value: float) -> bool:
    if key not in CRITERIA:
        return False
    lo, hi = CRITERIA[key]
    return lo <= value <= hi

# ───────────────────────────────────────────
# 피드백 메시지 (integrate_team.py FEEDBACK 통합)
# ───────────────────────────────────────────
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

# ───────────────────────────────────────────
# 오버레이 색상 (integrate_team.py 기준)
# ───────────────────────────────────────────
COLOR_GOOD   = (29,  158, 117)   # 민트 초록
COLOR_BAD    = (74,   50, 230)   # 파랑-보라
COLOR_TARGET = (50,  230, 180)   # 목표 민트
COLOR_NA     = (180, 180, 180)   # 회색

# ───────────────────────────────────────────
# 음성 안내
# ───────────────────────────────────────────
_tts_lock = threading.Lock()

def speak(text: str):
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

# ───────────────────────────────────────────
# AttentionMLP 모델 정의 (PyTorch)
# ───────────────────────────────────────────
class AttentionMLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=256, num_classes=1):
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

# ───────────────────────────────────────────
# 모델 로드
# ───────────────────────────────────────────
def load_models():
    """pose_yolo, mlp, env_yolo, device 반환"""
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    pose_yolo = YOLO("yolov8n-pose.pt")

    mlp = AttentionMLP()
    checkpoint = torch.load(POSE_MLP_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) \
                 if isinstance(checkpoint, dict) else checkpoint
    mlp.load_state_dict(state_dict)
    mlp.eval().to(device)

    env_yolo = YOLO(str(ENV_YOLO_PATH))
    print(f"[모델 로드 완료] device={device}")
    return pose_yolo, mlp, env_yolo, device

# ───────────────────────────────────────────
# 각도 계산 (YOLOv8-pose 17개 인덱스 기준)
# integrate_team.py 계산 방식 반영
# ───────────────────────────────────────────
def calc_angle_3pt(A, B, C) -> float:
    """삼점 내각 (B가 꼭짓점)"""
    v1 = np.array(A) - np.array(B)
    v2 = np.array(C) - np.array(B)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

def calc_vertical_angle(p1, p2) -> float:
    """수직축 기준 각도 (integrate 방식)"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))

def calc_cva(lm) -> float:
    """목굴곡각 CVA — 오른귀(4) → 오른어깨(6) vs 수직"""
    ear      = np.array([lm[4][0], lm[4][1]])
    shoulder = np.array([lm[6][0], lm[6][1]])
    return calc_vertical_angle(ear, shoulder)

def calc_tia(lm) -> float:
    """몸통굴곡각 TIA — 어깨중점 → 골반중점 vs 수직"""
    sh_mid  = np.array([(lm[5][0]+lm[6][0])/2,  (lm[5][1]+lm[6][1])/2])
    hip_mid = np.array([(lm[11][0]+lm[12][0])/2, (lm[11][1]+lm[12][1])/2])
    return calc_vertical_angle(sh_mid, hip_mid)

def calc_knee_angle(lm) -> float:
    """무릎각도 — 오른골반(12)-무릎(14)-발목(16)"""
    return calc_angle_3pt(lm[12][:2], lm[14][:2], lm[16][:2])

def calc_elbow_angle(lm) -> float | None:
    """팔꿈치 각도 — 오른어깨(6)-팔꿈치(8)-손목(10)"""
    try:
        return calc_angle_3pt(lm[6][:2], lm[8][:2], lm[10][:2])
    except Exception:
        return None

def calc_wrist_angle(lm) -> float | None:
    """손목 편차 — 팔꿈치(8)-손목(10)-손가락끝 근사"""
    try:
        inner = calc_angle_3pt(lm[6][:2], lm[8][:2], lm[10][:2])
        return float(abs(inner - 180.0))
    except Exception:
        return None

def calc_gaze_angle(lm, monitor_bbox) -> float | None:
    """모니터 시선각 — 눈중점 → 모니터중심"""
    if monitor_bbox is None:
        return None
    eye = np.array([(lm[1][0]+lm[2][0])/2, (lm[1][1]+lm[2][1])/2])
    mx  = (monitor_bbox[0] + monitor_bbox[2]) / 2
    my  = (monitor_bbox[1] + monitor_bbox[3]) / 2
    return float(np.degrees(np.arctan2(my - eye[1], mx - eye[0])))

def calc_desk_diff(lm, desk_bbox) -> float | None:
    """작업대 높이 비율 — 팔꿈치(8) vs 책상 상면"""
    if desk_bbox is None:
        return None
    elbow_y   = lm[8][1]
    desk_y    = desk_bbox[1]
    sh_mid_y  = (lm[5][1]+lm[6][1])/2
    hip_mid_y = (lm[11][1]+lm[12][1])/2
    ref       = abs(hip_mid_y - sh_mid_y)
    return float(abs(desk_y - elbow_y) / ref) if ref > 1e-4 else None

def calc_chair_gap(lm, chair_back_bbox) -> float | None:
    """등받이 거리 비율 — 오른골반(12) vs chair_back"""
    if chair_back_bbox is None:
        return None
    hip_x  = lm[12][0]
    back_x = chair_back_bbox[2]
    hip_w  = abs(lm[11][0] - lm[12][0])
    if hip_w < 0.05:
        hip_w = abs(lm[5][0] - lm[6][0])
    return float(abs(hip_x - back_x) / hip_w) if hip_w > 1e-4 else None

# ───────────────────────────────────────────
# 자세 MLP 판정
# ───────────────────────────────────────────
def predict_posture(mlp, cva: float, tia: float, device: str) -> str:
    cva_good = 1 if is_good("CVA", cva) else 0
    tia_good = 1 if is_good("TIA", tia) else 0
    feat = torch.tensor(
        [[cva, tia, cva_good, tia_good] + [0.0] * 12],
        dtype=torch.float32
    ).to(device)
    with torch.no_grad():
        out  = mlp(feat)
        pred = (torch.sigmoid(out) > 0.5).item()
    result = "GOOD" if pred else "BAD"
    # 디버그 출력 (확인 후 제거)
    print(f"[판정] CVA={cva:.1f}° ({'OK' if cva_good else 'NG'}), TIA={tia:.1f}° ({'OK' if tia_good else 'NG'}) → {result}")
    return result

# ───────────────────────────────────────────
# 자세 측정 (실시간 웹캠 프레임)
# ───────────────────────────────────────────
def run_posture(frame, pose_yolo, mlp, device, env_bboxes: dict) -> dict:
    """
    반환값 dict:
      result     : 'GOOD' | 'BAD' | None
      metrics    : {CVA, TIA, knee_angle, elbow_angle, wrist_angle,
                    gaze_angle, desk_diff, chair_gap}
      keypoints  : {귀, 어깨, 골반, 무릎, 발목, 팔꿈치, 손목} 픽셀 좌표
      gate_pass  : bool (CVA+TIA 모두 GOOD이면 True)
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

                cva          = calc_cva(lm)
                tia          = calc_tia(lm)
                knee         = calc_knee_angle(lm)
                elbow        = calc_elbow_angle(lm)
                wrist        = calc_wrist_angle(lm)
                gaze         = calc_gaze_angle(lm, env_bboxes.get("monitor"))
                desk_diff    = calc_desk_diff(lm, env_bboxes.get("desk_surface"))
                chair_gap    = calc_chair_gap(lm, env_bboxes.get("chair_back"))

                out["metrics"] = {
                    "CVA": cva, "TIA": tia,
                    "knee_angle": knee, "elbow_angle": elbow,
                    "wrist_angle": wrist, "gaze_angle": gaze,
                    "desk_diff": desk_diff, "chair_gap": chair_gap,
                }
                out["result"]    = predict_posture(mlp, cva, tia, device)
                out["gate_pass"] = is_good("CVA", cva) and is_good("TIA", tia)
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

# ───────────────────────────────────────────
# 환경 인식
# ───────────────────────────────────────────
def run_environment(frame, env_yolo) -> dict:
    """
    반환값 dict:
      detected : {label: conf}
      bboxes   : {label: [x1,y1,x2,y2]}

    땜빵 조치:
      - conf=0.5 이상만 탐지 (낮은 신뢰도 제거)
      - iou=0.3 으로 NMS 강화 (중복 박스 제거)
      - 클래스당 confidence 가장 높은 1개만 유지
    """
    detected, bboxes = {}, {}
    env_results = env_yolo(frame, verbose=False, conf=0.5, iou=0.3)
    for r in env_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = ENV_CLASSES.get(cls, str(cls))
            # 클래스당 confidence 가장 높은 1개만 유지
            if label not in detected or conf > detected[label]:
                detected[label] = conf
                bboxes[label]   = [x1, y1, x2, y2]
    return {"detected": detected, "bboxes": bboxes}