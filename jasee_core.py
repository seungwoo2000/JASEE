"""
jasee_core.py - 자세히봐 핵심 로직
위치: E:\python\Jasee\jasee_core.py

모델 로드, 각도 계산, 자세/환경 판정 담당
Streamlit UI(app.py)와 분리된 순수 로직 모듈
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
POSE_MLP_PATH = Path(r"E:\python\Jasee\Yolo_pose\04_final_model\output\final_attention_mlp.pt")
ENV_YOLO_PATH = Path(r"E:\python\Jasee\Yolo_env\images_data\runs\posture_v1\weights\best.pt")

ENV_CLASSES = {0: "chair_back", 1: "chair_seat", 2: "desk_surface", 3: "monitor"}
ENV_COLORS  = {0: (0,200,255),  1: (0,165,255),  2: (255,165,0),   3: (255,200,0)}

# ───────────────────────────────────────────
# 측정 기준 (RULA + VDT 고시 제2020-17호)
# ───────────────────────────────────────────
CRITERIA = {
    "CVA"        : (0,    20),    # 목굴곡각 0~20° Good
    "TIA"        : (-20,  20),    # 몸통굴곡각 -20~20° Good
    "knee_angle" : (85,   100),   # 무릎각도 85~100° Good
    "gaze_angle" : (10,   15),    # 시선각 10~15° 하방 Good
    "desk_diff"  : (0,    0.10),  # 작업대 높이 ±10% 이내
    "chair_gap"  : (0,    0.20),  # 등받이 거리 골반너비 20% 이내
}

def is_good(key: str, value: float) -> bool:
    lo, hi = CRITERIA[key]
    return lo <= value <= hi

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
# Attention MLP 모델 정의
# ───────────────────────────────────────────
class AttentionMLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=256, num_classes=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=1, batch_first=True
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),        # net.0
            nn.BatchNorm1d(hidden_dim),              # net.1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),  # net.4
            nn.BatchNorm1d(hidden_dim // 2),         # net.5
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # net.8
            nn.BatchNorm1d(hidden_dim // 4),              # net.9
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, num_classes),  # net.12
        )

    def forward(self, x):
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(1)
        return self.net(attn_out)

# ───────────────────────────────────────────
# 모델 로드
# ───────────────────────────────────────────
def load_models():
    """pose_yolo, mlp, env_yolo, device 반환"""
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    pose_yolo = YOLO("yolov8n-pose.pt")
    mlp       = AttentionMLP()
    
    checkpoint = torch.load(POSE_MLP_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        mlp.load_state_dict(checkpoint["model_state_dict"])
    else:
        mlp.load_state_dict(checkpoint)
    
    mlp.eval().to(device)
    env_yolo  = YOLO(str(ENV_YOLO_PATH))
    print(f"[모델 로드 완료] device={device}")
    return pose_yolo, mlp, env_yolo, device

# ───────────────────────────────────────────
# 각도 계산 (기준서 3.1절 - YOLOv8-pose 인덱스)
# ───────────────────────────────────────────
def calc_angle_3pt(A, B, C) -> float:
    """삼점 내각 (B가 꼭짓점) - 기준서 3.1.1"""
    v1 = np.array(A) - np.array(B)
    v2 = np.array(C) - np.array(B)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

def calc_cva(lm) -> float:
    """목굴곡각 CVA - 기준서 3.1.2
    YOLOv8-pose: 오른귀=4, 오른어깨=6
    """
    ear      = np.array([lm[4][0], lm[4][1]])
    shoulder = np.array([lm[6][0], lm[6][1]])
    vec      = shoulder - ear
    return float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]))))

def calc_tia(lm) -> float:
    """몸통굴곡각 TIA - 기준서 3.1.2
    YOLOv8-pose: 어깨=5,6 / 골반=11,12
    """
    sh_mid  = np.array([(lm[5][0]+lm[6][0])/2, (lm[5][1]+lm[6][1])/2])
    hip_mid = np.array([(lm[11][0]+lm[12][0])/2, (lm[11][1]+lm[12][1])/2])
    vec     = sh_mid - hip_mid
    return float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]))))

def calc_knee_angle(lm) -> float:
    """무릎각도 - 기준서 3.1.1
    YOLOv8-pose: 오른골반=12, 오른무릎=14, 오른발목=16
    """
    return calc_angle_3pt(lm[12][:2], lm[14][:2], lm[16][:2])

def calc_gaze_angle(lm, monitor_bbox) -> float | None:
    """모니터 시선각 - 기준서 3.1.3
    YOLOv8-pose: 왼눈=1, 오른눈=2
    """
    if monitor_bbox is None:
        return None
    eye = np.array([(lm[1][0]+lm[2][0])/2, (lm[1][1]+lm[2][1])/2])
    mx  = (monitor_bbox[0] + monitor_bbox[2]) / 2
    my  = (monitor_bbox[1] + monitor_bbox[3]) / 2
    return float(np.degrees(np.arctan2(my - eye[1], mx - eye[0])))

def calc_desk_diff(lm, desk_bbox) -> float | None:
    """작업대 높이 비율 - 기준서 3.2.2
    YOLOv8-pose: 오른팔꿈치=8
    """
    if desk_bbox is None:
        return None
    elbow_y   = lm[8][1]
    desk_y    = desk_bbox[1]
    sh_mid_y  = (lm[5][1]+lm[6][1])/2
    hip_mid_y = (lm[11][1]+lm[12][1])/2
    ref       = abs(hip_mid_y - sh_mid_y)
    return float(abs(desk_y - elbow_y) / ref) if ref > 1e-4 else None

def calc_chair_gap(lm, chair_back_bbox) -> float | None:
    """등받이 거리 비율 - 기준서 3.2.3
    YOLOv8-pose: 왼골반=11, 오른골반=12
    """
    if chair_back_bbox is None:
        return None
    hip_x     = lm[12][0]
    back_x    = chair_back_bbox[2]
    hip_w     = abs(lm[11][0] - lm[12][0])
    if hip_w < 0.05:
        hip_w = abs(lm[5][0] - lm[6][0])
    return float(abs(hip_x - back_x) / hip_w) if hip_w > 1e-4 else None

# ───────────────────────────────────────────
# 자세 MLP 판정
# ───────────────────────────────────────────
def predict_posture(mlp, cva: float, tia: float, device: str) -> str:
    cva_good = 1 if is_good("CVA", cva) else 0
    tia_good = 1 if is_good("TIA", tia) else 0
    feat = torch.tensor([[cva, tia, cva_good, tia_good] + [0.0]*12],
                         dtype=torch.float32).to(device)
    with torch.no_grad():
        out  = mlp(feat)
        pred = (torch.sigmoid(out) > 0.5).item()
    return "GOOD" if pred else "BAD"

# ───────────────────────────────────────────
# 자세 측정: 키포인트 추출 + 각도 계산
# ───────────────────────────────────────────
def run_posture(frame, pose_yolo, mlp, device, env_bboxes: dict) -> dict:
    """
    반환값 dict:
      result      : 'GOOD' | 'BAD' | None
      metrics     : {CVA, TIA, knee_angle, gaze_angle, desk_diff, chair_gap}
      keypoints   : {귀, 어깨, 골반, 무릎, 발목, 팔꿈치} 픽셀 좌표
    """
    out = {"result": None, "metrics": {}, "keypoints": {}}
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
                gaze      = calc_gaze_angle(lm, env_bboxes.get("monitor"))
                desk_diff = calc_desk_diff(lm, env_bboxes.get("desk_surface"))
                chair_gap = calc_chair_gap(lm, env_bboxes.get("chair_back"))

                out["metrics"] = {
                    "CVA": cva, "TIA": tia,
                    "knee_angle": knee, "gaze_angle": gaze,
                    "desk_diff": desk_diff, "chair_gap": chair_gap,
                }
                out["result"] = predict_posture(mlp, cva, tia, device)
                out["keypoints"] = {
                    "귀":    (int(lm[4][0]),  int(lm[4][1])),
                    "어깨":  (int(lm[6][0]),  int(lm[6][1])),
                    "골반":  (int(lm[12][0]), int(lm[12][1])),
                    "무릎":  (int(lm[14][0]), int(lm[14][1])),
                    "발목":  (int(lm[16][0]), int(lm[16][1])),
                    "팔꿈치":(int(lm[8][0]),  int(lm[8][1])),
                }
            except Exception:
                pass
            break  # 첫 번째 사람만
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
    """
    detected, bboxes = {}, {}
    env_results = env_yolo(frame, verbose=False)
    for r in env_results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = ENV_CLASSES.get(cls, str(cls))
            detected[label] = conf
            bboxes[label]   = [x1, y1, x2, y2]
    return {"detected": detected, "bboxes": bboxes}
