# =========================================================
# app_web.py — 자세히봐 웹 페이지 통합 버전
#
# 변경 사항 (app_mobile_team.py 기준):
#   1. integrate.py 의존 제거
#   2. analyze_image() → jasee_core 기반으로 교체
#   3. RealTimePostureProcessor._analyze_frame_background → jasee_core 기반
#   4. layout="wide", 웹 페이지 폭/배치 적용
#   5. 나머지 UI 전부 그대로 유지
#
# 실행: streamlit run app_web.py
# =========================================================

import os, sys, math, warnings, datetime, time, threading
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import streamlit.components.v1 as components
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

import json, hashlib, base64, html

def _html_escape(value):
    return html.escape(str(value), quote=True)

from urllib.parse import quote
import pandas as pd
import altair as alt
import requests

from streamlit_webrtc import webrtc_streamer
try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode
    import av
except Exception:
    VideoProcessorBase = object
    WebRtcMode = None
    av = None

from chatbot import process_rag_query

# streamlit-webrtc 일부 버전에서 컴포넌트가 재시작/정지될 때
# _polling_thread가 None인 상태로 stop()이 호출되어 앱 전체가 중단되는 문제 방지
try:
    from streamlit_webrtc.shutdown import SessionShutdownObserver

    _original_shutdown_observer_stop = SessionShutdownObserver.stop

    def _safe_shutdown_observer_stop(self):
        if getattr(self, "_polling_thread", None) is None:
            return
        return _original_shutdown_observer_stop(self)

    SessionShutdownObserver.stop = _safe_shutdown_observer_stop
except Exception:
    pass

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── jasee_core import ──────────────────────────────────────
from jasee_core import (
    load_models, speak, is_good, CRITERIA,
    ENV_CLASSES, ENV_COLORS, FEEDBACK, INDICATOR_NAMES, IND_UNITS,
    DISPLAY_ORDER, COLOR_GOOD, COLOR_BAD, COLOR_NA,
    run_posture, run_environment,
    predict_posture, AttentionMLP,
    cv2_put_korean, draw_image_overlay,
)

# =========================================================
# 1. 기본 설정 (웹: wide + sidebar collapsed)
# =========================================================
st.set_page_config(
    page_title="자세히봐 — AI 자세 분석",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# WebRTC가 로컬 네트워크/모바일 브라우저에서 Start 후 로딩 상태로 멈추는 문제를 줄이기 위한 STUN 설정
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}

# ── 모델 로드 (캐시) ───────────────────────────────────────
@st.cache_resource
def get_models():
    return load_models()

try:
    _pose_yolo, _mlp, _env_yolo, _device = get_models()
    _models_ok = True
except Exception as _e:
    _models_ok = False
    _model_error = str(_e)

if "env_bboxes" not in st.session_state:
    st.session_state.env_bboxes = {}

# =========================================================
# 2. analyze_image() — integrate.py 대체 (데스크탑과 동일)
# =========================================================
# cv2_put_korean → jasee_core.py로 이관


def analyze_image(pil_image: Image.Image) -> dict:
    if not _models_ok:
        return {"ok": False, "message": f"모델 로드 실패: {_model_error}"}
    try:
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {"ok": False, "message": f"이미지 변환 실패: {e}"}

    # 1차 자세 분석: 포즈 키포인트를 먼저 얻습니다.
    # 이 단계에서는 아직 최신 환경 bbox가 없을 수 있으므로,
    # 환경 지표(gaze_angle/desk_diff/chair_gap)는 아래에서 다시 계산합니다.
    posture_out = run_posture(
        frame, _pose_yolo, _mlp, _device,
        st.session_state.get("env_bboxes", {})
    )
    metrics = posture_out.get("metrics", {})

    if not metrics:
        return {"ok": False, "message": "관절 탐지 실패 — 측면 전신 이미지를 사용해주세요."}

    # 포즈 키포인트 기반으로 환경 객체를 인식합니다.
    _kp_raw = posture_out.get("keypoints_raw")  # jasee_core에서 raw numpy 배열
    env_out  = run_environment(frame, _env_yolo, pose_keypoints=_kp_raw)
    detected = env_out.get("detected", {}) or {}
    bboxes   = env_out.get("bboxes", {}) or {}

    # 최신 환경 bbox를 session_state에 저장합니다.
    # 환경 객체가 일시적으로 누락된 프레임에서는 직전 bbox를 유지해 계산 끊김을 줄입니다.
    prev_bboxes = st.session_state.get("env_bboxes", {}) or {}
    merged_bboxes = dict(prev_bboxes)
    merged_bboxes.update(bboxes)
    st.session_state.env_bboxes = merged_bboxes

    # 2차 자세 분석: 방금 인식한 최신 환경 bbox로 환경 지표까지 다시 계산합니다.
    # 이 과정을 거쳐 오버레이에는 객체가 보이는데 결과창은 '인식 불가'로 뜨는 문제를 방지합니다.
    posture_out_2 = run_posture(frame, _pose_yolo, _mlp, _device, merged_bboxes)
    if posture_out_2.get("metrics"):
        posture_out = posture_out_2
        metrics = posture_out.get("metrics", {})

    gate_pass = posture_out.get("gate_pass", False)
    kp        = posture_out.get("keypoints", {})

    # jasee_core 영어키 → 팀버전 한글키 매핑
    KR_MAP = {
        "CVA":         "CVA",
        "TIA":         "TIA",
        "knee_angle":  "무릎",
        "wrist_angle": "손목",
        "gaze_angle":  "시선각",
        "desk_diff":   "책상높이",
        "chair_gap":   "등받이",
    }
    POSTURE_KEYS = ["CVA", "TIA", "knee_angle", "wrist_angle"]
    ENV_KEYS     = ["gaze_angle", "desk_diff", "chair_gap"]

    def classify_raw(kr_key, raw):
        if raw is None: return "제외"
        raw = float(raw)
        if kr_key == "CVA":      return "정상" if 0 <= raw <= 20 else "위험"
        if kr_key == "TIA":      return "정상" if 0 <= raw <= 20 else "위험"
        if kr_key == "무릎":     return "정상" if 85 <= raw <= 100 else "위험"
        if kr_key == "손목":     return "정상" if -15 <= raw <= 15 else "위험"
        if kr_key == "시선각":   return "정상" if 10 <= raw <= 15 else "위험"
        if kr_key == "책상높이": return "정상" if raw <= 0.05 else "위험"
        if kr_key == "등받이":   return "정상" if raw <= 0.20 else "위험"
        return "정상"

    def to_tuple(eng_key):
        raw    = metrics.get(eng_key)
        kr_key = KR_MAP.get(eng_key, eng_key)
        unit   = "°" if eng_key in ("CVA","TIA","knee_angle","wrist_angle","gaze_angle") else ""
        level  = classify_raw(kr_key, raw)
        is_ok  = (level == "정상")
        val_str = f"{raw:.2f}{unit}" if raw is not None else "인식 불가"
        return (val_str, is_ok, raw)

    # 한글 키로 posture/env 딕셔너리 생성 (팀버전 FEEDBACK 키와 일치)
    posture = {KR_MAP[k]: to_tuple(k) for k in POSTURE_KEYS}
    env     = {KR_MAP[k]: to_tuple(k) for k in ENV_KEYS}

    gate_bad_items = []
    if not is_good("CVA", metrics.get("CVA", 999)):
        gate_bad_items.append(f"CVA 목굴곡각: {metrics.get('CVA', 0):.1f}° / BAD")
    if not is_good("TIA", metrics.get("TIA", 999)):
        gate_bad_items.append(f"TIA 몸통굴곡각: {metrics.get('TIA', 0):.1f}° / BAD")

    # 점수 계산 (한글 키 기준)
    all_kr = {**posture, **env}
    good_cnt = sum(1 for v in all_kr.values() if v[1])  # (val_str, is_ok, raw)
    total    = sum(1 for v in all_kr.values() if v[2] is not None)
    score    = round((good_cnt / total * 10), 1) if total else 0.0
    risk     = "안전" if score >= 8 else "주의" if score >= 6 else "위험"

    rv          = posture_out.get("result")
    all_kr      = {**posture, **env}

    # 오버레이 그리기 → jasee_core.draw_image_overlay() 에서 일괄 처리
    overlay_bgr = draw_image_overlay(frame, kp, all_kr, bboxes, detected, metrics)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return {
        "ok":             True,
        "gate_pass":      gate_pass,
        "gate_bad_items": gate_bad_items,
        "posture":        posture,
        "env":            env,
        "overlay":        overlay_rgb,
        "message":        f"자세: {rv}",
        "score":          score,
        "risk":           risk,
        "good_count":     good_cnt,
        "total_count":    total,
        "metrics":        metrics,
        "env_detected":   detected,
        "env_bboxes":     bboxes,
    }


# =========================================================
# 3. RealTimePostureProcessor — jasee_core 기반 (모바일)
# =========================================================
def build_ai_correction_comment(result):
    all_data = {**result["posture"], **result["env"]}

    GUIDE = {
        "CVA": {
            "part": "목·경추",
            "bad": "고개가 앞으로 기울어진 전방두부자세 가능성이 있습니다. 모니터 상단을 눈높이에 맞추고 1시간마다 목 스트레칭을 해주세요.",
            "goal": "모니터 높이를 눈높이에 맞추기",
        },
        "TIA": {
            "part": "몸통·허리",
            "bad": "몸통이 앞으로 과도하게 굽혀져 있습니다. 의자 깊숙이 앉아 허리를 등받이에 기대세요.",
            "goal": "골반을 의자 뒤쪽까지 넣고 등받이에 허리 밀착하기",
        },
        "무릎": {
            "part": "무릎·하체",
            "bad": "무릎 각도가 적절하지 않습니다. 의자 높이를 조절해 무릎이 90° 전후가 되도록 하세요.",
            "goal": "무릎이 90° 전후가 되도록 의자 높이와 발 위치 조정하기",
        },
        "손목": {
            "part": "손목",
            "bad": "손목이 과도하게 굽혀져 있습니다. 손목 받침대를 사용하고 키보드 앞 공간을 확보하세요.",
            "goal": "손목 받침대 사용하고 키보드 앞 공간 15cm 이상 확보하기",
        },
        "시선각": {
            "part": "시선·모니터",
            "bad": "모니터 위치가 적절하지 않아 목 부담이 커질 수 있습니다. 모니터 상단을 눈높이에 맞추세요.",
            "goal": "모니터 상단을 눈높이에 맞추고 화면 거리 40cm 이상 확보하기",
        },
        "책상높이": {
            "part": "작업대 높이",
            "bad": "책상 높이가 팔꿈치와 맞지 않습니다. 책상 또는 의자 높이를 조정하세요.",
            "goal": "팔꿈치와 책상면이 수평이 되도록 책상 또는 의자 높이 조정하기",
        },
        "등받이": {
            "part": "의자 등받이",
            "bad": "등받이 지지가 부족합니다. 의자 깊숙이 앉고 요추 부위를 등받이에 밀착하세요.",
            "goal": "의자 깊숙이 앉고 요추 부위를 등받이에 밀착하기",
        },
    }

    bad_items = []
    good_items = []
    goals = []

    for key, (value, is_good, raw) in all_data.items():
        if key not in GUIDE:
            continue

        if is_good:
            good_items.append(f"{GUIDE[key]['part']}({value})")
        else:
            bad_items.append((key, value, GUIDE[key]))
            goals.append(GUIDE[key]["goal"])

    if bad_items:
        first_key, first_value, first_item = bad_items[0]

        detail_html = ""
        for key, value, item in bad_items[:4]:
            detail_html += f"""
            <div style="padding:12px 0;border-top:1px solid #EEF2F6;">
                <div style="font-size:13px;font-weight:800;color:#172033;margin-bottom:4px;">
                    ⚠ {item["part"]} · 측정값 {value}
                </div>
                <div style="font-size:12.5px;line-height:1.7;color:#667085;">
                    {item["bad"]}
                </div>
            </div>
            """

        summary_html = f"""
        <div style="font-size:14px;line-height:1.85;color:#667085;margin-bottom:10px;">
            <b style="color:#172033;">가장 먼저 교정할 부위는 {first_item["part"]}입니다.</b><br>
            기준 범위를 벗어난 항목이 <b style="color:#D94A4A;">{len(bad_items)}개</b> 확인되었습니다.
        </div>
        """

    else:
        summary_html = """
        <div style="font-size:14px;line-height:1.85;color:#667085;">
            <b style="color:#172033;">전체 자세가 안정적입니다.</b><br>
            주요 자세 지표가 대부분 정상 범위에 있습니다.
        </div>
        """
        detail_html = ""
        goals = ["50분 작업 후 5분 스트레칭하기"]

    if good_items:
        good_html = f"""
        <div style="margin-top:12px;padding:12px;border-radius:12px;background:#F0FBF4;font-size:12.5px;line-height:1.7;color:#3B8C42;">
            <b>잘 유지되고 있는 항목</b><br>
            {" · ".join(good_items[:4])}
        </div>
        """
    else:
        good_html = ""

    default_goals = [
        "50분 작업 후 5분 스트레칭하기",
        "목과 어깨를 천천히 돌려 긴장 완화하기",
        "손목이 꺾이지 않도록 키보드와 마우스 위치 조정하기",
        "발바닥이 바닥에 닿는지 확인하기",
    ]

    for g in default_goals:
        if len(goals) >= 4:
            break
        if g not in goals:
            goals.append(g)

    goals_html = "".join([f"{i+1}. {goal}<br>" for i, goal in enumerate(goals[:4])])

    return f"""
<div class="fit-card">
    <div style="padding:14px;border-radius:14px;background:#F8FAFC;">
        <div style="font-size:15px;font-weight:900;color:#172033;margin-bottom:10px;">
            오늘의 실천 목표
        </div>
        <div style="font-size:13px;line-height:1.9;color:#667085;">
            {goals_html}
        </div>
    </div>
</div>
"""    


def render_ai_correction_comment(result):
    """
    AI 오버레이 결과 아래에 중복으로 표시되던
    오늘의 실천 목표 카드를 렌더링하지 않습니다.
    7개 측정 지표 결과 내부의 오늘의 실천 목표만 유지합니다.
    """
    return

def _render_chat_bubble(role, message):
    """
    Streamlit 기본 st.chat_message/st.write가 앱 전역 CSS의 영향을 받아
    챗봇 답변에 밑줄처럼 보이는 border/decoration이 생기는 문제를 막기 위한
    전용 채팅 말풍선 렌더러입니다.
    """
    role_label = "나" if role == "user" else "AI 챗봇"
    role_class = "user" if role == "user" else "assistant"

    safe_message = html.escape(str(message or ""))
    safe_message = safe_message.replace("\n", "<br>")

    st.markdown(
        f"""
<div class="jasee-chat-row {role_class}">
    <div class="jasee-chat-bubble {role_class}">
        <div class="jasee-chat-role">{role_label}</div>
        <div class="jasee-chat-text">{safe_message}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_mobile_chatbot():
    st.markdown("## 자세히봐 AI 챗봇")

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    # ===== 채팅 출력 =====
    for chat in st.session_state.rag_history:
        _render_chat_bubble("user", chat.get("user", ""))
        _render_chat_bubble("assistant", chat.get("assistant", ""))

    # ===== 하단 상담 유형 + 입력창 =====
    # st.chat_input을 컨테이너 안에 넣어 페이지 하단 고정이 아니라
    # 가장 최신 답변 바로 아래에 표시되도록 합니다.
    with st.container():
        st.markdown('<div class="chat-select-wrap">', unsafe_allow_html=True)

        selected_func_id = st.selectbox(
            "상담 유형",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "자세 분석 결과 설명",
                2: "RULA 점수 해석",
                3: "작업환경 적합성",
                4: "부위별 통증 완화",
                5: "운동 추천",
            }[x],
            key="chatbot_selectbox"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ===== 입력창 =====
        user_query = st.chat_input("궁금한 내용을 입력하세요")

    if user_query:
        _render_chat_bubble("user", user_query)

        with st.spinner("답변 생성 중..."):
            answer = process_rag_query(
                query=user_query,
                selected_func_id=selected_func_id,
                history=st.session_state.rag_history
            )

        _render_chat_bubble("assistant", answer)

        st.session_state.rag_history.append({
            "user": user_query,
            "assistant": answer
        })

        st.rerun()


# =========================================================
# 2. CSS — 첨부 HTML 느낌의 세련된 UI
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg: #F4F7FB;
    --panel: #FFFFFF;
    --card: #FFFFFF;
    --line: #E2ECF6;
    --text: #0F1E36;
    --sub: #5E718D;
    --blue: #2563EB;
    --blue-glow: rgba(37, 99, 235, 0.15);
    --blue2: #1E3A8A;
    --teal: #0D9488;
    --green: #10B981;
    --amber: #F59E0B;
    --red: #EF4444;
    --purple: #6366F1;
    --soft-blue: #EFF6FF;
    --soft-green: #ECFDF5;
    --soft-amber: #FFFBEB;
    --soft-red: #FEF2F2;
}

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

/* AI 챗봇 전용 말풍선: 앱 전역 CSS로 인한 밑줄/테두리 오염 방지 */
.jasee-chat-row,
.jasee-chat-row *,
.jasee-chat-bubble,
.jasee-chat-text,
.jasee-chat-role {
    text-decoration: none !important;
    border-bottom: none !important;
    box-shadow: none;
}

.jasee-chat-row {
    width: 100%;
    display: flex;
    margin: 10px 0;
}

.jasee-chat-row.user {
    justify-content: flex-end;
}

.jasee-chat-row.assistant {
    justify-content: flex-start;
}

.jasee-chat-bubble {
    max-width: 88%;
    padding: 13px 15px;
    border-radius: 18px;
    line-height: 1.75;
    box-shadow: 0 6px 18px rgba(15, 30, 54, 0.06);
    word-break: keep-all;
    overflow-wrap: anywhere;
}

.jasee-chat-bubble.user {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: #FFFFFF;
    border-top-right-radius: 6px;
}

.jasee-chat-bubble.assistant {
    background: #FFFFFF;
    color: #172033;
    border: 1px solid #E2ECF6 !important;
    border-top-left-radius: 6px;
}

.jasee-chat-role {
    font-size: 11px;
    font-weight: 900;
    margin-bottom: 5px;
    opacity: 0.72;
}

.jasee-chat-text {
    font-size: 14px;
    font-weight: 500;
    line-height: 1.75;
    white-space: normal;
}


/* AI 챗봇 하단 입력 영역 */
.jasee-chat-bottom-controls {
    margin: 18px auto 0 auto !important;
    padding: 16px 16px 0 16px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    background: #FFFFFF !important;
    border: 1px solid #DCE8F5 !important;
    border-bottom: 0 !important;
    border-radius: 24px 24px 0 0 !important;
    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.06) !important;
}

.jasee-chat-control-title {
    font-size: 13px !important;
    font-weight: 900 !important;
    color: #172033 !important;
    margin-bottom: 8px !important;
}

/* marker가 있는 챗봇 화면 안의 selectbox/form만 부드러운 모바일 UI로 보정 */
div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stSelectbox"] {
    margin-top: -1px !important;
    margin-bottom: 0 !important;
    padding: 0 16px 12px 16px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    background: #FFFFFF !important;
    border-left: 1px solid #DCE8F5 !important;
    border-right: 1px solid #DCE8F5 !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stSelectbox"] > label {
    display: none !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stSelectbox"] div[data-baseweb="select"] {
    width: 100% !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    width: 100% !important;
    min-height: 50px !important;
    border: 1px solid #D7E4F2 !important;
    border-radius: 18px !important;
    background: #FFFFFF !important;
    box-shadow: 0 6px 16px rgba(15, 30, 54, 0.03) !important;
    box-sizing: border-box !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stSelectbox"] div[data-baseweb="select"] div {
    font-size: 15px !important;
    font-weight: 800 !important;
    color: #172033 !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding: 8px 16px 16px 16px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    background: #FFFFFF !important;
    border: 1px solid #DCE8F5 !important;
    border-top: 0 !important;
    border-radius: 0 0 24px 24px !important;
    box-shadow: 0 18px 34px rgba(37, 99, 235, 0.06) !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] form {
    width: 100% !important;
}

/* 입력창 + 버튼 줄: selectbox와 같은 내부 너비로 맞춤 */
div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] div[data-testid="column"] {
    min-width: 0 !important;
    width: auto !important;
    flex: 1 1 auto !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] div[data-testid="column"]:last-child {
    flex: 0 0 52px !important;
    min-width: 52px !important;
    width: 52px !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] [data-testid="stTextInput"] {
    margin: 0 !important;
    width: 100% !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] [data-testid="stTextInput"] input {
    width: 100% !important;
    height: 52px !important;
    border-radius: 18px !important;
    border: 1px solid transparent !important;
    background: #F5F7FB !important;
    color: #172033 !important;
    font-size: 15px !important;
    padding: 0 16px !important;
    box-shadow: none !important;
    box-sizing: border-box !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] [data-testid="stTextInput"] input:focus {
    background: #FFFFFF !important;
    border-color: #2563EB !important;
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.10) !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stFormSubmitButton"] {
    width: 52px !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stFormSubmitButton"] button {
    width: 52px !important;
    height: 52px !important;
    min-height: 52px !important;
    padding: 0 !important;
    border-radius: 17px !important;
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    font-size: 22px !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28) !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.36) !important;
}

/* 컴포넌트 내부 스크롤 제거: iframe은 충분한 높이로 펼치고 페이지 전체 스크롤만 사용 */
iframe {
    width: 100% !important;
    border: 0 !important;
    overflow: hidden !important;
}

[data-testid="stIFrame"] {
    width: 100% !important;
    overflow: hidden !important;
}

[data-testid="stIFrame"] iframe {
    overflow: hidden !important;
}

/* Streamlit 기본 여백 조정 */
.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1930 0%, #152A4A 100%) !important;
    border-right: 1px solid #1E2E4A;
    min-width: 305px !important;
    max-width: 305px !important;
}

[data-testid="stSidebar"] > div:first-child {
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
    padding-top: 0rem;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* 사이드바 라디오 탭: 글씨 한 줄 고정 */
[data-testid="stSidebar"] div[role="radiogroup"] {
    width: 100% !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    width: 100% !important;
    min-height: 66px !important;
    display: flex !important;
    align-items: center !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(37, 99, 235, 0.15) !important;
    border-color: rgba(59, 130, 246, 0.3) !important;
    transform: translateX(4px);
}

[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, #1E40AF 0%, #2563EB 100%) !important;
    border-color: #3B82F6 !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: unset !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    margin: 0 !important;
    width: 100% !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {
    font-weight: 800 !important;
    color: #FFFFFF !important;
}

/* Streamlit Button Overrides */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1E40AF 0%, #2563EB 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100%;
}

div[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%) !important;
}

div[data-testid="stButton"] button:active {
    transform: translateY(0) !important;
}

/* Horizontal Radios (Tabs Override) */
div[data-testid="stRadio"][class*="horizontal"] > div {
    background: #F1F5F9 !important;
    padding: 6px !important;
    border-radius: 14px !important;
    border: 1px solid #E2E8F0 !important;
    display: inline-flex !important;
    gap: 4px !important;
}

div[data-testid="stRadio"][class*="horizontal"] label {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stRadio"][class*="horizontal"] label:hover {
    background: rgba(37, 99, 235, 0.05) !important;
}

div[data-testid="stRadio"][class*="horizontal"] label[data-checked="true"] {
    background: #FFFFFF !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
    border: 1px solid #E2E8F0 !important;
}

div[data-testid="stRadio"][class*="horizontal"] label[data-checked="true"] p {
    color: #2563EB !important;
    font-weight: 700 !important;
}

/* 로고 */
.logo-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 4px 14px 4px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 14px;
}

.logo-mark {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: linear-gradient(135deg, #2563EB, #0D9488);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 900;
}

.logo-title {
    font-size: 17px;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
}

.logo-sub {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 2px;
}

/* 제목 */
.page-title {
    font-size: 30px;
    font-weight: 900;
    color: var(--text);
    letter-spacing: -0.8px;
    margin-bottom: 8px;
    line-height: 1.35;
    position: relative;
    padding-left: 14px;
}

.page-title::before {
    content: '';
    position: absolute;
    left: 0;
    top: 15%;
    height: 70%;
    width: 5px;
    background: linear-gradient(180deg, #3B82F6, #2563EB);
    border-radius: 4px;
}

.page-sub {
    font-size: 14px;
    color: var(--sub);
    margin-bottom: 24px;
    padding-left: 14px;
}

/* 카드 */
.fit-card {
    background: var(--card);
    border: 1px solid rgba(37, 99, 235, 0.12);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.04);
    margin-bottom: 20px;
    transition: all 0.25s ease;
}

.fit-card:hover {
    box-shadow: 0 14px 40px rgba(37, 99, 235, 0.08);
    border-color: rgba(37, 99, 235, 0.25);
}

.fit-card-title {
    font-size: 16px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.fit-badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 12px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: -0.2px;
}

.badge-blue { background: var(--soft-blue); color: var(--blue); border: 1px solid rgba(37, 99, 235, 0.2); }
.badge-green { background: var(--soft-green); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.2); }
.badge-amber { background: var(--soft-amber); color: var(--amber); border: 1px solid rgba(245, 158, 11, 0.2); }
.badge-red { background: var(--soft-red); color: var(--red); border: 1px solid rgba(239, 68, 68, 0.2); }
.badge-gray { background: #F1F5F9; color: var(--sub); border: 1px solid #E2E8F0; }

/* 메트릭 */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid rgba(37, 99, 235, 0.1);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 8px 30px rgba(37, 99, 235, 0.03);
    transition: all 0.2s ease;
    border-bottom: 3px solid rgba(37, 99, 235, 0.2);
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 34px rgba(37, 99, 235, 0.07);
    border-bottom-color: var(--blue);
}

.metric-value {
    font-size: 28px;
    font-weight: 900;
    letter-spacing: -0.8px;
}

.metric-label {
    font-size: 13px;
    color: var(--sub);
    margin-top: 6px;
    font-weight: 600;
}

/* 결과 행 */
.result-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid #EEF2F6;
}

.result-row:last-child {
    border-bottom: 0;
}

.result-name {
    width: 104px;
    font-size: 13px;
    color: var(--sub);
    font-weight: 700;
    flex-shrink: 0;
}

.result-value {
    font-size: 14px;
    font-weight: 800;
    color: var(--text);
    width: 70px;
    flex-shrink: 0;
}

.bar-wrap {
    flex: 1;
    height: 10px;
    background: #F1F5F9;
    border-radius: 999px;
    overflow: hidden;
}

.bar {
    height: 10px;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.bar-green { background: var(--green); }
.bar-amber { background: var(--amber); }
.bar-red { background: var(--red); }
.bar-blue { background: var(--blue); }

/* 피드백 카드 */
.feedback-card {
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 12px;
    border: 1px solid rgba(37, 99, 235, 0.1);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.02);
}

.feedback-good {
    background: linear-gradient(135deg, #ECFDF5, #FFFFFF);
    border-left: 5px solid var(--green);
}

.feedback-bad {
    background: linear-gradient(135deg, #FEF2F2, #FFFFFF);
    border-left: 5px solid var(--red);
}

.feedback-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.feedback-name {
    font-size: 14px;
    font-weight: 800;
    color: var(--text);
}

.feedback-msg {
    font-size: 13px;
    line-height: 1.7;
    color: var(--sub);
    white-space: pre-line;
}

/* 업로드 영역 */
.upload-box {
    border: 2px dashed #3B82F6;
    background: #EFF6FF;
    border-radius: 20px;
    padding: 26px;
    text-align: center;
    color: var(--sub);
    transition: all 0.2s ease;
}

.upload-box:hover {
    border-color: var(--blue);
    background: #E0F2FE;
}

/* 표 */
.fit-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.fit-table th {
    text-align: left;
    color: var(--sub);
    font-weight: 700;
    border-bottom: 2px solid var(--line);
    padding: 12px 8px;
}

.fit-table td {
    border-bottom: 1px solid #EEF2F6;
    padding: 12px 8px;
    color: var(--text);
}

/* 모바일 */
@media (max-width: 900px) {
    .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}




/* 모바일에서는 Streamlit 컬럼을 한 줄씩 세로 배치해서 내용이 잘리지 않게 표시 */
@media (max-width: 700px) {
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 14px !important;
    }

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}



/* 챗봇 입력 영역 최종 정렬 보정: 전역 모바일 column CSS보다 우선 적용 */
@media (max-width: 700px) {
    div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 10px !important;
        width: 100% !important;
    }

    div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] div[data-testid="column"] {
        min-width: 0 !important;
        width: auto !important;
        flex: 1 1 auto !important;
    }

    div[data-testid="stVerticalBlock"]:has(.jasee-chat-bottom-controls) [data-testid="stForm"] div[data-testid="column"]:last-child {
        flex: 0 0 52px !important;
        min-width: 52px !important;
        width: 52px !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.main .block-container {
    transition: opacity 0.12s ease-in-out;
}

[data-testid="stAppViewContainer"] {
    background: #F4F7FB;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 로그인 / 회원가입 기능
# =========================================================

USER_DB_PATH = "users.json"


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    if not os.path.exists(USER_DB_PATH):
        return {}
    with open(USER_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)




def make_logo_transparent(filename="logo.png"):
    """
    app.py와 같은 폴더의 logo.png를 읽어서
    흰색/밝은 배경을 투명하게 만든 logo_transparent.png를 생성합니다.
    logo.png가 없어도 앱이 중단되지 않도록 None을 반환합니다.
    """
    try:
        base_dir = Path(__file__).resolve().parent
    except Exception:
        base_dir = Path.cwd()

    src = base_dir / filename
    out = base_dir / "logo_transparent.png"

    if not src.exists():
        return None

    try:
        img = Image.open(src).convert("RGBA")
        pixels = []

        for r, g, b, a in img.getdata():
            # 흰색 또는 거의 흰색 배경을 투명 처리
            if r >= 235 and g >= 235 and b >= 235:
                pixels.append((255, 255, 255, 0))
            else:
                pixels.append((r, g, b, a))

        img.putdata(pixels)
        img.save(out, "PNG")
        return out

    except Exception:
        return src


def get_logo_base64(filename="logo.png"):
    logo_path = make_logo_transparent(filename)

    if logo_path is None:
        return ""

    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

def render_logo():
    logo_base64 = get_logo_base64("logo.png")

    if logo_base64:
        logo_html = (
            '<div class="sidebar-logo-wrap">'
            f'<img class="sidebar-logo-img" src="data:image/png;base64,{logo_base64}">'
            '<div class="sidebar-logo-text">AI 자세 분석 서비스</div>'
            '</div>'
        )
    else:
        logo_html = (
            '<div class="sidebar-logo-fallback-wrap">'
            '<div class="sidebar-logo-fallback">F</div>'
            '<div class="sidebar-logo-text">AI 자세 분석 서비스</div>'
            '</div>'
        )

    st.sidebar.markdown(
        f"""
<style>
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
    margin-top: -54px !important;
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: 0 !important;
}}

.sidebar-header {{
    width: 100%;
    padding: 0 0 10px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 6px;
    text-align: center;
}}

.sidebar-logo-wrap {{
    position: relative;
    width: 100%;
    height: 245px;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    overflow: hidden;
}}

.sidebar-logo-img {{
    width: 255px;
    height: 255px;
    object-fit: contain;
    display: block;
}}

.sidebar-logo-text {{
    position: absolute;
    left: 50%;
    top: 74%;
    transform: translateX(-50%);
    font-size: 24px;
    font-weight: 900;
    color: #FFFFFF;
    white-space: nowrap;
    letter-spacing: -0.7px;
    text-align: center;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}}

.sidebar-logo-fallback-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.sidebar-logo-fallback {{
    width: 120px;
    height: 120px;
    border-radius: 28px;
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 46px;
    font-weight: 900;
    margin-bottom: 6px;
}}
</style>

<div class="sidebar-header">
    {logo_html}
</div>
""",
        unsafe_allow_html=True,
    )

def page_header(title, subtitle):
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-sub'>{subtitle}</div>", unsafe_allow_html=True)


def metric_card(value, label, color="#185FA5"):
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-value" style="color:{color}">{value}</div>
    <div class="metric-label">{label}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def status_badge(is_good):
    if is_good:
        return "<span class='fit-badge badge-green'>GOOD</span>"
    return "<span class='fit-badge badge-red'>BAD</span>"


def value_to_bar_width(key, raw):
    if raw is None:
        return 12

    ranges = {
        "CVA": 40,
        "TIA": 30,
        "무릎": 180,
        "손목": 30,
        "시선각": 45,
        "책상높이": 0.35,
        "등받이": 0.5,
    }

    max_v = ranges.get(key, 100)
    width = min(max(float(raw) / max_v * 100, 8), 100)
    return width


def render_result_rows(data):
    html = ""
    for key, (value, is_good, raw) in data.items():
        width = value_to_bar_width(key, raw)
        bar_class = "bar-green" if is_good else "bar-red"
        html += f"""
<div class="result-row">
    <div class="result-name">{FEEDBACK[key]["label"]}</div>
    <div class="result-value">{value}</div>
    <div class="bar-wrap">
        <div class="bar {bar_class}" style="width:{width}%"></div>
    </div>
    {status_badge(is_good)}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_feedback_cards(data):
    for key, (value, is_good, raw) in data.items():
        fb = FEEDBACK[key]
        msg = fb["good"] if is_good else fb["bad"]
        cls = "feedback-good" if is_good else "feedback-bad"
        badge = status_badge(is_good)

        st.markdown(
            f"""
<div class="feedback-card {cls}">
    <div class="feedback-top">
        <div class="feedback-name">{fb["no"]}. {fb["label"]} <span style="color:#667085;font-size:12px;">({fb["eng"]})</span></div>
        {badge}
    </div>
    <div style="font-size:12px;color:#98A2B3;margin-bottom:6px;">정상 범위: {fb["range"]} · 측정값: {value}</div>
    <div class="feedback-msg">{msg}</div>
</div>
""",
            unsafe_allow_html=True,
        )


CLINICAL_RULES = {
    "CVA": {
    "normal": "0° ~ 20°",
    "risk": "20° 초과",
    "basis": "RULA Neck Zone 및 VDT 화면 상단-눈높이/하방 시선 기준",
},

"TIA": {
    "normal": "0° ~ 20°",
    "risk": "20° 초과",
    "basis": "RULA Trunk Zone 및 등받이 지지 기준",
},
    "무릎": {
        "normal": "85° ~ 100°",
        "caution": "해당 없음",
        "risk": "85° 미만 또는 100° 초과",
        "basis": "VDT 무릎 내각 90° 전후 및 하지 지지 기준",
    },
    "손목": {
        "normal": "±15° 이내 손목 중립 자세 유지",
        "caution": "해당 없음",
        "risk": "±15° 초과",
        "basis": "RULA Wrist Zone 및 손목 중립 ±15° 기준",
    },
    "시선각": {
        "normal": "하방 10° ~ 15°",
        "caution": "해당 없음",
        "risk": "10° 미만 또는 15° 초과",
        "basis": "VDT 수평 하방 10~15° 시선 기준",
    },
    "책상높이": {
        "normal": "팔꿈치-책상면 차이 0 ~ 0.05",
        "caution": "해당 없음",
        "risk": "0.05 초과",
        "basis": "팔꿈치와 책상면 수평 정렬, ±5%/±10% 허용 기준",
    },
    "등받이": {
        "normal": "골반너비 20% 이내",
        "caution": "해당 없음",
        "risk": "20% 초과",
        "basis": "VDT 의자 깊숙이 착석 및 RULA Trunk 지지조건 기준",
    },
}


DISPLAY_METRIC_ORDER = [
    "CVA",
    "TIA",
    "무릎",
    "손목",
    "시선각",
    "책상높이",
    "등받이",
]


def is_three_level_metric(key):
    return False


def get_range_text_html(key, line_break="<br/>"):
    """모든 지표를 정상/위험 2단계 기준으로 표시합니다."""
    rule = CLINICAL_RULES.get(key, {})
    return (
        f"정상: {rule.get('normal', '-')}"
        f"{line_break}위험: {rule.get('risk', '-')}"
    )


def classify_posture_level(key, raw):
    if raw is None:
        return "제외"

    raw = float(raw)

    
    if key == "CVA":
        return "정상" if 0 <= raw <= 20 else "위험"

    if key == "TIA":
        return "정상" if 0 <= raw <= 20 else "위험"

    
    if key == "무릎":
        return "정상" if 85 <= raw <= 100 else "위험"

    if key == "손목":
        # 손목 중립 자세 기준: 측정값 자체를 중립에서 벗어난 각도로 보고 ±15° 이내를 정상으로 판정
        return "정상" if -15 <= raw <= 15 else "위험"

    if key == "시선각":
        return "정상" if 10 <= raw <= 15 else "위험"

    if key == "책상높이":
        return "정상" if raw <= 0.05 else "위험"

    if key == "등받이":
        return "정상" if raw <= 0.20 else "위험"

    return "정상"


def level_to_style(level):
    if level == "정상":
        return {
            "label": "정상",
            "class": "status-good",
            "color": "#45B86B",
            "desc": "양호",
            "score": 10,
            "marker": 17,
        }


    if level == "위험":
        return {
            "label": "위험",
            "class": "status-risk",
            "color": "#F2527D",
            "desc": "관리 필요",
            "score": 2,
            "marker": 83,
        }

    return {
        "label": "제외",
        "class": "status-none",
        "color": "#AEB6C2",
        "desc": "기준점 부족",
        "score": None,
        "marker": 50,
    }


def metric_status_for_card(key, is_good, raw):
    level = classify_posture_level(key, raw)
    return level_to_style(level)


def get_metric_range_html(key):
    """카드 안에 CVA/TIA는 3분류, 나머지는 2분류 기준을 표시합니다."""
    rule = CLINICAL_RULES.get(key, {})

    if is_three_level_metric(key):
        return f"""
        <div class="pretty-range-box">
            <div><b class="range-good">정상:</b> {rule.get("normal", "-")}</div>
            <div><b class="range-risk">위험:</b> {rule.get("risk", "-")}</div>
        </div>
        """

    return f"""
    <div class="pretty-range-box">
        <div><b class="range-good">정상:</b> {rule.get("normal", "-")}</div>
        <div><b class="range-risk">위험:</b> {rule.get("risk", "-")}</div>
    </div>
    """


def gauge_percent(key, raw):
    level = classify_posture_level(key, raw)
    return level_to_style(level)["marker"]


def calculate_clinical_score_from_items(posture, env):
    all_data = {**posture, **env}
    scores = []
    level_counts = {"정상": 0, "주의": 0, "위험": 0, "제외": 0}

    for key in DISPLAY_METRIC_ORDER:
        if key not in all_data:
            continue
        _, _, raw = all_data[key]
        level = classify_posture_level(key, raw)
        level_counts[level] += 1
        score = level_to_style(level)["score"]
        if score is not None:
            scores.append(score)

    final_score = round(sum(scores) / len(scores), 1) if scores else 0

    if final_score >= 8:
        risk = "양호"
    elif final_score >= 5:
        risk = "주의"
    else:
        risk = "위험"

    return final_score, risk, level_counts




def gauge_percent(key, raw):
    level = classify_posture_level(key, raw)
    return level_to_style(level)["marker"]

def image_to_base64_src(path):
    try:
        path = Path(path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def metric_icon_svg(key, color=None):
    base_dir = Path(__file__).resolve().parent

    icon_map = {
        "CVA": base_dir / "assets" / "metric_icons" / "cva.png",
        "TIA": base_dir / "assets" / "metric_icons" / "tia.png", 
        "무릎": base_dir / "assets" / "metric_icons" / "knee.png",
        "손목": base_dir / "assets" / "metric_icons" / "wrist.png",
        "시선각": base_dir / "assets" / "metric_icons" / "gaze.png",
        "책상높이": base_dir / "assets" / "metric_icons" / "desk.png",
        "등받이": base_dir / "assets" / "metric_icons" / "chair.png",
    }

    img_src = image_to_base64_src(icon_map.get(key, ""))

    if img_src:
        return f'<img src="{img_src}" class="metric-img-icon">'

    return '<div class="metric-img-placeholder">이미지 없음</div>'

def image_to_base64_src(path):
    try:
        path = Path(path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except:
        return ""

def render_pretty_7_metric_dashboard(result):
    all_data = {**result["posture"], **result["env"]}

    posture_keys = ["CVA", "TIA", "무릎", "손목"]
    env_keys = ["시선각", "책상높이", "등받이"]

    posture_cards_html = ""
    env_cards_html = ""

    metric_order = posture_keys + env_keys

    for idx, key in enumerate(metric_order, start=1):
        if key not in all_data:
            continue

        value, is_good, raw = all_data[key]
        fb = FEEDBACK[key]

        display_raw = raw
        status = metric_status_for_card(key, is_good, raw)
        icon = metric_icon_svg(key, status["color"])

        bar_configs = {
            "CVA": {
                "min": 0, "max": 40, "normal_min": 0, "normal_max": 20,
                "ticks": [(0, "0°"), (20, "20°"), (40, "40°")],
            },
            "TIA": {
                "min": 0, "max": 45, "normal_min": 0, "normal_max": 20,
                "ticks": [(0, "0°"), (20, "20°"), (45, "45°")],
            },
            "무릎": {
                "min": 60, "max": 120, "normal_min": 85, "normal_max": 100,
                "ticks": [(85, "85°"), (100, "100°")],
            },
            "손목": {
                "min": -30, "max": 30, "normal_min": -15, "normal_max": 15,
                "ticks": [(-15, "-15°"), (0, "중립"), (15, "+15°")],
            },
            "시선각": {
                "min": 0, "max": 25, "normal_min": 10, "normal_max": 15,
                "ticks": [(10, "10°"), (15, "15°")],
            },
            "책상높이": {
                "min": -0.10, "max": 0.10, "normal_min": 0, "normal_max": 0.05,
                "ticks": [(0, "0"), (0.05, "0.05")],
            },
            "등받이": {
                "min": 0, "max": 0.40, "normal_min": 0, "normal_max": 0.20,
                "ticks": [(0, "0%"), (0.20, "20%")],
            },
        }

        bar_cfg = bar_configs[key]

        bar_min = bar_cfg["min"]
        bar_max = bar_cfg["max"]
        bar_span = bar_max - bar_min if bar_max != bar_min else 1

        marker_raw = display_raw if display_raw is not None else bar_min
        marker_value = min(max(float(marker_raw), bar_min), bar_max)
        percent = (marker_value - bar_min) / bar_span * 100

        normal_left = (bar_cfg["normal_min"] - bar_min) / bar_span * 100
        normal_width = (
            (bar_cfg["normal_max"] - bar_cfg["normal_min"]) / bar_span * 100
        )

        marker_color = status["color"]

        tick_html = ""
        for tick_value, tick_label in bar_cfg["ticks"]:
            tick_left = (tick_value - bar_min) / bar_span * 100

            tick_html += f"""
                <div class="report-bar-tick" style="left:{tick_left}%;">
                    <div class="report-bar-tick-line"></div>
                    <div class="report-bar-tick-label">{tick_label}</div>
                </div>
            """

        gauge_html = f"""
        <div class="report-bar-wrap">
            <div class="report-bar-track">
                <div
                    class="report-bar-normal"
                    style="left:{normal_left}%; width:{normal_width}%;"
                ></div>

                <div
                    class="report-bar-marker"
                    style="left:{percent}%; border-color:{marker_color};"
                ></div>
            </div>

            <div class="report-bar-ticks">
                {tick_html}
            </div>
        </div>
        """

        msg = fb["good"] if is_good else fb["bad"]
        msg_html = msg.replace("\n", "<br>")

        card_html = f"""
        <div class="pretty-metric-card">
            <div class="pretty-card-top">
                <div class="pretty-card-title {status["class"]}">
                    {idx}. {fb["label"]}
                </div>
                <div class="pretty-card-eng">{fb["eng"]}</div>
            </div>

            <div class="pretty-icon">
                {icon}
            </div>

            <div class="pretty-value-bg"
                 style="background:linear-gradient(180deg, {status["color"]}18, {status["color"]}08);">
                <div class="pretty-value" style="color:{status["color"]};">
                    {value}
                </div>

                <div class="pretty-status {status["class"]}">
                    {status["label"]}
                </div>
            </div>

            {gauge_html}

            <div class="pretty-feedback-box">
                <div class="pretty-feedback-title">맞춤 피드백</div>
                <div class="pretty-feedback-text">{msg_html}</div>
            </div>
        </div>
        """

        if key in posture_keys:
            posture_cards_html += card_html
        else:
            env_cards_html += card_html


    score = result.get("score", 0)
    risk = result.get("risk", "-")
    good_count = result.get("good_count", 0)
    total_count = result.get("total_count", 0)
    caution_count = max(total_count - good_count, 0)
    measured_time = datetime.datetime.now().strftime("%Y.%m.%d %H:%M")

    try:
        score_float = float(score)
    except Exception:
        score_float = 0.0
    score_deg = max(0, min(score_float * 36, 360))

    bad_items = []
    good_items = []
    for key in metric_order:
        if key not in all_data or key not in FEEDBACK:
            continue
        value, is_good, raw = all_data[key]
        if raw is None:
            continue
        if is_good:
            good_items.append(f"{FEEDBACK[key]['label']} {value}로 안정적인 범위예요.")
        else:
            first_guide = FEEDBACK[key].get("bad", "기준을 벗어났어요.").split("\n")[0]
            bad_items.append(f"{FEEDBACK[key]['label']} {value} - {first_guide}")

    if not bad_items:
        bad_items = ["현재 개선이 필요한 핵심 항목이 거의 없어요."]
    if not good_items:
        good_items = ["측정 가능한 양호 항목이 부족해요."]

    bad_items_html = "".join([f"<li>{item}</li>" for item in bad_items[:4]])
    good_items_html = "".join([f"<li>{item}</li>" for item in good_items[:4]])

    exercise_map = {
        "CVA": "목 스트레칭",
        "TIA": "허리 스트레칭",
        "무릎": "하체 스트레칭",
        "손목": "손목 스트레칭",
        "시선각": "목 스트레칭",
        "책상높이": "어깨 이완",
        "등받이": "허리 스트레칭",
    }
    recommend = []
    for key in metric_order:
        if key in all_data:
            _, is_good, raw = all_data[key]
            if raw is not None and not is_good:
                item = exercise_map.get(key)
                if item and item not in recommend:
                    recommend.append(item)
    for default_item in ["목 스트레칭", "허리 스트레칭", "손목 스트레칭"]:
        if len(recommend) >= 3:
            break
        if default_item not in recommend:
            recommend.append(default_item)

    # Streamlit의 st.markdown HTML <img>에서는 로컬 상대경로가 깨질 수 있으므로
    # PNG 파일을 base64(data URI)로 변환해서 넣습니다.
    stretch_image_map = {
        "하체 스트레칭": image_to_base64_src(Path(BASE_DIR) / "assets" / "stretch" / "lower_body.png"),
        "손목 스트레칭": image_to_base64_src(Path(BASE_DIR) / "assets" / "stretch" / "wrist.png"),
        "목 스트레칭": image_to_base64_src(Path(BASE_DIR) / "assets" / "stretch" / "neck.png"),
        "허리 스트레칭": image_to_base64_src(Path(BASE_DIR) / "assets" / "stretch" / "waist.png"),
        "어깨 이완": image_to_base64_src(Path(BASE_DIR) / "assets" / "stretch" / "shoulder.png"),
    }

    def stretch_img_html(label):
        img_src = stretch_image_map.get(label) or stretch_image_map.get("목 스트레칭", "")
        if img_src:
            return f'<img class="stretch-img" src="{img_src}" alt="{label}">'
        return '<div class="stretch-img-missing">이미지 없음</div>'

    stretch_items_html = "".join([
        (
            f'<div class="jasee-stretch-chip-wrapper">'
            f'<div class="jasee-stretch-chip">'
            f'{stretch_img_html(label)}'
            f'<div>{_html_escape(label)}</div>'
            f'</div>'
            f'</div>'
        )
        for label in recommend[:3]
    ])

    goal_items = []
    for key in metric_order:
        if key in all_data:
            _, is_good, raw = all_data[key]
            if raw is not None and not is_good:
                if key == "CVA":
                    goal_items.append("모니터 높이를 눈높이에 맞추기")
                elif key == "TIA":
                    goal_items.append("골반을 의자 뒤쪽까지 넣고 등받이에 허리 밀착하기")
                elif key == "무릎":
                    goal_items.append("무릎이 90° 전후가 되도록 의자 높이와 발 위치 조정하기")
                elif key == "손목":
                    goal_items.append("손목 받침대 사용하고 키보드 앞 공간 15cm 이상 확보하기")
                elif key == "시선각":
                    goal_items.append("모니터 상단을 눈높이에 맞추고 화면 거리 40cm 이상 확보하기")
                elif key == "책상높이":
                    goal_items.append("팔꿈치와 책상면이 수평이 되도록 책상 또는 의자 높이 조정하기")
                elif key == "등받이":
                    goal_items.append("의자 깊숙이 앉고 요추 부위를 등받이에 밀착하기")

    for default_goal in [
        "50분 작업 후 5분 스트레칭하기",
        "목과 어깨를 천천히 돌려 긴장 완화하기",
        "손목이 꺾이지 않도록 키보드와 마우스 위치 조정하기",
        "발바닥이 바닥에 닿는지 확인하기",
    ]:
        if len(goal_items) >= 4:
            break
        if default_goal not in goal_items:
            goal_items.append(default_goal)

    goals_html = "".join([f"<li>{goal}</li>" for goal in goal_items[:4]])

    css_head = """
    <html>
    <head>
    <style>
    body {
        margin:0;
        padding:0;
        font-family:'Pretendard', Arial, sans-serif;
        background:transparent;
        color:#0F1E36;
    }

    .pretty-dashboard {
        background:#FFFFFF;
        border:1px solid #E2ECF6;
        border-radius:20px;
        padding:18px 14px 18px 14px;
        box-shadow:0 12px 34px rgba(37,99,235,0.04);
        box-sizing:border-box;
    }

    .pretty-dashboard-title {
        text-align:center;
        font-size:30px;
        font-weight:900;
        color:#0F1E36;
        letter-spacing:-1px;
        margin-bottom:6px;
    }

    .pretty-dashboard-sub {
        text-align:center;
        font-size:14px;
        color:#5E718D;
        margin-bottom:18px;
    }

    .pretty-section-title{
        font-size:28px;
        font-weight:900;
        color:#0F1E36;
        margin:10px 0 18px 4px;
        letter-spacing:-0.7px;
    }
    
    .pretty-legend {
        display:flex;
        justify-content:center;
        gap:20px;
        align-items:center;
        font-size:13px;
        color:#5E718D;
        margin-bottom:26px;
    }

    .legend-dot {
        width:11px;
        height:11px;
        border-radius:50%;
        display:inline-block;
        margin-right:6px;
    }

    .pretty-grid {
        display:grid;
        grid-template-columns:repeat(4, minmax(0, 1fr));
        gap:16px;
    }

    .env-grid {
        grid-template-columns:repeat(3, minmax(0, 1fr));
    }

    .pretty-metric-card,
    .pretty-guide {
        background:#FFFFFF;
        border:1px solid #E2ECF6;
        border-radius:20px;
        padding:18px;
        box-shadow:0 10px 26px rgba(37,99,235,0.025);
        min-height:430px;
        box-sizing:border-box;
        transition:all 0.18s ease;
    }

    .pretty-metric-card:hover {
        transform:translateY(-4px);
        box-shadow:0 16px 34px rgba(37,99,235,0.08);
        border-color:rgba(37, 99, 235, 0.25);
    }

    .pretty-card-top {
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:8px;
        margin-bottom:10px;
    }

    .pretty-card-title {
        font-size:22px;
        font-weight:900;
        letter-spacing:-0.4px;
        line-height:1.2;
    }

    .pretty-card-eng {
        font-size:11px;
        color:#94A3B8;
        font-weight:700;
    }

    .status-good { color:#10B981; }
    .status-warn { color:#6366F1; }
    .status-risk { color:#EF4444; }
    .status-none { color:#94A3B8; }

    .pretty-icon {
        height:110px;
        display:flex;
        align-items:center;
        justify-content:center;
        margin:8px 0 14px 0;
        background:linear-gradient(180deg, #F8FAFC, #FFFFFF);
        border-radius:18px;
    }

    .metric-img-icon {
        width:132px;
        height:102px;
        object-fit:contain;
        display:block;
        filter:drop-shadow(0 8px 12px rgba(37,99,235,0.1));
    }

    .metric-img-placeholder {
        font-size:12px;
        color:#94A3B8;
        background:#F1F5F9;
        padding:10px 14px;
        border-radius:999px;
    }

    .pretty-value-bg {
        width:150px;
        height:76px;
        border-radius:90px 90px 0 0;
        margin:0 auto 12px auto;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        border-bottom:1px solid #E2ECF6;
    }

    .pretty-value {
        font-size:31px;
        font-weight:950;
        line-height:1;
    }

    .pretty-status {
        margin-top:8px;
        font-size:14px;
        font-weight:900;
    }

    .pretty-gauge-track {
        position:relative;
        height:12px;
        border-radius:999px;
        display:flex;
        background:#F1F5F9;
    }

    .pretty-zone {
        height:12px;
    }

    .zone-good {
        width:34%;
        background:#10B981;
        border-radius:999px 0 0 999px;
    }

    .zone-warn {
        width:33%;
        background:#6366F1;
    }

    .zone-risk {
        width:33%;
        background:#EF4444;
        border-radius:0 999px 999px 0;
    }

    .zone-good-two {
        width:50%;
        background:#10B981;
        border-radius:999px 0 0 999px;
    }

    .zone-risk-two {
        width:50%;
        background:#EF4444;
        border-radius:0 999px 999px 0;
    }

    .pretty-marker {
        position:absolute;
        top:-5px;
        width:10px;
        height:22px;
        border-radius:999px;
        transform:translateX(-50%);
        box-shadow:0 3px 8px rgba(0,0,0,0.18);
    }

    .pretty-range {
        display:flex;
        justify-content:space-between;
        font-size:10.5px;
        color:#5E718D;
        margin-top:7px;
    }

    .pretty-desc {
        background:#F8FAFC;
        border-radius:12px;
        padding:10px 11px;
        margin-top:12px;
        font-size:12px;
        line-height:1.6;
        color:#5E718D;
    }

    .pretty-range-box {
        display:flex;
        flex-direction:column;
        gap:4px;
        font-size:12px;
        line-height:1.55;
        color:#475569;
    }

    .range-good {
        color:#10B981;
        font-weight:900;
    }

    .range-warn {
        color:#6366F1;
        font-weight:900;
    }

    .range-risk {
        color:#EF4444;
        font-weight:900;
    }

    .current-status {
        margin-top:8px;
        padding-top:8px;
        border-top:1px solid #E2ECF6;
        font-size:12px;
        color:#5E718D;
    }

    .pretty-feedback-box {
        margin-top:12px;
        background:#FEF2F2;
        border-left:4px solid #EF4444;
        border-radius:12px;
        padding:11px 12px;
    }

    .pretty-feedback-title {
        font-size:12px;
        font-weight:900;
        color:#0F1E36;
        margin-bottom:6px;
    }

    .pretty-feedback-text {
        font-size:12.5px;
        line-height:1.65;
        color:#5E718D;
    }

    .pretty-guide {
        background:linear-gradient(135deg, #EFF6FF, #FFFFFF);
    }

    .pretty-guide-title {
        font-size:17px;
        font-weight:900;
        color:#0F1E36;
        margin-bottom:14px;
    }

    .pretty-guide-row {
        display:flex;
        gap:10px;
        font-size:13px;
        line-height:1.7;
        color:#5E718D;
        margin-bottom:12px;
    }

    .report-bar-wrap {
        width: 100%;
        margin: 16px 0 14px 0;
        padding: 0 2px 22px 2px;
        box-sizing: border-box;
    }

    .report-bar-track {
        position: relative;
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: #EF4444;
        overflow: visible;
    }

    .report-bar-normal {
        position: absolute;
        top: 0;
        height: 8px;
        border-radius: 999px;
        background: #10B981;
        z-index: 1;
    }

    .report-bar-marker {
        position: absolute;
        top: 50%;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #FFFFFF;
        border: 4px solid #10B981;
        transform: translate(-50%, -50%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.20);
        z-index: 3;
        box-sizing: border-box;
    }

    .report-bar-ticks {
        position: relative;
        width: 100%;
        height: 30px;
        margin-top: 4px;
    }

    .report-bar-tick {
        position: absolute;
        top: 0;
        transform: translateX(-50%);
        text-align: center;
        white-space: nowrap;
    }

    .report-bar-tick-line {
        width: 1px;
        height: 13px;
        background: #CBD5E1;
        margin: 0 auto 2px auto;
    }

    .report-bar-tick-label {
        font-size: 11px;
        font-weight: 700;
        color: #2563EB;
        line-height: 1;
    }


    .result-head {
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:4px;
        margin-bottom:22px;
    }
    .result-title {
        font-size:24px;
        font-weight:950;
        color:#0F1E36;
        letter-spacing:-0.7px;
        margin-bottom:6px;
    }
    .result-sub {
        font-size:10px;
        color:#2563EB;
        font-weight:650;
        line-height:1.15;
        letter-spacing:-0.6px;
        white-space:nowrap;
    }
    .result-time {
        font-size:13px;
        color:#64748B;
        white-space:nowrap;
        padding-top:4px;
    }
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        align-items: center;
    }

   .score-card {
        grid-column: auto;
        display:flex;
        align-items:center;
        justify-content:center;
        height:100%;
    }
    .summary-card {
        min-width: 0;
    }
    .score-ring {
        width:108px;
        height:108px;
        border-radius:50%;
        background:conic-gradient(#2563EB 0deg var(--score-deg), #EFF6FF 0deg 360deg);
        display:flex;
        align-items:center;
        justify-content:center;
        position:relative;
    }
    .score-ring::after {
        content:"";
        position:absolute;
        width:84px;
        height:84px;
        border-radius:50%;
        background:#FFFFFF;
    }
    .score-inner {
        position:relative;
        z-index:2;
        text-align:center;
    }
    .score-label {
        font-size:12px;
        color:#334155;
        font-weight:850;
    }
    .score-number {
        font-size:28px;
        font-weight:950;
        line-height:1.05;
        color:#0F1E36;
    }
    .score-number span {
        font-size:15px;
        font-weight:850;
        color:#334155;
    }
    .risk-badge {
        display:inline-flex;
        margin-top:8px;
        padding:4px 12px;
        border-radius:999px;
        background:#EFF6FF;
        color:#2563EB;
        font-size:12px;
        font-weight:950;
    }
    .summary-card {
        min-height:108px;
        border-radius:18px;
        padding:16px;
        border:1px solid #EFF6FF;
        background:linear-gradient(135deg,#EFF6FF,#FFFFFF);
    }
    .summary-card.bad {
        border-color:#FEE2E2;
        background:linear-gradient(135deg,#FEF2F2,#FFFFFF);
    }
    .summary-card.good {
        border-color:#D1FAE5;
        background:linear-gradient(135deg,#ECFDF5,#FFFFFF);
    }
    .summary-icon {
        font-size:23px;
        margin-bottom:8px;
    }
    .summary-title {
        font-size:15px;
        font-weight:950;
        margin-bottom:13px;
    }
    .summary-title.bad { color:#EF4444; }
    .summary-title.good { color:#10B981; }
    .summary-count {
        font-size:28px;
        font-weight:950;
        color:#0F1E36;
        line-height:1;
    }
    .summary-count span { font-size:16px; }
    .summary-desc {
        margin-top:10px;
        color:#475569;
        font-size:10px;
        font-weight:750;
        line-height:1.2;
        white-space:nowrap;
        letter-spacing:-0.4px;
    }
    .section-row {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:16px;
        border-top:1px solid #E2ECF6;
        padding-top:18px;
        margin-top:4px;
    }
    .feedback-title-wrap {
        display:flex;
        align-items:center;
        gap:8px;
        font-size:18px;
        font-weight:950;
        color:#2563EB;
        margin:22px 0 12px 0;
        border-top:1px solid #E2ECF6;
        padding-top:18px;
    }
    .feedback-grid {
        display:grid;
        grid-template-columns:1.15fr 1fr 1fr;
        gap:14px;
    }
    .ai-feedback-box {
        min-height:150px;
        border-radius:16px;
        border:1px solid #E2ECF6;
        padding:16px 18px;
        background:#FFFFFF;
        box-sizing:border-box;
    }
    .ai-feedback-box.bad { background:linear-gradient(135deg,#FFFFFF,#FEF2F2); }
    .ai-feedback-box.good { background:linear-gradient(135deg,#FFFFFF,#ECFDF5); }
    .ai-feedback-box.goal { background:linear-gradient(135deg,#FFFFFF,#F8FAFC); }
    .ai-feedback-box.blue { background:linear-gradient(135deg,#FFFFFF,#EFF6FF); }
    .ai-feedback-subtitle {
        font-size:14px;
        font-weight:950;
        margin-bottom:10px;
    }
    .ai-feedback-subtitle.bad { color:#EF4444; }
    .ai-feedback-subtitle.good { color:#10B981; }
    .ai-feedback-subtitle.goal { color:#0F1E36; }
    .ai-feedback-subtitle.blue { color:#2563EB; }
    .ai-feedback-box ul { margin:0; padding-left:18px; }
    .ai-feedback-box li {
        margin:0 0 6px 0;
        font-size:12.5px;
        line-height:1.6;
        color:#475569;
    }
    .ai-feedback-box li::marker { color:#2563EB; }
    .stretch-row {
        display:flex;
        gap:10px;
        align-items:stretch;
    }
    .stretch-chip {
        flex:1;
        min-height:76px;
        border:1px solid #E2ECF6;
        border-radius:12px;
        background:#FFFFFF;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        gap:0;
        font-size:12px;
        font-weight:850;
        color:#334155;
        text-align:center;
    }
    .stretch-img {
        width:72px;
        height:72px;
        object-fit:contain;
        margin-bottom:4px;
        display:block;
    }
    .stretch-img-missing {
        width:72px;
        height:72px;
        border:1px dashed #CBD5E1;
        border-radius:10px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:9px;
        color:#94A3B8;
        margin-bottom:4px;
    }
    .more-link {
        text-align:center;
        margin-top:10px;
        color:#2563EB;
        font-size:13px;
        font-weight:950;
    }

    @media (max-width:1100px) {
        .pretty-grid, .feedback-grid {
            grid-template-columns:1fr !important;
        }

        .summary-grid {
            grid-template-columns: repeat(3, 1fr) !important;
            gap: 8px !important;
        }
        .pretty-dashboard { padding:16px 14px 18px 14px !important; border-radius:18px !important; }
        .pretty-dashboard-title, .result-title { font-size:22px !important; }
        .pretty-section-title { font-size:21px !important; }
        .pretty-metric-card, .pretty-guide { min-height:auto !important; padding:15px !important; }
        .pretty-card-title { font-size:18px !important; }
        .section-row { flex-direction:column !important; align-items:flex-start !important; gap:8px !important; }
        .pretty-legend { flex-wrap:wrap !important; justify-content:flex-start !important; gap:10px !important; }
        .summary-wrap { grid-template-columns:1fr !important; }
        .stretch-row { flex-direction:column !important; }

        /* 결과 iframe 안쪽 하단 여백 축소 */
        .pretty-dashboard {
            padding-bottom: 10px !important;
        }
        .feedback-title-wrap {
            margin: 18px 0 10px 0 !important;
            padding-top: 14px !important;
        }
        .feedback-grid {
            gap: 10px !important;
        }
        .ai-feedback-box {
            min-height: auto !important;
            padding: 14px 16px !important;
        }
        .ai-feedback-box li {
            margin-bottom: 4px !important;
            line-height: 1.5 !important;
        }
    }
    </style>
    </head>

    <body>
    <div class="pretty-dashboard">
        <div class="result-head">
            <div>
                <div class="result-title">자세 측정 결과</div>
                <div class="result-sub">AI가 분석한 7가지 자세 및 작업환경 지표입니다.</div>
            </div>
            <div class="result-time">측정 시간: {measured_time}</div>
        </div>

        <div class="summary-grid">
            <div class="score-card">
                <div class="score-ring" style="--score-deg:{score_deg:.1f}deg;">
                    <div class="score-inner">
                        <div class="score-label">종합 점수</div>
                        <div class="score-number">{score}<span> /10</span></div>
                        <div class="risk-badge">{risk}</div>
                    </div>
                </div>
            </div>
            <div class="summary-card bad">
                <div class="summary-icon">⚠️</div>
                <div class="summary-title bad">주의 항목</div>
                <div class="summary-count">{caution_count}<span> 개</span></div>
                <div class="summary-desc">개선이 필요한 항목</div>
            </div>
            <div class="summary-card good">
                <div class="summary-icon">🙂</div>
                <div class="summary-title good">양호 항목</div>
                <div class="summary-count">{good_count}<span> 개</span></div>
                <div class="summary-desc">올바른 자세 유지</div>
            </div>
        </div>

        <div class="section-row">
            <div class="pretty-section-title" style="margin:0;">자세 지표</div>
            <div class="pretty-legend" style="margin:0;">
                <span><span class="legend-dot" style="background:#EF4444;"></span>위험</span>
                <span><span class="legend-dot" style="background:#10B981;"></span>정상</span>
                <span><span class="legend-dot" style="background:#94A3B8;"></span>제외</span>
            </div>
        </div>
    """

    

    css_head = (css_head
        .replace("{measured_time}", str(measured_time))
        .replace("{score_deg:.1f}", f"{score_deg:.1f}")
        .replace("{score}", str(score))
        .replace("{risk}", str(risk))
        .replace("{caution_count}", str(caution_count))
        .replace("{good_count}", str(good_count))
    )

    html = css_head + f"""

    <div class="pretty-grid" style="margin-top:18px;">
        {posture_cards_html}
    </div>

    <div class="pretty-section-title" style="margin-top:34px;">
        작업환경 지표
    </div>

    <div class="pretty-grid">
        {env_cards_html}
    </div>

    <div class="feedback-title-wrap">🛡️ AI 맞춤 피드백</div>
    <div class="feedback-grid">
        <div class="ai-feedback-box bad">
            <div class="ai-feedback-subtitle bad">개선이 필요한 항목</div>
            <ul>{bad_items_html}</ul>
        </div>
        <div class="ai-feedback-box good">
            <div class="ai-feedback-subtitle good">잘하고 있는 항목</div>
            <ul>{good_items_html}</ul>
        </div>
        <div class="ai-feedback-box goal">
            <div class="ai-feedback-subtitle goal">오늘의 실천 목표</div>
            <ul>{goals_html}</ul>
        </div>
        
    </div>

    </div>
    </body>
    </html>
    """

    # 실시간/이미지 자세측정 결과 공통:
    # 5200px처럼 과도하게 큰 iframe 높이는 「오늘의 실천 목표」 아래에
    # 큰 빈 여백을 만들기 때문에, 모바일 세로 카드가 잘리지 않는 선에서 높이를 줄입니다.
    components.html(html, height=4300, scrolling=False)

    # =========================================================
    # 추천 운동 및 스트레칭 클릭 팝업
    # - components.html() 내부 iframe에서는 Streamlit 이벤트가 안정적으로 동작하지 않으므로
    #   운동 박스만 Streamlit 본문 HTML로 다시 렌더링합니다.
    # - 기존 카드 UI는 유지하고, 카드 전체 위에 투명 링크 버튼을 겹쳐 클릭 가능하게 만듭니다.
    # =========================================================
    all_exercise_labels = [
        "하체 스트레칭",
        "손목 스트레칭",
        "목 스트레칭",
        "허리 스트레칭",
        "어깨 이완",
    ]

    if "exercise_popup_label" not in st.session_state:
        st.session_state.exercise_popup_label = None

    # Streamlit 버튼을 투명 레이어처럼 카드 위에 겹쳐 사용합니다.
    # href/query_params를 사용하지 않기 때문에 클릭해도 로그인 화면으로 튕기지 않습니다.
    st.markdown(
        """
<style>
.jasee-exercise-box {
    min-height:0 !important;
    border-radius:16px;
    border:1px solid #E2ECF6;
    padding:14px 16px 12px 16px;
    background:linear-gradient(135deg,#FFFFFF,#EFF6FF);
    box-sizing:border-box;
    margin-top:0 !important;
    margin-bottom:12px !important;
}
.jasee-exercise-subtitle {
    font-size:14px;
    font-weight:950;
    margin-bottom:10px;
    color:#2563EB;
}
.jasee-stretch-chip {
    min-height:172px;
    border:1px solid #E2ECF6;
    border-radius:12px;
    background:#FFFFFF;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    gap:6px;
    font-size:12px;
    font-weight:850;
    color:#334155;
    text-align:center;
    box-sizing:border-box;
    transition:all .18s ease;
    margin-bottom:0 !important;
}
.jasee-stretch-chip .stretch-img {
    width:140px;
    height:140px;
    object-fit:contain;
    margin-bottom:4px;
    display:block;
}
.jasee-stretch-chip .stretch-img-missing {
    width:140px;
    height:140px;
    border:1px dashed #CBD5E1;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:9px;
    color:#94A3B8;
    margin-bottom:4px;
}
/* 운동 카드 투명 클릭 레이어: st.markdown의 div는 Streamlit 요소를 실제로 감싸지 못하므로
   같은 vertical block 안의 marker를 기준으로 버튼을 카드 위에 겹칩니다. */
div[data-testid="column"] > div[data-testid="stVerticalBlock"]:has(.jasee-stretch-click-target) div[data-testid="stButton"] {
    margin-top:-172px !important;
    height:172px !important;
    position:relative !important;
    z-index:30 !important;
}
div[data-testid="column"] > div[data-testid="stVerticalBlock"]:has(.jasee-stretch-click-target) div[data-testid="stButton"] button {
    height:172px !important;
    min-height:172px !important;
    width:100% !important;
    opacity:0 !important;
    background:transparent !important;
    border:0 !important;
    box-shadow:none !important;
    padding:0 !important;
    cursor:pointer !important;
}
div[data-testid="column"] > div[data-testid="stVerticalBlock"]:has(.jasee-stretch-click-target) div[data-testid="stButton"] button:hover,
div[data-testid="column"] > div[data-testid="stVerticalBlock"]:has(.jasee-stretch-click-target) div[data-testid="stButton"] button:focus,
div[data-testid="column"] > div[data-testid="stVerticalBlock"]:has(.jasee-stretch-click-target) div[data-testid="stButton"] button:active {
    opacity:0 !important;
    background:transparent !important;
    box-shadow:none !important;
    border:0 !important;
    transform:none !important;
}
div[data-testid="column"] > div[data-testid="stVerticalBlock"]:has(.jasee-stretch-click-target):hover .jasee-stretch-chip {
    transform:translateY(-2px);
    box-shadow:0 8px 18px rgba(37, 99, 235, 0.12);
    border-color:rgba(37, 99, 235, 0.35);
}
.jasee-stretch-chip:hover {
    transform:translateY(-2px);
    box-shadow:0 8px 18px rgba(37, 99, 235, 0.12);
    border-color:rgba(37, 99, 235, 0.35);
}
.jasee-more-button-wrap div[data-testid="stButton"] button {
    opacity:1 !important;
    height:38px !important;
    min-height:38px !important;
    background:transparent !important;
    color:#2563EB !important;
    box-shadow:none !important;
    border:0 !important;
    font-size:13px !important;
    font-weight:950 !important;
    padding:0 !important;
}
.jasee-more-button-wrap div[data-testid="stButton"] button:hover {
    text-decoration:underline !important;
    transform:none !important;
    box-shadow:none !important;
}

/* 자세 결과 iframe 아래 여백 최소화 */
div[data-testid="stIFrame"] {
    margin-bottom:0 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="jasee-exercise-box"><div class="jasee-exercise-subtitle">추천 운동 및 스트레칭</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(recommend[:3]))
    for col, label in zip(cols, recommend[:3]):
        with col:
            st.markdown(
                f'<div class="jasee-stretch-click-target"></div><div class="jasee-stretch-chip">{stretch_img_html(label)}</div>',
                unsafe_allow_html=True,
            )
            if st.button(label, key=f"exercise_overlay_btn_{label}", use_container_width=True):
                st.session_state.exercise_popup_label = label
                st.rerun()

    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
    if st.button("운동 더 보기 ›", key="exercise_more_overlay_btn", use_container_width=True):
        st.session_state.exercise_popup_label = "all"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    @st.dialog("맞춤 운동 추천")
    def show_exercise_rag_popup(label):
        if label == "all":
            query = """
            사용자의 자세측정 결과를 바탕으로
            하체 스트레칭, 손목 스트레칭, 목 스트레칭, 허리 스트레칭, 어깨 이완
            5가지 전체 운동을 추천해줘.

            각 항목별로 아래 형식으로 정리해줘.
            - 운동명
            - 운동 방법
            - 횟수 및 시간
            - 효과
            - 주의사항
            """
        else:
            query = f"""
            사용자의 자세측정 결과를 바탕으로 '{label}'에 맞는 운동과 스트레칭을 추천해줘.

            아래 형식으로 정리해줘.
            - 운동명
            - 운동 방법
            - 횟수 및 시간
            - 효과
            - 주의사항
            """

        with st.spinner("RAG 기반 운동 추천 생성 중..."):
            answer = process_rag_query(
                query=query,
                selected_func_id=5,
                history=[]
            )

        st.markdown(answer)

        if st.button("닫기", key=f"close_exercise_popup_{label}", use_container_width=True):
            st.session_state.exercise_popup_label = None
            st.rerun()

    clicked_exercise = st.session_state.get("exercise_popup_label")
    if clicked_exercise:
        if clicked_exercise == "all" or clicked_exercise in all_exercise_labels:
            show_exercise_rag_popup(clicked_exercise)


def init_history():
    # =========================================================
# Session State 초기화
# =========================================================

    if "history" not in st.session_state:
        st.session_state.history = []

    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "history" not in st.session_state:
        st.session_state.history = []


HISTORY_DB_PATH = "user_history.json"
CHALLENGE_DB_PATH = "challenge_results.json"


def load_json_file(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_current_username():
    return st.session_state.get("username", "익명")


def save_history_overlay_image(result, username, now_text):
    """측정 결과의 overlay 이미지를 파일로 저장하고, 측정이력에서 다시 보여줄 경로를 반환합니다."""
    overlay = result.get("overlay")
    if overlay is None:
        return result.get("image_path")

    try:
        history_dir = Path(BASE_DIR) / "history_images"
        history_dir.mkdir(parents=True, exist_ok=True)

        safe_user = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(username))
        safe_time = now_text.replace("-", "").replace(":", "").replace(" ", "_")
        file_path = history_dir / f"{safe_user}_{safe_time}_{datetime.datetime.now().strftime('%f')}.png"

        if isinstance(overlay, Image.Image):
            img = overlay.convert("RGB")
        else:
            arr = np.array(overlay)
            if arr.ndim == 2:
                img = Image.fromarray(arr).convert("RGB")
            else:
                if arr.shape[-1] == 4:
                    img = Image.fromarray(arr.astype("uint8"), "RGBA").convert("RGB")
                else:
                    img = Image.fromarray(arr.astype("uint8"), "RGB")

        img.save(file_path, "PNG")
        return str(file_path)
    except Exception:
        return result.get("image_path")


def save_history(result):
    username = get_current_username()
    histories = load_json_file(HISTORY_DB_PATH, {})

    if username not in histories:
        histories[username] = []

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    image_path = save_history_overlay_image(result, username, now)

    histories[username].insert(
        0,
        {
            "time": now,
            "source": result.get("source", "이미지 자세 분석"),
            "score": result["score"],
            "risk": result["risk"],
            "good": result["good_count"],
            "total": result["total_count"],
            "missing_items": result.get("missing_items", []),
            "image_path": image_path,
        },
    )

    save_json_file(HISTORY_DB_PATH, histories)


def load_challenge_results():
    if not os.path.exists(CHALLENGE_DB_PATH):
        return []

    try:
        with open(CHALLENGE_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_challenge_results(results):
    with open(CHALLENGE_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def sync_result_to_challenge(result):
    username = st.session_state.get("username", "익명")

    all_data = {**result.get("posture", {}), **result.get("env", {})}
    bad_items = [
        FEEDBACK[key]["label"]
        for key, (_, is_good, _) in all_data.items()
        if key in FEEDBACK and not is_good
    ]

    new_record = {
        "name": username,
        "score": result["score"],
        "risk": result["risk"],
        "good": result["good_count"],
        "total": result["total_count"],
        "bad_items": bad_items[:3],
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    results = load_challenge_results()

    # 같은 사용자는 최신 측정 결과 1개만 유지
    results = [r for r in results if r.get("name") != username]
    results.insert(0, new_record)

    save_challenge_results(results)

def render_measurement_coverage(result_or_history):
    """양호 지표가 몇 개 기준으로 계산됐는지 설명합니다."""
    missing_items = result_or_history.get("missing_items", []) or []
    total = result_or_history.get("total_count", result_or_history.get("total", 0))
    good = result_or_history.get("good_count", result_or_history.get("good", 0))

    if missing_items:
        missing_text = "<br>".join(
            [f"- {item['label']}: {item['reason']}" for item in missing_items]
        )
        badge = "일부 제외"
        badge_class = "badge-amber"
        body = (
            f"양호 지표는 <b>{good}/{total}</b>입니다.<br>"
            f"총 7개 항목 중 <b>{len(missing_items)}개 항목</b>은 사진에서 기준점이 부족해 계산에서 제외했습니다.<br><br>"
            f"<b>제외된 항목</b><br>{missing_text}"
        )
    else:
        badge = "전체 측정"
        badge_class = "badge-green"
        body = (
            f"양호 지표는 <b>{good}/{total}</b>입니다.<br>"
            f"총 7개 항목이 모두 인식되었고, 그중 <b>{good}개 항목</b>이 정상 범위로 판정되었습니다."
        )

    st.markdown(
        f"""
<div class="fit-card" style="padding:16px 18px;">
    <div class="fit-card-title" style="margin-bottom:8px;">
        <span>양호 지표 계산 기준</span>
        <span class="fit-badge {badge_class}">{badge}</span>
    </div>
    <div style="font-size:13px;line-height:1.8;color:#667085;">
{body}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ===============================
# 바른자세 챌린지 JSON 구조 오류 해결
# 기존 함수와 교체해서 복붙하세요
# ===============================

def load_challenge_results():
    data = load_json_file(CHALLENGE_DB_PATH, {})

    # 예전 버전(list 구조) 자동 변환
    if isinstance(data, list):
        converted = {}

        for item in data:
            name = item.get("name", "익명")
            score = item.get("score", 0)
            point = int(round(score * 10))

            if name not in converted:
                converted[name] = {
                    "name": name,
                    "total_point": 0,
                    "count": 0,
                    "records": []
                }

            converted[name]["total_point"] += point
            converted[name]["count"] += 1
            converted[name]["records"].append(
                {
                    "score": score,
                    "point": point,
                    "risk": item.get("risk", "-"),
                    "good": item.get("good", 0),
                    "total": item.get("total", 0),
                    "bad_items": item.get("bad_items", []),
                    "time": item.get("time", "-"),
                }
            )

        save_challenge_results(converted)
        return converted

    # 새 버전(dict 구조)
    if isinstance(data, dict):
        return data

    return {}


def save_challenge_results(results):
    save_json_file(CHALLENGE_DB_PATH, results)


def sync_result_to_challenge(result):
    username = get_current_username()

    all_data = {**result.get("posture", {}), **result.get("env", {})}

    bad_items = [
        FEEDBACK[key]["label"]
        for key, (_, is_good, _) in all_data.items()
        if key in FEEDBACK and not is_good
    ]

    point = int(round(result["score"] * 10))

    challenge = load_challenge_results()

    if username not in challenge:
        challenge[username] = {
            "name": username,
            "total_point": 0,
            "count": 0,
            "records": [],
        }

    challenge[username]["total_point"] += point
    challenge[username]["count"] += 1
    challenge[username]["records"].insert(
        0,
        {
            "score": result["score"],
            "point": point,
            "risk": result["risk"],
            "good": result["good_count"],
            "total": result["total_count"],
            "bad_items": bad_items[:3],
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    )

    save_challenge_results(challenge)


# =========================================================
# 5-1. 로그인/회원가입 화면 최종 모바일 수정 오버라이드
# =========================================================

def make_logo_transparent(filename="logo.png"):
    """
    logo.png의 흰 배경을 투명 처리하고, 이미지 주변의 큰 여백을 잘라
    로그인 화면에서 로고가 과하게 커지거나 아래로 밀리지 않게 만듭니다.
    """
    try:
        base_dir = Path(__file__).resolve().parent
    except Exception:
        base_dir = Path.cwd()

    src = base_dir / filename
    out = base_dir / "logo_transparent_cropped.png"

    if not src.exists():
        return None

    try:
        img = Image.open(src).convert("RGBA")
        new_pixels = []
        for r, g, b, a in img.getdata():
            if r >= 238 and g >= 238 and b >= 238:
                new_pixels.append((255, 255, 255, 0))
            else:
                new_pixels.append((r, g, b, a))
        img.putdata(new_pixels)

        alpha = img.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            img = img.crop(bbox)

        img.save(out, "PNG")
        return out
    except Exception:
        return src


def risk_style(is_good):
    if is_good:
        return {
            "label": "양호",
            "color": "#3B8C42",
            "badge": "badge-green",
            "emoji": "🟢",
        }
    return {
        "label": "관리 필요",
        "color": "#D94A4A",
        "badge": "badge-red",
        "emoji": "🔴",
    }


def get_priority_items(result):
    all_data = {**result["posture"], **result["env"]}
    bad_items = []

    for key, value in all_data.items():
        measured_value, is_good, raw = value
        if not is_good:
            bad_items.append((key, measured_value, raw))

    return bad_items


def render_dashboard():
    page_header(
        "나의 자세 현황 대시보드",
        "최근 자세 분석 결과를 바탕으로 위험 부위와 교정 우선순위를 확인합니다.",
    )

    result = st.session_state.get("latest_result", None)
    history = st.session_state.get("history", [])

    # 아직 측정 결과가 없는 경우
    if result is None:
        st.markdown(
            """
<div class="fit-card">
    <div class="fit-card-title">
        <span>아직 분석 결과가 없습니다</span>
        <span class="fit-badge badge-blue">Ready</span>
    </div>
    <div style="font-size:14px;line-height:1.8;color:#667085;">
        먼저 왼쪽 메뉴에서 <b style="color:#172033;">자세측정</b>을 실행하면,
        이 대시보드에 최근 자세 점수, 위험 부위, 교정 우선순위가 자동으로 표시됩니다.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    all_data = {**result["posture"], **result["env"]}
    bad_items = get_priority_items(result)

    good_rate = round(result["good_count"] / result["total_count"] * 100) if result["total_count"] else 0
    bad_count = result["total_count"] - result["good_count"]

    # 상단 핵심 지표
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(result["score"], "최근 자세 점수", "#2563EB")
    with c2:
        metric_card(result["risk"], "종합 위험도", "#F59E0B")
    with c3:
        metric_card(f"{bad_count}개", "관리 필요 지표", "#EF4444")
    with c4:
        metric_card(f"{good_rate}%", "정상 범위 비율", "#10B981")

    render_measurement_coverage(result)

    left, right = st.columns([1.15, 0.85])

    # 실제 신체 부위별 위험 현황
    with left:
        rows = ""

        label_map = {
            "CVA": "목·경추",
            "TIA": "몸통·허리",
            "팔꿈치": "팔꿈치",
            "무릎": "무릎",
            "손목": "손목",
            "시선각": "시선·모니터",
            "책상높이": "책상 높이",
            "등받이": "의자 등받이",
        }

        for key, (value, is_good, raw) in all_data.items():
            style = risk_style(is_good)
            width = value_to_bar_width(key, raw)

            rows += f"""
<div class="result-row">
    <div class="result-name">{label_map.get(key, key)}</div>
    <div class="result-value">{value}</div>
    <div class="bar-wrap">
        <div class="bar" style="width:{width}%; background:{style["color"]};"></div>
    </div>
    <span class="fit-badge {style["badge"]}">{style["label"]}</span>
</div>
"""

        st.markdown(
            f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>최근 측정 기반 신체 부위별 위험 현황</span>
        <span class="fit-badge badge-blue">Live Result</span>
    </div>
    {rows}
</div>
""",
            unsafe_allow_html=True,
        )

    # 교정 우선순위
    with right:
        if bad_items:
            priority_html = ""

            for i, (key, value, raw) in enumerate(bad_items[:3], start=1):
                fb = FEEDBACK[key]
                msg = fb["bad"].split("\n")[0]

                priority_html += f"""
<div style="padding:12px 0;border-bottom:1px solid #EEF2F6;">
    <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="font-size:14px;font-weight:800;color:#172033;">
            {i}. {fb["label"]}
        </div>
        <span class="fit-badge badge-red">{value}</span>
    </div>
    <div style="font-size:12.5px;color:#667085;line-height:1.6;margin-top:5px;">
        {msg}
    </div>
</div>
"""

            guide_title = "오늘의 교정 우선순위"
            guide_badge = "집중관리"
        else:
            priority_html = """
<div style="font-size:14px;line-height:1.8;color:#667085;">
    현재 모든 주요 지표가 정상 범위에 있습니다.<br>
    지금 자세를 유지하면서 50분마다 가벼운 스트레칭을 해주세요.
</div>
"""
            guide_title = "오늘의 자세 상태"
            guide_badge = "양호"

        st.markdown(
            f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>{guide_title}</span>
        <span class="fit-badge badge-amber">{guide_badge}</span>
    </div>
    {priority_html}
</div>
""",
            unsafe_allow_html=True,
        )

    # 중단: 최근 측정 이미지 + AI 요약
    st.markdown("### AI 분석 요약")

    img_col, summary_col = st.columns([1, 1])

    with img_col:
        if "overlay" in result:
            st.markdown(
                """
    <div class="fit-card">
        <div class="fit-card-title">
            <span>최근 AI 오버레이</span>
            <span class="fit-badge badge-green">Analyzed</span>
        </div>
    </div>
    """,
            unsafe_allow_html=True,
        )
        st.image(result["overlay"], use_container_width=True)

    with summary_col:
        render_ai_correction_comment(result)


def _init_realtime_state():
    defaults = {
        "rt_phase": "idle",
        "rt_ready_result": None,
        "rt_saved": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_realtime_feedback_card(snapshot):
    status = snapshot.get("last_status", "WAIT")
    elapsed = int(snapshot.get("elapsed", 0))
    remain = max(10 - elapsed, 0)
    good_hold = min(snapshot.get("good_hold_seconds", 0), 3)

    if status == "GOOD":
        badge_class = "badge-green"
        color = "#3B8C42"
    elif status == "BAD":
        badge_class = "badge-red"
        color = "#D94A4A"
    else:
        badge_class = "badge-amber"
        color = "#BA7517"

    st.markdown(
        f"""
<div class="fit-card" style="border-left:6px solid {color};">
    <div class="fit-card-title">
        <span>실시간 자세 판정</span>
        <span class="fit-badge {badge_class}">{status}</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
        <div style="padding:14px;border-radius:14px;background:#F8FAFC;text-align:center;">
            <div style="font-size:12px;color:#667085;">남은 촬영 시간</div>
            <div style="font-size:34px;font-weight:950;color:#172033;">{remain}초</div>
        </div>
        <div style="padding:14px;border-radius:14px;background:#F8FAFC;text-align:center;">
            <div style="font-size:12px;color:#667085;">GOOD 유지 시간</div>
            <div style="font-size:34px;font-weight:950;color:{color};">{good_hold:.1f}초</div>
        </div>
    </div>
    <div style="font-size:13px;line-height:1.8;color:#667085;">
        {snapshot.get("last_feedback", "측정 대기 중입니다.")}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_measurement_coverage(result_or_history):
    """양호 지표가 몇 개 기준으로 계산됐는지 설명합니다."""
    missing_items = result_or_history.get("missing_items", []) or []
    total = result_or_history.get("total_count", result_or_history.get("total", 0))
    good = result_or_history.get("good_count", result_or_history.get("good", 0))

    if missing_items:
        missing_text = "<br>".join(
            [f"- {item['label']}: {item['reason']}" for item in missing_items]
        )
        badge = "일부 제외"
        badge_class = "badge-amber"
        body = (
            f"양호 지표는 <b>{good}/{total}</b>입니다.<br>"
            f"총 7개 항목 중 <b>{len(missing_items)}개 항목</b>은 사진에서 기준점이 부족해 계산에서 제외했습니다.<br><br>"
            f"<b>제외된 항목</b><br>{missing_text}"
        )
    else:
        badge = "전체 측정"
        badge_class = "badge-green"
        body = (
            f"양호 지표는 <b>{good}/{total}</b>입니다.<br>"
            f"총 7개 항목이 모두 인식되었고, 그중 <b>{good}개 항목</b>이 정상 범위로 판정되었습니다."
        )

    st.markdown(
        f"""
<div class="fit-card" style="padding:16px 18px;">
    <div class="fit-card-title" style="margin-bottom:8px;">
        <span>양호 지표 계산 기준</span>
        <span class="fit-badge {badge_class}">{badge}</span>
    </div>
    <div style="font-size:13px;line-height:1.8;color:#667085;">
{body}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ===============================
# 바른자세 챌린지 JSON 구조 오류 해결
# 기존 함수와 교체해서 복붙하세요
# ===============================

def load_challenge_results():
    data = load_json_file(CHALLENGE_DB_PATH, {})

    # 예전 버전(list 구조) 자동 변환
    if isinstance(data, list):
        converted = {}

        for item in data:
            name = item.get("name", "익명")
            score = item.get("score", 0)
            point = int(round(score * 10))

            if name not in converted:
                converted[name] = {
                    "name": name,
                    "total_point": 0,
                    "count": 0,
                    "records": []
                }

            converted[name]["total_point"] += point
            converted[name]["count"] += 1
            converted[name]["records"].append(
                {
                    "score": score,
                    "point": point,
                    "risk": item.get("risk", "-"),
                    "good": item.get("good", 0),
                    "total": item.get("total", 0),
                    "bad_items": item.get("bad_items", []),
                    "time": item.get("time", "-"),
                }
            )

        save_challenge_results(converted)
        return converted

    # 새 버전(dict 구조)
    if isinstance(data, dict):
        return data

    return {}


def save_challenge_results(results):
    save_json_file(CHALLENGE_DB_PATH, results)


def sync_result_to_challenge(result):
    username = get_current_username()

    all_data = {**result.get("posture", {}), **result.get("env", {})}

    bad_items = [
        FEEDBACK[key]["label"]
        for key, (_, is_good, _) in all_data.items()
        if key in FEEDBACK and not is_good
    ]

    point = int(round(result["score"] * 10))

    challenge = load_challenge_results()

    if username not in challenge:
        challenge[username] = {
            "name": username,
            "total_point": 0,
            "count": 0,
            "records": [],
        }

    challenge[username]["total_point"] += point
    challenge[username]["count"] += 1
    challenge[username]["records"].insert(
        0,
        {
            "score": result["score"],
            "point": point,
            "risk": result["risk"],
            "good": result["good_count"],
            "total": result["total_count"],
            "bad_items": bad_items[:3],
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    )

    save_challenge_results(challenge)


# =========================================================
# 5-1. 로그인/회원가입 화면 최종 모바일 수정 오버라이드
# =========================================================

def make_logo_transparent(filename="logo.png"):
    """
    logo.png의 흰 배경을 투명 처리하고, 이미지 주변의 큰 여백을 잘라
    로그인 화면에서 로고가 과하게 커지거나 아래로 밀리지 않게 만듭니다.
    """
    try:
        base_dir = Path(__file__).resolve().parent
    except Exception:
        base_dir = Path.cwd()

    src = base_dir / filename
    out = base_dir / "logo_transparent_cropped.png"

    if not src.exists():
        return None

    try:
        img = Image.open(src).convert("RGBA")
        new_pixels = []
        for r, g, b, a in img.getdata():
            if r >= 238 and g >= 238 and b >= 238:
                new_pixels.append((255, 255, 255, 0))
            else:
                new_pixels.append((r, g, b, a))
        img.putdata(new_pixels)

        alpha = img.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            img = img.crop(bbox)

        img.save(out, "PNG")
        return out
    except Exception:
        return src


def render_auth_page():
    """
    로그인/회원가입 화면
    - 로그인 후 모바일 화면과 같은 430px 프레임 유지
    - 로고가 과하게 커지거나 위아래 여백이 생기는 문제 해결
    - auth 라벨 숨김
    - 로그인/회원가입 탭, 입력창, 버튼을 모바일 폭에 맞게 정렬
    """

    st.markdown(
        """
<style id="fitmeup-auth-final-fix">
html, body {
    background: #DCE7F5 !important;
    overflow-x: hidden !important;
}

.stApp,
[data-testid="stAppViewContainer"] {
    max-width: 520px !important;
    width: 100% !important;
    min-height: 100vh !important;
    margin: 0 auto !important;
    background: #F4F7FB !important;
    box-shadow: 0 0 0 1px rgba(15, 30, 54, 0.08), 0 20px 70px rgba(15, 30, 54, 0.18) !important;
    overflow-x: hidden !important;
}

[data-testid="stHeader"],
[data-testid="collapsedControl"],
[data-testid="stSidebar"],
section[data-testid="stSidebar"] {
    display: none !important;
}

.main .block-container,
.block-container {
    max-width: 520px !important;
    width: 100% !important;
    padding: 1.05rem 1rem 2rem 1rem !important;
    margin: 0 auto !important;
    box-sizing: border-box !important;
}

/* 이전 로그인 화면에서 생기던 상단 흰색 캡슐/불필요한 요소 숨김 */
.element-container:has(.auth-top-spacer) + .element-container,
.auth-top-spacer {
    display: none !important;
}

.auth-card {
    width: 100%;
    box-sizing: border-box;
    background: #FFFFFF;
    border: 1px solid #E2ECF6;
    border-radius: 24px;
    padding: 22px 18px 20px 18px;
    box-shadow: 0 14px 34px rgba(37, 99, 235, 0.08);
    margin: 0 auto;
}

.auth-logo-wrap {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 4px 0 8px 0;
}

.auth-logo-wrap img {
    width: 176px !important;
    max-width: 58% !important;
    height: auto !important;
    display: block !important;
    object-fit: contain !important;
}

.auth-brand-fallback {
    text-align: center;
    font-size: 32px;
    font-weight: 950;
    letter-spacing: -1px;
    color: #0F1E36;
    margin: 4px 0 2px 0;
}

.auth-guide-text {
    text-align: center;
    color: #667085;
    font-size: 12.5px;
    line-height: 1.55;
    margin: 6px 0 14px 0;
}

.auth-section-title {
    text-align: center;
    font-size: 20px;
    font-weight: 900;
    color: #172033;
    margin: 12px 0 12px 0;
}

/* Streamlit 라디오의 auth 라벨 완전 숨김 */
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] label[data-testid="stWidgetLabel"],
div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 로그인/회원가입 탭 */
div[data-testid="stRadio"] {
    width: 100% !important;
    margin: 0 0 12px 0 !important;
}

div[data-testid="stRadio"] > div {
    width: 100% !important;
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 8px !important;
    background: #EEF4FB !important;
    border: 1px solid #D7E3F2 !important;
    border-radius: 16px !important;
    padding: 5px !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

div[data-testid="stRadio"] label {
    width: 100% !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 10px 6px !important;
    border-radius: 12px !important;
    background: transparent !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    white-space: nowrap !important;
}

div[data-testid="stRadio"] label[data-checked="true"] {
    background: #FFFFFF !important;
    box-shadow: 0 6px 14px rgba(15, 30, 54, 0.08) !important;
}

div[data-testid="stRadio"] label p {
    font-size: 13.5px !important;
    font-weight: 850 !important;
    color: #0F1E36 !important;
    white-space: nowrap !important;
    margin: 0 !important;
}

/* 입력창/버튼 */
div[data-testid="stTextInput"],
div[data-testid="stButton"] {
    width: 100% !important;
    max-width: 100% !important;
}

div[data-testid="stTextInput"] label p {
    font-size: 13.5px !important;
    font-weight: 800 !important;
    color: #0F1E36 !important;
}

div[data-testid="stTextInput"] input {
    min-height: 45px !important;
    border-radius: 14px !important;
    font-size: 14px !important;
    background: #F8FAFC !important;
    border: 1px solid #E2ECF6 !important;
}

div[data-testid="stButton"] button {
    width: 100% !important;
    min-height: 47px !important;
    border-radius: 15px !important;
    font-size: 14px !important;
    font-weight: 900 !important;
}

.element-container {
    max-width: 100% !important;
    margin-bottom: 0.45rem !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    logo_path = make_logo_transparent("logo.png")

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    if logo_path is not None:
        logo_b64 = base64.b64encode(Path(logo_path).read_bytes()).decode("utf-8")
        st.markdown(
            f'<div class="auth-logo-wrap"><img src="data:image/png;base64,{logo_b64}" alt="자세히봐 logo"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="auth-brand-fallback">자세히봐</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="auth-guide-text">로그인 후 AI 자세 분석 서비스를 이용하세요.</div>',
        unsafe_allow_html=True,
    )

    auth_tab = st.radio(
        "",
        ["로그인", "회원가입"],
        horizontal=True,
        label_visibility="collapsed",
        key="auth_tab_fixed",
    )

    users = load_users()

    if auth_tab == "로그인":
        st.markdown('<div class="auth-section-title">로그인</div>', unsafe_allow_html=True)
        username = st.text_input("아이디", key="login_id")
        password = st.text_input("비밀번호", type="password", key="login_pw")

        if st.button("로그인", use_container_width=True):
            if username not in users:
                st.error("존재하지 않는 아이디입니다.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            if users[username]["password"] != hash_password(password):
                st.error("비밀번호가 일치하지 않습니다.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"{username}님, 로그인되었습니다.")
            st.rerun()
    else:
        st.markdown('<div class="auth-section-title">회원가입</div>', unsafe_allow_html=True)
        new_username = st.text_input("아이디", key="signup_id")
        new_password = st.text_input("비밀번호", type="password", key="signup_pw")
        new_password_check = st.text_input("비밀번호 확인", type="password", key="signup_pw_check")

        if st.button("회원가입", use_container_width=True):
            if not new_username or not new_password:
                st.error("아이디와 비밀번호를 입력해주세요.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            if new_username in users:
                st.error("이미 존재하는 아이디입니다.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            if new_password != new_password_check:
                st.error("비밀번호가 일치하지 않습니다.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            users[new_username] = {
                "password": hash_password(new_password),
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_users(users)
            st.success("회원가입이 완료되었습니다. 로그인해주세요.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 6. 웹 페이지용 상단 메뉴
# =========================================================

# 웹 페이지 화면에서는 상단 메뉴를 유지하고, 본문 폭과 그리드만 웹 화면에 맞게 확장합니다.

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    render_auth_page()
    st.stop()

if "username" not in st.session_state or st.session_state.username is None:
    st.session_state.username = "익명"

MENU_OPTIONS = [
    "📸 자세측정",
    "💬 AI 챗봇",
    "🧘 운동 및 스트레칭 추천",
    "📈 측정이력",
    "🧾 예상 영수증",
    "🎯 바른자세 챌린지",
    "📄 근골격계 리포트",
    "🛒 제품 추천",
]

st.markdown(
    f"""
<div class="web-app-header">
    <div class="web-app-logo">자세히봐</div>
    <div class="web-app-sub">AI 자세 분석 서비스</div>
    <div class="web-app-user">로그인 계정 · {st.session_state.username}</div>
</div>
""",
    unsafe_allow_html=True,
)

menu = st.selectbox(
    "메뉴 선택",
    MENU_OPTIONS,
    key="web_menu_select",
    label_visibility="collapsed",
)

if st.button("로그아웃", key="web_logout_btn", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

st.markdown('<div class="web-menu-spacer"></div>', unsafe_allow_html=True)



# =========================================================
# MOBILE APP MODE — Streamlit 화면을 모바일 앱 폭으로 강제 적용
# =========================================================
st.markdown(
    """
<style id="jasee-web-page-mode">

/* === WEB PAGE MODE: 기능은 그대로 두고 화면 폭/배치만 데스크톱용으로 확장 === */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
}

html, body {
    background: #F4F7FB !important;
    overflow-x: hidden !important;
}

[data-testid="stAppViewContainer"] {
    width: 100% !important;
    max-width: 100% !important;
    min-height: 100vh !important;
    margin: 0 !important;
    background: #F4F7FB !important;
    box-shadow: none !important;
    overflow-x: hidden !important;
}

[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
}

.main .block-container,
.block-container {
    width: 100% !important;
    max-width: 1240px !important;
    padding: 1.5rem 2.5rem 4rem 2.5rem !important;
    margin: 0 auto !important;
    box-sizing: border-box !important;
}

.web-app-header {
    width: 100%;
    padding: 26px 34px;
    margin: 0 0 18px 0;
    border-radius: 28px;
    background: linear-gradient(135deg, #0B1930 0%, #152A4A 55%, #2563EB 100%);
    color: #FFFFFF;
    box-shadow: 0 18px 42px rgba(15, 30, 54, 0.16);
    box-sizing: border-box;
    display: grid;
    grid-template-columns: 1fr auto;
    grid-template-areas: "logo user" "sub user";
    gap: 6px 28px;
    align-items: center;
}
.web-app-logo {
    grid-area: logo;
    font-size: 34px;
    font-weight: 950;
    letter-spacing: -1px;
    line-height: 1.05;
}
.web-app-sub {
    grid-area: sub;
    margin-top: 2px;
    font-size: 16px;
    font-weight: 800;
    color: #E2E8F0;
}
.web-app-user {
    grid-area: user;
    padding: 10px 16px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.13);
    border: 1px solid rgba(255, 255, 255, 0.18);
    font-size: 13px;
    font-weight: 800;
    color: #EAF2FF;
    white-space: nowrap;
}
.web-menu-spacer { height: 12px; }

/* 메뉴/로그아웃 영역은 웹 폭에서 너무 넓어지지 않게 정돈 */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    min-height: 52px !important;
    border-radius: 16px !important;
    border: 1px solid #D7E3F2 !important;
    background: #FFFFFF !important;
    font-weight: 800 !important;
}

div[data-testid="stButton"] button[kind] { white-space: nowrap !important; }

/* 웹에서는 Streamlit 컬럼을 원래처럼 가로 배치 */
div[data-testid="stHorizontalBlock"] {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 1.25rem !important;
    align-items: stretch !important;
}
div[data-testid="stHorizontalBlock"] > div,
[data-testid="column"] {
    min-width: 0 !important;
}

/* 데스크톱 그리드 확장 */
.metric-grid,
.summary-grid,
.history-grid {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
    gap: 18px !important;
}
.pretty-grid,
.env-grid,
.feedback-grid,
.product-grid,
.exercise-grid {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 18px !important;
}

.fit-card,
.metric-card,
.feedback-card,
.pretty-metric-card,
.pretty-guide {
    width: 100% !important;
    max-width: 100% !important;
    border-radius: 22px !important;
    padding: 22px !important;
    box-sizing: border-box !important;
    margin-bottom: 18px !important;
    overflow: visible !important;
}

.page-title {
    font-size: 34px !important;
    line-height: 1.25 !important;
    margin-top: 18px !important;
    margin-bottom: 8px !important;
}
.page-sub {
    font-size: 15px !important;
    margin-bottom: 24px !important;
}
.metric-value { font-size: 30px !important; }

.result-row {
    display: flex !important;
    grid-template-columns: none !important;
    gap: 12px !important;
    align-items: center !important;
}
.result-row .fit-badge { width: fit-content !important; }
.result-name { width: 104px !important; font-size: 13px !important; }
.result-value { width: 70px !important; font-size: 14px !important; }

img, video, canvas,
[data-testid="stImage"],
[data-testid="stImage"] img {
    max-width: 100% !important;
    height: auto !important;
    box-sizing: border-box !important;
}

/* 버튼/입력창 웹 스타일 */
div[data-testid="stButton"] button,
button[kind="primary"],
button[kind="secondary"] {
    min-height: 48px !important;
    border-radius: 14px !important;
    font-size: 15px !important;
    width: 100% !important;
}

div[data-testid="stTextInput"],
div[data-testid="stTextArea"],
div[data-testid="stSelectbox"],
div[data-testid="stFileUploader"] {
    width: 100% !important;
    max-width: 100% !important;
}

/* 탭/라디오: 웹에서는 가로 배치 유지 */
div[data-testid="stRadio"] > div {
    max-width: 100% !important;
    overflow-x: visible !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
}
div[data-testid="stRadio"] label { white-space: nowrap !important; }
div[data-testid="stRadio"] label p { white-space: nowrap !important; font-size: 14px !important; }

/* components.html iframe */
div[data-testid="stIFrame"],
div[data-testid="stIFrame"] iframe,
iframe[title="streamlit.components.v1.html"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow: visible !important;
    border: 0 !important;
}

.fit-table,
.ms-table,
.report-table,
table {
    width: 100% !important;
    max-width: 100% !important;
    table-layout: auto !important;
    word-break: keep-all !important;
}

p, span, div, li {
    word-break: keep-all;
    overflow-wrap: anywhere;
}

.pretty-dashboard-title,
.result-title { font-size: 28px !important; line-height: 1.25 !important; }
.pretty-section-title { font-size: 24px !important; line-height: 1.25 !important; }
.pretty-card-title { font-size: 18px !important; line-height: 1.25 !important; }
.pretty-value { font-size: 30px !important; }
.pretty-feedback-text,
.ai-feedback-box,
.summary-desc,
.result-sub { font-size: 13px !important; line-height: 1.65 !important; }

/* 로그인 화면은 웹에서도 카드형으로 중앙 배치 */
.auth-card {
    max-width: 520px !important;
    margin: 36px auto !important;
}
.auth-guide-text,
.auth-section-title { text-align: center !important; }

.element-container { max-width: 100% !important; margin-bottom: 0.5rem !important; }
div[data-testid="stVerticalBlock"] { gap: 0.55rem !important; }

@media (max-width: 900px) {
    .main .block-container,
    .block-container { padding: 1rem 1rem 3rem 1rem !important; }
    .web-app-header {
        display: block;
        text-align: center;
        padding: 18px 16px;
        border-radius: 22px;
    }
    .web-app-logo { font-size: 25px; }
    .web-app-sub { font-size: 14px; margin-top: 5px; }
    .web-app-user { display: inline-block; margin-top: 12px; font-size: 12px; }
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    div[data-testid="stHorizontalBlock"] > div,
    [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
    .metric-grid,
    .pretty-grid,
    .env-grid,
    .feedback-grid,
    .summary-grid,
    .product-grid,
    .exercise-grid,
    .history-grid { grid-template-columns: 1fr !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# =========================================================
# 7. 페이지 함수형 렌더링 구조
# =========================================================

def risk_style(is_good):
    if is_good:
        return {
            "label": "양호",
            "color": "#3B8C42",
            "badge": "badge-green",
            "emoji": "🟢",
        }
    return {
        "label": "관리 필요",
        "color": "#D94A4A",
        "badge": "badge-red",
        "emoji": "🔴",
    }


def get_priority_items(result):
    all_data = {**result["posture"], **result["env"]}
    bad_items = []

    for key, value in all_data.items():
        measured_value, is_good, raw = value
        if not is_good:
            bad_items.append((key, measured_value, raw))

    return bad_items


def render_dashboard():
    page_header(
        "나의 자세 현황 대시보드",
        "최근 자세 분석 결과를 바탕으로 위험 부위와 교정 우선순위를 확인합니다.",
    )

    result = st.session_state.get("latest_result", None)
    history = st.session_state.get("history", [])

    # 아직 측정 결과가 없는 경우
    if result is None:
        st.markdown(
            """
<div class="fit-card">
    <div class="fit-card-title">
        <span>아직 분석 결과가 없습니다</span>
        <span class="fit-badge badge-blue">Ready</span>
    </div>
    <div style="font-size:14px;line-height:1.8;color:#667085;">
        먼저 왼쪽 메뉴에서 <b style="color:#172033;">자세측정</b>을 실행하면,
        이 대시보드에 최근 자세 점수, 위험 부위, 교정 우선순위가 자동으로 표시됩니다.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    all_data = {**result["posture"], **result["env"]}
    bad_items = get_priority_items(result)

    good_rate = round(result["good_count"] / result["total_count"] * 100) if result["total_count"] else 0
    bad_count = result["total_count"] - result["good_count"]

    # 상단 핵심 지표
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(result["score"], "최근 자세 점수", "#2563EB")
    with c2:
        metric_card(result["risk"], "종합 위험도", "#F59E0B")
    with c3:
        metric_card(f"{bad_count}개", "관리 필요 지표", "#EF4444")
    with c4:
        metric_card(f"{good_rate}%", "정상 범위 비율", "#10B981")

    render_measurement_coverage(result)

    left, right = st.columns([1.15, 0.85])

    # 실제 신체 부위별 위험 현황
    with left:
        rows = ""

        label_map = {
            "CVA": "목·경추",
            "TIA": "몸통·허리",
            "팔꿈치": "팔꿈치",
            "무릎": "무릎",
            "손목": "손목",
            "시선각": "시선·모니터",
            "책상높이": "책상 높이",
            "등받이": "의자 등받이",
        }

        for key, (value, is_good, raw) in all_data.items():
            style = risk_style(is_good)
            width = value_to_bar_width(key, raw)

            rows += f"""
<div class="result-row">
    <div class="result-name">{label_map.get(key, key)}</div>
    <div class="result-value">{value}</div>
    <div class="bar-wrap">
        <div class="bar" style="width:{width}%; background:{style["color"]};"></div>
    </div>
    <span class="fit-badge {style["badge"]}">{style["label"]}</span>
</div>
"""

        st.markdown(
            f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>최근 측정 기반 신체 부위별 위험 현황</span>
        <span class="fit-badge badge-blue">Live Result</span>
    </div>
    {rows}
</div>
""",
            unsafe_allow_html=True,
        )

    # 교정 우선순위
    with right:
        if bad_items:
            priority_html = ""

            for i, (key, value, raw) in enumerate(bad_items[:3], start=1):
                fb = FEEDBACK[key]
                msg = fb["bad"].split("\n")[0]

                priority_html += f"""
<div style="padding:12px 0;border-bottom:1px solid #EEF2F6;">
    <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="font-size:14px;font-weight:800;color:#172033;">
            {i}. {fb["label"]}
        </div>
        <span class="fit-badge badge-red">{value}</span>
    </div>
    <div style="font-size:12.5px;color:#667085;line-height:1.6;margin-top:5px;">
        {msg}
    </div>
</div>
"""

            guide_title = "오늘의 교정 우선순위"
            guide_badge = "집중관리"
        else:
            priority_html = """
<div style="font-size:14px;line-height:1.8;color:#667085;">
    현재 모든 주요 지표가 정상 범위에 있습니다.<br>
    지금 자세를 유지하면서 50분마다 가벼운 스트레칭을 해주세요.
</div>
"""
            guide_title = "오늘의 자세 상태"
            guide_badge = "양호"

        st.markdown(
            f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>{guide_title}</span>
        <span class="fit-badge badge-amber">{guide_badge}</span>
    </div>
    {priority_html}
</div>
""",
            unsafe_allow_html=True,
        )

    # 중단: 최근 측정 이미지 + AI 요약
    st.markdown("### AI 분석 요약")

    img_col, summary_col = st.columns([1, 1])

    with img_col:
        if "overlay" in result:
            st.markdown(
                """
    <div class="fit-card">
        <div class="fit-card-title">
            <span>최근 AI 오버레이</span>
            <span class="fit-badge badge-green">Analyzed</span>
        </div>
    </div>
    """,
            unsafe_allow_html=True,
        )
        st.image(result["overlay"], use_container_width=True)

    with summary_col:
        render_ai_correction_comment(result)


def _init_realtime_state():
    defaults = {
        "rt_phase": "idle",
        "rt_ready_result": None,
        "rt_saved": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_realtime_feedback_card(snapshot):
    status = snapshot.get("last_status", "WAIT")
    elapsed = int(snapshot.get("elapsed", 0))
    remain = max(10 - elapsed, 0)
    good_hold = min(snapshot.get("good_hold_seconds", 0), 3)

    if status == "GOOD":
        badge_class = "badge-green"
        color = "#3B8C42"
    elif status == "BAD":
        badge_class = "badge-red"
        color = "#D94A4A"
    else:
        badge_class = "badge-amber"
        color = "#BA7517"

    st.markdown(
        f"""
<div class="fit-card" style="border-left:6px solid {color};">
    <div class="fit-card-title">
        <span>실시간 자세 판정</span>
        <span class="fit-badge {badge_class}">{status}</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
        <div style="padding:14px;border-radius:14px;background:#F8FAFC;text-align:center;">
            <div style="font-size:12px;color:#667085;">남은 촬영 시간</div>
            <div style="font-size:34px;font-weight:950;color:#172033;">{remain}초</div>
        </div>
        <div style="padding:14px;border-radius:14px;background:#F8FAFC;text-align:center;">
            <div style="font-size:12px;color:#667085;">GOOD 유지 시간</div>
            <div style="font-size:34px;font-weight:950;color:{color};">{good_hold:.1f}초</div>
        </div>
    </div>
    <div style="font-size:13px;line-height:1.8;color:#667085;">
        {snapshot.get("last_feedback", "측정 대기 중입니다.")}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _init_realtime_state():
    defaults = {
        "rt_phase":        "idle",
        "rt_ready_result": None,
        "rt_saved":        False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_realtime_feedback_card(snapshot):
    status    = snapshot.get("last_status", "WAIT")
    elapsed   = int(snapshot.get("elapsed", 0))
    remain    = max(20 - elapsed, 0)
    good_hold = min(snapshot.get("good_hold_seconds", 0), 5)
    phase     = snapshot.get("phase", "idle")
    env_det   = snapshot.get("env_detected", {})

    if status == "GOOD":
        badge_class, color = "badge-green", "#3B8C42"
    elif status == "BAD":
        badge_class, color = "badge-red", "#D94A4A"
    else:
        badge_class, color = "badge-amber", "#BA7517"

    if phase == "posture":
        # 기존의 "실시간 자세 판정 / 남은 측정 시간 / GOOD 유지 시간" 카드 제거
        # → 동일한 위치에 "측정 진행 상황" 카드만 표시
        posture_ratio = min(max(elapsed / 20, 0), 1)
        progress_pct = int(posture_ratio * 100)
        st.markdown(f"""
<div class="fit-card">
    <div style="font-size:14px;font-weight:900;color:#172033;margin-bottom:12px;">
        측정 진행 상황
    </div>
    <div style="width:100%;height:14px;background:#D9DEE7;border-radius:999px;overflow:hidden;margin-bottom:12px;">
        <div style="width:{progress_pct}%;height:100%;background:#33B76A;border-radius:999px;"></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;text-align:center;gap:8px;">
        <div>
            <div style="font-size:12.5px;font-weight:800;color:#33B76A;">자세 측정</div>
            <div style="font-size:12px;color:#334155;margin-top:4px;">(20초)</div>
        </div>
        <div>
            <div style="font-size:12.5px;font-weight:800;color:#667085;">작업환경 측정</div>
            <div style="font-size:12px;color:#334155;margin-top:4px;">(자동 전환)</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    elif phase in ("env_transition", "environment"):
        items_kr = {"chair_back":"등받이","chair_seat":"의자시트",
                    "desk_surface":"책상","monitor":"모니터"}
        items_html = ""
        for k, label in items_kr.items():
            ok    = k in env_det
            c     = "#3B8C42" if ok else "#667085"
            icon  = "✅" if ok else "⬜"
            items_html += f'<div style="font-size:13px;color:{c};margin:4px 0;">{icon} {label}</div>'

        found = sum(1 for item in items_kr if item in env_det)
        env_pct = min(50 + int((found / 4) * 50), 100)
        st.markdown(f"""
<div class="fit-card">
    <div style="font-size:14px;font-weight:900;color:#172033;margin-bottom:12px;">
        측정 진행 상황
    </div>
    <div style="width:100%;height:14px;background:#D9DEE7;border-radius:999px;overflow:hidden;margin-bottom:12px;">
        <div style="width:{env_pct}%;height:100%;background:#33B76A;border-radius:999px;"></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;text-align:center;gap:8px;margin-bottom:12px;">
        <div>
            <div style="font-size:12.5px;font-weight:800;color:#33B76A;">자세 측정</div>
            <div style="font-size:12px;color:#334155;margin-top:4px;">완료</div>
        </div>
        <div>
            <div style="font-size:12.5px;font-weight:800;color:#33B76A;">작업환경 측정</div>
            <div style="font-size:12px;color:#334155;margin-top:4px;">{found}/4개 감지</div>
        </div>
    </div>
    <div style="font-size:13px;color:#667085;margin-bottom:8px;">
        책상·의자·모니터 4개가 동시에 잡히면 자동 완료됩니다.
    </div>
    {items_html}
</div>""", unsafe_allow_html=True)



class RealTimePostureProcessor(VideoProcessorBase):
    """WebRTC 기반 실시간 자세 측정.
    app_mobile.py의 조건(20초 안에 GOOD 5초 유지 → 환경 자동 전환 → 4개 객체 감지 완료)을 유지합니다.
    """

    POSTURE_TOTAL = 20
    GOOD_HOLD = 5
    REQUIRED_ENV_ITEMS = ["chair_back", "chair_seat", "desk_surface", "monitor"]

    # 분석 주기: 0.30초 → 0.07초
    # 단, 이전 분석 스레드가 끝나기 전에는 새 분석을 만들지 않아
    # CPU 폭주 없이 가능한 한 빠르게 최신 프레임을 분석합니다.
    ANALYZE_INTERVAL = 0.07

    # 오래된 분석 오버레이 프레임을 오래 붙잡으면 카메라가 뚝뚝 끊겨 보입니다.
    # 최신 분석 결과만 짧게 반영하고, 나머지는 현재 카메라 프레임을 그대로 출력합니다.
    OVERLAY_FRESH_SECONDS = 0.14

    def __init__(self):
        self.lock = threading.Lock()
        self.started_at = None
        self.last_analyzed_at = 0
        self.good_streak_started_at = None
        self.good_hold_seconds = 0
        self.last_status = "WAIT"
        self.last_feedback = "자세 측정 시작 버튼을 누른 뒤 측면 자세를 유지해주세요."
        self.latest_result = None
        self.latest_overlay = None
        self.latest_overlay_at = 0
        self.good_ready = False
        self.env_ready = False
        self.finished_bad = False
        self.analysis_running = False
        self.phase = "idle"
        self.env_detected = {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        with self.lock:
            if self.started_at is None:
                self.latest_overlay = img.copy()
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        should_analyze = False
        with self.lock:
            should_analyze = (
                now - self.last_analyzed_at >= self.ANALYZE_INTERVAL
                and not self.finished_bad
                and not self.env_ready
                and not self.analysis_running
            )
            if should_analyze:
                self.last_analyzed_at = now
                self.analysis_running = True

        if should_analyze:
            threading.Thread(
                target=self._analyze_frame_background,
                args=(img.copy(),),
                daemon=True,
            ).start()

        with self.lock:
            if self.phase == "posture" and not self.good_ready and not self.finished_bad:
                elapsed_for_finish = time.time() - self.started_at
                if elapsed_for_finish >= self.POSTURE_TOTAL:
                    self.finished_bad = True
                    self.phase = "bad"
                    self.last_status = "BAD"
                    self.last_feedback = "20초 동안 GOOD 자세가 5초 이상 유지되지 않았습니다. 자세를 교정한 뒤 재측정해주세요."

            # 카메라 화면이 1초 단위로 뚝뚝 끊기지 않도록 현재 프레임을 기본으로 사용합니다.
            # YOLO 오버레이는 최신 분석 결과가 충분히 최근일 때만 잠깐 반영합니다.
            # 오래된 오버레이 프레임을 계속 재사용하면 카메라가 멈춘 것처럼 보이므로 제한합니다.
            overlay_is_fresh = (
                self.latest_overlay is not None
                and self.latest_overlay_at
                and (time.time() - self.latest_overlay_at <= self.OVERLAY_FRESH_SECONDS)
            )
            out = self.latest_overlay.copy() if overlay_is_fresh else img.copy()
            elapsed = 0 if self.started_at is None else min(int(time.time() - self.started_at), self.POSTURE_TOTAL)
            remain = max(self.POSTURE_TOTAL - elapsed, 0)
            status = self.last_status
            phase = self.phase
            env_detected = dict(self.env_detected)
            good_hold = self.good_hold_seconds

        h, w = out.shape[:2]
        scale_x = w / 640.0
        scale_y = h / 480.0
        scale = min(scale_x, scale_y)
        border_thickness = max(2, int(round(3 * scale)))
        text_thickness = max(2, int(round(2 * scale)))

        # 좌측 상단 GOOD/BAD 표시: 네모 박스 없이 글자만 표시
        status_x = int(24 * scale_x)
        status_y = int(48 * scale_y)
        status_color = (16, 185, 129) if status == "GOOD" else (0, 0, 255) if status == "BAD" else (245, 158, 11)
        # 가독성을 위해 얇은 흰색 외곽선만 추가하고 배경 박스는 만들지 않습니다.
        cv2.putText(out, status, (status_x, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05 * scale, (255, 255, 255), max(3, text_thickness + 2))
        cv2.putText(out, status, (status_x, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05 * scale, status_color, text_thickness)

        # 우측 상단 시계형 타이머: 자세 단계에서만 20초 표시
        clock_radius = max(24, int(34 * scale))
        clock_cx = w - int(52 * scale_x)
        clock_cy = int(52 * scale_y)
        cv2.circle(out, (clock_cx, clock_cy), clock_radius, (255, 255, 255), -1)
        cv2.circle(out, (clock_cx, clock_cy), clock_radius, (0, 0, 0), border_thickness)
        knob_w = int(16 * scale)
        knob_h = int(8 * scale)
        cv2.rectangle(out, (clock_cx - knob_w // 2, clock_cy - clock_radius - knob_h),
                      (clock_cx + knob_w // 2, clock_cy - clock_radius + int(2 * scale)), (0, 0, 0), -1)
        # 타이머가 끝난 뒤에도 ENV/OK 같은 문구로 바꾸지 않고 숫자 0에 머무르게 합니다.
        timer_text = str(remain)
        timer_font_scale = 0.9 * scale if len(timer_text) >= 2 else 1.05 * scale
        (tw, th), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, timer_font_scale, text_thickness)
        cv2.putText(out, timer_text, (clock_cx - tw // 2, clock_cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, timer_font_scale, (0, 0, 0), text_thickness)

        # 하단 GOOD 유지 진행바
        if phase == "posture" and good_hold > 0:
            ratio = min(good_hold / self.GOOD_HOLD, 1.0)
            bar_x1, bar_y1 = int(24 * scale_x), h - int(42 * scale_y)
            bar_x2, bar_y2 = w - int(24 * scale_x), h - int(22 * scale_y)
            cv2.rectangle(out, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 40), -1)
            cv2.rectangle(out, (bar_x1, bar_y1), (bar_x1 + int((bar_x2 - bar_x1) * ratio), bar_y2), (16, 185, 129), -1)
            cv2.putText(out, f"GOOD {good_hold:.1f}s/{self.GOOD_HOLD}s", (bar_x1, bar_y1 - int(8 * scale_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (16, 185, 129), max(1, text_thickness - 1))

        # 환경 단계 안내
        if phase == "environment":
            items_kr = {"chair_back": "chair back", "chair_seat": "seat", "desk_surface": "desk", "monitor": "monitor"}
            base_y = int(92 * scale_y)
            for i, key in enumerate(self.REQUIRED_ENV_ITEMS):
                ok = key in env_detected
                color = (16, 185, 129) if ok else (150, 150, 150)
                label = f"{'OK' if ok else '--'} {items_kr[key]}"
                cv2.putText(out, label, (int(18 * scale_x), base_y + int(i * 26 * scale_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, color, max(1, text_thickness - 1))

        return av.VideoFrame.from_ndarray(out, format="bgr24")

    def _analyze_frame_background(self, img):
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = analyze_image(Image.fromarray(rgb))
            now = time.time()

            with self.lock:
                if self.started_at is None or self.finished_bad or self.env_ready:
                    return

                if isinstance(result, dict) and result.get("ok"):
                    self.latest_result = result
                    overlay_rgb = result.get("overlay")
                    if overlay_rgb is not None:
                        self.latest_overlay = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                        self.latest_overlay_at = time.time()

                if self.phase == "posture":
                    if isinstance(result, dict) and result.get("ok") and result.get("gate_pass", False):
                        if self.good_streak_started_at is None:
                            self.good_streak_started_at = now
                        self.good_hold_seconds = now - self.good_streak_started_at
                        self.last_status = "GOOD"
                        self.last_feedback = f"GOOD 자세 유지 중: {self.good_hold_seconds:.1f}초 / {self.GOOD_HOLD}초"

                        if self.good_hold_seconds >= self.GOOD_HOLD:
                            self.good_ready = True
                            self.phase = "environment"
                            self.last_feedback = "GOOD 자세 유지 완료. 작업환경 측정으로 자동 전환되었습니다."
                    else:
                        self.good_streak_started_at = None
                        self.good_hold_seconds = 0
                        self.last_status = "BAD"
                        if isinstance(result, dict) and result.get("gate_bad_items"):
                            self.last_feedback = " / ".join(result.get("gate_bad_items", [])[:2])
                        elif isinstance(result, dict):
                            self.last_feedback = result.get("message", "자세가 BAD로 판정되었습니다.")
                        else:
                            self.last_feedback = "자세가 BAD로 판정되었습니다."

                elif self.phase == "environment":
                    detected = result.get("env_detected", {}) if isinstance(result, dict) else {}
                    self.env_detected = detected or {}
                    if all(item in self.env_detected for item in self.REQUIRED_ENV_ITEMS):
                        self.env_ready = True
                        self.phase = "stopped"
                        self.last_status = "GOOD"
                        self.last_feedback = "작업환경 인식 완료. 기록하기를 눌러 측정 이력에 저장하세요."
                    else:
                        found = sum(1 for item in self.REQUIRED_ENV_ITEMS if item in self.env_detected)
                        self.last_status = "GOOD"
                        self.last_feedback = f"작업환경 인식 중입니다. {found}/4개 감지됨."
        finally:
            with self.lock:
                self.analysis_running = False

    def snapshot(self):
        with self.lock:
            return {
                "started_at": self.started_at,
                "elapsed": 0 if self.started_at is None else min(time.time() - self.started_at, self.POSTURE_TOTAL),
                "good_hold_seconds": self.good_hold_seconds,
                "last_status": self.last_status,
                "last_feedback": self.last_feedback,
                "latest_result": self.latest_result,
                "good_ready": self.good_ready,
                "env_ready": self.env_ready,
                "finished_bad": self.finished_bad,
                "phase": self.phase,
                "env_detected": dict(self.env_detected),
            }

    def start(self):
        with self.lock:
            self.started_at = time.time()
            self.last_analyzed_at = 0
            self.good_streak_started_at = None
            self.good_hold_seconds = 0
            self.last_status = "WAIT"
            self.last_feedback = "20초 안에 GOOD 자세를 5초 이상 유지해주세요."
            self.latest_result = None
            self.latest_overlay = None
            self.latest_overlay_at = 0
            self.good_ready = False
            self.env_ready = False
            self.finished_bad = False
            self.analysis_running = False
            self.phase = "posture"
            self.env_detected = {}

    def reset(self):
        with self.lock:
            self.started_at = None
            self.last_analyzed_at = 0
            self.good_streak_started_at = None
            self.good_hold_seconds = 0
            self.last_status = "WAIT"
            self.last_feedback = "재측정 준비 완료. 자세 측정 시작 버튼을 눌러주세요."
            self.latest_result = None
            self.latest_overlay = None
            self.latest_overlay_at = 0
            self.good_ready = False
            self.env_ready = False
            self.finished_bad = False
            self.analysis_running = False
            self.phase = "idle"
            self.env_detected = {}


def render_measure():
    page_header(
        "자세 측정",
        "실시간 자세 분석과 이미지 자세 분석을 선택해서 사용할 수 있습니다.",
    )

    _init_realtime_state()

    analysis_tab = st.radio(
        "분석 방식 선택",
        ["1. 실시간 자세 분석", "2. 이미지 자세 분석"],
        horizontal=True,
        key="measure_mode_tabs",
    )

    if analysis_tab == "1. 실시간 자세 분석":
        st.markdown("### 1. 실시간 자세 분석")
        st.markdown(
            """
<div class="fit-card">
    <div class="fit-card-title">
        <span>측정 흐름</span>
        <span class="fit-badge badge-blue">Real-time</span>
    </div>
    <div style="font-size:13px;line-height:1.8;color:#667085;">
        <b>자세 측정 시작</b> → 20초 안에 <b>GOOD 5초 유지</b> →
        작업환경 측정 자동 전환 → 책상·의자·모니터 4개 동시 감지 완료 → <b>기록하기</b><br>
        20초 안에 GOOD이 안 나오면 BAD로 저장 후 재측정 안내
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        camera_type = st.radio(
            "카메라 선택",
            ["전면 카메라", "후면 카메라"],
            horizontal=True,
            key="realtime_camera_type",
        )
        facing_mode = "user" if camera_type == "전면 카메라" else "environment"

        media_stream_constraints = {
            "video": {
                "facingMode": {"ideal": facing_mode},
                # 해상도를 낮추고 FPS를 높여 WebRTC 프리뷰가 더 부드럽게 나오도록 조정
                "width": {"ideal": 480, "max": 640},
                "height": {"ideal": 360, "max": 480},
                "frameRate": {"ideal": 60, "max": 60},
            },
            "audio": False,
        }

        webrtc_kwargs = dict(
            key=f"realtime-posture-{facing_mode}",
            media_stream_constraints=media_stream_constraints,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=RealTimePostureProcessor,
            async_processing=True,
        )
        if WebRtcMode is not None:
            webrtc_kwargs["mode"] = WebRtcMode.SENDRECV

        try:
            ctx = webrtc_streamer(**webrtc_kwargs)
        except TypeError:
            webrtc_kwargs.pop("mode", None)
            webrtc_kwargs.pop("rtc_configuration", None)
            ctx = webrtc_streamer(**webrtc_kwargs)

        processor = ctx.video_processor if ctx else None

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            start_btn = st.button("▶ 자세 측정 시작", use_container_width=True, key="rt_start_btn", type="primary")
        with c2:
            stop_btn = st.button("⏹ STOP", use_container_width=True, key="rt_stop_btn")
        with c3:
            retry_btn = st.button("🔄 재측정", use_container_width=True, key="rt_retry_btn")

        if processor is None:
            st.info(
                "카메라 권한을 허용한 뒤에도 로딩만 계속되면, 브라우저 주소가 https 또는 localhost인지 확인하고 "
                "카메라를 사용하는 다른 앱/탭을 닫은 다음 다시 Start를 눌러주세요."
            )
            return

        if start_btn:
            processor.start()
            st.session_state.rt_phase = "counting"
            st.session_state.rt_ready_result = None
            st.session_state.rt_saved = False
            speak("측정을 시작합니다. 측면을 카메라에 맞춰주세요.")

        if retry_btn:
            processor.reset()
            st.session_state.rt_phase = "idle"
            st.session_state.rt_ready_result = None
            st.session_state.rt_saved = False
            st.rerun()

        snapshot = processor.snapshot()
        if snapshot.get("started_at") is not None and st.session_state.get("rt_phase") not in ["stopped", "saved"]:
            _render_realtime_feedback_card(snapshot)

        if snapshot.get("good_ready") and not snapshot.get("env_ready") and not snapshot.get("finished_bad"):
            st.success("5초 이상 GOOD 자세가 유지되었습니다. 작업환경 측정으로 자동 전환되었습니다.")

        if snapshot.get("env_ready"):
            result = snapshot.get("latest_result")
            if result:
                result["source"] = "실시간 자세 분석"
                st.session_state.rt_ready_result = result
                if st.session_state.get("rt_phase") not in ["stopped", "saved"]:
                    st.session_state.rt_phase = "stopped"
                st.success("작업환경 인식이 완료되었습니다. 아래 기록하기 버튼을 눌러 저장하세요.")

        if stop_btn:
            snap = processor.snapshot()
            result = snap.get("latest_result") or st.session_state.get("rt_ready_result")
            if result and (snap.get("env_ready") or snap.get("good_ready")):
                result["source"] = "실시간 자세 분석"
                st.session_state.rt_phase = "stopped"
                st.session_state.rt_ready_result = result
                st.success("측정이 정지되었습니다. 아래 기록하기 버튼을 누르면 현재 오버레이 이미지와 함께 측정 이력에 저장됩니다.")
            else:
                st.warning("아직 GOOD 5초 유지 조건이 충족되지 않았습니다. 조건 충족 후 STOP을 눌러주세요.")

        if snapshot.get("finished_bad"):
            st.error("20초 동안 GOOD 자세가 5초 이상 유지되지 않아 작업환경 측정으로 전환하지 않았습니다.")
            bad_result = snapshot.get("latest_result")
            if bad_result and bad_result.get("overlay") is not None:
                st.image(bad_result["overlay"], use_container_width=True, caption="BAD 자세 — 파란 화살표 방향으로 교정해주세요")
            if bad_result and bad_result.get("ok"):
                st.session_state.rt_ready_result = bad_result
                st.session_state.rt_phase = "stopped"
            st.info("재측정 버튼을 누르면 20초 측정을 다시 시작할 수 있습니다.")

        result_to_show = st.session_state.get("rt_ready_result")
        if result_to_show and result_to_show.get("ok"):
            st.markdown("---")
            st.markdown("### 📊 실시간 자세 오버레이 및 통합 결과")
            good = result_to_show.get("good_count", 0)
            total = result_to_show.get("total_count", 0)
            rate = round(good / total * 100) if total else 0
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card(result_to_show.get("score", 0), "종합 자세 점수", "#2563EB")
            with c2:
                metric_card(result_to_show.get("risk", "-"), "위험도", "#F59E0B")
            with c3:
                metric_card(f"{good}/{total}", "양호 지표", "#10B981")
            with c4:
                metric_card(f"{rate}%", "정상 범위 비율", "#6366F1")

            # 실시간 결과창에서는 "오늘의 실천 목표" 카드와 iframe 고정 높이로 생기던
            # 큰 여백을 렌더링하지 않고, 오버레이 이미지만 표시합니다.
            overlay = result_to_show.get("overlay")
            if overlay is not None:
                caption = "실시간 자세 오버레이" if result_to_show.get("gate_pass") else "BAD 자세 — 파란 화살표 방향으로 교정해주세요"
                st.image(overlay, use_container_width=True, caption=caption)

            if st.session_state.get("rt_phase") in ["stopped", "saved"]:
                if st.button("💾 기록하기", use_container_width=True, key="rt_save_history_btn"):
                    if not st.session_state.get("rt_saved"):
                        result_to_show["source"] = result_to_show.get("source", "실시간 자세 분석")
                        st.session_state.latest_result = result_to_show
                        save_history(result_to_show)
                        sync_result_to_challenge(result_to_show)
                        st.session_state.rt_saved = True
                        st.session_state.rt_phase = "saved"
                        st.success("✅ 측정 이력과 바른자세 챌린지 포인트에 기록되었습니다.")
                    else:
                        st.info("이미 기록된 측정 결과입니다.")
            else:
                st.info("작업환경 4개 객체가 모두 감지되거나 STOP을 누르면 기록할 수 있습니다.")

            render_pretty_7_metric_dashboard(result_to_show)

        return

    # ==============================
    # 2. 이미지 자세 분석
    # ==============================
    st.markdown("### 2. 이미지 자세 분석")

    left, right = st.columns([0.95, 1.05])

    with left:
        st.markdown(
            """
<div class="fit-card">
    <div class="fit-card-title">
        <span>측면 사진 업로드</span>
        <span class="fit-badge badge-blue">Image</span>
    </div>
    <div style="font-size:13px;color:#667085;line-height:1.7;margin-bottom:12px;">
        의자, 책상, 모니터, 전신 측면이 최대한 함께 보이도록 촬영해주세요.
        발목·무릎·골반·어깨·귀가 보이면 분석 정확도가 좋아집니다.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "이미지 업로드",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="measure_uploader",
        )

        run_btn = st.button(
            "AI 자세 분석 실행",
            use_container_width=True,
            key="run_posture_analysis",
        )

    with right:
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="업로드된 측면 사진", use_container_width=True)
        else:
            st.markdown(
                """
<div class="upload-box">
    <div style="font-size:34px;margin-bottom:8px;">📸</div>
    <div style="font-weight:700;color:#172033;margin-bottom:4px;">측면 사진을 업로드하세요</div>
    <div style="font-size:13px;">AI 오버레이 분석 결과가 이 영역에 표시됩니다.</div>
</div>
""",
                unsafe_allow_html=True,
            )

    if run_btn:
        if not uploaded:
            st.warning("먼저 이미지를 업로드해주세요.")
            return

        with st.spinner("AI가 자세를 분석하는 중입니다..."):
            result = analyze_image(Image.open(uploaded))

        if not result["ok"]:
            st.error(result["message"])
            return

        if not result.get("gate_pass", True):
            st.session_state.latest_result = None
            gate_metrics = result.get("gate_metrics", {})
            cva = gate_metrics.get("CVA", {})
            tia = gate_metrics.get("TIA", {})

            def _status_color(status):
                return "#3B8C42" if status == "GOOD" else "#D94A4A"

            img_col, info_col = st.columns([1, 1])

            with img_col:
                overlay_img = result.get("overlay")
                if overlay_img is not None:
                    st.markdown(
                        """
<div style="border-radius:16px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
""", unsafe_allow_html=True)
                    st.image(overlay_img, use_container_width=True,
                             caption="현재 자세(빨강) vs 목표 자세(민트)")
                    st.markdown("</div>", unsafe_allow_html=True)

            with info_col:
                st.markdown(
                    f"""
<div class="fit-card" style="border-left:6px solid #D94A4A;
     background:linear-gradient(135deg,#FFF1F1,#FFFFFF);height:100%;">
    <div class="fit-card-title">
        <span>자세 교정이 필요합니다</span>
        <span class="fit-badge badge-red">BAD</span>
    </div>
    <div style="font-size:14px;line-height:1.8;color:#172033;
         font-weight:700;margin-bottom:10px;">
        CVA 또는 TIA가 BAD 판정으로<br>
        환경 분석 결과를 제공하지 않습니다.
    </div>
    <div style="font-size:12.5px;line-height:1.8;color:#667085;margin-bottom:16px;">
        이미지의 <b style="color:#D94A4A;">빨간 선</b>이 현재 자세,
        <b style="color:#2ec4b6;">민트 선</b>이 목표 자세입니다.<br>
        목표 자세에 맞게 교정 후 다시 촬영해주세요.
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;">
        <div style="padding:14px;border-radius:14px;background:#FFFFFF;
             border:2px solid {_status_color(cva.get('status','BAD'))};text-align:center;">
            <div style="font-size:11px;color:#667085;margin-bottom:4px;">CVA 목굴곡각</div>
            <div style="font-size:26px;font-weight:900;color:#172033;">
                {cva.get('value', '측정불가')}
            </div>
            <div style="font-size:13px;font-weight:900;
                 color:{_status_color(cva.get('status','BAD'))};">
                {cva.get('status', 'BAD')} · 정상 0°~20°
            </div>
        </div>
        <div style="padding:14px;border-radius:14px;background:#FFFFFF;
             border:2px solid {_status_color(tia.get('status','BAD'))};text-align:center;">
            <div style="font-size:11px;color:#667085;margin-bottom:4px;">TIA 몸통굴곡각</div>
            <div style="font-size:26px;font-weight:900;color:#172033;">
                {tia.get('value', '측정불가')}
            </div>
            <div style="font-size:13px;font-weight:900;
                 color:{_status_color(tia.get('status','BAD'))};">
                {tia.get('status', 'BAD')} · 정상 0°~20°
            </div>
        </div>
    </div>
    <div style="padding:12px;border-radius:10px;background:#FFF8E1;
         border-left:4px solid #F59E0B;font-size:12.5px;color:#92400E;line-height:1.7;">
        💡 측면에서 <b>코·어깨·골반</b>이 잘 보이도록 촬영하면<br>
        더 정확한 분석이 가능합니다.
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.warning("자세를 교정하고 다시 촬영해주세요.")
            return

        st.session_state.latest_result = result
        result["source"] = "이미지 자세 분석"
        save_history(result)
        sync_result_to_challenge(result)
        st.success("분석이 완료되었습니다. 측정이력과 바른자세 챌린지 포인트에 반영되었습니다.")

    result = st.session_state.get("latest_result")

    if result:
        st.markdown("---")


        img_col, summary_col = st.columns([1.05, 0.95])

        with img_col:
            st.markdown(
                """
<div class="fit-card">
    <div class="fit-card-title">
        <span>AI 오버레이 결과</span>
        <span class="fit-badge badge-green">AI</span>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.image(result["overlay"], use_container_width=True)

        with summary_col:
            render_ai_correction_comment(result)

        st.markdown("### 7개 측정 지표 결과")
        render_pretty_7_metric_dashboard(result)

def render_history_image_card(record, caption="측정 이미지"):
    image_path = record.get("image_path")
    if not image_path:
        return

    try:
        path = Path(image_path)
        if not path.exists():
            return
        st.image(str(path), use_container_width=True, caption=caption)
    except Exception:
        return


def render_history():
    page_header(
        "측정 이력",
        "현재 로그인한 계정의 자세 분석 결과만 확인합니다.",
    )

    username = get_current_username()
    histories = load_json_file(HISTORY_DB_PATH, {})
    user_history = histories.get(username, [])

    if not user_history:
        st.info(f"{username}님의 측정 이력이 아직 없습니다.")
        return

    latest = user_history[0]

    # 상단 핵심 지표
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(latest["score"], "최근 자세 점수", "#2563EB")
    with c2:
        metric_card(latest["risk"], "최근 위험도", "#F59E0B")
    with c3:
        metric_card(f"{latest['good']}/{latest['total']}", "양호 지표", "#10B981")
    with c4:
        rate = round(latest["good"] / latest["total"] * 100)
        metric_card(f"{rate}%", "정상 비율", "#6366F1")

    # ==================================================
    # 1. 내 점수 추이 (최근 측정기록 위로 이동)
    # ==================================================
    if len(user_history) >= 2:
        st.markdown("### 내 점수 추이")

        chart_data = pd.DataFrame(
            [
                {
                    "회차": i + 1,
                    "점수": float(h["score"]),
                    "측정시간": h["time"],
                }
                for i, h in enumerate(reversed(user_history))
            ]
        )

        base = alt.Chart(chart_data).encode(
            x=alt.X(
                "회차:O",
                title="측정 회차",
                axis=alt.Axis(labelAngle=0, labelPadding=10, titlePadding=16),
            ),
            y=alt.Y(
                "점수:Q",
                title="자세 점수",
                scale=alt.Scale(domain=[0, 10]),
                axis=alt.Axis(values=[0, 2, 4, 6, 8, 10], titlePadding=32, labelPadding=12),
            ),
            tooltip=[
                alt.Tooltip("측정시간:N", title="측정 시간"),
                alt.Tooltip("점수:Q", title="점수", format=".1f"),
            ],
        )

        line = base.mark_line(
            strokeWidth=4,
            interpolate="monotone",
        )

        points = base.mark_circle(
            size=95,
            opacity=1,
        )

        labels = base.mark_text(
            align="center",
            baseline="bottom",
            dy=-10,
            fontSize=13,
            fontWeight="bold",
        ).encode(
            text=alt.Text("점수:Q", format=".1f")
        )

        chart = (
            (line + points + labels)
            .properties(
                height=430,
                padding={"left": 78, "right": 28, "top": 24, "bottom": 20},
            )
            .configure_view(strokeWidth=0)
            .configure_axis(
                gridColor="#E5EAF2",
                labelColor="#667085",
                titleColor="#172033",
                labelFontSize=13,
                titleFontSize=15,
                titleFontWeight="bold",
            )
        )

        st.altair_chart(chart, use_container_width=True)

    # ==================================================
    # 2. 최근 측정기록
    # ==================================================
    st.markdown(f"### {username}님의 최근 측정 기록")

    risk = latest["risk"]

    if risk == "양호":
        color = "#3B8C42"
    elif risk == "주의":
        color = "#BA7517"
    else:
        color = "#D94A4A"

    latest_rate = round(latest["good"] / latest["total"] * 100)

    missing_items = latest.get("missing_items", []) or []
    if missing_items:
        coverage_note = f"측정 제외: {', '.join([item['label'] for item in missing_items])}"
    else:
        coverage_note = "7개 항목 전체 측정"

    st.markdown(
        f"""
<div class="fit-card" style="border-left:5px solid {color};">
<b>{latest["time"]}</b><br><br>
종합 점수: <b>{latest["score"]}/10</b><br>
위험도: <b>{risk}</b><br>
양호 지표: <b>{latest["good"]}/{latest["total"]}</b><br>
정상 비율: <b>{latest_rate}%</b><br>
<span style="font-size:12px;color:#667085;">{coverage_note}</span>

<div style="margin-top:10px;height:8px;background:#EEF2F6;border-radius:999px;">
<div style="height:8px;width:{latest_rate}%;background:{color};border-radius:999px;"></div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    render_history_image_card(latest, caption=f"최근 측정 이미지 · {latest.get('source', '자세 분석')}")

    # ==================================================
    # 3. 전체 측정이력
    # ==================================================
    st.markdown("### 전체 측정이력")

    for h in user_history:
        risk = h["risk"]

        if risk == "양호":
            color = "#3B8C42"
        elif risk == "주의":
            color = "#BA7517"
        else:
            color = "#D94A4A"

        rate = round(h["good"] / h["total"] * 100)

        missing_items = h.get("missing_items", []) or []
        if missing_items:
            coverage_note = f"측정 제외: {', '.join([item['label'] for item in missing_items])}"
        else:
            coverage_note = "7개 항목 전체 측정"

        st.markdown(
            f"""
<div class="fit-card" style="border-left:5px solid {color};">
<b>{h["time"]}</b><br>
<span class="fit-badge badge-blue" style="margin-top:8px;">{h.get("source", "자세 분석")}</span><br><br>
종합 점수: <b>{h["score"]}/10</b><br>
위험도: <b>{risk}</b><br>
양호 지표: <b>{h["good"]}/{h["total"]}</b><br>
정상 비율: <b>{rate}%</b><br>

<span style="font-size:12px;color:#667085;">{coverage_note}</span>

<div style="margin-top:10px;height:8px;background:#EEF2F6;border-radius:999px;">
<div style="height:8px;width:{rate}%;background:{color};border-radius:999px;"></div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )
        render_history_image_card(h, caption=f"측정 이미지 · {h.get('source', '자세 분석')}")



# =========================================================
# 7-1. 자세 분석 결과 기반 비급여 예상 영수증
# =========================================================

NONPAY_CODES = {
    "경추": ["도수", "체외", "증식척추"],
    "요추": ["도수", "체외", "증식척추"],
    "손목": ["체외", "증식사지"],
}

NONPAY_INFO = {
    "도수": {"name": "🛏 도수치료", "avg": 107999},
    "체외": {"name": "⚡ 체외충격파", "avg": 91145},
    "증식척추": {"name": "💉 증식치료 (척추)", "avg": 93469},
    "증식사지": {"name": "💉 증식치료 (사지)", "avg": 90000},
}

PERIODS = [
    {"label": "1회 치료", "sessions": 1},
    {"label": "2주 (4회)", "sessions": 4},
    {"label": "1개월 (8회)", "sessions": 8},
    {"label": "3개월 (24회)", "sessions": 24},
    {"label": "6개월 (48회)", "sessions": 48},
]


def map_result_to_disease_locations(result):
    if result is None:
        return []
    all_data = {**result.get("posture", {}), **result.get("env", {})}
    bad_keys = [key for key, (_, is_good, _) in all_data.items() if not is_good]
    active = []
    if any(key in bad_keys for key in ["CVA", "시선각"]):
        active.append("경추")
    if any(key in bad_keys for key in ["TIA", "등받이"]):
        active.append("요추")
    if any(key in bad_keys for key in ["손목", "팔꿈치", "책상높이"]):
        active.append("손목")
    return active


def build_receipt_html(active_d, period_label, t_dosu=True, t_shock=True, t_prolo=True, result=None):
    selected_period = next(p for p in PERIODS if p["label"] == period_label)
    sessions = selected_period["sessions"]

    used_np = []
    for d in active_d:
        for code in NONPAY_CODES.get(d, []):
            if code not in used_np:
                used_np.append(code)

    nonpay_total = 0
    np_rows_html = ""
    unit_html = ""
    for code in used_np:
        info = NONPAY_INFO[code]
        is_active = (code == "도수" and t_dosu) or (code == "체외" and t_shock) or (code.startswith("증식") and t_prolo)
        if not is_active:
            continue
        total = info["avg"] * sessions
        nonpay_total += total
        np_rows_html += f"""
        <div class='r-row non'><span>{info['name']}</span><span>×{sessions}회</span><b>{total:,}원</b></div>
        """
        unit_html += f"""
        <div class='r-row sub'><span>{info['name']}</span><span>1회</span><span>{info['avg']:,}원</span></div>
        """

    if not np_rows_html:
        np_rows_html = "<div class='r-row'><span>현재 자동 청구 예상 항목 없음</span><span>-</span><b>0원</b></div>"
        unit_html = "<div class='r-row sub'><span>정상 범위 유지 시 예방 관리 권장</span><span>-</span><span>0원</span></div>"

    warn_msgs = [
        (0, "⚠ 이 자세를 계속 유지하면 위 비용이 발생할 수 있습니다", "지금 자세를 교정하세요"),
        (500000, "💸 월급의 상당 부분이 병원비로 사라질 수 있습니다", "만성 통증으로 이어지기 전에 예방하세요"),
        (1500000, "🚨 해외여행 경비가 통째로 날아갈 수 있습니다", "치료보다 예방이 훨씬 저렴합니다"),
        (3000000, "🔴 분기 의료비가 차 한 대 값에 육박할 수 있습니다", "이제 자세 교정이 투자입니다"),
        (6000000, "☠️ 연봉의 상당 부분을 병원에 내야 할 수 있습니다", "지금 당장 작업환경을 바꾸세요"),
    ]
    wm = warn_msgs[0]
    for msg in reversed(warn_msgs):
        if nonpay_total >= msg[0]:
            wm = msg
            break

    now = datetime.datetime.now()
    dt_str = f"{now.year}.{now.month:02d}.{now.day:02d}  {now.hour:02d}:{now.minute:02d}"
    disease_str = " · ".join(active_d) + " 질환" if active_d else "관리 필요 질환 없음"
    barcode_num = f"FITMEUP-VDT-{str(nonpay_total).zfill(9)}"
    score_line = ""
    if result is not None:
        score_line = f"자세점수 : {result.get('score', '-')} / 10<br>위험도 : {result.get('risk', '-')}<br>"

    return f"""
<!DOCTYPE html><html lang='ko'><head><meta charset='UTF-8'>
<link href='https://fonts.googleapis.com/css2?family=Nanum+Gothic+Coding:wght@400;700&family=Pretendard:wght@300;400;500;600;700;900&display=swap' rel='stylesheet'>
<style>
:root {{ --paper:#fefcf6; --ink:#0F1E36; --red:#2563EB; --mono:'Nanum Gothic Coding',monospace; }}
body {{ margin:0; padding:10px; background:transparent; display:flex; justify-content:center; }}
.receipt-outer {{ width:100%; max-width:430px; filter:drop-shadow(0 8px 24px rgba(37,99,235,0.08)); }}
.zig-top {{ height:18px; background:var(--paper); clip-path:polygon(0% 100%,4% 0%,8% 100%,12% 0%,16% 100%,20% 0%,24% 100%,28% 0%,32% 100%,36% 0%,40% 100%,44% 0%,48% 100%,52% 0%,56% 100%,60% 0%,64% 100%,68% 0%,72% 100%,76% 0%,80% 100%,84% 0%,88% 100%,92% 0%,96% 100%,100% 0%); }}
.zig-bot {{ height:18px; background:var(--paper); clip-path:polygon(0% 0%,4% 100%,8% 0%,12% 100%,16% 0%,20% 100%,24% 0%,28% 100%,32% 0%,36% 100%,40% 0%,44% 100%,48% 0%,52% 100%,56% 0%,60% 100%,64% 0%,68% 100%,72% 0%,76% 100%,80% 0%,84% 100%,88% 0%,92% 100%,96% 0%,100% 100%); }}
.body {{ background:var(--paper); padding:8px 24px 22px; font-family:var(--mono); color:var(--ink); }}
.center {{ text-align:center; }} .store {{ font-size:16px; font-weight:700; letter-spacing:3px; }} .subt {{ font-size:10px; color:#5E718D; letter-spacing:2px; }}
.warn {{ font-size:11px; font-weight:700; color:var(--red); border:2px solid var(--red); padding:4px 8px; display:inline-block; margin:8px 0 4px; }}
.dash {{ color:#E2ECF6; font-size:11px; white-space:nowrap; overflow:hidden; }} .meta {{ font-size:10px; color:#5E718D; line-height:1.9; margin:8px 0; }}
.hd {{ font-size:10px; font-weight:700; color:#5E718D; letter-spacing:1px; margin:8px 0 4px; }}
.r-row {{ display:flex; justify-content:space-between; gap:8px; align-items:baseline; font-size:11.5px; margin-bottom:4px; }}
.r-row span:first-child {{ flex:1; color:#0F1E36; }} .r-row span:nth-child(2) {{ color:#94A3B8; font-size:10px; white-space:nowrap; }} .r-row b {{ color:var(--red); white-space:nowrap; }}
.sub {{ font-size:10px; color:#94A3B8; }} .sgl {{ border-top:1px dashed #E2ECF6; margin:7px 0; }} .dbl {{ border-top:2px solid #0F1E36; margin:8px 0 4px; }}
.total {{ font-size:15px; font-weight:700; color:var(--red); }} .barcode {{ font-size:36px; line-height:.8; letter-spacing:-1px; opacity:.85; color:#0F1E36; }} .bcnum {{ font-size:9px; letter-spacing:3px; color:#94A3B8; margin-top:4px; }}
.notice {{ font-size:9px; color:#94A3B8; line-height:1.8; margin-top:12px; }} .notice p {{ margin:0; }} .notice p:before {{ content:'* '; }}
</style></head><body><div class='receipt-outer'><div class='zig-top'></div><div class='body'>
<div class='center' style='padding:14px 0 8px'><div class='store'>비급여 의료비 예상 청구서</div><div class='subt'>POSTURE LINKED MEDICAL COST</div><span class='warn'>⚠ 경 고 ⚠</span><div class='dash'>────────────────────────</div><div style='font-size:11px;color:#5E718D;margin-top:4px'>{disease_str}</div></div>
<div class='sgl'></div><div class='meta'>발행일시 : {dt_str}<br>{score_line}치료기간 : {selected_period['label']}<br>치료방식 : 주 2회 집중 치료 기준<br>자동연동 : 자세측정 BAD 항목 기반</div>
<div class='sgl'></div><div class='hd'>[ 비급여 항목 · 전액 본인부담 ]</div>{np_rows_html}
<div class='sgl'></div><div class='r-row'><span>비급여 소계</span><span></span><b>{nonpay_total:,}원</b></div><div class='dbl'></div><div class='r-row total'><span>TOTAL</span><b>{nonpay_total:,}원</b></div>
<div class='sgl'></div><div class='hd'>[ 1회 단가 참고 ]</div>{unit_html}
<div class='sgl'></div><div class='center' style='margin:10px 0'><div style='font-size:10px;color:var(--red);font-weight:700'>{wm[1]}</div><div style='font-size:9px;color:#94A3B8;margin-top:4px'>{wm[2]}</div></div>
<div class='sgl'></div><div class='center'><div class='barcode'>▌▌ ▌▌▌ ▌ ▌▌▌▌ ▌ ▌▌ ▌▌▌ ▌▌</div><div class='bcnum'>{barcode_num}</div></div>
<div class='notice'><p>본 청구서는 예상 비용이며 실제 금액과 다를 수 있습니다</p><p>자세 분석 BAD 항목을 경추·요추·손목 질환 위치로 자동 매핑했습니다</p><p>비급여 금액은 앱 내 평균 단가 기준입니다</p><p>자세 분석은 전문 의료 진단을 대체하지 않습니다</p></div>
</div><div class='zig-bot'></div></div></body></html>
"""

def minutes_until_next_alarm(selected_times):
    now = datetime.datetime.now()
    candidates = []

    for t in selected_times:
        hour, minute = map(int, t.split(":"))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if target <= now:
            target += datetime.timedelta(days=1)

        candidates.append(target)

    if not candidates:
        return None, None

    next_time = min(candidates)
    diff_min = int((next_time - now).total_seconds() // 60)

    return next_time, diff_min

def render_alarm_effect(selected_times):
    now = datetime.datetime.now().strftime("%H:%M")

    dismissed_param = st.query_params.get("alarm_dismissed", "")
    if dismissed_param == now:
        st.session_state.alarm_dismissed_time = now
        st.query_params.clear()
        st.rerun()

    if now not in selected_times:
        return

    if st.session_state.get("alarm_dismissed_time", "") == now:
        return

    components.html(
        f"""
<script>
const alarmTime = "{now}";

function removeOldAlarm() {{
    const old = window.parent.document.getElementById("fitmeupAlarmOverlay");
    if (old) old.remove();

    const oldStyle = window.parent.document.getElementById("fitmeupAlarmStyle");
    if (oldStyle) oldStyle.remove();
}}

function closeAlarm() {{
    const overlay = window.parent.document.getElementById("fitmeupAlarmOverlay");
    if (overlay) overlay.remove();

    const params = new URLSearchParams(window.parent.location.search);
    params.set("alarm_dismissed", alarmTime);
    window.parent.location.search = params.toString();
}}

removeOldAlarm();

const style = window.parent.document.createElement("style");
style.id = "fitmeupAlarmStyle";
style.innerHTML = `
#fitmeupAlarmOverlay {{
    position: fixed;
    inset: 0;
    z-index: 2147483647;
    background: rgba(15, 23, 42, 0.38);
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fitAlarmBg 0.75s infinite alternate;
}}

#fitmeupAlarmModal {{
    position: relative;
    width: 560px;
    max-width: 84vw;
    padding: 48px 38px;
    border-radius: 30px;
    background: #FCEBEB;
    border: 4px solid #D94A4A;
    box-shadow: 0 24px 90px rgba(217, 74, 74, 0.45);
    text-align: center;
    font-family: Pretendard, sans-serif;
    animation: fitAlarmPulse 0.75s infinite alternate;
}}

#fitmeupAlarmClose {{
    position: absolute;
    top: 16px;
    right: 20px;
    border: none;
    background: transparent;
    color: #D94A4A;
    font-size: 30px;
    font-weight: 900;
    cursor: pointer;
}}

.fitmeup-alarm-icon {{
    font-size: 64px;
    margin-bottom: 14px;
}}

.fitmeup-alarm-title {{
    font-size: 36px;
    font-weight: 900;
    color: #D94A4A;
    line-height: 1.35;
}}

.fitmeup-alarm-sub {{
    margin-top: 14px;
    font-size: 18px;
    font-weight: 700;
    color: #8E2424;
}}

@keyframes fitAlarmBg {{
    from {{ background: rgba(15, 23, 42, 0.30); }}
    to {{ background: rgba(217, 74, 74, 0.42); }}
}}

@keyframes fitAlarmPulse {{
    from {{ transform: scale(1); opacity: 1; }}
    to {{ transform: scale(1.04); opacity: 0.86; }}
}}
`;
window.parent.document.head.appendChild(style);

const overlay = window.parent.document.createElement("div");
overlay.id = "fitmeupAlarmOverlay";
overlay.innerHTML = `
    <div id="fitmeupAlarmModal">
        <button id="fitmeupAlarmClose">✕</button>
        <div class="fitmeup-alarm-icon">🔔</div>
        <div class="fitmeup-alarm-title">바른자세 체크 시간입니다!</div>
        <div class="fitmeup-alarm-sub">지금 자세를 확인하고 바로 측정해보세요.</div>
    </div>
`;
window.parent.document.body.appendChild(overlay);

window.parent.document
    .getElementById("fitmeupAlarmClose")
    .addEventListener("click", closeAlarm);

window.parent.document.addEventListener("keydown", function(e) {{
    if (e.key === "Escape") {{
        closeAlarm();
    }}
}});
</script>

<audio autoplay>
    <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
</audio>
""",
        height=1,
    )

def render_posture_challenge():
    st.markdown("## 바른자세 챌린지")
    st.caption("자세측정을 할 때마다 점수가 포인트로 누적됩니다.")

    if "challenge_times" not in st.session_state:
        st.session_state.challenge_times = []

    left, right = st.columns([0.9, 1.1])

    with left:
        st.markdown("### 알림 시간 설정")

        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            hour = st.selectbox("시", range(24), format_func=lambda x: "{:02d}".format(x), key="challenge_hour")

        with c2:
            minute = st.selectbox("분", range(60), format_func=lambda x: "{:02d}".format(x), key="challenge_minute")

        with c3:
            st.write("")
            st.write("")
            if st.button("추가", use_container_width=True, key="add_challenge_alarm"):
                t = "{:02d}:{:02d}".format(hour, minute)

                if t not in st.session_state.challenge_times:
                    st.session_state.challenge_times.append(t)
                    st.success("{} 알림 추가".format(t))
                else:
                    st.warning("이미 추가된 시간입니다.")

        if st.session_state.challenge_times:
            st.markdown("#### 설정된 알림")

            for t in sorted(st.session_state.challenge_times):
                c_time, c_delete = st.columns([4, 1])

                with c_time:
                    st.write("⏰ {}".format(t))

                with c_delete:
                    if st.button("삭제", key="delete_alarm_{}".format(t)):
                        st.session_state.challenge_times.remove(t)
                        st.rerun()

            render_alarm_effect(st.session_state.challenge_times)
        else:
            st.info("알림 시간이 아직 없습니다.")

    with right:
        st.markdown("### 4팀 척추처척추")
        st.caption("누적 포인트가 높을수록 결승선에 가까워집니다.")

        challenge = load_challenge_results()

        if not challenge:
            st.info("아직 자세측정을 완료한 팀원이 없습니다.")
            return

        if isinstance(challenge, list):
            converted = {}
            for item in challenge:
                name = item.get("name", "익명")
                score = item.get("score", 0)
                point = int(round(score * 10))

                if name not in converted:
                    converted[name] = {
                        "name": name,
                        "total_point": 0,
                        "count": 0,
                        "records": [],
                    }

                converted[name]["total_point"] += point
                converted[name]["count"] += 1
                converted[name]["records"].insert(0, {
                    "score": score,
                    "point": point,
                    "risk": item.get("risk", "-"),
                    "good": item.get("good", 0),
                    "total": item.get("total", 0),
                    "bad_items": item.get("bad_items", []),
                    "time": item.get("time", "-"),
                })

            challenge = converted
            save_challenge_results(challenge)

        members = sorted(
            challenge.values(),
            key=lambda x: x.get("total_point", 0),
            reverse=True,
        )

        max_point = max([m.get("total_point", 0) for m in members]) or 1
        icons = ["🐰", "🐢", "🦊", "🐻", "🐼", "🐯", "🐸", "🐹"]

        race_html = """
<div style="background:#F8FAFC;border:1px solid #E5EAF2;border-radius:22px;padding:18px;margin-bottom:18px;">
<div style="font-size:18px;font-weight:900;color:#172033;margin-bottom:14px;">🐰 자세 포인트 레이스 🐢</div>
"""

        for idx, member in enumerate(members):
            name = member.get("name", "익명")
            total_point = member.get("total_point", 0)
            count = member.get("count", 0)
            records = member.get("records", [])
            latest = records[0] if records else {}

            latest_score = latest.get("score", "-")
            latest_risk = latest.get("risk", "-")

            percent = int((total_point / max_point) * 100)
            percent = max(8, min(percent, 100))
            icon = icons[idx % len(icons)]

            race_html += """
<div style="margin-bottom:18px;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
<div style="font-size:14px;font-weight:800;color:#172033;">{rank}. {name}</div>
<div style="font-size:13px;font-weight:800;color:#185FA5;">{point}P · {count}회</div>
</div>
<div style="position:relative;height:38px;background:#EAF0F7;border-radius:999px;overflow:hidden;">
<div style="position:absolute;left:0;top:0;height:38px;width:{percent}%;background:linear-gradient(90deg,#DFF3FF,#B9E6FF);border-radius:999px;"></div>
<div style="position:absolute;left:calc({percent}% - 22px);top:2px;font-size:28px;">{icon}</div>
<div style="position:absolute;right:10px;top:8px;font-size:18px;">🏁</div>
</div>
<div style="font-size:12px;color:#667085;margin-top:5px;">최근 점수 {score}/10 · 상태 {risk}</div>
</div>
""".format(
                rank=idx + 1,
                name=name,
                point=total_point,
                count=count,
                percent=percent,
                icon=icon,
                score=latest_score,
                risk=latest_risk,
            )

        race_html += "</div>"
        st.markdown(race_html, unsafe_allow_html=True)

        st.markdown("### 누적 포인트 순위")

        for i, member in enumerate(members, start=1):
            name = member.get("name", "익명")
            total_point = member.get("total_point", 0)
            count = member.get("count", 0)
            records = member.get("records", [])
            latest = records[0] if records else {}

            latest_time = latest.get("time", "-")
            latest_score = latest.get("score", "-")
            latest_risk = latest.get("risk", "-")
            bad_items = latest.get("bad_items", [])

            if latest_risk == "양호":
                badge = "badge-green"
            elif latest_risk == "주의":
                badge = "badge-amber"
            else:
                badge = "badge-red"

            bad_text = " · ".join(bad_items) if bad_items else "관리 필요 항목 없음"

            card_html = """
<div class="fit-card">
<div class="fit-card-title">
<span>{rank}. {name}</span>
<span class="fit-badge {badge}">{risk}</span>
</div>
<div style="font-size:28px;font-weight:900;color:#185FA5;margin-bottom:8px;">{point}P</div>
<div style="font-size:14px;line-height:1.8;color:#172033;">
측정 횟수: <b>{count}회</b><br>
최근 점수: <b>{score}/10</b><br>
최근 측정: <b>{time}</b><br>
교정 필요: <b>{bad_text}</b>
</div>
</div>
""".format(
                rank=i,
                name=name,
                badge=badge,
                risk=latest_risk,
                point=total_point,
                count=count,
                score=latest_score,
                time=latest_time,
                bad_text=bad_text,
            )

            st.markdown(card_html, unsafe_allow_html=True)


def calculate_receipt_total(active_d, period_label, t_dosu=True, t_shock=True, t_prolo=True):
    """영수증에 표시될 비급여 TOTAL 금액을 계산합니다."""
    selected_period = next(p for p in PERIODS if p["label"] == period_label)
    sessions = selected_period["sessions"]

    used_np = []
    for d in active_d:
        for code in NONPAY_CODES.get(d, []):
            if code not in used_np:
                used_np.append(code)

    total_cost = 0
    for code in used_np:
        info = NONPAY_INFO[code]
        is_active = (
            (code == "도수" and t_dosu)
            or (code == "체외" and t_shock)
            or (code.startswith("증식") and t_prolo)
        )
        if is_active:
            total_cost += info["avg"] * sessions

    return total_cost

def render_receipt_page():
    page_header("비급여 의료비 예상 영수증", "자세측정 결과에서 기준 범위를 벗어난 부위를 경추·요추·손목 항목으로 자동 연결합니다.")
    result = st.session_state.get("latest_result")
    if result is None:
        st.info("영수증을 생성하려면 먼저 왼쪽 메뉴의 자세측정에서 AI 자세 분석을 실행해주세요.")
        return

    auto_diseases = map_result_to_disease_locations(result)
    all_data = {**result["posture"], **result["env"]}
    bad_labels = [FEEDBACK[k]["label"] for k, v in all_data.items() if not v[1]]

    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown(f"""
<div class='fit-card'><div class='fit-card-title'><span>자세 분석 자동 매핑</span><span class='fit-badge badge-blue'>Auto</span></div>
<div style='font-size:13px;line-height:1.8;color:#667085;'><b style='color:#172033;'>BAD 측정 항목</b><br>{' · '.join(bad_labels) if bad_labels else '현재 BAD 항목 없음'}<br><br><b style='color:#172033;'>영수증 반영 위치</b><br>{' · '.join(auto_diseases) if auto_diseases else '관리 필요 질환 없음'}</div></div>
""", unsafe_allow_html=True)
        st.subheader("📍 질환 위치")
        st.caption("자세 분석 결과에 따라 기본값이 자동 선택됩니다. 필요하면 직접 수정할 수 있습니다.")
        c1, c2, c3 = st.columns(3)
        with c1:
            d_neck = st.checkbox("경추", value=("경추" in auto_diseases), key="receipt_neck")
        with c2:
            d_waist = st.checkbox("요추", value=("요추" in auto_diseases), key="receipt_waist")
        with c3:
            d_wrist = st.checkbox("손목", value=("손목" in auto_diseases), key="receipt_wrist")
        st.divider()
        st.subheader("💉 비급여 치료 선택")
        t_dosu = st.checkbox("🛏 도수치료", value=True, key="receipt_dosu")
        t_shock = st.checkbox("⚡ 체외충격파", value=True, key="receipt_shock")
        t_prolo = st.checkbox("💉 증식치료", value=True, key="receipt_prolo")
        st.divider()
        st.subheader("⏱ 치료 기간")
        default_period = "3개월 (24회)" if result.get("risk") == "위험" else ("1회 치료" if result.get("risk") == "양호" else "1개월 (8회)")
        period_label = st.select_slider("치료 기간", options=[p["label"] for p in PERIODS], value=default_period, label_visibility="collapsed", key="receipt_period")

    active_d = []
    if d_neck:
        active_d.append("경추")
    if d_waist:
        active_d.append("요추")
    if d_wrist:
        active_d.append("손목")

    receipt_html = build_receipt_html(active_d, period_label, t_dosu, t_shock, t_prolo, result)
    total_cost = calculate_receipt_total(active_d, period_label, t_dosu, t_shock, t_prolo)

    with right:
        st.markdown(
            f"""
<div style="
    width:100%;
    max-width:430px;
    margin:0 auto 14px auto;
    text-align:left;
">
    <div style="
        font-size:28px;
        font-weight:900;
        line-height:1.2;
        color:#667085;
        margin-bottom:6px;
        letter-spacing:-0.8px;
    ">
        예상 총 진료비
    </div>
    <div style="
        font-size:42px;
        font-weight:950;
        color:#D94A4A;
        line-height:1.05;
        letter-spacing:-1.2px;
    ">
        {total_cost:,}원
    </div>
    <div style="
        margin-top:8px;
        font-size:12.5px;
        color:#98A2B3;
        line-height:1.5;
    ">
        선택한 질환 위치 · 치료 방식 · 치료 기간 기준
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        components.html(receipt_html, height=3600, scrolling=False)

# =========================================================
# 제품 추천 탭
# =========================================================

PRODUCT_RECOMMENDATIONS = {
    "경추": [
        {
            "name": "목 지지대 + 높낮이 조절 의자",
            "reason": "목·경추 부담이 크거나 등받이 지지가 부족할 때 추천",
            "url": "https://www.coupang.com/vp/products/7830801595?itemId=21297117616&vendorItemId=88356855899&q=시디즈+의자",
        },
        {
            "name": "모니터 받침대",
            "reason": "시선각이 맞지 않거나 모니터가 낮아 목이 앞으로 숙여질 때 추천",
            "url": "https://www.coupang.com/vp/products/8641572134?itemId=25078452009&vendorItemId=92082407026",
        },
    ],
    "요추": [
        {
            "name": "등받이 요추 쿠션",
            "reason": "몸통이 앞으로 굽거나 허리 지지가 부족할 때 추천",
            "url": "https://drohbros.com/product/바른자세-허리쿠션-룸바/29/",
        },
        {
            "name": "허리 보호대",
            "reason": "허리 부담이 크고 장시간 앉아 있는 경우 보조용으로 추천",
            "url": "https://www.coupang.com/vp/products/8525671118?itemId=24684684526&vendorItemId=91509747922",
        },
    ],
        

    "손목": [
        {
            "name": "손목 받침대",
            "reason": "손목이 꺾이거나 키보드 사용 시 손목 부담이 클 때 추천",
            "url": "https://www.coupang.com/vp/products/8604154504?itemId=24950395992&vendorItemId=91962355876",
        },
        {
            "name": "버티컬 마우스",
            "reason": "마우스 사용 시 손목 회전 부담이 크거나 손목 통증 예방이 필요할 때 추천",
            "url": "https://www.coupang.com/vp/products/7295558262?itemId=20340965740&vendorItemId=86330525952&q=버티컬+마우스",
        },
    ],
    "무릎": [
        {
            "name": "사무실 발받침대",
            "reason": "무릎 각도가 맞지 않거나 발이 바닥에 안정적으로 닿지 않을 때 추천",
            "url": "https://www.coupang.com/vp/products/5227999402?itemId=23156160593&vendorItemId=74642538655&q=사무실+발받침대",
        },
    ],
}

# =========================================================
# 운동 및 스트레칭 추천 — RAG + LLM 기반
# =========================================================

# RAG 핵심 아이디어
# 1) 자세측정 결과에서 위험 항목을 추출합니다.
# 2) 검증된 운동 지식 DB(EXERCISE_KNOWLEDGE_BASE)에서 관련 운동만 검색합니다.
# 3) LLM은 검색된 운동 자료만 사용해서 추천 문장을 정리합니다.
# → LLM이 이상한 운동명/문장을 새로 만들어내는 문제를 줄입니다.

EXERCISE_KNOWLEDGE_BASE = [
    {
        "id": "cva_01",
        "targets": ["CVA", "목", "경추", "시선각"],
        "part": "목·경추",
        "name": "턱 당기기 운동",
        "method": "허리를 세우고 정면을 바라본 상태에서 턱을 목 쪽으로 천천히 당깁니다. 목 뒤가 길어진다는 느낌으로 유지한 뒤 힘을 풉니다.",
        "count": "10초 유지 × 5회",
        "effect": "앞으로 나온 머리 위치를 교정하고 목 뒤쪽 긴장을 완화하는 데 도움을 줍니다.",
        "caution": "고개를 아래로 숙이지 말고 턱만 뒤로 당기세요. 통증이 있으면 즉시 중단하세요.",
        "keywords": ["목굴곡각", "전방두부자세", "거북목", "목", "경추", "모니터"]
    },
    {
        "id": "cva_02",
        "targets": ["CVA", "목", "경추", "시선각"],
        "part": "목·어깨",
        "name": "상부 승모근 스트레칭",
        "method": "한 손을 의자 아래나 허벅지에 두고, 반대손으로 머리를 옆으로 천천히 기울여 목 옆을 늘립니다.",
        "count": "20초 유지 × 좌우 2회",
        "effect": "목과 어깨 위쪽 근육의 긴장을 줄이는 데 도움을 줍니다.",
        "caution": "반동을 주지 말고 천천히 움직이세요. 저림이 생기면 중단하세요.",
        "keywords": ["목굴곡각", "목", "어깨", "승모근", "경추", "시선각"]
    },
    {
        "id": "tia_01",
        "targets": ["TIA", "몸통", "허리", "등받이"],
        "part": "등·흉추",
        "name": "앉아서 가슴 열기",
        "method": "의자에 앉아 양손을 등 뒤로 깍지 끼고, 가슴을 천천히 열며 어깨를 뒤로 보냅니다.",
        "count": "20초 유지 × 2회",
        "effect": "등이 말리는 자세를 줄이고 가슴과 어깨 앞쪽 긴장을 완화하는 데 도움을 줍니다.",
        "caution": "허리를 과하게 꺾지 말고 가슴을 부드럽게 여세요.",
        "keywords": ["몸통굴곡각", "몸통", "허리", "등", "흉추", "굽은등"]
    },
    {
        "id": "tia_02",
        "targets": ["TIA", "몸통", "허리", "등받이"],
        "part": "허리·골반",
        "name": "골반 전후 기울이기",
        "method": "의자에 앉은 상태에서 골반을 천천히 앞으로 기울였다가 뒤로 말아줍니다. 허리가 부드럽게 움직이는 정도로만 반복합니다.",
        "count": "10회 반복",
        "effect": "오래 앉아 굳어진 허리와 골반 주변을 부드럽게 움직이는 데 도움을 줍니다.",
        "caution": "허리 통증이 심한 경우 범위를 작게 하거나 중단하세요.",
        "keywords": ["몸통굴곡각", "허리", "골반", "등받이", "요추"]
    },
    {
        "id": "wrist_01",
        "targets": ["손목"],
        "part": "손목·전완부",
        "name": "손목 굴곡근 스트레칭",
        "method": "팔을 앞으로 뻗고 손바닥이 위를 향하게 합니다. 반대손으로 손가락을 아래쪽으로 천천히 당겨 손목 안쪽을 늘립니다.",
        "count": "20초 유지 × 좌우 2회",
        "effect": "키보드와 마우스 사용으로 긴장된 손목 안쪽과 전완부를 이완하는 데 도움을 줍니다.",
        "caution": "손목을 억지로 꺾지 말고 편안하게 늘어나는 범위에서 실시하세요.",
        "keywords": ["손목각도", "손목", "키보드", "마우스", "전완부"]
    },
    {
        "id": "wrist_02",
        "targets": ["손목"],
        "part": "손목·전완부",
        "name": "손목 신전근 스트레칭",
        "method": "팔을 앞으로 뻗고 손등이 위를 향하게 합니다. 반대손으로 손등을 몸 쪽으로 천천히 당겨 손목 바깥쪽을 늘립니다.",
        "count": "20초 유지 × 좌우 2회",
        "effect": "손목 바깥쪽과 전완부의 긴장을 완화하는 데 도움을 줍니다.",
        "caution": "저림이나 날카로운 통증이 느껴지면 즉시 중단하세요.",
        "keywords": ["손목각도", "손목", "키보드", "마우스", "전완부"]
    },
    {
        "id": "knee_01",
        "targets": ["무릎"],
        "part": "무릎·하체",
        "name": "앉아서 햄스트링 스트레칭",
        "method": "의자 앞쪽에 앉아 한쪽 다리를 앞으로 뻗고 발끝을 몸 쪽으로 당깁니다. 허리를 세운 상태에서 상체를 살짝 앞으로 기울입니다.",
        "count": "20초 유지 × 좌우 2회",
        "effect": "허벅지 뒤쪽 긴장을 줄이고 오래 앉아 생기는 하체 뻣뻣함을 완화하는 데 도움을 줍니다.",
        "caution": "무릎을 억지로 펴지 말고 통증 없는 범위에서 실시하세요.",
        "keywords": ["무릎각도", "무릎", "하체", "햄스트링", "의자높이"]
    },
    {
        "id": "knee_02",
        "targets": ["무릎"],
        "part": "발목·종아리",
        "name": "발목 펌프 운동",
        "method": "의자에 앉아 발뒤꿈치를 바닥에 둔 채 발끝을 들어 올렸다가 내립니다. 이어서 발끝을 바닥에 두고 발뒤꿈치를 들어 올립니다.",
        "count": "20회 반복",
        "effect": "종아리 근육을 움직여 하체 순환을 돕습니다.",
        "caution": "발목에 통증이 있으면 움직임을 작게 하세요.",
        "keywords": ["무릎각도", "하체", "종아리", "발목", "순환"]
    },
    {
        "id": "gaze_01",
        "targets": ["시선각", "CVA"],
        "part": "눈·목",
        "name": "20초 원거리 보기",
        "method": "모니터에서 시선을 떼고 6m 이상 떨어진 곳을 20초 동안 편안하게 바라봅니다. 이후 목과 어깨 힘을 가볍게 풉니다.",
        "count": "작업 중 20~30분마다 1회",
        "effect": "눈의 피로와 목 주변 긴장을 줄이는 데 도움을 줍니다.",
        "caution": "어지러움이 있으면 눈을 감고 잠시 쉬세요.",
        "keywords": ["시선각", "모니터", "눈", "목", "VDT"]
    },
    {
        "id": "desk_01",
        "targets": ["책상높이", "손목"],
        "part": "어깨·상지",
        "name": "어깨 올렸다 내리기",
        "method": "양쪽 어깨를 귀 쪽으로 천천히 올린 뒤, 힘을 빼며 아래로 부드럽게 내립니다.",
        "count": "10회 반복",
        "effect": "책상 높이와 키보드 사용으로 긴장된 어깨 주변 근육을 이완하는 데 도움을 줍니다.",
        "caution": "목에 힘을 과하게 주지 말고 어깨만 부드럽게 움직이세요.",
        "keywords": ["작업대높이", "책상높이", "어깨", "상지", "키보드"]
    },
    {
        "id": "chair_01",
        "targets": ["등받이", "TIA"],
        "part": "허리·등",
        "name": "의자 등받이 기대기 연습",
        "method": "엉덩이를 의자 뒤쪽까지 넣고 허리를 등받이에 가볍게 붙입니다. 턱을 살짝 당기고 어깨 힘을 빼며 30초간 유지합니다.",
        "count": "30초 유지 × 3회",
        "effect": "허리 지지를 회복하고 몸통이 앞으로 굽는 습관을 줄이는 데 도움을 줍니다.",
        "caution": "등받이에 기대도 허리가 불편하면 쿠션 높이나 의자 깊이를 조절하세요.",
        "keywords": ["등받이", "의자", "허리", "요추", "몸통굴곡각"]
    }
]


def _html_escape(text):
    text = str(text)
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
    )


def extract_bad_posture_items(result):
    all_data = {**result.get("posture", {}), **result.get("env", {})}
    bad_items = []

    for key, value in all_data.items():
        if key not in FEEDBACK:
            continue

        measured_value, is_good, raw = value

        if measured_value == "인식 불가":
            continue

        if not is_good:
            bad_items.append({
                "key": key,
                "label": FEEDBACK[key]["label"],
                "value": measured_value,
                "range": FEEDBACK[key]["range"],
                "reason": FEEDBACK[key]["bad"],
            })

    return bad_items


def retrieve_exercises_by_rag(bad_items, top_k=6):
    """
    간단한 RAG 검색 함수입니다.
    외부 라이브러리 없이 현재 자세 위험 항목과 운동 지식 DB를 매칭합니다.
    추후 ChromaDB/FAISS로 바꿔도 이 함수만 교체하면 됩니다.
    """
    if bad_items:
        query_terms = []
        for item in bad_items:
            query_terms.extend([
                item.get("key", ""),
                item.get("label", ""),
                item.get("reason", ""),
            ])
    else:
        query_terms = ["목", "허리", "손목", "어깨", "하체", "사무실"]

    query = " ".join(query_terms)
    scored = []

    for ex in EXERCISE_KNOWLEDGE_BASE:
        score = 0

        for target in ex.get("targets", []):
            if target and target in query:
                score += 5

        for kw in ex.get("keywords", []):
            if kw and kw in query:
                score += 3

        for item in bad_items:
            if item.get("key") in ex.get("targets", []):
                score += 8
            if item.get("label") in ex.get("keywords", []):
                score += 4

        if not bad_items:
            score += 1

        if score > 0:
            scored.append((score, ex))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 같은 운동 중복 제거
    selected = []
    seen_names = set()
    for _, ex in scored:
        if ex["name"] in seen_names:
            continue
        selected.append(ex)
        seen_names.add(ex["name"])
        if len(selected) >= top_k:
            break

    return selected


def format_retrieved_exercises_for_prompt(exercises):
    if not exercises:
        return "검색된 운동 자료가 없습니다."

    rows = []
    for i, ex in enumerate(exercises, start=1):
        rows.append(
            f"[{i}] 운동명: {ex['name']}\n"
            f"- 관리 부위: {ex['part']}\n"
            f"- 방법: {ex['method']}\n"
            f"- 횟수/시간: {ex['count']}\n"
            f"- 효과: {ex['effect']}\n"
            f"- 주의사항: {ex['caution']}"
        )
    return "\n\n".join(rows)


def build_rag_exercise_prompt(bad_items, retrieved_exercises):
    if bad_items:
        anomaly_text = "\n".join([
            f"- {item['label']} / 측정값: {item['value']} / 기준: {item['range']} / 문제: {item['reason'].replace(chr(10), ' ')}"
            for item in bad_items
        ])
    else:
        anomaly_text = "- 기준 범위를 벗어난 항목은 없습니다. 예방 목적의 가벼운 루틴을 추천합니다."

    retrieved_text = format_retrieved_exercises_for_prompt(retrieved_exercises)

    return f"""
너는 사무직 사용자의 자세 개선을 돕는 운동 추천 전문가다.
아래 [검색된 운동 자료]에 있는 내용만 사용해서 추천문을 작성해라.

[절대 규칙]
- [검색된 운동 자료]에 없는 운동명, 방법, 횟수, 효과, 주의사항을 새로 만들지 마라.
- 운동 설명은 자료의 문장을 자연스럽게 정리하는 정도만 허용한다.
- 반드시 한국어만 사용한다.
- 영어, 일본어, 중국어, 독일어를 사용하지 않는다.
- 문장은 짧고 자연스럽게 작성한다.
- 통증, 저림, 어지러움이 있으면 즉시 중단하고 전문가와 상담하라는 문구를 포함한다.
- 의료 진단처럼 단정하지 않는다.

[자세측정 위험 항목]
{anomaly_text}

[검색된 운동 자료]
{retrieved_text}

[출력 형식]
## 오늘의 핵심 관리 부위
- 2줄 이내로 작성

## 추천 운동 및 스트레칭
### 1. 운동명
- 방법:
- 횟수/시간:
- 효과:
- 주의사항:

### 2. 운동명
- 방법:
- 횟수/시간:
- 효과:
- 주의사항:

### 3. 운동명
- 방법:
- 횟수/시간:
- 효과:
- 주의사항:

## 사무실 1분 루틴
1.
2.
3.

## 주의
- 통증·저림·어지러움이 있으면 즉시 중단하고 전문가와 상담하세요.
"""


def call_local_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "repeat_penalty": 1.08,
                },
            },
            timeout=180,
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()

        st.error(f"Ollama 응답 오류: {response.status_code}")
        st.code(response.text)
        return None

    except requests.exceptions.ConnectionError:
        st.error("Ollama 서버가 실행 중이 아닙니다. PowerShell에서 `ollama serve`를 실행해주세요.")
        return None

    except requests.exceptions.Timeout:
        st.error("Ollama 응답 시간이 초과되었습니다. qwen2.5:3b처럼 더 작은 모델을 사용해주세요.")
        return None

    except Exception as e:
        st.error(f"Ollama 연결 오류: {e}")
        return None


def build_retrieved_exercise_cards_html(exercises):
    if not exercises:
        return ""

    html = ""
    for ex in exercises:
        html += f"""
<div class="exercise-card">
    <div class="exercise-part">{_html_escape(ex['part'])}</div>
    <div class="exercise-name">{_html_escape(ex['name'])}</div>
    <div class="exercise-desc">{_html_escape(ex['method'])}</div>
    <div class="exercise-count">{_html_escape(ex['count'])}</div>
</div>
"""
    return html


def render_exercise_recommendation_page():
    page_header(
        "운동 및 스트레칭 추천",
        "자세측정 결과에서 위험 부위를 추출하고, 운동 지식 DB를 검색한 뒤 LLM으로 개인 맞춤 루틴을 생성합니다.",
    )

    result = st.session_state.get("latest_result")

    if result is None:
        st.info("운동 및 스트레칭 추천을 보려면 먼저 왼쪽 메뉴의 자세측정에서 AI 자세 분석을 실행해주세요.")
        return

    bad_items = extract_bad_posture_items(result)
    bad_labels = [item["label"] for item in bad_items]
    retrieved_exercises = retrieve_exercises_by_rag(bad_items, top_k=6)
    prompt = build_rag_exercise_prompt(bad_items, retrieved_exercises)

    st.markdown(
        f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>추천 기준</span>
        <span class="fit-badge badge-blue">RAG + LLM</span>
    </div>
    <div style="font-size:14px;line-height:1.8;color:#5E718D;">
        기준 범위를 벗어난 항목:
        <b style="color:#0F1E36;">{" · ".join(bad_labels) if bad_labels else "없음"}</b><br>
        검색된 운동 자료:
        <b style="color:#2563EB;">{len(retrieved_exercises)}개</b><br>
        운동명과 방법은 운동 지식 DB에서 가져오고, LLM은 검색된 자료 안에서만 추천문을 정리합니다.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<style>
.exercise-grid {
    display:grid;
    grid-template-columns:repeat(3, minmax(0, 1fr));
    gap:16px;
    margin-top:10px;
}
.exercise-card {
    background:#FFFFFF;
    border:1px solid rgba(37,99,235,0.12);
    border-radius:20px;
    padding:20px;
    box-shadow:0 10px 30px rgba(37,99,235,0.03);
    min-height:210px;
    transition:all 0.2s ease;
}
.exercise-card:hover {
    transform:translateY(-4px);
    box-shadow:0 14px 40px rgba(37,99,235,0.08);
    border-color:rgba(37,99,235,0.25);
}
.exercise-part {
    display:inline-flex;
    padding:5px 12px;
    border-radius:999px;
    background:#ECFDF5;
    color:#10B981;
    border:1px solid rgba(16,185,129,0.2);
    font-size:11px;
    font-weight:800;
    margin-bottom:12px;
}
.exercise-name {
    font-size:18px;
    font-weight:900;
    color:#0F1E36;
    margin-bottom:8px;
}
.exercise-desc {
    font-size:13px;
    line-height:1.7;
    color:#5E718D;
    min-height:86px;
    margin-bottom:10px;
}
.exercise-count {
    font-size:13px;
    font-weight:800;
    color:#2563EB;
}
@media (max-width: 1000px) {
    .exercise-grid {
        grid-template-columns:repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 700px) {
    .exercise-grid {
        grid-template-columns:1fr;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )

    with st.expander("RAG 검색 결과 보기"):
        st.code(format_retrieved_exercises_for_prompt(retrieved_exercises), language="text")

    with st.expander("LLM에 전달되는 RAG 프롬프트 보기"):
        st.code(prompt, language="text")

    if st.button("RAG 기반 LLM 운동 추천 생성", use_container_width=True):
        with st.spinner("운동 지식 DB 검색 결과를 바탕으로 LLM이 추천 루틴을 정리하는 중입니다..."):
            llm_answer = call_local_llm(prompt)

        if llm_answer:
            st.markdown(
                f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>RAG 기반 맞춤 운동 추천</span>
        <span class="fit-badge badge-green">Generated</span>
    </div>
    <div style="font-size:14px;line-height:1.9;color:#172033;white-space:pre-wrap;">
{_html_escape(llm_answer)}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.warning("LLM 연결이 되지 않아 검색된 운동 자료 카드만 표시합니다. Ollama와 모델 이름을 확인해주세요.")

    st.markdown("### 검색된 운동 자료")
    st.markdown('<div class="exercise-grid">', unsafe_allow_html=True)
    st.markdown(build_retrieved_exercise_cards_html(retrieved_exercises), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("※ 운동 추천은 자세 개선 참고용이며, 통증·저림·어지러움이 있으면 즉시 중단하고 전문가와 상담하세요.")

def map_result_to_product_categories(result):
    all_data = {**result.get("posture", {}), **result.get("env", {})}
    categories = []

    # 경추: 목굴곡각, 시선각 문제
    if not all_data.get("CVA", ("", True, None))[1] or not all_data.get("시선각", ("", True, None))[1]:
        categories.append("경추")

    # 요추: 몸통굴곡각, 등받이 문제
    if not all_data.get("TIA", ("", True, None))[1] or not all_data.get("등받이", ("", True, None))[1]:
        categories.append("요추")

    # 팔꿈치: 팔꿈치 각도, 책상높이 문제
    if not all_data.get("팔꿈치", ("", True, None))[1] or not all_data.get("책상높이", ("", True, None))[1]:
        categories.append("팔꿈치")

    # 손목
    if not all_data.get("손목", ("", True, None))[1]:
        categories.append("손목")

    # 무릎
    if not all_data.get("무릎", ("", True, None))[1]:
        categories.append("무릎")

    return list(dict.fromkeys(categories))


def render_product_recommendation_page():
    page_header(
        "제품 추천",
        "자세측정 결과에서 기준 범위를 벗어난 부위에 맞춰 필요한 제품을 추천합니다.",
    )

    result = st.session_state.get("latest_result")

    if result is None:
        st.info("제품 추천을 보려면 먼저 왼쪽 메뉴의 자세측정에서 AI 자세 분석을 실행해주세요.")
        return

    categories = map_result_to_product_categories(result)

    if not categories:
        st.success("현재 자세측정 결과상 필수 추천 제품은 없습니다. 현재 작업환경을 잘 유지해주세요.")
        return

    all_data = {**result.get("posture", {}), **result.get("env", {})}
    bad_labels = [
        FEEDBACK[k]["label"]
        for k, v in all_data.items()
        if k in FEEDBACK and not v[1]
    ]

    st.markdown(
        f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>추천 기준</span>
        <span class="fit-badge badge-blue">AI Product Match</span>
    </div>
    <div style="font-size:14px;line-height:1.8;color:#5E718D;">
        기준 범위를 벗어난 항목:
        <b style="color:#0F1E36;">{" · ".join(bad_labels) if bad_labels else "없음"}</b><br>
        추천 카테고리:
        <b style="color:#2563EB;">{" · ".join(categories)}</b>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<style>
.product-grid {
    display:grid;
    grid-template-columns:repeat(3, minmax(0, 1fr));
    gap:16px;
    margin-top:10px;
}
.product-card {
    background:#FFFFFF;
    border:1px solid rgba(37,99,235,0.12);
    border-radius:20px;
    padding:20px;
    box-shadow:0 10px 30px rgba(37,99,235,0.03);
    min-height:210px;
    transition:all 0.2s ease;
}
.product-card:hover {
    transform:translateY(-4px);
    box-shadow:0 14px 40px rgba(37,99,235,0.08);
    border-color:rgba(37,99,235,0.25);
}
.product-category {
    display:inline-flex;
    padding:5px 12px;
    border-radius:999px;
    background:var(--soft-blue);
    color:var(--blue);
    border:1px solid rgba(37,99,235,0.2);
    font-size:11px;
    font-weight:800;
    margin-bottom:12px;
}
.product-name {
    font-size:18px;
    font-weight:900;
    color:#0F1E36;
    margin-bottom:8px;
    letter-spacing:-0.4px;
}
.product-reason {
    font-size:13px;
    line-height:1.7;
    color:#5E718D;
    min-height:68px;
    margin-bottom:14px;
}
.product-link {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:100%;
    padding:10px 12px;
    border-radius:12px;
    background:linear-gradient(135deg, #1E40AF 0%, #2563EB 100%) !important;
    color:white !important;
    text-decoration:none !important;
    font-size:13px;
    font-weight:750;
    transition: all 0.2s ease;
}
.product-link:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
}
@media (max-width: 1000px) {
    .product-grid {
        grid-template-columns:repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 700px) {
    .product-grid {
        grid-template-columns:1fr;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )

    html = '<div class="product-grid">'

    for category in categories:
        for product in PRODUCT_RECOMMENDATIONS.get(category, []):
            html += f"""
<div class="product-card">
    <div class="product-category">{category}</div>
    <div class="product-name">{product["name"]}</div>
    <div class="product-reason">{product["reason"]}</div>
    <a class="product-link" href="{product["url"]}" target="_blank">
        제품 보러가기
    </a>
</div>
"""

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

    st.caption("※ 제품 추천은 자세측정 결과 기반의 작업환경 개선 참고용이며, 의료적 진단이나 치료 목적이 아닙니다.")    

# =========================================================
# 근골격계 리포트 전용 UI — 직관형 그래프 + 2단계 상세 피드백
# =========================================================

def _report_level_style(level):
    if level == "정상":
        return {"color": "#45B86B", "soft": "#F0FBF4", "badge": "정상", "desc": "양호", "marker": 17}
    if level == "위험":
        return {"color": "#F2527D", "soft": "#FFF1F5", "badge": "위험", "desc": "관리 필요", "marker": 83}
    return {"color": "#AEB6C2", "soft": "#F2F4F7", "badge": "제외", "desc": "기준점 부족", "marker": 50}


def _report_feedback_text(key, is_normal):
    fb = FEEDBACK.get(key, {})
    text = fb.get("good", "") if is_normal else fb.get("bad", "")
    return text.replace("\n", "<br>")


def _report_metric_icon(key):
    base_dir = Path(__file__).resolve().parent

    icon_map = {
        "CVA": base_dir / "assets" / "metric_icons" / "cva.png",
        "TIA": base_dir / "assets" / "metric_icons" / "tia.png",
        "팔꿈치": base_dir / "assets" / "metric_icons" / "elbow.png",
        "무릎": base_dir / "assets" / "metric_icons" / "knee.png",
        "손목": base_dir / "assets" / "metric_icons" / "wrist.png",
        "시선각": base_dir / "assets" / "metric_icons" / "gaze.png",
        "책상높이": base_dir / "assets" / "metric_icons" / "desk.png",
        "등받이": base_dir / "assets" / "metric_icons" / "chair.png",
    }

    img_src = image_to_base64_src(icon_map.get(key, ""))
    if img_src:
        return f"<img class='ms-metric-img' src='{img_src}' alt='{key}'>"

    return "📍"


def _report_range_html(key):
    rule = CLINICAL_RULES.get(key, {})
    if is_three_level_metric(key):
        return f'''
        <div class="report-range-box">
            <div><b class="report-range-good">정상:</b> {rule.get("normal", "-")}</div>
            <div><b class="report-range-risk">위험:</b> {rule.get("risk", "-")}</div>
        </div>
        '''
    return f'''
    <div class="report-range-box">
        <div><b class="report-range-good">정상:</b> {rule.get("normal", "-")}</div>
        <div><b class="report-range-risk">위험:</b> {rule.get("risk", "-")}</div>
    </div>
    '''


def generate_legal_pdf(is_vdt_over_4h: bool):
    """
    법정 근골격계부담작업 체크리스트 PDF 생성
    - landscape A4
    - 제1호~제11호 전체 항목 포함
    - 사용자가 제1호 VDT 작업 4시간 이상 여부를 선택하면
      단위작업명 '사무작업' 행의 제1호 칸만 O/X로 반영
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

    buffer = BytesIO()
    font_name = get_korean_pdf_font()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=7 * mm,
        rightMargin=7 * mm,
        topMargin=7 * mm,
        bottomMargin=7 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="legal_title",
        fontName=font_name,
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.black,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="legal_subtitle",
        fontName=font_name,
        fontSize=14,
        leading=18,
        alignment=0,
        textColor=colors.black,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="legal_cell",
        fontName=font_name,
        fontSize=5.7,
        leading=7.0,
        alignment=1,
        textColor=colors.black,
        wordWrap="CJK",
    ))
    styles.add(ParagraphStyle(
        name="legal_header",
        fontName=font_name,
        fontSize=6.5,
        leading=8.0,
        alignment=1,
        textColor=colors.black,
        wordWrap="CJK",
    ))
    styles.add(ParagraphStyle(
        name="legal_mark",
        fontName=font_name,
        fontSize=9,
        leading=10,
        alignment=1,
        textColor=colors.black,
    ))
    styles.add(ParagraphStyle(
        name="legal_note",
        fontName=font_name,
        fontSize=9,
        leading=13,
        alignment=0,
        textColor=colors.black,
        wordWrap="CJK",
    ))

    def P(text, style="legal_cell"):
        text = "" if text is None else str(text)
        return Paragraph(text.replace("\n", "<br/>"), styles[style])

    story = []
    story.append(Paragraph("3. 근골격계부담작업 체크리스트 작성 방법", styles["legal_title"]))
    story.append(Paragraph("3-1. 근골격계부담작업 체크리스트 예시", styles["legal_subtitle"]))

    # ── 상단 고정 정보 표 ───────────────────────────────────────────
    top_table = Table(
        [
            [P("사업장명", "legal_header"), P("아시아경제교육센터", "legal_header"),
             P("조사 일자", "legal_header"), P("2026년 4월 30일", "legal_header"),
             P("조사자", "legal_header"), P("김OO", "legal_header")],
            [P("부서명", "legal_header"), P("4팀", "legal_header"),
             P("작업 내용", "legal_header"), P("사무작업", "legal_header"), "", ""],
        ],
        colWidths=[24*mm, 68*mm, 24*mm, 54*mm, 24*mm, 62*mm],
        rowHeights=[8*mm, 8*mm],
    )
    top_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
        ("BOX", (0, 0), (-1, -1), 1.2, colors.black),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAEAEA")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EAEAEA")),
        ("BACKGROUND", (4, 0), (4, -1), colors.HexColor("#EAEAEA")),
        ("SPAN", (4, 1), (5, 1)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(top_table)
    story.append(Spacer(1, 2))

    # ── 첨부 사진 양식에 맞춘 제1호~제11호 표 ───────────────────────
    office_mark = "O" if is_vdt_over_4h else "X"

    # 표 구조: [좌측 항목명 1칸] + [단위작업명 1칸] + [제1호~제11호 11칸]
    # 기존에는 제1호 칸 앞에 단위작업명 칸이 빠져서 O/X가 한 칸씩 오른쪽으로 밀렸습니다.
    # 아래처럼 모든 행을 13칸으로 맞추고, 상단 설명 행은 좌측 2칸을 병합합니다.
    header = ["구분", ""] + [f"{i})" for i in range(1, 12)]

    image_row = [
        "", "",
        "💻", "🔁", "🙆", "↯", "🧎", "🤏", "✋", "📦", "🏋", "📦", "🔨",
    ]

    exposure_time = [
        "노출 시간", "",
        "하루에\n총 4시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "-",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
    ]

    exposure_freq = [
        "노출 빈도", "",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "하루에\n총 10회 이상",
        "하루에\n총 25회 이상",
        "분당\n2회 이상",
        "시간당\n10회 이상",
    ]

    body_part = [
        "신체 부위", "",
        "손, 손가락,\n팔, 어깨",
        "목, 어깨, 손목,\n손, 팔꿈치",
        "어깨, 팔",
        "목, 허리",
        "다리, 무릎",
        "손가락",
        "손",
        "허리",
        "손, 무릎",
        "허리",
        "목, 무릎,\n팔꿈치",
    ]

    work_detail = [
        "작업 자세\n및\n내용", "",
        "집중적인\n입력 작업\n(마우스·키보드\n사용)",
        "같은 동작\n반복 작업",
        "머리 위에 손\n또는 팔꿈치가\n몸통 뒤쪽에\n위치",
        "구부리거나\n비트는 자세\n(지지되지 않은\n상태, 자세변경\n불가)",
        "쪼그리고\n앉거나 무릎을\n굽힘",
        "한 손가락 집어\n올리거나 쥐는\n작업\n(지지되지\n않은 상태)",
        "물건을 한손으로\n들거나 잡는\n작업",
        "물건을 드는\n작업",
        "어깨 위에서\n팔을 뻗은\n상태에서\n물건을 드는\n작업",
        "물건을 드는\n작업",
        "반복적인 충격",
    ]

    weight = [
        "무게", "",
        "-",
        "-",
        "-",
        "-",
        "-",
        "1kg 이상의\n물건\n또는 2kg 이상에\n상응하는 힘으로\n쥐기",
        "4.5kg 이상의\n물건 들기\n또는 동일한\n힘으로 쥐기",
        "25kg 이상",
        "10kg 이상",
        "4.5kg 이상",
        "-",
    ]

    table_data = [
        [P(x, "legal_header") for x in header],
        [P(x, "legal_header") for x in image_row],
        [P(x, "legal_cell") for x in exposure_time],
        [P(x, "legal_cell") for x in exposure_freq],
        [P(x, "legal_cell") for x in body_part],
        [P(x, "legal_cell") for x in work_detail],
        [P(x, "legal_cell") for x in weight],
        [P("단위작업명", "legal_header"), P("설계작업", "legal_header")] + [P("X", "legal_mark") for _ in range(11)],
        ["", P("사무작업", "legal_header")] + [P(office_mark if i == 1 else "X", "legal_mark") for i in range(1, 12)],
        ["", P("현장작업", "legal_header")] + [P("X", "legal_mark") for _ in range(11)],
        ["", P("시운전", "legal_header")] + [P("X", "legal_mark") for _ in range(11)],
    ]

    col_widths = [20*mm, 18*mm] + [21.6*mm] * 11
    row_heights = [8*mm, 10*mm, 9*mm, 10*mm, 10*mm, 24*mm, 24*mm, 8*mm, 8*mm, 8*mm, 8*mm]

    main_table = Table(table_data, colWidths=col_widths, rowHeights=row_heights, repeatRows=1)
    main_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("GRID", (0, 0), (-1, -1), 0.55, colors.black),
        ("BOX", (0, 0), (-1, -1), 1.2, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#E6E6E6")),
        ("BACKGROUND", (0, 2), (1, 6), colors.HexColor("#E6E6E6")),
        ("BACKGROUND", (0, 7), (1, 10), colors.HexColor("#F2F2F2")),
        ("SPAN", (0, 0), (1, 0)),
        ("SPAN", (0, 1), (1, 1)),
        ("SPAN", (0, 2), (1, 2)),
        ("SPAN", (0, 3), (1, 3)),
        ("SPAN", (0, 4), (1, 4)),
        ("SPAN", (0, 5), (1, 5)),
        ("SPAN", (0, 6), (1, 6)),
        ("SPAN", (0, 7), (0, 10)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.4),
        ("TOPPADDING", (0, 0), (-1, -1), 1.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
    ]))

    story.append(main_table)
    story.append(Spacer(1, 5))

    story.append(Paragraph(
        "아래 체크리스트는 작업 내용, 단위작업명은 본인 부서에 맞게 작성하고, "
        "해당 작업 유무(O/X) 표시는 근골격계부담작업 체크리스트 평가 방법을 참고하여 작성합니다.",
        styles["legal_note"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


def render_legal_checklist_page():
    page_header(
        "근골격계부담작업 체크리스트",
        "제1호 VDT 작업 해당 여부만 확인하고 법정 PDF를 생성합니다.",
    )

    st.markdown("""
<div class="fit-card">
    <div class="fit-card-title">
        <span>제1호 VDT 작업 확인</span>
        <span class="fit-badge badge-blue">Legal Checklist</span>
    </div>
    <div style="font-size:14px;line-height:1.8;color:#667085;">
        <b>제1호 기준</b><br>
        하루에 4시간 이상 집중적으로 자료입력 등을 위해 키보드 또는 마우스를 조작하는 작업에 해당하는지 확인합니다.<br><br>
        화면에는 제1호(VDT 작업) 안내만 표시하고, PDF에는 법정 양식에 맞춰 제1호부터 제11호까지 전체 표를 포함합니다.
    </div>
</div>
""", unsafe_allow_html=True)

    is_vdt_over_4h = st.radio(
        "하루에 4시간 이상 집중적으로 자료입력 등을 위해 키보드 또는 마우스를 조작하는 작업을 했나요?",
        ["네", "아니오"],
        horizontal=True,
        key="legal_vdt_over_4h",
    ) == "네"

    if is_vdt_over_4h:
        st.success("제1호 VDT 작업에 해당합니다. PDF의 단위작업명 '사무작업' 행 제1호 칸에 O가 표시됩니다.")
    else:
        st.info("제1호 VDT 작업에 해당하지 않습니다. PDF의 단위작업명 '사무작업' 행 제1호 칸에 X가 표시됩니다.")

    pdf_buffer = generate_legal_pdf(is_vdt_over_4h)

    st.download_button(
        label="📋 근골격계부담작업 체크리스트 PDF 다운로드",
        data=pdf_buffer,
        file_name="legal_musculoskeletal_checklist_20260430.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def render_report():
    page_header(
        "근골격계 리포트",
        "7가지 항목의 자세 및 환경을 종합적으로 분석했습니다.",
    )

    result = st.session_state.get("latest_result")

    if result is None:
        st.info("리포트를 생성하려면 먼저 자세 측정을 실행해주세요.")
        return

    pdf_buffer = make_musculoskeletal_report_pdf(result)
    st.download_button(
        label="📄 근골격계 리포트 PDF 저장",
        data=pdf_buffer,
        file_name=f"fit_me_up_musculoskeletal_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    all_data = {**result["posture"], **result["env"]}
    
    st.markdown("---")
    st.subheader(":clipboard: 법정 문서(근골격계부담작업 체크리스트) 자동 생성")
    st.caption("법정 유해요인 조사 결과(제1호)를 PDF로 출력합니다")

    is_vdt_over_4h_report = st.radio(
        "하루에 4시간 이상 집중적으로 자료입력 등을 위해 키보드 또는 마우스를 조작했나요?",
        ["네", "아니오"],
        horizontal=True,
        key="report_legal_vdt_over_4h",
    ) == "네"

    legal_pdf_buffer = generate_legal_pdf(is_vdt_over_4h_report)

    st.download_button(
        label="📋 법정 유해요인 조사 PDF 다운로드",
        data=legal_pdf_buffer,
        file_name="legal_musculoskeletal_checklist_20260430.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    def _level_info(level):
        if level == "정상":
            return {"label": "GOOD", "kr": "양호", "color": "#2FB35A", "soft": "#EAF8EF", "class": "good"}
        if level == "위험":
            return {"label": "BAD", "kr": "위험", "color": "#F43F5E", "soft": "#FFE8EE", "class": "bad"}
        return {"label": "EXCLUDED", "kr": "제외", "color": "#98A2B3", "soft": "#F2F4F7", "class": "none"}

    def _fmt_value(key, value, raw):
        if raw is None:
            return "인식 불가"
        if key in ["책상높이"]:
            return f"{float(raw):.3f}"
        if key in ["등받이"]:
            return f"{float(raw) * 100:.1f}%"
        return value

    def _subtitle(key):
        return {
            "CVA": "목의 전방 기울기 각도",
            "TIA": "몸통의 전방 굴곡 각도",
            "팔꿈치": "팔꿈치 굴곡 각도",
            "무릎": "무릎 굴곡 각도",
            "손목": "손목 굴곡 각도",
            "시선각": "수평선 대비 시선 각도",
            "책상높이": "팔꿈치 대비 작업대 높이",
            "등받이": "등받이 지지 비율",
        }.get(key, "측정 지표")

    def _icon(key):
        base_dir = Path(__file__).resolve().parent

        icon_map = {
            "CVA": base_dir / "assets" / "metric_icons" / "cva.png",
            "TIA": base_dir / "assets" / "metric_icons" / "tia.png",
            "팔꿈치": base_dir / "assets" / "metric_icons" / "elbow.png",
            "무릎": base_dir / "assets" / "metric_icons" / "knee.png",
            "손목": base_dir / "assets" / "metric_icons" / "wrist.png",
            "시선각": base_dir / "assets" / "metric_icons" / "gaze.png",
            "책상높이": base_dir / "assets" / "metric_icons" / "desk.png",
            "등받이": base_dir / "assets" / "metric_icons" / "chair.png",
        }

        img_src = image_to_base64_src(icon_map.get(key, ""))
        if img_src:
            return f"<img class='ms-metric-img' src='{img_src}' alt='{key}'>"

        return "📍"

    def _bar_meta(key):
        # min/max는 그래프 표시용 범위입니다. 실제 판정은 classify_posture_level() 기준을 사용합니다.
        return {
            "CVA": {"min": 0, "max": 40, "unit": "°", "segments": [(0, 20, "good"), (20, 40, "bad")], "ticks": [0, 20, 40]},
            "TIA": {"min": 0, "max": 45, "unit": "°", "segments": [(0, 20, "good"), (20, 45, "bad")], "ticks": [0, 20, 45]},
            "팔꿈치": {"min": 70, "max": 140, "unit": "°", "segments": [(70, 90, "bad"), (90, 120, "good"), (120, 140, "bad")], "ticks": [90, 120]},
            "무릎": {"min": 65, "max": 125, "unit": "°", "segments": [(65, 85, "bad"), (85, 100, "good"), (100, 125, "bad")], "ticks": [85, 100]},
            "손목": {"min": -30, "max": 30, "unit": "°", "segments": [(-30, -15, "bad"), (-15, 15, "good"), (15, 30, "bad")], "ticks": [-15, 0, 15]},
            "시선각": {"min": -10, "max": 45, "unit": "°", "segments": [(-10, 10, "bad"), (10, 15, "good"), (15, 45, "bad")], "ticks": [10, 15]},
            "책상높이": {"min": -0.05, "max": 0.15, "unit": "", "segments": [(-0.05, 0.05, "good"), (0.05, 0.15, "bad")], "ticks": [0, 0.05]},
            "등받이": {"min": 0, "max": 0.50, "unit": "%", "segments": [(0, 0.20, "good"), (0.20, 0.50, "bad")], "ticks": [0, 0.20]},
        }.get(key)

    def _pct(value, min_v, max_v):
        if max_v == min_v:
            return 0
        return max(0, min(100, (float(value) - min_v) / (max_v - min_v) * 100))

    def _tick_label(key, v, unit):
        if key == "등받이":
            return f"{int(round(v * 100))}%"
        if key == "책상높이":
            return "0" if abs(v) < 1e-9 else f"{v:.2f}"
        if key == "손목":
            if abs(v) < 1e-9:
                return "중립"
            return f"{'+' if v > 0 else ''}{int(v) if float(v).is_integer() else v}°"
        return f"{int(v) if float(v).is_integer() else v}{unit}"

    def _range_lines(key):
        rule = CLINICAL_RULES.get(key, {})
        return f"""
        <div><b class='range-good'>정상</b><span>{rule.get('normal', '-')}</span></div>
        <div><b class='range-bad'>위험</b><span>{rule.get('risk', '-')}</span></div>
        """


    def _bar_html(key, raw, level):
        meta = _bar_meta(key)
        if meta is None:
            return ""
        min_v, max_v = meta["min"], meta["max"]
        seg_html = ""
        for s, e, cls in meta["segments"]:
            left = _pct(s, min_v, max_v)
            width = max(0, _pct(e, min_v, max_v) - left)
            seg_html += f"<div class='ms-seg seg-{cls}' style='left:{left:.4f}%;width:{width:.4f}%;'></div>"

        tick_html = ""
        for t in meta["ticks"]:
            left = _pct(t, min_v, max_v)
            tick_html += f"<div class='ms-tick' style='left:{left:.4f}%;'></div><div class='ms-tick-label' style='left:{left:.4f}%;'>{_tick_label(key, t, meta['unit'])}</div>"

        if raw is None:
            marker_html = "<div class='ms-missing-marker'></div>"
        else:
            marker_left = _pct(raw, min_v, max_v)
            marker_html = f"<div class='ms-marker marker-{_level_info(level)['class']}' style='left:{marker_left:.4f}%;'></div>"

        return f"""
        <div class='ms-bar-wrap'>
            <div class='ms-bar'>
                {seg_html}
                {tick_html}
                {marker_html}
            </div>
        </div>
        """

    rows_html = ""
    counts = {"정상": 0, "주의": 0, "위험": 0, "제외": 0}

    for key in DISPLAY_METRIC_ORDER:
        if key not in all_data:
            continue
        value, is_good, raw = all_data[key]
        level = classify_posture_level(key, raw)
        counts[level] = counts.get(level, 0) + 1
        info = _level_info(level)
        display_value = _fmt_value(key, value, raw)
        standard_text = CLINICAL_RULES.get(key, {}).get("normal", "-")

        rows_html += f"""
        <div class='ms-row'>
            <div class='ms-item'>
                <div class='ms-icon icon-{info['class']}'>{_icon(key)}</div>
                <div>
                    <div class='ms-name'>{FEEDBACK[key]['label']} <span>({FEEDBACK[key]['eng']})</span></div>
                    <div class='ms-sub'>{_subtitle(key)}</div>
                </div>
            </div>
            <div class='ms-value-box'>
                <div class='ms-value' style='color:{info['color']};'>{display_value}</div>
                <div class='ms-standard'>기준 {standard_text}</div>
            </div>
            <div class='ms-status-graph'>
                {_bar_html(key, raw, level)}
            </div>
            <div class='ms-status-box'>
                <div class='ms-badge badge-{info['class']}'>{info['label']}</div>
                <div class='ms-status-kr'>{info['kr']}</div>
            </div>
            <div class='ms-range-box'>
                {_range_lines(key)}
            </div>
        </div>
        """

    good_count = counts.get("정상", 0)
    bad_count = counts.get("위험", 0)
    excluded_count = counts.get("제외", 0)

    logo_src = image_to_base64_src(Path(__file__).resolve().parent / "logo.png")
    if logo_src:
        logo_html = f"<img class='ms-main-logo' src='{logo_src}'>"
    else:
        logo_html = "<div class='ms-main-icon'>🧬</div>"

    html = f"""
    <html>
    <head>
    <style>
    * {{ box-sizing: border-box; }}
    body {{
        margin: 0;
        padding: 0;
        font-family: Pretendard, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: transparent;
        color: #0F1F3A;
    }}
    .ms-report {{
        width: 100%;
        background: linear-gradient(180deg, #F7FAFF 0%, #FFFFFF 100%);
        border: 1px solid #E3EAF5;
        border-radius: 24px;
        padding: 22px 24px 14px 24px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }}
    .ms-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 4px 0 22px 0;
    }}
    .ms-title-wrap {{
        display: flex;
        align-items: center;
        gap: 18px;
    }}
    .ms-main-icon {{
        width: 60px;
        height: 60px;
        border-radius: 18px;
        background: #EEF5FF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 34px;
        color: #1C64F2;
        box-shadow: inset 0 0 0 1px #D9E8FF;
    }}

    .ms-main-logo {{
        width: 60px;
        height: 60px;
        object-fit: contain;
        border-radius: 18px;
        background: #EEF5FF;
        padding: 8px;
        box-shadow: inset 0 0 0 1px #D9E8FF;
        box-sizing: border-box;
    }}
    .ms-title {{
        font-size: 30px;
        font-weight: 950;
        letter-spacing: -1.2px;
        color: #0B1B38;
        line-height: 1.1;
    }}
    .ms-desc {{
        margin-top: 8px;
        font-size: 15px;
        color: #58708F;
        font-weight: 500;
    }}
    .ms-summary {{
        min-width: 760px;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid #E4ECF7;
        border-radius: 18px;
        padding: 14px 18px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }}
    .ms-sum-item {{
        display: flex;
        align-items: center;
        gap: 11px;
    }}
    .ms-sum-dot {{
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 20px;
        font-weight: 950;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
    }}
    .sum-good {{ background: linear-gradient(135deg, #10B981, #34D399); }}
    .sum-bad {{ background: linear-gradient(135deg, #EF4444, #F87171); }}
    .ms-sum-num {{ font-size: 18px; font-weight: 950; color: #17233F; line-height: 1; }}
    .ms-sum-label {{ font-size: 12px; font-weight: 900; color: #304766; margin-top: 4px; }}
    .ms-sum-kr {{ font-size: 12px; color: #667A99; margin-top: 3px; }}
    .ms-table {{
        background: #FFFFFF;
        border: 1px solid #E4ECF7;
        border-radius: 18px;
        overflow: hidden;
    }}
    .ms-head, .ms-row {{
        display: grid;
        grid-template-columns: 1.35fr 0.72fr 1.75fr 0.42fr 0.92fr;
        align-items: center;
        gap: 18px;
    }}
    .ms-head {{
        height: 54px;
        padding: 0 22px;
        background: #FFFFFF;
        border-bottom: 1px solid #E6EDF7;
        color: #445D7F;
        font-size: 13px;
        font-weight: 900;
        text-align: center;
    }}
    .ms-head div:first-child {{ text-align: center; }}
    .ms-row {{
        min-height: 92px;
        padding: 12px 14px 12px 22px;
        border-bottom: 1px solid #EAF0F8;
        background: rgba(255,255,255,0.98);
    }}
    .ms-row:last-child {{ border-bottom: 0; }}
    .ms-item {{
        display: flex;
        align-items: center;
        gap: 16px;
        min-width: 0;
    }}
    .ms-icon {{
        width: 52px;
        height: 52px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        flex-shrink: 0;
    }}
    .ms-metric-img {{
        width: 38px;
        height: 38px;
        object-fit: contain;
        display: block;
        filter: drop-shadow(0 4px 7px rgba(15, 23, 42, 0.10));
    }}
    .icon-good {{ background: #ECFDF5; color: #10B981; }}
    .icon-bad {{ background: #FEF2F2; color: #EF4444; }}
    .icon-none {{ background: #F2F4F7; color: #98A2B3; }}
    .ms-name {{
        font-size: 17px;
        font-weight: 950;
        color: #0F1F3A;
        letter-spacing: -0.4px;
        white-space: nowrap;
    }}
    .ms-name span {{ color: #405877; font-size: 14px; font-weight: 850; }}
    .ms-sub {{ margin-top: 8px; font-size: 13px; color: #526986; font-weight: 600; }}
    .ms-value-box {{ text-align: center; }}
    .ms-value {{ font-size: 26px; font-weight: 950; letter-spacing: -0.6px; line-height: 1; }}
    .ms-standard {{ margin-top: 10px; color: #526986; font-size: 13px; font-weight: 700; }}
    .ms-status-graph {{ padding: 0 2px; }}
    .ms-bar-wrap {{ position: relative; height: 48px; }}
    .ms-bar {{
        position: relative;
        height: 8px;
        top: 15px;
        border-radius: 999px;
        background: #E7ECF5;
    }}
    .ms-seg {{ position: absolute; height: 8px; top: 0; }}
    .seg-good {{ background: #10B981; }}
    .seg-bad {{ background: #EF4444; }}
    .ms-seg:first-child {{ border-radius: 999px 0 0 999px; }}
    .ms-seg:last-of-type {{ border-radius: 0 999px 999px 0; }}
    .ms-marker {{
        position: absolute;
        top: -5px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        transform: translateX(-50%);
        border: 3px solid #FFFFFF;
        box-shadow: 0 4px 10px rgba(15,23,42,0.18);
        z-index: 5;
    }}
    .marker-good {{ background: #10B981; }}
    .marker-bad {{ background: #EF4444; }}
    .marker-none {{ background: #98A2B3; }}
    .ms-missing-marker {{
        position: absolute;
        left: 0;
        top: -1px;
        height: 10px;
        width: 56px;
        border-radius: 999px;
        background: #EF4444;
        box-shadow: 0 3px 8px rgba(239,68,68,0.24);
    }}
    .ms-tick {{
        position: absolute;
        top: 15px;
        width: 1px;
        height: 13px;
        background: #B8C5D8;
        transform: translateX(-50%);
    }}
    .ms-tick-label {{
        position: absolute;
        top: 27px;
        transform: translateX(-50%);
        color: #546B8D;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
    }}
    .ms-status-box {{ text-align: center; }}
    .ms-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 64px;
        height: 30px;
        border-radius: 10px;
        color: #FFFFFF;
        font-size: 14px;
        font-weight: 950;
        letter-spacing: -0.2px;
        box-shadow: 0 8px 16px rgba(15,23,42,0.10);
    }}
    .badge-good {{ background: linear-gradient(135deg, #10B981, #34D399); }}
    .badge-bad {{ background: linear-gradient(135deg, #EF4444, #F87171); }}
    .badge-none {{ background: linear-gradient(135deg, #98A2B3, #CBD5E1); }}
    .ms-status-kr {{ margin-top: 7px; font-size: 13px; color: #526986; font-weight: 800; }}
    .ms-range-box {{
        border: 1px solid #E2EAF5;
        border-radius: 12px;
        padding: 10px 14px;
        background: #FFFFFF;
        font-size: 12.5px;
        line-height: 1.65;
        color: #4B6382;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.65);
    }}
    .ms-range-box div {{ display: grid; grid-template-columns: 42px 1fr; gap: 8px; }}
    .ms-range-box b {{ font-weight: 950; }}
    .range-good {{ color: #10B981; }}
    .range-bad {{ color: #EF4444; }}
    .ms-tip {{
        margin-top: 12px;
        border: 1px solid #CFE1FF;
        background: linear-gradient(135deg, #F2F7FF, #FFFFFF);
        border-radius: 12px;
        padding: 13px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        color: #20569B;
        font-size: 14px;
        font-weight: 800;
    }}
    .ms-legend {{ display: flex; align-items: center; gap: 16px; color: #445D7F; font-size: 13px; font-weight: 800; white-space: nowrap; }}
    .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 6px; vertical-align: -1px; }}
    @media (max-width: 980px) {{
        .ms-top {{ flex-direction: column; align-items: stretch; }}
        .ms-summary {{ min-width: 0; }}
        .ms-head {{ display: none; }}
        .ms-row {{ grid-template-columns: 1fr; gap: 12px; }}
        .ms-range-box {{ max-width: none; }}
        .ms-tip {{ flex-direction: column; align-items: flex-start; }}
    }}
    </style>
    </head>
    <body>
        <div class='ms-report'>
            <div class='ms-top'>
                <div class='ms-title-wrap'>
                    {logo_html}
                    <div>
                        <div class='ms-title'>근골격계 측정 리포트</div>
                        <div class='ms-desc'>7가지 항목의 자세 및 환경을 종합적으로 분석했습니다.</div>
                    </div>
                </div>
                <div class='ms-summary'>
                    <div class='ms-sum-item'>
                        <div class='ms-sum-dot sum-good'>{good_count}</div>
                        <div><div class='ms-sum-num'>{good_count}</div><div class='ms-sum-label'>GOOD</div><div class='ms-sum-kr'>양호</div></div>
                    </div>
                    <div class='ms-sum-item'>
                        <div class='ms-sum-dot sum-bad'>{bad_count}</div>
                        <div><div class='ms-sum-num'>{bad_count}</div><div class='ms-sum-label'>BAD</div><div class='ms-sum-kr'>위험</div></div>
                    </div>
                </div>
            </div>

            <div class='ms-table'>
                <div class='ms-head'>
                    <div>항목</div>
                    <div>측정값</div>
                    <div>상태</div>
                    <div></div>
                    <div>기준 범위</div>
                </div>
                {rows_html}
            </div>

            <div class='ms-tip'>
                <div>💡 <b>TIP</b>&nbsp;&nbsp; 빨간색 항목부터 우선적으로 교정하는 것이 자세 개선에 효과적입니다.</div>
                <div class='ms-legend'>
                    <span><span class='legend-dot' style='background:#2FB35A;'></span>GOOD(양호)</span>
                    <span><span class='legend-dot' style='background:#F43F5E;'></span>BAD(위험)</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    components.html(html, height=3600, scrolling=False)

    if excluded_count > 0:
        st.caption(f"※ 기준점 또는 사물 인식이 부족한 {excluded_count}개 항목은 그래프에서 제외로 표시됩니다.")


    


def get_korean_pdf_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("KoreanFont", font_path))
                return "KoreanFont"
            except Exception:
                pass

    return "Helvetica"


def clean_pdf_text(text):
    if text is None:
        return "-"
    return str(text).replace("<br>", "\n").replace("·", "-").replace("→", "->")


def make_musculoskeletal_report_pdf(result):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )

    buffer = BytesIO()
    font_name = get_korean_pdf_font()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="KTitle",
            fontName=font_name,
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#172033"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="KSub",
            fontName=font_name,
            fontSize=10.5,
            leading=16,
            textColor=colors.HexColor("#667085"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="KBody",
            fontName=font_name,
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#172033"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="KSmall",
            fontName=font_name,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#667085"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="KSection",
            fontName=font_name,
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#172033"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )

    story = []

    all_data = {**result.get("posture", {}), **result.get("env", {})}
    level_counts = result.get("level_counts", {})
    username = st.session_state.get("username", "익명")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    story.append(Paragraph("자세히봐 근골격계 리포트", styles["KTitle"]))
    story.append(
        Paragraph(
            f"사용자: {username}  |  생성일시: {now}<br/>"
            "본 리포트는 AI 자세 분석 기반 참고 자료이며, 의료 진단을 대체하지 않습니다.",
            styles["KSub"],
        )
    )

    summary_data = [
        ["종합 점수", "종합 위험도", "정상 지표", "관리 필요 지표"],
        [
            f"{result.get('score', 0)}/10",
            result.get("risk", "-"),
            f"{result.get('good_count', 0)}개",
            f"{result.get('total_count', 0) - result.get('good_count', 0)}개",
        ],
    ]

    summary_table = Table(summary_data, colWidths=[40 * mm, 40 * mm, 40 * mm, 40 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#667085")),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#172033")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, 1), 15),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5EAF2")),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. 부위별 측정 결과", styles["KSection"]))

    metric_rows = [["번호", "지표", "측정값", "판정", "판정 기준"]]

    for idx, key in enumerate(DISPLAY_METRIC_ORDER, start=1):
        if key not in all_data:
            continue

        value, _, raw = all_data[key]
        level = classify_posture_level(key, raw)
        rule = CLINICAL_RULES.get(key, {})

        range_text = get_range_text_html(key, line_break="<br/>")

        metric_rows.append(
            [
                str(idx),
                FEEDBACK[key]["label"],
                value,
                level,
                Paragraph(range_text, styles["KSmall"]),
            ]
        )

    metric_table = Table(
        metric_rows,
        colWidths=[12 * mm, 30 * mm, 25 * mm, 22 * mm, 78 * mm],
        repeatRows=1,
    )
    metric_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172033")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (3, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E5EAF2")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(metric_table)

    story.append(PageBreak())
    story.append(Paragraph("2. 상세 피드백", styles["KSection"]))

    for key in DISPLAY_METRIC_ORDER:
        if key not in all_data:
            continue

        value, is_good, raw = all_data[key]
        level = classify_posture_level(key, raw)
        fb = FEEDBACK[key]
        msg = fb["good"] if level == "정상" else fb["bad"]
        rule = CLINICAL_RULES.get(key, {})

        color = "#45B86B" if level == "정상" else "#7467F0" if level == "주의" else "#F2527D" if level == "위험" else "#AEB6C2"

        block = Table(
            [
                [
                    Paragraph(
                        f"<b>{fb['no']}. {fb['label']} ({fb['eng']})</b>",
                        styles["KBody"],
                    ),
                    Paragraph(f"<b>{level}</b>", styles["KBody"]),
                ],
                [
                    Paragraph(
                        f"측정값: {value}<br/>"
                        f"{get_range_text_html(key, line_break='<br/>')}",
                        styles["KSmall"],
                    ),
                    "",
                ],
                [
                    Paragraph(clean_pdf_text(msg).replace("\n", "<br/>"), styles["KBody"]),
                    "",
                ],
            ],
            colWidths=[135 * mm, 28 * mm],
        )

        block.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("SPAN", (0, 1), (1, 1)),
                    ("SPAN", (0, 2), (1, 2)),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7F8") if level != "정상" else colors.HexColor("#F0FBF4")),
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(color)),
                    ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor(color)),
                    ("ALIGN", (1, 0), (1, 0), "CENTER"),
                    ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor(color)),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )

        story.append(block)
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer

# =========================================================
# 8. 실제 페이지 출력부 — 잔상 방지 핵심
# =========================================================

init_history()

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
page_placeholder = st.empty()

if "challenge_times" not in st.session_state:
    st.session_state.challenge_times = []

render_alarm_effect(st.session_state.challenge_times)

with page_placeholder.container():

    if menu == "📸 자세측정":
        render_measure()
    
    elif menu == "💬 AI 챗봇":
        render_mobile_chatbot()
        
    elif menu == "🧘 운동 및 스트레칭 추천":
        render_exercise_recommendation_page()

    elif menu == "📈 측정이력":
        render_history()

    elif menu == "📄 근골격계 리포트":
        render_report()

    elif menu == "🧾 예상 영수증":
        render_receipt_page()

    elif menu == "🎯 바른자세 챌린지":
        render_posture_challenge()
    elif menu == "🛒 제품 추천":
        render_product_recommendation_page()

# 챗봇 입력 영역 최종 정렬 오버라이드
st.markdown(
    """
<style>
/* ===== JASEE Chatbot final clean layout ===== */
div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) {
    max-width: 420px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Streamlit가 빈 마커용 element-container를 높이로 잡지 않게 처리 */
div[data-testid="stElementContainer"]:has(.jasee-chatbox-clean-scope) {
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* 상담 유형 영역 */
div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stSelectbox"] {
    width: 100% !important;
    margin: 12px 0 0 0 !important;
    padding: 18px 16px 14px 16px !important;
    background: #FFFFFF !important;
    border: 1px solid #D7E6F8 !important;
    border-bottom: 0 !important;
    border-radius: 22px 22px 0 0 !important;
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.06) !important;
    box-sizing: border-box !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stSelectbox"] > label {
    display: block !important;
    margin: 0 0 8px 0 !important;
    padding: 0 !important;
    color: #172033 !important;
    font-size: 13px !important;
    font-weight: 900 !important;
    line-height: 1.2 !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stSelectbox"] div[data-baseweb="select"] {
    width: 100% !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    height: 50px !important;
    min-height: 50px !important;
    width: 100% !important;
    background: #FFFFFF !important;
    border: 1px solid #CFE0F5 !important;
    border-radius: 16px !important;
    box-shadow: none !important;
    box-sizing: border-box !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #0F1E36 !important;
    font-size: 15px !important;
    font-weight: 900 !important;
}

/* 입력 카드: 상담 유형과 같은 외곽 너비 */
div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] {
    width: 100% !important;
    margin: 0 0 14px 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] form {
    width: 100% !important;
    margin: 0 !important;
    padding: 16px !important;
    background: #FFFFFF !important;
    border: 1px solid #D7E6F8 !important;
    border-top: 1px solid #EEF4FB !important;
    border-radius: 0 0 22px 22px !important;
    box-shadow: 0 18px 36px rgba(37, 99, 235, 0.08) !important;
    box-sizing: border-box !important;
}

/* 입력창 + 버튼 한 줄 정렬 */
div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] div[data-testid="column"] {
    width: auto !important;
    min-width: 0 !important;
    flex: 1 1 auto !important;
    padding: 0 !important;
    margin: 0 !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] div[data-testid="column"]:last-child {
    flex: 0 0 52px !important;
    width: 52px !important;
    min-width: 52px !important;
    max-width: 52px !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] [data-testid="stTextInput"] {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] [data-testid="stTextInput"] input {
    width: 100% !important;
    height: 48px !important;
    min-height: 48px !important;
    margin: 0 !important;
    padding: 0 16px !important;
    background: #F7FAFE !important;
    border: 1px solid #E5EEF8 !important;
    border-radius: 14px !important;
    color: #172033 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    box-sizing: border-box !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] [data-testid="stTextInput"] input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
    outline: none !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stFormSubmitButton"] {
    width: 52px !important;
    height: 48px !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stFormSubmitButton"] button {
    width: 52px !important;
    height: 48px !important;
    min-height: 48px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: #FFFFFF !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    box-shadow: 0 10px 20px rgba(37, 99, 235, 0.26) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stFormSubmitButton"] button p {
    font-size: 18px !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.34) !important;
}

@media (max-width: 700px) {
    div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) {
        max-width: calc(100vw - 32px) !important;
    }

    div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-wrap: nowrap !important;
    }

    div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] div[data-testid="column"] {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        width: auto !important;
    }

    div[data-testid="stVerticalBlock"]:has(.jasee-chatbox-clean-scope) [data-testid="stForm"] div[data-testid="column"]:last-child {
        flex: 0 0 52px !important;
        width: 52px !important;
        min-width: 52px !important;
        max-width: 52px !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)
