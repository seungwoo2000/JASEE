"""
app_mobile.py — 자세히봐 모바일 Streamlit 앱
위치: E:\python\Jasee\app_mobile.py
실행: streamlit run app_mobile.py
"""

import os, sys, math, warnings, datetime, time, threading, json, hashlib, base64
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import pandas as pd
import altair as alt

from jasee_core import (
    load_models, speak, is_good, CRITERIA, ENV_CLASSES, ENV_COLORS,
    run_posture, run_environment
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────
# 기본 경로
# ─────────────────────────────────────────
BASE_DIR       = Path(r"E:\python\Jasee")
USERS_FILE     = BASE_DIR / "users.json"
HISTORY_FILE   = BASE_DIR / "user_history.json"
CHALLENGE_FILE = BASE_DIR / "challenge_results.json"
LOGO_PATH      = BASE_DIR / "logo_transparent.png"

# ─────────────────────────────────────────
# 피드백 / 기준
# ─────────────────────────────────────────
FEEDBACK = {
    "CVA":       {"no":"01","label":"목굴곡각","eng":"CVA",
                  "range":"정상 0°~20° · 위험 20° 초과","cat":"posture",
                  "good":"머리·경추 수직 정렬 유지\n경추 부담 최소화 상태",
                  "bad":"전방두부자세(FHP) 의심\n모니터를 눈높이로 올리세요\n1시간마다 목 스트레칭 시행"},
    "TIA":       {"no":"02","label":"몸통굴곡각","eng":"TIA",
                  "range":"정상 0°~20° · 위험 20° 초과","cat":"posture",
                  "good":"척추 수직 정렬 양호\n요추 압박 최소화 상태",
                  "bad":"과도한 몸통 전굴 감지\n등받이에 허리 완전 밀착\n의자 깊숙이 앉으세요"},
    "knee_angle":{"no":"03","label":"무릎 각도","eng":"Knee",
                  "range":"정상 85°~100° · 위험 범위 이탈","cat":"posture",
                  "good":"하지 혈액순환 원활\n하체 부담 최소화 상태",
                  "bad":"무릎 각도 기준 이탈\n의자 높이 조절 필요\n발받침대 사용 권장"},
    "gaze_angle":{"no":"04","label":"모니터 시선각","eng":"Gaze",
                  "range":"정상 하방 10°~15° · 위험 범위 이탈","cat":"env",
                  "good":"시선각 기준 충족\n경추 부담 최소화",
                  "bad":"시선각 기준 이탈\n모니터 상단을 눈높이에 맞추세요\n화면 거리 40cm 이상 권장"},
    "desk_diff": {"no":"05","label":"작업대 높이","eng":"Desk",
                  "range":"정상 팔꿈치 ±10% 이내","cat":"env",
                  "good":"작업대·팔꿈치 정렬 양호\n상지 부담 최소화",
                  "bad":"작업대 높이 불일치\n책상 높이 또는 의자 높이 조정 필요"},
    "chair_gap": {"no":"06","label":"의자 등받이","eng":"Chair",
                  "range":"정상 골반너비 20% 이내","cat":"env",
                  "good":"등받이 지지 충분\n요추 안정성 확보",
                  "bad":"등받이 지지 부족\n의자 깊숙이 착석\n허리 완전 밀착 필요"},
}

INDICATOR_NAMES = {k: v["label"] for k, v in FEEDBACK.items()}
IND_UNITS = {"CVA":"°","TIA":"°","knee_angle":"°","gaze_angle":"°","desk_diff":"","chair_gap":""}
DISPLAY_ORDER = ["CVA","TIA","knee_angle","gaze_angle","desk_diff","chair_gap"]

# ─────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────
def classify_level(key, raw):
    if raw is None: return "제외"
    raw = float(raw)
    if key == "CVA":        return "정상" if 0 <= raw <= 20 else "위험"
    if key == "TIA":        return "정상" if 0 <= raw <= 20 else "위험"
    if key == "knee_angle": return "정상" if 85 <= raw <= 100 else "위험"
    if key == "gaze_angle": return "정상" if 10 <= raw <= 15 else "위험"
    if key == "desk_diff":  return "정상" if raw <= 0.10 else "위험"
    if key == "chair_gap":  return "정상" if raw <= 0.20 else "위험"
    return "정상"

def calc_score(metrics):
    total, good = 0, 0
    for key in DISPLAY_ORDER:
        val = metrics.get(key)
        if val is None: continue
        total += 1
        if classify_level(key, val) == "정상": good += 1
    score = round((good / total) * 10, 1) if total else 0.0
    risk  = "안전" if score >= 8 else "주의" if score >= 6 else "위험"
    return score, risk, good, total

def score_color(risk):
    return {"안전":"#1D9E75","주의":"#e67e22","위험":"#e74c3c"}.get(risk, "#888")

def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def logo_b64():
    return base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else ""

def save_history(username, metrics):
    history = load_json(HISTORY_FILE)
    score, risk, good, total = calc_score(metrics)
    entry = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
             "score": score, "risk": risk, "good": good, "total": total, "missing_items": []}
    history.setdefault(username, []).insert(0, entry)
    save_json(HISTORY_FILE, history)

def sync_challenge(username, metrics):
    score, risk, good, total = calc_score(metrics)
    point = int(round(score * 10))
    ch = load_json(CHALLENGE_FILE)
    if isinstance(ch, list): ch = {}
    if username not in ch:
        ch[username] = {"name": username, "total_point": 0, "count": 0, "records": []}
    ch[username]["total_point"] += point
    ch[username]["count"] += 1
    ch[username]["records"].insert(0, {"score": score, "point": point, "risk": risk,
        "good": good, "total": total, "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
    save_json(CHALLENGE_FILE, ch)

# ─────────────────────────────────────────
# page_config (모바일: centered + 사이드바 숨김)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="자세히봐 — AI 자세 분석",
    layout="centered",
    page_icon="🪑",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
# 모바일 CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Pretendard',sans-serif;}
.stApp{background:#F5F7FB;}
.block-container{padding-top:3rem!important;max-width:520px;margin:0 auto;}

/* 헤더 */
.hc-header{background:linear-gradient(135deg,#1D9E75,#0d7a5a);padding:16px 18px;
  border-radius:14px;color:white;margin-bottom:16px;}
.hc-header h1{margin:0;font-size:1.3rem;font-weight:800;}
.hc-header p{margin:3px 0 0;font-size:.78rem;opacity:.85;}

/* 카드 */
.hc-card{background:#fff;border-radius:14px;padding:14px 16px;
  box-shadow:0 2px 10px rgba(0,0,0,.07);margin-bottom:12px;}
.hc-card-title{font-size:13px;font-weight:700;color:#172033;margin-bottom:8px;
  display:flex;justify-content:space-between;align-items:center;}

/* 배지 */
.badge-good{background:#d4f7e7;color:#0a5e3a;border-radius:6px;padding:3px 8px;font-size:.7rem;font-weight:700;}
.badge-bad{background:#fde8e8;color:#7a1010;border-radius:6px;padding:3px 8px;font-size:.7rem;font-weight:700;}
.badge-na{background:#f0f0f0;color:#666;border-radius:6px;padding:3px 8px;font-size:.7rem;font-weight:700;}
.badge-blue{background:#e6f1fb;color:#0c447c;border-radius:6px;padding:3px 8px;font-size:.7rem;font-weight:700;}

/* 메트릭 */
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;}
.metric-card{background:#fff;border:1px solid #E5EAF2;border-radius:12px;padding:14px;
  box-shadow:0 3px 12px rgba(15,23,42,.04);}
.metric-value{font-size:22px;font-weight:900;letter-spacing:-.5px;}
.metric-label{font-size:11px;color:#667085;margin-top:3px;}

/* 점수 박스 */
.score-box{border-radius:14px;padding:18px;text-align:center;color:white;margin-bottom:14px;}
.score-num{font-size:2.8rem;font-weight:800;line-height:1;}
.score-lbl{font-size:.82rem;opacity:.9;margin-top:3px;}

/* 지표 행 */
.result-row{display:flex;align-items:center;gap:8px;padding:9px 0;border-bottom:1px solid #EEF2F6;}
.result-row:last-child{border-bottom:0;}
.result-name{width:80px;font-size:11px;color:#667085;flex-shrink:0;}
.result-value{font-size:12px;font-weight:800;color:#172033;width:55px;flex-shrink:0;}
.bar-wrap{flex:1;height:7px;background:#EEF2F6;border-radius:999px;overflow:hidden;}
.bar{height:7px;border-radius:999px;}

/* 모바일 네비 */
.mobile-nav{position:sticky;top:0;z-index:999;background:#fff;
  padding:8px 0 6px;border-bottom:1px solid #E5EAF2;margin-bottom:16px;}

/* 버튼 */
.stButton>button{border-radius:10px!important;font-weight:700!important;}

/* 로그인 */
.login-wrap{max-width:340px;margin:40px auto;background:#fff;border-radius:16px;
  padding:28px 24px;box-shadow:0 4px 24px rgba(0,0,0,.10);}

/* 이력 카드 */
.hist-card{background:#fff;border-radius:12px;padding:12px;margin-bottom:10px;
  box-shadow:0 2px 8px rgba(0,0,0,.05);}

/* 챌린지 */
.race-bar{position:relative;height:32px;background:#EAF0F7;border-radius:999px;
  overflow:visible;margin-top:5px;}
.race-fill{position:absolute;left:0;top:0;height:32px;
  background:linear-gradient(90deg,#DFF3FF,#B9E6FF);border-radius:999px;}
.race-icon{position:absolute;top:2px;font-size:22px;}
.race-flag{position:absolute;right:8px;top:6px;font-size:16px;}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 로그인 페이지
# ─────────────────────────────────────────
def page_login():
    b64 = logo_b64()
    if b64:
        st.markdown(f'<div style="text-align:center;margin-bottom:12px;"><img src="data:image/png;base64,{b64}" width="180"></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align:center;color:#1D9E75;'>🪑 자세히봐</h2>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#888;font-size:.82rem;margin-bottom:20px;'>RULA + VDT 기준 자세 & 작업환경 측정</p>",
                unsafe_allow_html=True)

    tab_li, tab_su = st.tabs(["로그인","회원가입"])
    with tab_li:
        u = st.text_input("이름", key="li_u")
        p = st.text_input("비밀번호", type="password", key="li_p")
        if st.button("로그인", use_container_width=True, type="primary"):
            users = load_json(USERS_FILE)
            if u in users and users[u]["password"] == hash_pw(p):
                st.session_state.logged_in = True
                st.session_state.username  = u
                st.rerun()
            else:
                st.error("이름 또는 비밀번호가 올바르지 않습니다.")
    with tab_su:
        nu  = st.text_input("이름", key="su_u")
        np_ = st.text_input("비밀번호", type="password", key="su_p")
        np2 = st.text_input("비밀번호 확인", type="password", key="su_p2")
        if st.button("회원가입", use_container_width=True):
            users = load_json(USERS_FILE)
            if not nu: st.error("이름을 입력해주세요.")
            elif nu in users: st.error("이미 존재하는 사용자입니다.")
            elif np_ != np2: st.error("비밀번호가 일치하지 않습니다.")
            else:
                users[nu] = {"password": hash_pw(np_),
                             "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                save_json(USERS_FILE, users)
                st.success("가입 완료! 로그인해주세요.")

# ─────────────────────────────────────────
# 프레임 오버레이
# ─────────────────────────────────────────
def draw_posture_frame(frame, posture_out, good_start, hold_sec, elapsed, total_sec):
    result = frame.copy(); h, w = result.shape[:2]
    rv  = posture_out["result"]; kp = posture_out["keypoints"]; met = posture_out["metrics"]
    col = (29,158,117) if rv=="GOOD" else (74,50,230) if rv=="BAD" else (150,150,150)
    if kp:
        for pt in kp.values():
            cv2.circle(result, pt, 8, col, -1)
            cv2.circle(result, pt, 10, (255,255,255), 2)
        for a, b in [("귀","어깨"),("어깨","골반"),("골반","무릎"),("무릎","발목"),("어깨","팔꿈치")]:
            if a in kp and b in kp: cv2.line(result, kp[a], kp[b], col, 2)
        if rv == "BAD" and "어깨" in kp:
            sh = kp["어깨"]
            cv2.arrowedLine(result, sh, (sh[0], sh[1]-60), (255,80,0), 2, tipLength=.35)
            cv2.putText(result, "자세를 바로 하세요", (sh[0]-100, sh[1]-68),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255,80,0), 2)
    # 상단 상태바
    ov = result.copy()
    bc = (29,158,117) if rv=="GOOD" else (74,50,230) if rv=="BAD" else (50,50,50)
    cv2.rectangle(ov, (0,0), (w,46), bc, -1)
    cv2.addWeighted(ov, .78, result, .22, 0, result)
    cv2.putText(result, f"자세:{rv}" if rv else "인식 대기...", (12,30),
                cv2.FONT_HERSHEY_SIMPLEX, .85, (255,255,255), 2)
    cv2.putText(result, f"남은:{max(0,total_sec-elapsed):.0f}s", (w-115, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
    # GOOD 프로그레스바
    if rv == "GOOD" and good_start:
        held = time.time() - good_start
        ratio = min(held / hold_sec, 1.)
        bw = int((w-30) * ratio)
        cv2.rectangle(result, (15, h-35), (w-15, h-18), (40,40,40), -1)
        cv2.rectangle(result, (15, h-35), (15+bw, h-18), (29,158,117), -1)
        cv2.putText(result, f"GOOD: {held:.1f}s/{hold_sec}s", (15, h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (29,158,117), 2)
    return result

def draw_env_frame(frame, env_out):
    result = frame.copy(); h, w = result.shape[:2]
    det = env_out["detected"]; bb = env_out["bboxes"]
    for label, bbox in bb.items():
        x1,y1,x2,y2 = bbox
        cls = [k for k,v in ENV_CLASSES.items() if v==label]
        c = ENV_COLORS.get(cls[0], (200,200,200)) if cls else (200,200,200)
        conf = det.get(label, 0)
        cv2.rectangle(result, (x1,y1), (x2,y2), c, 2)
        cv2.rectangle(result, (x1,y1-22), (x2,y1), c, -1)
        cv2.putText(result, f"{label} {conf:.0%}", (x1+3, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 2)
    for i, label in enumerate(ENV_CLASSES.values()):
        ok = label in det; c = (29,158,117) if ok else (120,120,120)
        cv2.putText(result, f"{'v' if ok else 'o'} {label}",
                    (12, 65+i*26), cv2.FONT_HERSHEY_SIMPLEX, .58, c, 2)
    ov = result.copy()
    cv2.rectangle(ov, (0,0), (w,46), (13,110,90), -1)
    cv2.addWeighted(ov, .78, result, .22, 0, result)
    cv2.putText(result, "작업환경 인식 중...", (12,30),
                cv2.FONT_HERSHEY_SIMPLEX, .85, (255,255,255), 2)
    return result

# ─────────────────────────────────────────
# 측정 페이지
# ─────────────────────────────────────────
def render_measure(pose_yolo, mlp, env_yolo, device, username):
    st.markdown("""<div class="hc-header"><h1>📷 자세 측정</h1>
    <p>GOOD 자세 5초 유지 → 작업환경 인식 → 완료</p></div>""", unsafe_allow_html=True)

    defaults = dict(phase="posture", good_start=None, env_bboxes={},
                    running=False, last_posture=None, last_metrics={})
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    POSTURE_TOTAL = 20; GOOD_HOLD = 5

    # 단계 / 상태
    phase_box  = st.empty()
    result_box = st.empty()
    good_bar   = st.empty()
    frame_box  = st.empty()

    # 지표
    st.markdown("**📐 측정 지표**")
    metric_box = st.empty()
    env_box    = st.empty()
    st.markdown("---")

    # 버튼
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ 시작" if not st.session_state.running else "⏹ 중지",
                     use_container_width=True, type="primary"):
            st.session_state.running = not st.session_state.running
            if st.session_state.running:
                speak("측정을 시작합니다. 측면을 카메라에 맞춰주세요.")
    with c2:
        if st.button("🔄 초기화", use_container_width=True):
            for k, v in defaults.items(): st.session_state[k] = v
            st.rerun()

    # 기준
    with st.expander("📋 RULA+VDT 측정 기준"):
        for key, fb in FEEDBACK.items():
            st.markdown(f"• **{fb['label']}**: {fb['range']}")

    # 완료 결과
    if not st.session_state.running and st.session_state.last_metrics:
        m = st.session_state.last_metrics
        score, risk, good, total = calc_score(m)
        sc = score_color(risk)
        st.markdown(f"""<div class="score-box" style="background:{sc};">
          <div class="score-num">{score}</div>
          <div class="score-lbl">/ 10점 | {risk} | {good}/{total}개 양호</div>
        </div>""", unsafe_allow_html=True)
        for key in DISPLAY_ORDER:
            val = m.get(key); unit = IND_UNITS.get(key,""); fb = FEEDBACK.get(key,{})
            level = classify_level(key, val)
            if val is None:
                badge='<span class="badge-na">제외</span>'; border="#ccc"; msg="측정불가"
            elif level == "정상":
                badge='<span class="badge-good">✓ 정상</span>'; border="#1D9E75"
                msg=fb.get("good","")
            else:
                badge='<span class="badge-bad">✗ 위험</span>'; border="#e74c3c"
                msg=fb.get("bad","")
            vs = f"{val:.2f}{unit}" if val is not None else "—"
            st.markdown(f"""<div class="hc-card" style="border-left:4px solid {border};">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:700;font-size:.85rem;">{fb.get('label',key)}</span>{badge}</div>
              <div style="font-size:1.3rem;font-weight:800;color:{border};margin:5px 0;">{vs}</div>
              <div style="font-size:.72rem;color:#555;line-height:1.4;">{msg.replace(chr(10),'<br>')}</div>
            </div>""", unsafe_allow_html=True)

    # 카메라 루프
    if st.session_state.running:
        cap = cv2.VideoCapture(0); start_time = time.time()
        out = {"metrics": {}}  # 초기화

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret: st.warning("카메라를 찾을 수 없어요."); break
            frame   = cv2.flip(frame, 1)
            elapsed = time.time() - start_time

            if st.session_state.phase == "posture":
                phase_box.info("🔵 **1단계: 자세 측정 중**")
                out = run_posture(frame, pose_yolo, mlp, device, st.session_state.env_bboxes)
                rv  = out["result"]; met = out["metrics"]
                result_frame = draw_posture_frame(frame, out, st.session_state.good_start,
                                                  GOOD_HOLD, elapsed, POSTURE_TOTAL)
                if met:
                    metric_box.markdown("".join([
                        f"{'🟢' if is_good(k,v) else '🔴'} **{INDICATOR_NAMES.get(k,k)}**: {v:.2f}{IND_UNITS.get(k,'')}\n\n"
                        for k,v in met.items() if v is not None]))
                if rv:
                    result_box.markdown(f"## {'🟢' if rv=='GOOD' else '🔴'} {rv}")
                    if rv != st.session_state.last_posture:
                        st.session_state.last_posture = rv
                        speak("좋은 자세입니다. 5초간 유지해주세요." if rv=="GOOD" else "자세를 바로 해주세요.")
                if rv == "GOOD":
                    if not st.session_state.good_start: st.session_state.good_start = time.time()
                    held = time.time() - st.session_state.good_start
                    good_bar.progress(min(held/GOOD_HOLD, 1.), text=f"GOOD 유지: {held:.1f}s / {GOOD_HOLD}s")
                    if held >= GOOD_HOLD:
                        speak("자세 완료! 작업환경 인식 시작합니다.")
                        st.session_state.phase = "environment"
                else:
                    st.session_state.good_start = None
                    good_bar.progress(0., text="BAD 자세 — 다시 시도")
                if elapsed >= POSTURE_TOTAL:
                    speak("시간 종료. 다시 시도해주세요."); st.session_state.running = False

            elif st.session_state.phase == "environment":
                phase_box.info("🟠 **2단계: 작업환경 인식 중**")
                env_out = run_environment(frame, env_yolo)
                det = env_out["detected"]; st.session_state.env_bboxes = env_out["bboxes"]
                result_frame = draw_env_frame(frame, env_out)
                items = list(ENV_CLASSES.values())
                env_box.markdown("**감지 현황**\n\n" +
                                 "\n\n".join([f"{'✅' if i in det else '⬜'} {i}" for i in items]))
                if all(i in det for i in items):
                    final = out.get("metrics", {})
                    st.session_state.last_metrics = final
                    save_history(username, final); sync_challenge(username, final)
                    speak("환경 인식 완료. 측정을 종료합니다.")
                    st.session_state.running = False; st.balloons()

            frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            frame_box.image(frame_rgb, channels="RGB", use_container_width=True)
        cap.release()

# ─────────────────────────────────────────
# 이력 페이지
# ─────────────────────────────────────────
def render_history(username):
    st.markdown("""<div class="hc-header"><h1>📈 측정 이력</h1>
    <p>자세 분석 결과 이력</p></div>""", unsafe_allow_html=True)
    history = load_json(HISTORY_FILE); records = history.get(username, [])
    if not records: st.info("아직 측정 이력이 없어요!"); return

    scores = [r["score"] for r in records]; avg = round(sum(scores)/len(scores), 1)
    st.markdown(f"""<div class="metric-grid">
      <div class="metric-card"><div class="metric-value" style="color:#185FA5;">{len(records)}회</div><div class="metric-label">총 측정 횟수</div></div>
      <div class="metric-card"><div class="metric-value" style="color:#1D9E75;">{avg}</div><div class="metric-label">평균 점수 / 10</div></div>
    </div>""", unsafe_allow_html=True)

    if len(records) >= 2:
        st.markdown("**점수 추이**")
        df = pd.DataFrame([{"회차":i+1,"점수":float(h["score"])} for i,h in enumerate(reversed(records))])
        chart = (alt.Chart(df).mark_line(strokeWidth=3,interpolate="monotone",color="#1D9E75") +
                 alt.Chart(df).mark_circle(size=60,color="#1D9E75")
                ).encode(x=alt.X("회차:O"), y=alt.Y("점수:Q",scale=alt.Scale(domain=[0,10]))
                ).properties(height=200).configure_view(strokeWidth=0)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("**전체 이력**")
    for r in records[:15]:
        sc = score_color(r["risk"]); rate = round(r["good"]/r["total"]*100)
        st.markdown(f"""<div class="hc-card" style="border-left:4px solid {sc};">
          <div style="display:flex;justify-content:space-between;">
            <span style="font-size:.8rem;color:#666;">{r['time']}</span>
            <span style="font-weight:800;color:{sc};">{r['score']}점 ({r['risk']})</span>
          </div>
          <div style="font-size:.75rem;color:#888;margin-top:3px;">양호 {r['good']}/{r['total']}개</div>
          <div style="margin-top:7px;height:6px;background:#EEF2F6;border-radius:999px;">
            <div style="height:6px;width:{rate}%;background:{sc};border-radius:999px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 챌린지 페이지
# ─────────────────────────────────────────
def render_challenge():
    st.markdown("""<div class="hc-header"><h1>🎯 바른자세 챌린지</h1>
    <p>측정할 때마다 포인트가 쌓입니다.</p></div>""", unsafe_allow_html=True)
    ch = load_json(CHALLENGE_FILE)
    if isinstance(ch, list): ch = {}
    if not ch: st.info("아직 참여자가 없습니다."); return
    members = sorted(ch.values(), key=lambda x: x.get("total_point",0), reverse=True)
    max_pt  = max([m.get("total_point",0) for m in members]) or 1
    icons   = ["🐰","🐢","🦊","🐻","🐼","🐯"]

    st.markdown('<div style="background:#F8FAFC;border:1px solid #E5EAF2;border-radius:16px;padding:16px;">', unsafe_allow_html=True)
    st.markdown("**🏁 자세 포인트 레이스**")
    for idx, member in enumerate(members):
        pct = max(8, min(int(member.get("total_point",0)/max_pt*100), 100))
        icon = icons[idx % len(icons)]
        recs = member.get("records",[]); latest = recs[0] if recs else {}
        st.markdown(f"""
        <div style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:800;">
            <span>{idx+1}. {member.get('name','익명')}</span>
            <span style="color:#185FA5;">{member.get('total_point',0)}P · {member.get('count',0)}회</span>
          </div>
          <div class="race-bar">
            <div class="race-fill" style="width:{pct}%;"></div>
            <div class="race-icon" style="left:calc({pct}% - 20px);">{icon}</div>
            <div class="race-flag">🏁</div>
          </div>
          <div style="font-size:11px;color:#888;margin-top:3px;">최근 {latest.get('score','-')}/10 · {latest.get('risk','-')}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 순위")
    for i, member in enumerate(members, 1):
        recs = member.get("records",[]); latest = recs[0] if recs else {}
        risk = latest.get("risk","—"); sc = score_color(risk)
        bc = "badge-good" if risk=="안전" else ("badge-bad" if risk=="위험" else "badge-na")
        st.markdown(f"""<div class="hc-card">
          <div class="hc-card-title"><span>{i}. {member.get('name','익명')}</span>
            <span class="{bc}">{risk}</span></div>
          <div style="font-size:24px;font-weight:900;color:#185FA5;">{member.get('total_point',0)}P</div>
          <div style="font-size:13px;color:#667085;">측정 {member.get('count',0)}회 | 최근 {latest.get('score','-')}/10</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 앱 진입점
# ─────────────────────────────────────────
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username"  not in st.session_state: st.session_state.username  = ""

if not st.session_state.logged_in:
    page_login(); st.stop()

username = st.session_state.username

@st.cache_resource
def get_models(): return load_models()

with st.spinner("모델 로딩 중..."):
    try:
        pose_yolo, mlp, env_yolo, device = get_models()
    except Exception as e:
        st.error(f"모델 로드 실패: {e}"); st.stop()

# 모바일 상단 수평 탭 네비게이션
st.markdown('<div class="mobile-nav">', unsafe_allow_html=True)
menu = st.radio("메뉴", ["📷 측정","📈 이력","🎯 챌린지"],
                horizontal=True, label_visibility="collapsed", key="mobile_nav")
st.markdown('</div>', unsafe_allow_html=True)

# 계정 정보 (접기)
with st.expander("👤 계정 정보"):
    st.caption(f"로그인: {username}")
    if st.button("로그아웃", use_container_width=True):
        st.session_state.logged_in = False; st.session_state.username = ""; st.rerun()

# 페이지 렌더링
if "측정" in menu:
    render_measure(pose_yolo, mlp, env_yolo, device, username)
elif "이력" in menu:
    render_history(username)
elif "챌린지" in menu:
    render_challenge()
