"""
app.py — Fit Me Up / 자세히봐 통합 Streamlit 앱
위치: E:\python\Jasee\app.py
실행: streamlit run app.py
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

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
except Exception:
    VideoProcessorBase = object
    av = None

from jasee_core import (
    load_models, speak, is_good, CRITERIA, ENV_CLASSES, ENV_COLORS,
    run_posture, run_environment, AttentionMLP, predict_posture,
    calc_cva, calc_tia, calc_knee_angle, calc_gaze_angle,
    calc_desk_diff, calc_chair_gap
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────
# 기본 경로
# ─────────────────────────────────────────
BASE_DIR     = Path(r"E:\python\Jasee")
USERS_FILE   = BASE_DIR / "users.json"
HISTORY_FILE = BASE_DIR / "user_history.json"
CHALLENGE_FILE = BASE_DIR / "challenge_results.json"
LOGO_PATH    = BASE_DIR / "logo_transparent.png"

# ─────────────────────────────────────────
# 피드백 / 기준 (구버전 integrate.py 기반)
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

CLINICAL_RULES = {
    "CVA":       {"normal":"0° ~ 20°","risk":"20° 초과","basis":"RULA Neck Zone + VDT 기준"},
    "TIA":       {"normal":"0° ~ 20°","risk":"20° 초과","basis":"RULA Trunk Zone + 등받이 지지"},
    "knee_angle":{"normal":"85° ~ 100°","risk":"범위 이탈","basis":"VDT 무릎 내각 90° 전후"},
    "gaze_angle":{"normal":"하방 10° ~ 15°","risk":"범위 이탈","basis":"VDT 수평 하방 기준"},
    "desk_diff": {"normal":"팔꿈치 기준 ±10%","risk":"10% 초과","basis":"팔꿈치-책상면 수평 정렬"},
    "chair_gap": {"normal":"골반너비 20% 이내","risk":"20% 초과","basis":"VDT 의자 착석 기준"},
}

DISPLAY_ORDER = ["CVA","TIA","knee_angle","gaze_angle","desk_diff","chair_gap"]

# ─────────────────────────────────────────
# 점수 / 위험도
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

def calc_score(metrics: dict):
    total, good, missing = 0, 0, []
    for key in DISPLAY_ORDER:
        val = metrics.get(key)
        if val is None: missing.append(key); continue
        total += 1
        if classify_level(key, val) == "정상": good += 1
    score = round((good / total) * 10, 1) if total else 0.0
    risk  = "안전" if score >= 8 else "주의" if score >= 6 else "위험"
    return score, risk, good, total

def score_color(risk):
    return {"안전":"#1D9E75","주의":"#e67e22","위험":"#e74c3c"}.get(risk, "#888")

def level_color(level):
    return {"정상":"#1D9E75","위험":"#e74c3c","제외":"#aaa"}.get(level, "#aaa")

# ─────────────────────────────────────────
# JSON 유틸
# ─────────────────────────────────────────
def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

# ─────────────────────────────────────────
# 이력 저장
# ─────────────────────────────────────────
def save_history(username, metrics):
    history = load_json(HISTORY_FILE)
    score, risk, good, total = calc_score(metrics)
    missing_items = [
        {"key":k,"label":INDICATOR_NAMES[k],"reason":f"{INDICATOR_NAMES[k]} 측정 불가"}
        for k in DISPLAY_ORDER if metrics.get(k) is None
    ]
    entry = {"time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
             "score":score,"risk":risk,"good":good,"total":total,
             "missing_items":missing_items}
    history.setdefault(username,[]).insert(0, entry)
    save_json(HISTORY_FILE, history)

def sync_challenge(username, metrics):
    score, risk, good, total = calc_score(metrics)
    point = int(round(score * 10))
    ch = load_json(CHALLENGE_FILE)
    if isinstance(ch, list): ch = {}
    if username not in ch:
        ch[username] = {"name":username,"total_point":0,"count":0,"records":[]}
    ch[username]["total_point"] += point
    ch[username]["count"] += 1
    ch[username]["records"].insert(0,{
        "score":score,"point":point,"risk":risk,"good":good,"total":total,
        "time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
    save_json(CHALLENGE_FILE, ch)

# ─────────────────────────────────────────
# 로고
# ─────────────────────────────────────────
def logo_base64():
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return ""

# ─────────────────────────────────────────
# CSS 주입
# ─────────────────────────────────────────
def inject_css(mobile=False):
    grid_cols = "repeat(2,1fr)" if mobile else "repeat(4,1fr)"
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;600;700;800;900&display=swap');
html,body,[class*="css"]{{font-family:'Pretendard',sans-serif;}}
.stApp{{background:#F5F7FB;}}
.block-container{{padding-top:4rem!important;max-width:{'880px' if mobile else '1280px'};}}
[data-testid="stSidebar"]{{background:#fff;border-right:1px solid #E5EAF2;}}

/* 헤더 */
.hc-header{{background:linear-gradient(135deg,#1D9E75,#0d7a5a);padding:18px 24px;
  border-radius:14px;color:white;margin-bottom:20px;}}
.hc-header h1{{margin:0;font-size:1.6rem;font-weight:800;}}
.hc-header p{{margin:4px 0 0;font-size:0.82rem;opacity:.85;}}

/* 카드 */
.hc-card{{background:#fff;border-radius:14px;padding:18px;
  box-shadow:0 2px 12px rgba(0,0,0,.07);margin-bottom:14px;}}
.hc-card-title{{font-size:14px;font-weight:700;color:#172033;margin-bottom:10px;
  display:flex;justify-content:space-between;align-items:center;}}

/* 배지 */
.badge-good{{background:#d4f7e7;color:#0a5e3a;border-radius:6px;
  padding:3px 9px;font-size:.73rem;font-weight:700;}}
.badge-bad{{background:#fde8e8;color:#7a1010;border-radius:6px;
  padding:3px 9px;font-size:.73rem;font-weight:700;}}
.badge-na{{background:#f0f0f0;color:#666;border-radius:6px;
  padding:3px 9px;font-size:.73rem;font-weight:700;}}
.badge-blue{{background:#e6f1fb;color:#0c447c;border-radius:6px;
  padding:3px 9px;font-size:.73rem;font-weight:700;}}

/* 메트릭 그리드 */
.metric-grid{{display:grid;grid-template-columns:{grid_cols};gap:12px;margin-bottom:16px;}}
.metric-card{{background:#fff;border:1px solid #E5EAF2;border-radius:14px;padding:16px;
  box-shadow:0 4px 16px rgba(15,23,42,.04);}}
.metric-value{{font-size:26px;font-weight:900;letter-spacing:-.6px;}}
.metric-label{{font-size:12px;color:#667085;margin-top:4px;}}

/* 점수 박스 */
.score-box{{border-radius:14px;padding:20px;text-align:center;color:white;margin-bottom:14px;}}
.score-num{{font-size:3.2rem;font-weight:800;line-height:1;}}
.score-lbl{{font-size:.88rem;opacity:.9;margin-top:4px;}}

/* 지표 바 */
.result-row{{display:flex;align-items:center;gap:10px;padding:10px 0;
  border-bottom:1px solid #EEF2F6;}}
.result-row:last-child{{border-bottom:0;}}
.result-name{{width:90px;font-size:12px;color:#667085;flex-shrink:0;}}
.result-value{{font-size:13px;font-weight:800;color:#172033;width:60px;flex-shrink:0;}}
.bar-wrap{{flex:1;height:8px;background:#EEF2F6;border-radius:999px;overflow:hidden;}}
.bar{{height:8px;border-radius:999px;}}

/* 피드백 카드 */
.fb-card{{border-radius:14px;padding:14px 16px;margin-bottom:10px;border:1px solid #E5EAF2;}}
.fb-good{{background:linear-gradient(135deg,#F0FBF4,#fff);border-left:4px solid #1D9E75;}}
.fb-bad{{background:linear-gradient(135deg,#FFF1F1,#fff);border-left:4px solid #e74c3c;}}
.fb-top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}}
.fb-name{{font-size:14px;font-weight:800;color:#172033;}}
.fb-msg{{font-size:12.5px;line-height:1.65;color:#667085;white-space:pre-line;}}

/* 이력 */
.hist-card{{background:#fff;border-radius:12px;padding:12px 16px;margin-bottom:10px;
  box-shadow:0 2px 8px rgba(0,0,0,.05);}}

/* 버튼 */
.stButton>button{{border-radius:10px!important;font-weight:700!important;}}

/* 로그인 */
.login-wrap{{max-width:380px;margin:50px auto;background:#fff;border-radius:18px;
  padding:36px 32px;box-shadow:0 4px 28px rgba(0,0,0,.10);}}

/* 챌린지 레이스 */
.race-box{{background:#F8FAFC;border:1px solid #E5EAF2;border-radius:20px;padding:18px;}}
.race-row{{margin-bottom:18px;}}
.race-bar{{position:relative;height:36px;background:#EAF0F7;border-radius:999px;overflow:visible;margin-top:6px;}}
.race-fill{{position:absolute;left:0;top:0;height:36px;background:linear-gradient(90deg,#DFF3FF,#B9E6FF);border-radius:999px;}}
.race-icon{{position:absolute;top:3px;font-size:26px;}}
.race-flag{{position:absolute;right:10px;top:7px;font-size:18px;}}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 프레임 오버레이
# ─────────────────────────────────────────
def draw_posture_frame(frame, posture_out, good_start, hold_sec, elapsed, total_sec):
    result = frame.copy(); h, w = result.shape[:2]
    rv  = posture_out["result"]; kp = posture_out["keypoints"]; met = posture_out["metrics"]
    col = (29,158,117) if rv=="GOOD" else (74,50,230) if rv=="BAD" else (150,150,150)
    if kp:
        for pt in kp.values():
            cv2.circle(result,pt,9,col,-1); cv2.circle(result,pt,11,(255,255,255),2)
        for a,b in [("귀","어깨"),("어깨","골반"),("골반","무릎"),("무릎","발목"),("어깨","팔꿈치")]:
            if a in kp and b in kp: cv2.line(result,kp[a],kp[b],col,3)
        if rv=="BAD" and "어깨" in kp:
            sh=kp["어깨"]; cv2.arrowedLine(result,sh,(sh[0],sh[1]-70),(255,80,0),3,tipLength=.35)
            cv2.putText(result,"자세를 바로 하세요",(sh[0]-110,sh[1]-82),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,80,0),2)
    y_off=70
    for key,label,unit in [("CVA","CVA","deg"),("TIA","TIA","deg"),
                            ("knee_angle","무릎","deg"),("gaze_angle","시선각","deg"),
                            ("desk_diff","작업대","ratio"),("chair_gap","등받이","ratio")]:
        val=met.get(key)
        if val is None: continue
        g=is_good(key,val); c=(29,158,117) if g else (74,50,230)
        tag="Good" if g else "Bad"
        txt=f"{label}:{val:.2f}({tag})" if unit=="ratio" else f"{label}:{val:.1f}deg({tag})"
        cv2.putText(result,txt,(w-295,y_off),cv2.FONT_HERSHEY_SIMPLEX,.52,c,2); y_off+=26
    ov=result.copy()
    bc=(29,158,117) if rv=="GOOD" else (74,50,230) if rv=="BAD" else (50,50,50)
    cv2.rectangle(ov,(0,0),(w,50),bc,-1); cv2.addWeighted(ov,.78,result,.22,0,result)
    cv2.putText(result,f"자세:{rv}" if rv else "인식 대기중...",(15,33),cv2.FONT_HERSHEY_SIMPLEX,1.,(255,255,255),2)
    cv2.putText(result,f"남은:{max(0,total_sec-elapsed):.0f}s",(w-130,33),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    if rv=="GOOD" and good_start:
        held=time.time()-good_start; ratio=min(held/hold_sec,1.)
        bw=int((w-40)*ratio)
        cv2.rectangle(result,(20,h-40),(w-20,h-20),(40,40,40),-1)
        cv2.rectangle(result,(20,h-40),(20+bw,h-20),(29,158,117),-1)
        cv2.putText(result,f"GOOD 유지:{held:.1f}s/{hold_sec}s",(20,h-45),cv2.FONT_HERSHEY_SIMPLEX,.55,(29,158,117),2)
    return result

def draw_env_frame(frame, env_out):
    result=frame.copy(); h,w=result.shape[:2]
    det=env_out["detected"]; bb=env_out["bboxes"]
    for label,bbox in bb.items():
        x1,y1,x2,y2=bbox
        cls=[k for k,v in ENV_CLASSES.items() if v==label]
        c=ENV_COLORS.get(cls[0],(200,200,200)) if cls else (200,200,200)
        conf=det.get(label,0)
        cv2.rectangle(result,(x1,y1),(x2,y2),c,2); cv2.rectangle(result,(x1,y1-24),(x2,y1),c,-1)
        cv2.putText(result,f"{label} {conf:.0%}",(x1+4,y1-6),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),2)
    for i,label in enumerate(ENV_CLASSES.values()):
        ok=label in det; c=(29,158,117) if ok else (120,120,120)
        cv2.putText(result,f"{'v' if ok else 'o'} {label}",(15,75+i*28),cv2.FONT_HERSHEY_SIMPLEX,.65,c,2)
    ov=result.copy(); cv2.rectangle(ov,(0,0),(w,50),(13,110,90),-1); cv2.addWeighted(ov,.78,result,.22,0,result)
    cv2.putText(result,"작업환경 인식 중...",(15,33),cv2.FONT_HERSHEY_SIMPLEX,1.,(255,255,255),2)
    return result

# ─────────────────────────────────────────
# AI 코멘트 (구버전 기반)
# ─────────────────────────────────────────
def build_ai_comment(metrics):
    GUIDE = {
        "CVA":       {"part":"목·경추","bad":"전방두부자세 가능성. 모니터를 눈높이에 맞추고 목 스트레칭을 해주세요.","goal":"모니터 높이를 눈높이에 맞추기"},
        "TIA":       {"part":"몸통·허리","bad":"몸통이 앞으로 굽었습니다. 의자 깊숙이 앉아 등받이에 허리를 기대세요.","goal":"등받이에 허리 밀착하기"},
        "knee_angle":{"part":"무릎·하체","bad":"무릎 각도 이탈. 의자 높이를 조절해 90° 전후를 맞추세요.","goal":"의자 높이 조절하기"},
        "gaze_angle":{"part":"시선·모니터","bad":"모니터 높이 이탈. 모니터 상단을 눈높이에 맞추세요.","goal":"모니터 높이 조정하기"},
        "desk_diff": {"part":"작업대 높이","bad":"책상-팔꿈치 불일치. 책상 또는 의자 높이를 조정하세요.","goal":"팔꿈치와 책상면 수평 맞추기"},
        "chair_gap": {"part":"의자 등받이","bad":"등받이 지지 부족. 의자 깊숙이 앉아 허리를 밀착하세요.","goal":"의자 깊숙이 앉기"},
    }
    bad_items=[(k,v,GUIDE[k]) for k,v in metrics.items() if k in GUIDE and v is not None and classify_level(k,v)=="위험"]
    good_items=[GUIDE[k]["part"] for k,v in metrics.items() if k in GUIDE and v is not None and classify_level(k,v)=="정상"]
    goals=[item[2]["goal"] for item in bad_items]
    default_goals=["50분 작업 후 5분 스트레칭","목과 어깨를 천천히 돌리기","손목 받침대 사용하기","발바닥이 바닥에 닿는지 확인하기"]
    for g in default_goals:
        if len(goals)>=4: break
        if g not in goals: goals.append(g)
    detail_html=""
    for k,v,item in bad_items[:4]:
        unit=IND_UNITS.get(k,"")
        detail_html+=f"""
        <div style="padding:10px 0;border-top:1px solid #EEF2F6;">
          <div style="font-size:13px;font-weight:800;color:#172033;margin-bottom:4px;">⚠ {item['part']} · {v:.2f}{unit}</div>
          <div style="font-size:12.5px;line-height:1.7;color:#667085;">{item['bad']}</div>
        </div>"""
    summary = f"""<div style="font-size:14px;line-height:1.85;color:#667085;margin-bottom:10px;">
      <b style="color:#172033;">가장 먼저 교정할 부위는 {bad_items[0][2]['part']}입니다.</b><br>
      기준 범위를 벗어난 항목이 <b style="color:#D94A4A;">{len(bad_items)}개</b> 확인되었습니다.
    </div>""" if bad_items else """<div style="font-size:14px;color:#667085;">
      <b style="color:#172033;">전체 자세가 안정적입니다.</b><br>지금 상태를 유지해주세요.</div>"""
    good_html=f"""<div style="margin-top:12px;padding:12px;border-radius:12px;background:#F0FBF4;font-size:12.5px;color:#3B8C42;">
      <b>잘 유지되고 있는 항목</b><br>{" · ".join(good_items[:4])}</div>""" if good_items else ""
    goals_html="".join([f"{i+1}. {g}<br>" for i,g in enumerate(goals[:4])])
    return f"""
<div class="hc-card">
  <div class="hc-card-title"><span>맞춤 교정 코멘트</span><span class="badge-blue">AI Guide</span></div>
  {summary}{detail_html}{good_html}
  <div style="margin-top:16px;padding:14px;border-radius:14px;background:#F8FAFC;">
    <div style="font-size:13px;font-weight:800;color:#172033;margin-bottom:8px;">오늘의 실천 목표</div>
    <div style="font-size:13px;line-height:1.8;color:#667085;">{goals_html}</div>
  </div>
</div>"""

# ─────────────────────────────────────────
# 7지표 대시보드 (구버전 render_pretty_7_metric_dashboard 기반)
# ─────────────────────────────────────────
def render_metric_dashboard(metrics):
    BAR_CFG = {
        "CVA":       {"min":0,"max":40,"norm_min":0,"norm_max":20,"ticks":[(0,"0°"),(20,"20°"),(40,"40°")]},
        "TIA":       {"min":0,"max":45,"norm_min":0,"norm_max":20,"ticks":[(0,"0°"),(20,"20°"),(45,"45°")]},
        "knee_angle":{"min":60,"max":120,"norm_min":85,"norm_max":100,"ticks":[(85,"85°"),(100,"100°")]},
        "gaze_angle":{"min":0,"max":25,"norm_min":10,"norm_max":15,"ticks":[(10,"10°"),(15,"15°")]},
        "desk_diff": {"min":-0.10,"max":0.10,"norm_min":0,"norm_max":0.10,"ticks":[(0,"0"),(0.10,"0.10")]},
        "chair_gap": {"min":0,"max":0.40,"norm_min":0,"norm_max":0.20,"ticks":[(0,"0%"),(0.20,"20%")]},
    }
    cards_html=""
    for idx,key in enumerate(DISPLAY_ORDER,1):
        val=metrics.get(key); fb=FEEDBACK.get(key,{}); unit=IND_UNITS.get(key,"")
        level=classify_level(key,val); c=level_color(level)
        badge="✓ 정상" if level=="정상" else ("✗ 위험" if level=="위험" else "— 제외")
        badge_cls="badge-good" if level=="정상" else ("badge-bad" if level=="위험" else "badge-na")
        val_str=f"{val:.2f}{unit}" if val is not None else "인식불가"
        msg=(fb.get("good","") if level=="정상" else fb.get("bad","")).replace("\n","<br>")
        cfg=BAR_CFG.get(key,{}); span=cfg.get("max",1)-cfg.get("min",0) or 1
        def pct(v): return max(0,min(100,(v-cfg.get("min",0))/span*100))
        nl=pct(cfg.get("norm_min",0)); nw=pct(cfg.get("norm_max",1))-nl
        mp=pct(float(val)) if val is not None else 0
        tick_html="".join([f"<div class='rb-tick' style='left:{pct(t):.2f}%;'></div><div class='rb-tick-lbl' style='left:{pct(t):.2f}%;'>{l}</div>" for t,l in cfg.get("ticks",[])])
        cards_html+=f"""
<div class="dm-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <span style="font-size:15px;font-weight:900;color:{c};">{idx}. {fb.get('label',key)}</span>
    <span class="{badge_cls}">{badge}</span>
  </div>
  <div style="font-size:28px;font-weight:900;color:{c};margin-bottom:10px;">{val_str}</div>
  <div class="rb-wrap">
    <div class="rb-track">
      <div class="rb-norm" style="left:{nl:.2f}%;width:{nw:.2f}%;"></div>
      <div class="rb-marker" style="left:{mp:.2f}%;border-color:{c};"></div>
      {tick_html}
    </div>
  </div>
  <div class="dm-fb">
    <div style="font-size:11.5px;font-weight:800;color:#172033;margin-bottom:4px;">피드백</div>
    <div style="font-size:11.5px;line-height:1.6;color:#667085;">{msg}</div>
  </div>
  <div style="font-size:11px;color:#aaa;margin-top:6px;">정상: {fb.get('range','—')}</div>
</div>"""

    html=f"""<html><head><style>
body{{margin:0;padding:0;font-family:Pretendard,sans-serif;background:transparent;}}
.dm-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;padding:4px;}}
.dm-card{{background:#fff;border:1px solid #E5EAF2;border-radius:18px;padding:18px;
  box-shadow:0 8px 24px rgba(15,23,42,.05);min-height:340px;}}
.rb-wrap{{width:100%;margin:10px 0 14px;padding-bottom:22px;}}
.rb-track{{position:relative;width:100%;height:8px;border-radius:999px;background:#F43F5E;}}
.rb-norm{{position:absolute;top:0;height:8px;border-radius:999px;background:#2DBE60;z-index:1;}}
.rb-marker{{position:absolute;top:50%;width:14px;height:14px;border-radius:50%;background:#fff;
  border:4px solid;transform:translate(-50%,-50%);box-shadow:0 4px 10px rgba(15,23,42,.2);z-index:3;box-sizing:border-box;}}
.rb-tick{{position:absolute;top:15px;width:1px;height:12px;background:#CBD5E1;transform:translateX(-50%);}}
.rb-tick-lbl{{position:absolute;top:27px;transform:translateX(-50%);font-size:11px;font-weight:700;color:#185FA5;white-space:nowrap;}}
.dm-fb{{background:#FFF7F8;border-left:4px solid #F2527D;border-radius:10px;padding:10px 12px;margin-top:10px;}}
@media(max-width:900px){{.dm-grid{{grid-template-columns:repeat(2,1fr);}}}}
</style></head><body>
<div class="dm-grid">{cards_html}</div>
</body></html>"""
    components.html(html, height=1100, scrolling=False)

# ─────────────────────────────────────────
# 로그인 페이지
# ─────────────────────────────────────────
def page_login():
    b64=logo_base64()
    logo_html=f'<img src="data:image/png;base64,{b64}" width="240">' if b64 else \
              '<div style="font-size:2rem;font-weight:900;color:#1D9E75;">🪑 자세히봐</div>'
    col=st.columns([1,2,1])[1]
    with col:
        st.markdown(logo_html, unsafe_allow_html=True)
        st.markdown("<p style='color:#888;font-size:.85rem;margin-top:6px;'>RULA + VDT 기준 자세 & 작업환경 측정</p>", unsafe_allow_html=True)
        tab_li, tab_su = st.tabs(["로그인","회원가입"])
        with tab_li:
            u=st.text_input("이름",key="li_u"); p=st.text_input("비밀번호",type="password",key="li_p")
            if st.button("로그인",use_container_width=True,type="primary"):
                users=load_json(USERS_FILE)
                if u in users and users[u]["password"]==hash_pw(p):
                    st.session_state.logged_in=True; st.session_state.username=u; st.rerun()
                else: st.error("이름 또는 비밀번호가 올바르지 않습니다.")
        with tab_su:
            nu=st.text_input("이름",key="su_u"); np_=st.text_input("비밀번호",type="password",key="su_p")
            np2=st.text_input("비밀번호 확인",type="password",key="su_p2")
            if st.button("회원가입",use_container_width=True):
                users=load_json(USERS_FILE)
                if not nu: st.error("이름을 입력해주세요.")
                elif nu in users: st.error("이미 존재하는 사용자입니다.")
                elif np_!=np2: st.error("비밀번호가 일치하지 않습니다.")
                else:
                    users[nu]={"password":hash_pw(np_),"created_at":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    save_json(USERS_FILE,users); st.success("가입 완료! 로그인해주세요.")

# ─────────────────────────────────────────
# 사이드바 로고
# ─────────────────────────────────────────
def render_sidebar_logo():
    b64=logo_base64()
    if b64:
        st.sidebar.markdown(f"""
<div style="text-align:center;padding-bottom:10px;border-bottom:1px solid #E5EAF2;margin-bottom:10px;">
  <img src="data:image/png;base64,{b64}" width="160">
  <div style="font-size:13px;color:#667085;margin-top:4px;">AI 자세 분석 서비스</div>
</div>""", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("## 🪑 자세히봐")

# ─────────────────────────────────────────
# 측정 탭
# ─────────────────────────────────────────
def render_measure(pose_yolo, mlp, env_yolo, device, username):
    st.markdown("""<div class="hc-header"><h1>📷 자세 측정</h1>
    <p>GOOD 자세 5초 유지 → 작업환경 인식 → 완료</p></div>""", unsafe_allow_html=True)

    defaults=dict(phase="posture",good_start=None,env_bboxes={},
                  running=False,last_posture=None,last_metrics={})
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

    POSTURE_TOTAL=20; GOOD_HOLD=5
    col_cam,col_info=st.columns([3,1])

    with col_info:
        phase_box=st.empty(); result_box=st.empty(); good_bar=st.empty()
        st.markdown("---")
        st.markdown("**📐 측정 지표**"); metric_box=st.empty()
        st.markdown("---"); env_box=st.empty(); st.markdown("---")
        c1,c2=st.columns(2)
        with c1:
            if st.button("▶ 시작" if not st.session_state.running else "⏹ 중지",
                         use_container_width=True, type="primary"):
                st.session_state.running=not st.session_state.running
                if st.session_state.running:
                    speak("측정을 시작합니다. 측면을 카메라에 맞춰주세요.")
        with c2:
            if st.button("🔄 초기화",use_container_width=True):
                for k,v in defaults.items(): st.session_state[k]=v
                st.rerun()
        st.markdown("---")
        for key,fb in FEEDBACK.items():
            st.markdown(f"<small>• **{fb['label']}**: {fb['range']}</small>", unsafe_allow_html=True)

    with col_cam:
        frame_box=st.empty()

        # 측정 완료 결과 카드
        if not st.session_state.running and st.session_state.last_metrics:
            m=st.session_state.last_metrics
            score,risk,good,total=calc_score(m); sc=score_color(risk)
            st.markdown(f"""<div class="score-box" style="background:{sc};">
              <div class="score-num">{score}</div>
              <div class="score-lbl">/ 10점 | {risk} | {good}/{total}개 양호</div>
            </div>""", unsafe_allow_html=True)
            cols=st.columns(3)
            for i,(key,name) in enumerate(INDICATOR_NAMES.items()):
                val=m.get(key); unit=IND_UNITS.get(key,""); fb=FEEDBACK.get(key,{})
                level=classify_level(key,val)
                if val is None:
                    bc,vc,msg,border="#f0f0f0","#aaa","측정불가","#ccc"
                    badge='<span class="badge-na">제외</span>'
                elif level=="정상":
                    bc,vc,msg,border="#d4f7e7","#1D9E75",fb.get("good",""),"#1D9E75"
                    badge='<span class="badge-good">✓ 정상</span>'
                else:
                    bc,vc,msg,border="#fde8e8","#e74c3c",fb.get("bad",""),"#e74c3c"
                    badge='<span class="badge-bad">✗ 위험</span>'
                vs=f"{val:.2f}{unit}" if val is not None else "—"
                with cols[i%3]:
                    st.markdown(f"""<div class="hc-card" style="border-left:4px solid {border};">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-weight:700;font-size:.85rem;">{name}</span>{badge}</div>
                      <div style="font-size:1.4rem;font-weight:800;color:{vc};margin:6px 0;">{vs}</div>
                      <div style="font-size:.72rem;color:#555;line-height:1.4;">{msg.replace(chr(10),'<br>')}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown(build_ai_comment(m), unsafe_allow_html=True)
            render_metric_dashboard(m)

        if st.session_state.running:
            cap=cv2.VideoCapture(0); start_time=time.time()
            while st.session_state.running:
                ret,frame=cap.read()
                if not ret: st.warning("카메라를 찾을 수 없어요."); break
                frame=cv2.flip(frame,1); elapsed=time.time()-start_time

                if st.session_state.phase=="posture":
                    phase_box.info("🔵 **1단계: 자세 측정 중**")
                    out=run_posture(frame,pose_yolo,mlp,device,st.session_state.env_bboxes)
                    rv=out["result"]; met=out["metrics"]
                    result_frame=draw_posture_frame(frame,out,st.session_state.good_start,GOOD_HOLD,elapsed,POSTURE_TOTAL)
                    if met:
                        md="".join([f"{'🟢' if is_good(k,v) else '🔴'} **{INDICATOR_NAMES.get(k,k)}**: {v:.2f}{IND_UNITS.get(k,'')}\n\n"
                                    for k,v in met.items() if v is not None])
                        metric_box.markdown(md)
                    if rv:
                        result_box.markdown(f"## {'🟢' if rv=='GOOD' else '🔴'} {rv}")
                        if rv!=st.session_state.last_posture:
                            st.session_state.last_posture=rv
                            speak("좋은 자세입니다. 5초간 유지해주세요." if rv=="GOOD" else "자세를 바로 해주세요.")
                    if rv=="GOOD":
                        if not st.session_state.good_start: st.session_state.good_start=time.time()
                        held=time.time()-st.session_state.good_start
                        good_bar.progress(min(held/GOOD_HOLD,1.), text=f"GOOD 유지: {held:.1f}s / {GOOD_HOLD}s")
                        if held>=GOOD_HOLD:
                            speak("자세 완료! 이제 작업환경을 인식합니다.")
                            st.session_state.phase="environment"
                    else:
                        st.session_state.good_start=None
                        good_bar.progress(0., text="BAD 자세 — 다시 시도")
                    if elapsed>=POSTURE_TOTAL:
                        speak("시간 종료. 다시 시도해주세요."); st.session_state.running=False

                elif st.session_state.phase=="environment":
                    phase_box.info("🟠 **2단계: 작업환경 인식 중**")
                    env_out=run_environment(frame,env_yolo)
                    det=env_out["detected"]; st.session_state.env_bboxes=env_out["bboxes"]
                    result_frame=draw_env_frame(frame,env_out)
                    items=list(ENV_CLASSES.values())
                    env_box.markdown("**감지 현황**\n\n"+"\n\n".join([f"{'✅' if i in det else '⬜'} {i}" for i in items]))
                    if all(i in det for i in items):
                        final=out.get("metrics",{}) if "out" in dir() else {}
                        st.session_state.last_metrics=final
                        save_history(username,final); sync_challenge(username,final)
                        speak("환경 인식 완료. 측정을 종료합니다.")
                        st.session_state.running=False; st.balloons()

                frame_rgb=cv2.cvtColor(result_frame,cv2.COLOR_BGR2RGB)
                frame_box.image(frame_rgb,channels="RGB",use_container_width=True)
            cap.release()

# ─────────────────────────────────────────
# 이력 탭
# ─────────────────────────────────────────
def render_history(username):
    st.markdown("""<div class="hc-header"><h1>📈 측정 이력</h1>
    <p>자세 분석 결과 이력을 확인하세요.</p></div>""", unsafe_allow_html=True)
    history=load_json(HISTORY_FILE); records=history.get(username,[])
    if not records: st.info("아직 측정 이력이 없어요. 측정을 시작해보세요!"); return
    scores=[r["score"] for r in records]; avg=round(sum(scores)/len(scores),1)
    m1,m2,m3=st.columns(3)
    m1.metric("총 측정 횟수",f"{len(records)}회")
    m2.metric("평균 점수",f"{avg} / 10")
    m3.metric("최근 위험도",records[0]["risk"])
    if len(records)>=2:
        st.markdown("### 점수 추이")
        df=pd.DataFrame([{"회차":i+1,"점수":float(h["score"]),"측정시간":h["time"]} for i,h in enumerate(reversed(records))])
        base=alt.Chart(df).encode(x=alt.X("회차:O",title="측정 회차"),y=alt.Y("점수:Q",scale=alt.Scale(domain=[0,10])),tooltip=["측정시간:N","점수:Q"])
        chart=(base.mark_line(strokeWidth=4,interpolate="monotone")+base.mark_circle(size=80)+
               base.mark_text(align="center",baseline="bottom",dy=-10,fontSize=12,fontWeight="bold").encode(text=alt.Text("점수:Q",format=".1f"))
              ).properties(height=300).configure_view(strokeWidth=0)
        st.altair_chart(chart,use_container_width=True)
    st.markdown("### 전체 이력")
    for r in records[:20]:
        sc=score_color(r["risk"]); rate=round(r["good"]/r["total"]*100)
        missing_str=f" | 미측정: {', '.join([m['label'] for m in r.get('missing_items',[])])}" if r.get("missing_items") else ""
        st.markdown(f"""<div class="hc-card" style="border-left:4px solid {sc};">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:.85rem;color:#666;">{r['time']}</span>
            <span style="font-weight:800;color:{sc};font-size:1.1rem;">{r['score']}점 ({r['risk']})</span>
          </div>
          <div style="font-size:.78rem;color:#888;margin-top:4px;">양호 {r['good']}/{r['total']}개{missing_str}</div>
          <div style="margin-top:8px;height:6px;background:#EEF2F6;border-radius:999px;">
            <div style="height:6px;width:{rate}%;background:{sc};border-radius:999px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 챌린지 탭
# ─────────────────────────────────────────
def render_challenge():
    st.markdown("""<div class="hc-header"><h1>🎯 바른자세 챌린지</h1>
    <p>측정할 때마다 점수가 포인트로 누적됩니다.</p></div>""", unsafe_allow_html=True)
    ch=load_json(CHALLENGE_FILE)
    if isinstance(ch,list): ch={}
    if not ch: st.info("아직 참여자가 없습니다."); return
    members=sorted(ch.values(),key=lambda x:x.get("total_point",0),reverse=True)
    max_pt=max([m.get("total_point",0) for m in members]) or 1
    icons=["🐰","🐢","🦊","🐻","🐼","🐯"]
    race_html='<div class="race-box"><div style="font-size:18px;font-weight:900;margin-bottom:14px;">🏁 자세 포인트 레이스</div>'
    for idx,member in enumerate(members):
        pct=max(8,min(int(member.get("total_point",0)/max_pt*100),100))
        icon=icons[idx%len(icons)]
        race_html+=f"""<div class="race-row">
          <div style="display:flex;justify-content:space-between;font-size:14px;font-weight:800;">
            <span>{idx+1}. {member.get('name','익명')}</span>
            <span style="color:#185FA5;">{member.get('total_point',0)}P · {member.get('count',0)}회</span>
          </div>
          <div class="race-bar">
            <div class="race-fill" style="width:{pct}%;"></div>
            <div class="race-icon" style="left:calc({pct}% - 22px);">{icon}</div>
            <div class="race-flag">🏁</div>
          </div>
        </div>"""
    race_html+='</div>'
    st.markdown(race_html, unsafe_allow_html=True)
    st.markdown("### 누적 포인트 순위")
    for i,member in enumerate(members,1):
        recs=member.get("records",[]); latest=recs[0] if recs else {}
        risk=latest.get("risk","—"); sc=score_color(risk)
        bc="badge-good" if risk=="안전" else ("badge-bad" if risk=="위험" else "badge-na")
        st.markdown(f"""<div class="hc-card">
          <div class="hc-card-title"><span>{i}. {member.get('name','익명')}</span>
            <span class="{bc}">{risk}</span></div>
          <div style="font-size:28px;font-weight:900;color:#185FA5;margin-bottom:8px;">{member.get('total_point',0)}P</div>
          <div style="font-size:14px;line-height:1.8;color:#172033;">
            측정 횟수: <b>{member.get('count',0)}회</b><br>
            최근 점수: <b>{latest.get('score','-')}/10</b></div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# page_config — 모바일/데스크탑 자동 감지
# ─────────────────────────────────────────
# URL 파라미터 ?mobile=1 이면 모바일 레이아웃
query_params = st.query_params
is_mobile = query_params.get("mobile","0") == "1"

st.set_page_config(
    page_title="자세히봐 — AI 자세 분석",
    layout="centered" if is_mobile else "wide",
    page_icon="🪑",
    initial_sidebar_state="collapsed" if is_mobile else "expanded",
)
inject_css(mobile=is_mobile)

# ─────────────────────────────────────────
# 세션 초기화
# ─────────────────────────────────────────
if "logged_in" not in st.session_state: st.session_state.logged_in=False
if "username"  not in st.session_state: st.session_state.username=""

if not st.session_state.logged_in:
    page_login(); st.stop()

username=st.session_state.username

# ─────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────
@st.cache_resource
def get_models(): return load_models()

with st.spinner("모델 로딩 중..."):
    try:
        pose_yolo,mlp,env_yolo,device=get_models()
    except Exception as e:
        st.error(f"모델 로드 실패: {e}"); st.stop()

# ─────────────────────────────────────────
# 네비게이션
# ─────────────────────────────────────────
if is_mobile:
    # 모바일: 상단 탭
    menu=st.radio("메뉴",["📷 측정","📈 이력","🎯 챌린지"],horizontal=True,label_visibility="collapsed")
else:
    # 데스크탑: 사이드바
    render_sidebar_logo()
    menu=st.sidebar.radio("메뉴",["📷 측정","📈 이력","🎯 챌린지"])
    st.sidebar.markdown("---")
    st.sidebar.caption(f"로그인: {username}")
    if st.sidebar.button("로그아웃",use_container_width=True):
        st.session_state.logged_in=False; st.session_state.username=""; st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 서비스 정보")
    st.sidebar.caption("YOLOv8 Pose + Attention MLP 기반")
    st.sidebar.caption("RULA + VDT 고시 제2020-17호 기준")
    st.sidebar.caption("의료 진단 대체 불가 — 참고용")

# ─────────────────────────────────────────
# 페이지 렌더링
# ─────────────────────────────────────────
if "측정" in menu:
    render_measure(pose_yolo,mlp,env_yolo,device,username)
elif "이력" in menu:
    render_history(username)
elif "챌린지" in menu:
    render_challenge()

if is_mobile:
    st.markdown("---")
    if st.button("로그아웃",use_container_width=True):
        st.session_state.logged_in=False; st.session_state.username=""; st.rerun()
