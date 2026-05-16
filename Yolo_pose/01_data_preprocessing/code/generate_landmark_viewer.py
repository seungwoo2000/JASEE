import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 라이브러리 체크 및 설치
try:
    import cv2
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("필요한 라이브러리를 설치합니다...")
    install("opencv-python")
    install("pandas")
    install("numpy")
    install("tqdm")
    import cv2
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

import base64

def generate_viewer(csv_path, image_dirs, output_html):
    print(f"--- Landmark Viewer 생성 시작 ---")
    
    if not os.path.exists(csv_path):
        print(f"[오류] CSV 파일을 찾을 수 없습니다: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 색상 정의 (BGR)
    COLORS = {
        'nose': (255, 255, 255),    # 흰색
        'eye': (235, 206, 135),     # 하늘색
        'ear': (0, 255, 255),       # 노란색
        'shoulder': (255, 0, 0),    # 파란색
        'elbow': (0, 165, 255),     # 주황색
        'wrist': (203, 192, 255),   # 분홍색
        'hip': (0, 255, 0),         # 초록색
        'knee': (128, 0, 128),      # 보라색
        'ankle': (0, 0, 255)        # 빨간색
    }

    CONNECTIONS = [
        (('nose_x', 'nose_y'), ('left_shoulder_x', 'left_shoulder_y'), COLORS['nose']),
        (('nose_x', 'nose_y'), ('right_shoulder_x', 'right_shoulder_y'), COLORS['nose']),
        (('left_ear_x', 'left_ear_y'), ('left_shoulder_x', 'left_shoulder_y'), COLORS['ear']),
        (('right_ear_x', 'right_ear_y'), ('right_shoulder_x', 'right_shoulder_y'), COLORS['ear']),
        (('left_shoulder_x', 'left_shoulder_y'), ('left_hip_x', 'left_hip_y'), COLORS['shoulder']),
        (('right_shoulder_x', 'right_shoulder_y'), ('right_hip_x', 'right_hip_y'), COLORS['shoulder']),
        (('left_hip_x', 'left_hip_y'), ('left_knee_x', 'left_knee_y'), COLORS['hip']),
        (('right_hip_x', 'right_hip_y'), ('right_knee_x', 'right_knee_y'), COLORS['hip']),
        (('left_knee_x', 'left_knee_y'), ('left_ankle_x', 'left_ankle_y'), COLORS['ankle']),
        (('right_knee_x', 'right_knee_y'), ('right_ankle_x', 'right_ankle_y'), COLORS['ankle']),
        (('left_shoulder_x', 'left_shoulder_y'), ('left_elbow_x', 'left_elbow_y'), COLORS['elbow']),
        (('right_shoulder_x', 'right_shoulder_y'), ('right_elbow_x', 'right_elbow_y'), COLORS['elbow']),
        (('left_elbow_x', 'left_elbow_y'), ('left_wrist_x', 'left_wrist_y'), COLORS['wrist']),
        (('right_elbow_x', 'right_elbow_y'), ('right_wrist_x', 'right_wrist_y'), COLORS['wrist'])
    ]

    KP_MAP = {
        'nose': 'nose', 'left_eye': 'eye', 'right_eye': 'eye',
        'left_ear': 'ear', 'right_ear': 'ear', 'left_shoulder': 'shoulder',
        'right_shoulder': 'shoulder', 'left_elbow': 'elbow', 'right_elbow': 'elbow',
        'left_wrist': 'wrist', 'right_wrist': 'wrist', 'left_hip': 'hip',
        'right_hip': 'hip', 'left_knee': 'knee', 'right_knee': 'knee',
        'left_ankle': 'ankle', 'right_ankle': 'ankle'
    }

    html_cards = []
    stats = {'full': 0, 'partial': 0, 'flipped': 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="이미지 처리 중"):
        filename = row['filename']
        img_path = None
        for d in image_dirs:
            # Recursively check subfolders if needed, but here we assume direct or depth-1
            trial = os.path.join(d, filename)
            if os.path.exists(trial):
                img_path = trial
                break
            # Try GOOD/BAD subfolders
            for sub in ['GOOD', 'BAD']:
                trial = os.path.join(d, sub, filename)
                if os.path.exists(trial):
                    img_path = trial
                    break
        
        if not img_path:
            continue

        image = cv2.imread(img_path)
        if image is None: continue
        
        if row.get('is_flipped', False):
            image = cv2.flip(image, 1)
            stats['flipped'] += 1

        h, w = image.shape[:2]
        recognized_count = 0

        # 선 그리기
        for start_kp, end_kp, color in CONNECTIONS:
            x1, y1 = row[start_kp[0]] * w, row[start_kp[1]] * h
            x2, y2 = row[end_kp[0]] * w, row[end_kp[1]] * h
            c1 = row[start_kp[0].replace('_x', '_conf')]
            c2 = row[end_kp[0].replace('_x', '_conf')]
            
            if c1 >= 0.5 and c2 >= 0.5:
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 점 그리기
        for kp, type_name in KP_MAP.items():
            x, y = row[f'{kp}_x'] * w, row[f'{kp}_y'] * h
            conf = row[f'{kp}_conf']
            color = COLORS[type_name]
            
            if conf >= 0.5:
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
                recognized_count += 1
            elif conf > 0:
                # 점선 대신 얇은 원으로 표시
                cv2.circle(image, (int(x), int(y)), 5, color, 1)

        if recognized_count == 17: stats['full'] += 1
        else: stats['partial'] += 1

        # 텍스트 오버레이
        cv2.putText(image, f"sitting_direction: {row['sitting_direction']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"is_flipped: {row['is_flipped']}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Base64 변환
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 카드 생성
        card = f"""
        <div class="card" data-dir="{row['sitting_direction']}" data-flipped="{row['is_flipped']}" data-full="{recognized_count == 17}">
            <img src="data:image/jpeg;base64,{img_base64}">
            <div class="info">
                <div class="filename">{filename}</div>
                <div>방향: {row['sitting_direction']} | 반전: {row['is_flipped']}</div>
                <div class="recon">인식 관절: {recognized_count}/17개</div>
            </div>
        </div>
        """
        html_cards.append(card)

    # HTML 템플릿
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>Landmark Viewer</title>
        <style>
            body {{ background-color: #0f172a; color: #f8fafc; font-family: sans-serif; margin: 2rem; }}
            .stats {{ background: #1e293b; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; display: flex; gap: 2rem; }}
            .filters {{ margin-bottom: 2rem; display: flex; gap: 1rem; }}
            button {{ background: #334155; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer; }}
            button.active {{ background: #6366f1; }}
            .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; }}
            .card {{ background: #1e293b; border-radius: 0.5rem; overflow: hidden; border: 1px solid #334155; }}
            .card img {{ width: 100%; display: block; }}
            .card .info {{ padding: 1rem; }}
            .filename {{ font-weight: bold; margin-bottom: 0.5rem; color: #22d3ee; word-break: break-all; }}
            .recon {{ color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem; }}
        </style>
    </head>
    <body>
        <h1>Pose Landmark Viewer</h1>
        
        <div class="stats">
            <div>전체 이미지: <b>{len(df)}장</b></div>
            <div>17개 완전 인식: <b style="color:#22d3ee">{stats['full']}장</b></div>
            <div>부분 인식: <b style="color:#f87171">{stats['partial']}장</b></div>
            <div>반전 처리: <b style="color:#fbbf24">{stats['flipped']}장</b></div>
        </div>

        <div class="filters">
            <button onclick="filter('all')" class="active">전체 보기</button>
            <button onclick="filter('LEFT')">LEFT만</button>
            <button onclick="filter('RIGHT')">RIGHT만</button>
            <button onclick="filter('flipped')">반전 이미지만</button>
            <button onclick="filter('partial')">부분인식만</button>
        </div>

        <div class="grid" id="grid">
            {''.join(html_cards)}
        </div>

        <script>
            function filter(type) {{
                const cards = document.querySelectorAll('.card');
                const buttons = document.querySelectorAll('button');
                buttons.forEach(b => b.classList.remove('active'));
                event.target.classList.add('active');

                cards.forEach(card => {{
                    card.style.display = 'none';
                    if (type === 'all') card.style.display = 'block';
                    else if (type === 'LEFT' && card.dataset.dir === 'LEFT') card.style.display = 'block';
                    else if (type === 'RIGHT' && card.dataset.dir === 'RIGHT') card.style.display = 'block';
                    else if (type === 'flipped' && card.dataset.flipped === 'True') card.style.display = 'block';
                    else if (type === 'partial' && card.dataset.full === 'false') card.style.display = 'block';
                }});
            }}
        </script>
    </body>
    </html>
    """

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" 전체 이미지: {len(df)}장")
    print(f" 시각화 완료: {len(html_cards)}장")
    print(f" 저장 경로: {output_html}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == "__main__":
    # 프로젝트 환경에 맞는 경로 설정
    CSV_FILE = r'yolo_landmarks_extracted.csv'
    IMG_DIRS = [
        r'D:\antigravity\semi2_contest\new images_data\images',
        r'd:\antigravity\semi2_contest\FOR_DA\FOR_DA\YOLO_full_body',
        r'd:\antigravity\semi2_contest\FOR_DA\FOR_DA\YOLO_ankle_visible'
    ]
    OUTPUT = 'landmark_viewer.html'
    
    generate_viewer(CSV_FILE, IMG_DIRS, OUTPUT)
