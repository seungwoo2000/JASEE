# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 1단계 - 데이터 전처리
# 설명: YOLOv8-Pose 고도화 랜드마크 추출 (좌우 반전 및 방향 판정 포함)
# 작성일: 2026.05.14
# ================================================
from ultralytics import YOLO
import pandas as pd
import os
import glob
import cv2
import numpy as np

def extract_landmarks(image_dir, model_path='yolov8n-pose.pt'):
    print(f"--- Landmark 추출 시작: {image_dir} ---")
    
    model = YOLO(model_path)
    images = glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True) + \
             glob.glob(os.path.join(image_dir, "**/*.png"), recursive=True)
    
    results_list = []
    failed_images = []
    flip_count = 0
    
    kp_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    for img_path in images:
        fname = os.path.basename(img_path)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            failed_images.append(f"{fname} (파일 로드 실패)")
            continue

        # 1차 추론 (방향 및 반전 여부 판단용)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = model(image_rgb, verbose=False)
        
        target_res = None
        for r in results:
            if r.keypoints is not None and len(r.keypoints.conf) > 0:
                target_res = r
                break
        
        if target_res is None:
            failed_images.append(f"{fname} (사람 감지 실패)")
            continue

        conf = target_res.keypoints.conf[0].cpu().numpy()
        left_ear_conf = conf[3]
        right_ear_conf = conf[4]
        
        is_flipped = False
        # 왼쪽 측면 이미지 판단 (왼쪽 귀가 더 잘 보일 때)
        if right_ear_conf < left_ear_conf:
            image_bgr = cv2.flip(image_bgr, 1) # 좌우 반전
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = model(image_rgb, verbose=False)
            
            target_res = None
            for r in results:
                if r.keypoints is not None and len(r.keypoints.conf) > 0:
                    target_res = r
                    break
            
            if target_res is None:
                failed_images.append(f"{fname} (반전 후 감지 실패)")
                continue
            
            conf = target_res.keypoints.conf[0].cpu().numpy()
            is_flipped = True
            flip_count += 1

        # 관절 좌표 및 신뢰도 추출
        kpts = target_res.keypoints.xyn[0].cpu().numpy()
        
        l_sh_conf = conf[5]
        r_sh_conf = conf[6]
        
        # 감지 실패 처리 (어깨 신뢰도 기준)
        if l_sh_conf < 0.5 or r_sh_conf < 0.5:
            failed_images.append(f"{fname} (어깨 신뢰도 낮음: L={l_sh_conf:.2f}, R={r_sh_conf:.2f})")
            continue

        # 앉은 방향 판정
        nose_x = kpts[0][0]
        l_sh_x = kpts[5][0]
        r_sh_x = kpts[6][0]
        shoulder_center_x = (l_sh_x + r_sh_x) / 2
        
        sitting_direction = "LEFT" if nose_x < shoulder_center_x else "RIGHT"

        # 결과 저장
        row = {
            'filename': fname,
            'sitting_direction': sitting_direction,
            'is_flipped': is_flipped
        }
        
        for i, name in enumerate(kp_names):
            row[f'{name}_x'] = kpts[i][0]
            row[f'{name}_y'] = kpts[i][1]
            row[f'{name}_conf'] = conf[i]
            
        results_list.append(row)

    # CSV 저장
    df = pd.DataFrame(results_list)
    output_csv = 'yolo_landmarks_extracted.csv'
    df.to_csv(output_csv, index=False)
    
    # 실패 목록 저장
    with open('failed_images.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(failed_images))

    # 콘솔 출력
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" 전체 이미지: {len(images)}장")
    print(f" 추출 성공: {len(results_list)}장")
    print(f" 감지 실패: {len(failed_images)}장")
    if failed_images:
        print(f" [실패 목록]")
        for f in failed_images:
            print(f"  - {f}")
    print(f" 왼쪽 측면 반전 처리: {flip_count}장")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" 결과가 '{output_csv}' 및 'failed_images.txt'에 저장되었습니다.")

# 실행 예시 (필요 시 주석 해제)
# extract_landmarks(r'D:\antigravity\semi2_contest\model_data\images')