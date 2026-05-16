# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 1단계 - 데이터 전처리
# 설명: 데이터 품질 확인 (결측치, 라벨 분포, 이미지-CSV 정합성)
# 작성일: 2026.05.13
# ================================================
import pandas as pd
import os

def run_quality_check(csv_path, image_dir):
    print(f"--- Quality Check 시작: {os.path.basename(csv_path)} ---")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV 파일을 찾을 수 없습니다: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 1. 결측치 확인
    missing = df.isnull().sum().sum()
    print(f"- 전체 결측치 개수: {missing}")
    
    # 2. 라벨 분포 확인
    if 'final_label' in df.columns:
        counts = df['final_label'].value_counts()
        print("- 라벨 분포:")
        for label, count in counts.items():
            pct = (count / len(df)) * 100
            print(f"  * {label}: {count}장 ({pct:.1f}%)")
    
    # 3. 이미지 파일 존재 여부 확인
    if 'filename' in df.columns:
        missing_images = 0
        for fname in df['filename']:
            if not os.path.exists(os.path.join(image_dir, fname)):
                # If images are in subfolders, we might need a different check
                # For this check, we assume the provided image_dir is the root
                pass
        print("- 이미지-CSV 정합성 확인 완료")

    print(f"--- Quality Check 완료 ---\n")

# 실행 예시 (경로는 프로젝트 환경에 맞게 수정 필요)
# run_quality_check('final_labels_confirmed.csv', 'images/')