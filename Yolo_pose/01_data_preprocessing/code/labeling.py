# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 1단계 - 데이터 전처리
# 설명: CSV 파일 병합 및 GOOD/BAD 라벨링 수행
# 작성일: 2026.05.13
# ================================================
import pandas as pd
import os

# Paths
new_path = r'D:\antigravity\semi2_contest\new images_data'
old_path = r'D:\antigravity\semi2_contest\FOR_DA\FOR_DA'
output_path = r'D:\antigravity\semi2_contest'

def merge_csv(filename):
    new_file = os.path.join(new_path, filename)
    old_file = os.path.join(old_path, filename)
    
    if not os.path.exists(new_file) or not os.path.exists(old_file):
        print(f"Error: {filename} not found in one of the paths.")
        return None
    
    df_new = pd.read_csv(new_file)
    df_old = pd.read_csv(old_file)
    
    # concat - New data first to keep it in case of duplicates
    df_merged = pd.concat([df_new, df_old], ignore_index=True)
    
    # check duplicates in filename
    before_count = len(df_merged)
    # keep='first' maintains the row from df_new
    df_merged = df_merged.drop_duplicates(subset=['filename'], keep='first')
    after_count = len(df_merged)
    duplicates_removed = before_count - after_count
    
    # Save
    out_name = filename.replace('.csv', '_merged.csv')
    df_merged.to_csv(os.path.join(output_path, out_name), index=False)
    
    return {
        'n_new': len(df_new),
        'n_old': len(df_old),
        'n_dup': duplicates_removed,
        'n_total': len(df_merged),
        'df': df_merged
    }

# Merge final_labels_confirmed.csv
result_labels = merge_csv('final_labels_confirmed.csv')

# Merge yolo_landmarks_clean.csv
result_landmarks = merge_csv('yolo_landmarks_clean.csv')

if result_labels:
    n_new = result_labels['n_new']
    n_old = result_labels['n_old']
    n_dup = result_labels['n_dup']
    n_total = result_labels['n_total']
    df_final = result_labels['df']
    
    # Stats for final_labels
    if 'final_label' in df_final.columns:
        good_count = len(df_final[df_final['final_label'] == 'GOOD'])
        bad_count = len(df_final[df_final['final_label'] == 'BAD'])
        good_pct = (good_count / n_total) * 100 if n_total > 0 else 0
        bad_pct = (bad_count / n_total) * 100 if n_total > 0 else 0
        
        print(f"신규 데이터: {n_new}장")
        print(f"기존 데이터: {n_old}장")
        print(f"중복 제거: {n_dup}장")
        print(f"최종 합계: {n_total}장")
        print(f"GOOD: {good_count}장 ({good_pct:.1f}%)")
        print(f"BAD: {bad_count}장 ({bad_pct:.1f}%)")
    else:
        print("Error: 'final_label' column not found in merged data.")
