# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 2단계 - 모델 비교 실험
# 설명: 4종의 머신러닝 모델 성능 비교 및 평가
# 작성일: 2026.05.13
# ================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score, precision_score

# 1. Load Data
data_dir = r'D:\antigravity\semi2_contest\model_data'
results_dir = r'D:\antigravity\semi2_contest\model_results'
os.makedirs(results_dir, exist_ok=True)

df_labels = pd.read_csv(os.path.join(data_dir, 'final_labels_confirmed_merged.csv'))
df_landmarks = pd.read_csv(os.path.join(data_dir, 'yolo_landmarks_clean_merged.csv'))
df = pd.merge(df_labels, df_landmarks, on='filename')
df['label'] = df['final_label'].map({'GOOD': 0, 'BAD': 1})

# Feature Selection (10 landmarks)
def get_features(row):
    side = row['sitting_direction']
    cols = ['left_ear_x', 'left_ear_y', 'left_shoulder_x', 'left_shoulder_y', 
            'left_hip_x', 'left_hip_y', 'left_knee_x', 'left_knee_y', 
            'left_ankle_x', 'left_ankle_y'] if side == 'LEFT' else \
           ['right_ear_x', 'right_ear_y', 'right_shoulder_x', 'right_shoulder_y', 
            'right_hip_x', 'right_hip_y', 'right_knee_x', 'right_knee_y', 
            'right_ankle_x', 'right_ankle_y']
    return pd.Series(row[cols].tolist())

X_raw = df.apply(get_features, axis=1)
y = df['label']

# 2. Split (80:20)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_raw, y, test_size=0.20, random_state=42, stratify=y)

print(f"Data Split Status:")
print(f"- Train: {len(X_train_val)}장")
print(f"- Test: {len(X_test)}장")

# 3. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)

# 4. Models & Params
# class_weight='balanced' handles GOOD(0)/BAD(1) imbalance
# XGBoost scale_pos_weight = count(0)/count(1)
spw = (y_train_val == 0).sum() / (y_train_val == 1).sum()

models = {
    'Logistic Regression': LogisticRegression(random_state=42, C=1.0, class_weight='balanced', max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1, 
                             subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, 
                                            min_samples_split=5, class_weight='balanced')
}

# 5. Training & Evaluation on Test Set
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train_val)
    
    preds_proba = model.predict_proba(X_test_scaled)[:, 1]
    preds = model.predict(X_test_scaled)
    
    res = {
        'Model': name,
        'Recall': recall_score(y_test, preds),
        'F1-Score': f1_score(y_test, preds),
        'AUC': roc_auc_score(y_test, preds_proba),
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds)
    }
    results.append(res)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='Recall', ascending=False)

# 6. Output
def highlight_max(s):
    is_max = s == s.max()
    return ['★ ' + f'{v:.4f}' if m else f'{v:.4f}' for v, m in zip(s, is_max)]

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("STEP 1. 머신러닝 모델 성능 결과 (Test Set 기준)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
df_print = df_results.copy()
for col in ['Recall', 'F1-Score', 'AUC', 'Accuracy', 'Precision']:
    df_print[col] = highlight_max(df_results[col])
print(df_print.to_string(index=False))

df_results.to_csv(os.path.join(results_dir, 'ml_results.csv'), index=False)

# Plot
plt.figure(figsize=(12, 8))
df_melt = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')
sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model')
plt.title('ML Model Performance (Test Set Only)')
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'ml_comparison.png'))

print(f"\n최종 추천 모델: {df_results.iloc[0]['Model']} (Recall 1순위 기준)")
