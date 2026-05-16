# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 3단계 - 모델 성능 개선
# 설명: Recall 향상을 위한 최적 임계값 탐색
# 작성일: 2026.05.13
# ================================================
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. Load Data & Prepare Test Set (Same as STEP 2 Final)
data_dir = r'D:\antigravity\semi2_contest\model_data'
results_dir = r'D:\antigravity\semi2_contest\model_results'

df_labels = pd.read_csv(os.path.join(data_dir, 'final_labels_confirmed_merged.csv'))
df_landmarks = pd.read_csv(os.path.join(data_dir, 'yolo_landmarks_clean_merged.csv'))
df = pd.merge(df_labels, df_landmarks, on='filename')
df['label'] = df['final_label'].map({'GOOD': 0, 'BAD': 1})

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
y = df['label'].values

# Split (20% Test fixed)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_raw, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
scaler.fit(X_train_val) # Use same scaling as original run
X_test_scaled = scaler.transform(X_test)
X_test_t = torch.FloatTensor(X_test_scaled)

# 2. Model Architecture (Attention MLP)
class AttentionMLP(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMLP, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        attn_out, _ = self.attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        return self.net(attn_out.squeeze(1))

model = AttentionMLP(X_test_t.shape[1])
model.load_state_dict(torch.load('best_Attention MLP.pt'))
model.eval()

# 3. Get Probabilities
with torch.no_grad():
    probs = model(X_test_t).squeeze().numpy()

# 4. Threshold Optimization
thresholds = np.arange(0.3, 0.71, 0.05)
results = []

for th in thresholds:
    preds = (probs >= th).astype(int)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    
    results.append({
        'Threshold': th,
        'Recall': rec,
        'Precision': prec,
        'F1-Score': f1,
        'AUC': auc,
        'Accuracy': acc
    })

df_th = pd.DataFrame(results)

# Filter: Precision >= 0.5, Sort: Recall (Desc)
df_filtered = df_th[df_th['Precision'] >= 0.5].copy()
df_sorted = df_filtered.sort_values(by=['Recall', 'F1-Score'], ascending=False)
best_th_row = df_sorted.iloc[0]
best_th = best_th_row['Threshold']

# 5. Output 1: Performance Table
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("[출력 1] 임계값별 성능표 (Attention MLP)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(df_th.to_string(index=False, formatters={
    'Threshold': '{:.2f}'.format,
    'Recall': '{:.4f}'.format,
    'Precision': '{:.4f}'.format,
    'F1-Score': '{:.4f}'.format,
    'AUC': '{:.4f}'.format,
    'Accuracy': '{:.4f}'.format
}))

# 6. Output 2: Selection
current_05 = df_th[np.isclose(df_th['Threshold'], 0.5)].iloc[0]
print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"[출력 2] 최적 임계값 선정 결과")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"[BEST] 최적 임계값: {best_th:.2f}")
print(f"성능: Recall {best_th_row['Recall']:.4f}, Precision {best_th_row['Precision']:.4f}")

# Performance change vs 0.5
rec_diff = (best_th_row['Recall'] - current_05['Recall']) / current_05['Recall'] * 100
prec_diff = (best_th_row['Precision'] - current_05['Precision']) / current_05['Precision'] * 100
print(f"0.5 대비 변화량: Recall {rec_diff:+.1f}% / Precision {prec_diff:+.1f}%")

# 7. Output 3: Visualization
plt.figure(figsize=(12, 8))
plt.plot(df_th['Threshold'], df_th['Recall'], marker='o', label='Recall', color='blue')
plt.plot(df_th['Threshold'], df_th['Precision'], marker='s', label='Precision', color='green')
plt.plot(df_th['Threshold'], df_th['F1-Score'], marker='^', label='F1-Score', color='red')

plt.axvline(x=best_th, color='orange', linestyle='-', alpha=0.7, label=f'Best ({best_th:.2f})')
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Current (0.50)')

plt.title('Threshold Optimization: Recall vs Precision')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'threshold_optimization.png'))

# 8. Output 4: Final Comparison Table
print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"[출력 4] 임계값 0.5 vs 최적 ({best_th:.2f}) 비교")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
comparison = []
metrics = ['Recall', 'F1-Score', 'AUC', 'Accuracy', 'Precision']
for m in metrics:
    v_old = current_05[m]
    v_new = best_th_row[m]
    comparison.append({
        '지표': m,
        '기존(0.5)': f"{v_old:.4f}",
        '최적 임계값': f"{v_new:.4f}",
        '변화량': f"{v_new - v_old:+.4f}"
    })
print(pd.DataFrame(comparison).to_string(index=False))
