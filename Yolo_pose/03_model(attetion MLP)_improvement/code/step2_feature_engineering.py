# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 3단계 - 모델 성능 개선
# 설명: 데이터 누수 제거 및 신규 피처 생성
# 작성일: 2026.05.13
# ================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score, precision_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. Load Data
data_dir = r'D:\antigravity\semi2_contest\model_data'
results_dir = r'D:\antigravity\semi2_contest\model_results'

df_labels = pd.read_csv(os.path.join(data_dir, 'final_labels_confirmed_merged.csv'))
df_landmarks = pd.read_csv(os.path.join(data_dir, 'yolo_landmarks_clean_merged.csv'))
df = pd.merge(df_labels, df_landmarks, on='filename')
df['label'] = df['final_label'].map({'GOOD': 0, 'BAD': 1})

# 2. Feature Engineering (16 features)
def get_engineered_features_16(row):
    side = row['sitting_direction']
    p = {
        'ear': (row['left_ear_x'], row['left_ear_y']),
        'shoulder': (row['left_shoulder_x'], row['left_shoulder_y']),
        'hip': (row['left_hip_x'], row['left_hip_y']),
        'knee': (row['left_knee_x'], row['left_knee_y']),
        'ankle': (row['left_ankle_x'], row['left_ankle_y'])
    } if side == 'LEFT' else {
        'ear': (row['right_ear_x'], row['right_ear_y']),
        'shoulder': (row['right_shoulder_x'], row['right_shoulder_y']),
        'hip': (row['right_hip_x'], row['right_hip_y']),
        'knee': (row['right_knee_x'], row['right_knee_y']),
        'ankle': (row['right_ankle_x'], row['right_ankle_y'])
    }
    
    # Raw features (10)
    raw = [p['ear'][0], p['ear'][1], p['shoulder'][0], p['shoulder'][1], p['hip'][0], p['hip'][1], p['knee'][0], p['knee'][1], p['ankle'][0], p['ankle'][1]]
    
    # 1. Distances (4)
    def dist(p1, p2): return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    d_ear_sh = dist(p['ear'], p['shoulder'])
    d_sh_hip = dist(p['shoulder'], p['hip'])
    d_hip_kn = dist(p['hip'], p['knee'])
    d_kn_ank = dist(p['knee'], p['ankle'])
    
    # 2. Angles (2)
    def angle(p1, p2): return np.degrees(np.arctan2(abs(p1[0]-p2[0]), abs(p1[1]-p2[1])))
    a_sh_hip_kn = angle(p['shoulder'], p['knee'])
    a_hip_kn_ank = angle(p['hip'], p['ankle'])
    
    return pd.Series(raw + [d_ear_sh, d_sh_hip, d_hip_kn, d_kn_ank, a_sh_hip_kn, a_hip_kn_ank])

X_16 = df.apply(get_engineered_features_16, axis=1)
y = df['label'].values

# 3. Split (64:16:20)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_16, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train_s)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_val_t = torch.FloatTensor(X_val_s)
y_val_t = torch.FloatTensor(y_val).view(-1, 1)
X_test_t = torch.FloatTensor(X_test_s)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# 4. Model (Attention MLP)
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

model = AttentionMLP(X_train_s.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 5. Training
best_loss = float('inf')
patience = 10
trigger = 0
history = {'train_acc': [], 'val_acc': []}

for epoch in range(100):
    model.train()
    correct_t = 0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = model(bx)
        loss = torch.neg(torch.mean(pos_weight * (by * torch.log(out + 1e-7)) + (1 - by) * torch.log(1 - out + 1e-7)))
        loss.backward()
        optimizer.step()
        correct_t += ((out > 0.5) == by).sum().item()
    
    train_acc = correct_t / len(y_train)
    
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)
        val_loss = torch.neg(torch.mean(pos_weight * (y_val_t * torch.log(val_out + 1e-7)) + (1 - y_val_t) * torch.log(1 - val_out + 1e-7))).item()
        val_acc = ((val_out > 0.5) == y_val_t).sum().item() / len(y_val_t)
    
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    if val_loss < best_loss:
        best_loss = val_loss
        trigger = 0
        torch.save(model.state_dict(), 'best_AttentionMLP_16.pt')
    else:
        trigger += 1
        if trigger >= patience: break

model.load_state_dict(torch.load('best_AttentionMLP_16.pt'))

# 6. Evaluation (Threshold 0.40)
model.eval()
with torch.no_grad():
    probs = model(X_test_t).squeeze().numpy()
    preds = (probs >= 0.40).astype(int)

res_new = {
    'Recall': recall_score(y_test, preds),
    'F1-Score': f1_score(y_test, preds),
    'AUC': roc_auc_score(y_test, probs),
    'Accuracy': accuracy_score(y_test, preds),
    'Precision': precision_score(y_test, preds),
    'Train/Val Diff': abs(history['train_acc'][-1] - history['val_acc'][-1])
}

# 7. Comparison with 18-feature baseline
baseline_18 = {
    'Recall': 0.9818, 'F1-Score': 0.9730, 'AUC': 0.9960, 'Accuracy': 0.9750, 'Precision': 0.9643, 'Train/Val Diff': 0.0457
}

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("피처 제거 전후 성능 비교표 (임계값 0.40)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
comp_rows = []
for m in baseline_18.keys():
    v_old = baseline_18[m]
    v_new = res_new[m]
    comp_rows.append({'지표': m, '18개(누수포함)': f"{v_old:.4f}", '16개(누수제거)': f"{v_new:.4f}", '변화량': f"{v_new - v_old:+.4f}"})
print(pd.DataFrame(comp_rows).to_string(index=False))
