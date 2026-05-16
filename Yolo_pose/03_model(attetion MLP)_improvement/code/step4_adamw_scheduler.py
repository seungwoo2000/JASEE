# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 3단계 - 모델 성능 개선
# 설명: AdamW 및 LRScheduler를 이용한 최종 모델 정제
# 작성일: 2026.05.13
# ================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score, precision_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. Load Data (16 features)
data_dir = r'D:\antigravity\semi2_contest\model_data'
results_dir = r'D:\antigravity\semi2_contest\model_results'

df_labels = pd.read_csv(os.path.join(data_dir, 'final_labels_confirmed_merged.csv'))
df_landmarks = pd.read_csv(os.path.join(data_dir, 'yolo_landmarks_clean_merged.csv'))
df = pd.merge(df_labels, df_landmarks, on='filename')
df['label'] = df['final_label'].map({'GOOD': 0, 'BAD': 1})

def get_engineered_features_16(row):
    side = row['sitting_direction']
    p = {
        'ear': (row['left_ear_x'], row['left_ear_y']), 'shoulder': (row['left_shoulder_x'], row['left_shoulder_y']),
        'hip': (row['left_hip_x'], row['left_hip_y']), 'knee': (row['left_knee_x'], row['left_knee_y']),
        'ankle': (row['left_ankle_x'], row['left_ankle_y'])
    } if side == 'LEFT' else {
        'ear': (row['right_ear_x'], row['right_ear_y']), 'shoulder': (row['right_shoulder_x'], row['right_shoulder_y']),
        'hip': (row['right_hip_x'], row['right_hip_y']), 'knee': (row['right_knee_x'], row['right_knee_y']),
        'ankle': (row['right_ankle_x'], row['right_ankle_y'])
    }
    raw = [p['ear'][0], p['ear'][1], p['shoulder'][0], p['shoulder'][1], p['hip'][0], p['hip'][1], p['knee'][0], p['knee'][1], p['ankle'][0], p['ankle'][1]]
    def dist(p1, p2): return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    def angle(p1, p2): return np.degrees(np.arctan2(abs(p1[0]-p2[0]), abs(p1[1]-p2[1])))
    return pd.Series(raw + [dist(p['ear'], p['shoulder']), dist(p['shoulder'], p['hip']), dist(p['hip'], p['knee']), dist(p['knee'], p['ankle']), 
                           angle(p['shoulder'], p['knee']), angle(p['hip'], p['ankle'])])

X_16 = df.apply(get_engineered_features_16, axis=1)
y = df['label'].values

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
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 2. Model (Dynamic structure from Optuna)
class DynamicAttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(DynamicAttentionMLP, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
            in_dim = h_dim
        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        attn_out, _ = self.attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        return self.net(attn_out.squeeze(1))

model = DynamicAttentionMLP(X_train_s.shape[1], [256, 128, 64], 0.42)

# 3. Optimizer & Scheduler (STEP 4 Update)
optimizer = optim.AdamW(model.parameters(), lr=0.0031, weight_decay=9.47e-05)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 4. Training
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': [], 'lr': [], 'train_acc': [], 'val_acc': []}
patience = 10
trigger = 0

print("Starting STEP 4 Refinement (AdamW + CosineAnnealingLR)...")
for epoch in range(100):
    model.train()
    train_loss = 0
    correct_t = 0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = model(bx)
        loss = torch.neg(torch.mean(pos_weight * (by * torch.log(out + 1e-7)) + (1 - by) * torch.log(1 - out + 1e-7)))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct_t += ((out > 0.5) == by).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = correct_t / len(y_train)
    
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)
        val_loss = torch.neg(torch.mean(pos_weight * (y_val_t * torch.log(val_out + 1e-7)) + (1 - y_val_t) * torch.log(1 - val_out + 1e-7))).item()
        val_acc = ((val_out > 0.5) == y_val_t).sum().item() / len(y_val_t)
        val_preds = (val_out >= 0.40).cpu().numpy().astype(int)
        val_recall = recall_score(y_val, val_preds)
        
    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['lr'].append(current_lr)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | lr: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Recall: {val_recall:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger = 0
        torch.save(model.state_dict(), 'best_AttentionMLP_Step4.pt')
    else:
        trigger += 1
        if trigger >= patience:
            print(f"Early Stopping at Epoch {epoch+1}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            break

# 5. Final Evaluation (Threshold 0.40)
model.load_state_dict(torch.load('best_AttentionMLP_Step4.pt'))
model.eval()
with torch.no_grad():
    probs = model(X_test_t).squeeze().numpy()
    preds = (probs >= 0.40).astype(int)

res_step4 = {
    'Recall': recall_score(y_test, preds),
    'F1-Score': f1_score(y_test, preds),
    'AUC': roc_auc_score(y_test, probs),
    'Accuracy': accuracy_score(y_test, preds),
    'Precision': precision_score(y_test, preds),
    'Train/Val Diff': abs(history['train_acc'][-1] - history['val_acc'][-1])
}

# 6. Comparison Output
step3_res = {'Recall': 0.9545, 'F1-Score': 0.9318, 'AUC': 0.9852, 'Accuracy': 0.9375, 'Precision': 0.9103, 'Train/Val Diff': 0.0384}

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("STEP 4 전후 성능 비교표")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
comp_rows = []
for m in step3_res.keys():
    v_old = step3_res[m]
    v_new = res_step4[m]
    comp_rows.append({'지표': m, 'STEP 3 결과': f"{v_old:.4f}", 'STEP 4 결과': f"{v_new:.4f}", '변화량': f"{v_new - v_old:+.4f}"})
print(pd.DataFrame(comp_rows).to_string(index=False))

# 7. Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_title('Training vs Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history['lr'], color='orange', label='Learning Rate')
axes[1].set_title('Learning Rate Schedule (CosineAnnealing)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('LR')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'step4_training_curve.png'))

# 8. Total History Table
history_rows = [
    {'단계': '기본 모델', 'Recall': 0.9273, 'F1': 0.9273, 'AUC': 0.9880, 'Acc': 0.9333, 'Diff': 0.0418},
    {'단계': '임계값 0.40', 'Recall': 0.9636, 'F1': 0.9422, 'AUC': 0.9871, 'Acc': 0.9458, 'Diff': 0.0418},
    {'단계': '피처 16개', 'Recall': 0.9182, 'F1': 0.9224, 'AUC': 0.9787, 'Acc': 0.9292, 'Diff': 0.0496},
    {'단계': 'Optuna 튜닝', 'Recall': 0.9545, 'F1': 0.9318, 'AUC': 0.9852, 'Acc': 0.9375, 'Diff': 0.0384},
    {'단계': 'STEP 4 (최종)', 'Recall': res_step4['Recall'], 'F1': res_step4['F1-Score'], 'AUC': res_step4['AUC'], 'Acc': res_step4['Accuracy'], 'Diff': res_step4['Train/Val Diff']}
]
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("전체 과정 최종 성능 비교표")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(pd.DataFrame(history_rows).to_string(index=False))
