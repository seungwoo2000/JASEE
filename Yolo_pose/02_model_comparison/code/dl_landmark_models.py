# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 2단계 - 모델 비교 실험
# 설명: 좌표 기반 딥러닝 모델(MLP, Attention, CNN) 비교
# 작성일: 2026.05.13
# ================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# 2. Split (64:16:20)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_raw, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val)

print(f"Data Split Status:")
print(f"- Train: {len(X_train)}장 | Val: {len(X_val)}장 | Test: {len(X_test)}장")

# 3. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_val_t = torch.FloatTensor(X_val_scaled)
y_val_t = torch.FloatTensor(y_val).view(-1, 1)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 4. Models
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

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

class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super(CNN1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(nn.Linear(128 * 2, 1), nn.Dropout(0.5), nn.Sigmoid())
    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 5. Training Logic
def weighted_bce(output, target, pos_weight):
    # loss = - [pos_weight * y * log(p) + (1-y) * log(1-p)]
    loss = pos_weight * (target * torch.log(output + 1e-7)) + \
           (1 - target) * torch.log(1 - output + 1e-7)
    return torch.neg(torch.mean(loss))

def train_model(name, model, train_loader, X_val, y_val, pos_weight, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_loss = float('inf')
    trigger = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(100):
        model.train()
        train_loss = 0
        correct_t = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = weighted_bce(out, by, pos_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct_t += ((out > 0.5) == by).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = correct_t / len(y_train)
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = weighted_bce(val_out, y_val, pos_weight).item()
            val_acc = ((val_out > 0.5) == y_val).sum().item() / len(y_val)
            
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"[{name}] Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_loss:
            best_loss = val_loss
            trigger = 0
            torch.save(model.state_dict(), f'best_{name}.pt')
        else:
            trigger += 1
            if trigger >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(f'best_{name}.pt'))
    return history

# 6. Run
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
input_dim = X_train_t.shape[1]

model_list = {'MLP': MLP(input_dim), 'Attention MLP': AttentionMLP(input_dim), '1D-CNN': CNN1D(input_dim)}
dl_results = []
histories = {}

for name, model in model_list.items():
    print(f"\nTraining {name}...")
    hist = train_model(name, model, train_loader, X_val_t, y_val_t, pos_weight)
    histories[name] = hist
    
    # Eval on TEST SET
    model.eval()
    with torch.no_grad():
        preds_proba = model(X_test_t).numpy()
        preds = (preds_proba > 0.5).astype(int)
    
    train_acc = hist['train_acc'][-1]
    val_acc = hist['val_acc'][-1]
    
    res = {
        'Model': name,
        'Recall': recall_score(y_test, preds),
        'F1-Score': f1_score(y_test, preds),
        'AUC': roc_auc_score(y_test, preds_proba),
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Train/Val Diff': abs(train_acc - val_acc)
    }
    dl_results.append(res)

df_dl = pd.DataFrame(dl_results).sort_values(by='Recall', ascending=False)

# 7. Output
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("STEP 2. 딥러닝(좌표) 모델 성능 결과 (Test Set 기준)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
cols = ['Recall', 'F1-Score', 'AUC', 'Accuracy', 'Precision', 'Train/Val Diff']
df_print = df_dl.copy()
for col in cols:
    is_max = df_dl[col] == df_dl[col].max()
    df_print[col] = ['★ ' + f'{v:.4f}' if m else f'{v:.4f}' for v, m in zip(df_dl[col], is_max)]
print(df_print.to_string(index=False))

df_dl.to_csv(os.path.join(results_dir, 'dl_landmark_results.csv'), index=False)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
df_melt = df_dl.melt(id_vars='Model', var_name='Metric', value_name='Score')
sns.barplot(data=df_melt[df_melt['Metric'] != 'Train/Val Diff'], x='Metric', y='Score', hue='Model', ax=axes[0])
axes[0].set_title('DL Landmark Performance (Test Set)')
axes[0].set_ylim(0, 1.1)

for name, hist in histories.items():
    axes[1].plot(hist['train_loss'], label=f'{name} Train')
    axes[1].plot(hist['val_loss'], linestyle='--', label=f'{name} Val')
axes[1].set_title('Training Curves (Loss)')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'dl_landmark_comparison.png'))
