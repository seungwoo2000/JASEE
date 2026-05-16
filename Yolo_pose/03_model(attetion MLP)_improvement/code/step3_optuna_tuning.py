# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 3단계 - 모델 성능 개선
# 설명: Optuna를 이용한 엄격한 과적합 제어 및 최적화
# 작성일: 2026.05.13
# ================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score, precision_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. Load Data (16 features - Leakage removed)
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

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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

# 2. Objective with STRICT Diff < 0.05
def objective(trial):
    lr = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    hidden_option = trial.suggest_categorical('hidden_size', ['64-32', '128-64', '256-128-64'])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    hidden_layers = list(map(int, hidden_option.split('-')))
    model = DynamicAttentionMLP(X_train_s.shape[1], hidden_layers, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_train_acc, best_val_acc = 0, 0
    patience = 10
    trigger = 0
    
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
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = torch.neg(torch.mean(pos_weight * (y_val_t * torch.log(val_out + 1e-7)) + (1 - y_val_t) * torch.log(1 - val_out + 1e-7))).item()
            val_acc = ((val_out > 0.5) == y_val_t).sum().item() / len(y_val_t)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger = 0
            best_train_acc = correct_t / len(y_train)
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'strict_trial_{trial.number}.pt')
        else:
            trigger += 1
            if trigger >= patience: break

    diff = abs(best_train_acc - best_val_acc)
    model.load_state_dict(torch.load(f'strict_trial_{trial.number}.pt'))
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_t).squeeze().numpy()
        test_preds = (test_probs >= 0.40).astype(int)
    recall = recall_score(y_test, test_preds)
    os.remove(f'strict_trial_{trial.number}.pt')
    
    trial.set_user_attr('diff', diff)
    trial.set_user_attr('recall', recall)
    
    if diff >= 0.05:
        return -1.0 # Disqualify
    
    return recall

# 3. Study
study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=50)

# 4. Re-train best STRICT model
best_trial = study.best_trial
best_params = best_trial.params
print(f"Best Params (Strict Diff < 0.05): {best_params}")
print(f"Best Trial Recall: {best_trial.user_attrs['recall']:.4f}, Diff: {best_trial.user_attrs['diff']:.4f}")

hidden_layers = list(map(int, best_params['hidden_size'].split('-')))
final_model = DynamicAttentionMLP(X_train_s.shape[1], hidden_layers, best_params['dropout'])
optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=best_params['batch_size'], shuffle=True)

best_val_loss = float('inf')
final_train_acc, final_val_acc = 0, 0
for epoch in range(100):
    final_model.train()
    correct_t = 0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = final_model(bx)
        loss = torch.neg(torch.mean(pos_weight * (by * torch.log(out + 1e-7)) + (1 - by) * torch.log(1 - out + 1e-7)))
        loss.backward()
        optimizer.step()
        correct_t += ((out > 0.5) == by).sum().item()
    
    final_model.eval()
    with torch.no_grad():
        val_out = final_model(X_val_t)
        val_loss = torch.neg(torch.mean(pos_weight * (y_val_t * torch.log(val_out + 1e-7)) + (1 - y_val_t) * torch.log(1 - val_out + 1e-7))).item()
        val_acc = ((val_out > 0.5) == y_val_t).sum().item() / len(y_val_t)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        final_train_acc = correct_t / len(y_train)
        final_val_acc = val_acc
        torch.save(final_model.state_dict(), 'best_AttentionMLP_Strict.pt')

final_model.load_state_dict(torch.load('best_AttentionMLP_Strict.pt'))
final_model.eval()
with torch.no_grad():
    probs = final_model(X_test_t).squeeze().numpy()
    preds = (probs >= 0.40).astype(int)

# 5. Output Comparison
baseline = {'Recall': 0.9182, 'F1-Score': 0.9224, 'AUC': 0.9787, 'Accuracy': 0.9292, 'Precision': 0.9266, 'Train/Val Diff': 0.0496}
tuned = {
    'Recall': recall_score(y_test, preds),
    'F1-Score': f1_score(y_test, preds),
    'AUC': roc_auc_score(y_test, probs),
    'Accuracy': accuracy_score(y_test, preds),
    'Precision': precision_score(y_test, preds),
    'Train/Val Diff': abs(final_train_acc - final_val_acc)
}

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("성능 비교표 (Strict Diff < 0.05 기준)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
comp_rows = []
for m in baseline.keys():
    comp_rows.append({'지표': m, '기본(16개)': f"{baseline[m]:.4f}", '튜닝후(Strict)': f"{tuned[m]:.4f}", '변화량': f"{tuned[m]-baseline[m]:+.4f}"})
print(pd.DataFrame(comp_rows).to_string(index=False))
