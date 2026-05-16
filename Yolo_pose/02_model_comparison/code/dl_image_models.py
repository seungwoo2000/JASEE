# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 2단계 - 모델 비교 실험
# 설명: 이미지 기반 전이학습 모델 성능 비교
# 작성일: 2026.05.13
# ================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score, precision_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. Config
data_dir = r'D:\antigravity\semi2_contest\model_data\images'
results_dir = r'D:\antigravity\semi2_contest\model_results'
batch_size = 16
epochs = 30
lr = 0.0001

# 2. Split Logic (Same as STEP 1 & 2)
full_dataset = datasets.ImageFolder(data_dir)
# Map: GOOD->0, BAD->1 (ImageFolder: alphabetical BAD=0, GOOD=1)
target_map = {full_dataset.class_to_idx['GOOD']: 0, full_dataset.class_to_idx['BAD']: 1}

# Get indices for split
indices = np.arange(len(full_dataset))
labels = [full_dataset.targets[i] for i in indices]

train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
# Split train_val into train and val
train_val_labels = [full_dataset.targets[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.20, random_state=42, stratify=train_val_labels)

print(f"Data Split Status:")
print(f"- Train: {len(train_idx)}장 | Val: {len(val_idx)}장 | Test: {len(test_idx)}장")

# 3. Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CustomSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_map = {dataset.class_to_idx['GOOD']: 0, dataset.class_to_idx['BAD']: 1}
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform: x = self.transform(x)
        return x, self.target_map[y]
    def __len__(self): return len(self.indices)

train_loader = DataLoader(CustomSubset(full_dataset, train_idx, transform_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CustomSubset(full_dataset, val_idx, transform_test), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(CustomSubset(full_dataset, test_idx, transform_test), batch_size=batch_size, shuffle=False)

# 4. Model Training Logic
def train_model(name, model, train_loader, val_loader, epochs=30):
    # Weighted CrossEntropy
    targets = [full_dataset.targets[i] for i in train_idx]
    mapped_t = [target_map[t] for t in targets]
    counts = np.bincount(mapped_t)
    # weights = inverse of freq
    weights = torch.FloatTensor([len(mapped_t)/c for c in counts])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_loss = float('inf')
    patience = 5
    trigger = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_t = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct_t += out.argmax(1).eq(by).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = correct_t / len(train_idx)
        
        model.eval()
        val_loss = 0
        correct_v = 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item()
                correct_v += out.argmax(1).eq(by).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct_v / len(val_idx)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0:
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

def evaluate_model(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for bx, by in loader:
            out = model(bx)
            probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(by.cpu().numpy())
    return {
        'Recall': recall_score(labels, preds),
        'F1-Score': f1_score(labels, preds),
        'AUC': roc_auc_score(labels, probs),
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds)
    }

# 5. Models
def get_model(name):
    if name == 'EfficientNetB0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for param in m.parameters(): param.requires_grad = False
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.classifier[1].in_features, 2))
    elif name == 'ResNet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in m.parameters(): param.requires_grad = False
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, 2))
    elif name == 'MobileNetV3Small':
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        for param in m.parameters(): param.requires_grad = False
        m.classifier = nn.Sequential(nn.Linear(m.classifier[0].in_features, 1024), nn.Hardswish(), nn.Dropout(0.5), nn.Linear(1024, 2))
    return m

model_names = ['EfficientNetB0', 'ResNet50', 'MobileNetV3Small']
img_results = []
histories = {}

for name in model_names:
    print(f"\nTraining {name}...")
    model = get_model(name)
    hist = train_model(name, model, train_loader, val_loader, epochs=epochs)
    histories[name] = hist
    
    metrics = evaluate_model(model, test_loader)
    train_acc = hist['train_acc'][-1]
    val_acc = hist['val_acc'][-1]
    img_results.append({'Model': name, **metrics, 'Train/Val Diff': abs(train_acc - val_acc)})

# 6. Save
df_img = pd.DataFrame(img_results).sort_values(by='Recall', ascending=False)
df_img.to_csv(os.path.join(results_dir, 'dl_image_results.csv'), index=False)

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("STEP 3. 이미지 기반 딥러닝 모델 성능 결과 (Test Set 기준)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(df_img.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
df_melt = df_img.melt(id_vars='Model', var_name='Metric', value_name='Score')
sns.barplot(data=df_melt[df_melt['Metric'] != 'Train/Val Diff'], x='Metric', y='Score', hue='Model', ax=axes[0])
axes[0].set_ylim(0, 1.1)

for name, hist in histories.items():
    axes[1].plot(hist['train_loss'], label=f'{name} Train')
    axes[1].plot(hist['val_loss'], linestyle='--', label=f'{name} Val')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'dl_image_comparison.png'))
