# 최종 모델 사양
- **모델명**: Attention MLP
- **구조**: Input(16) → Self-Attention → 256 → 128 → 64 → 1
- **임계값**: 0.40
- **성능**: Recall 0.9545, AUC 0.9852, Diff 0.0384
- **로드 방법**:
```python
import torch
checkpoint = torch.load('final_attention_mlp.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```