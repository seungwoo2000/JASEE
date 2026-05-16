"""
Step 2: 데이터 증강 (train만)
- dataset/train 을 입력으로 받아 증강
- 결과: images_data/augmented/train/images, labels
- 원본도 augmented에 포함됨 (원본 + 증강본)
- 필요 패키지: pip install albumentations opencv-python
"""

import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path
import albumentations as A

# ───────────────────────────────────────────
# 경로 설정
# ───────────────────────────────────────────
BASE_DIR     = Path(r"E:\python\Jasee\Yolo_env\images_data")
TRAIN_IMG    = BASE_DIR / "dataset" / "train" / "images"
TRAIN_LBL    = BASE_DIR / "dataset" / "train" / "labels"
OUT_IMG      = BASE_DIR / "augmented" / "train" / "images"
OUT_LBL      = BASE_DIR / "augmented" / "train" / "labels"

AUG_PER_IMAGE = 2   # 이미지 1장당 증강 횟수 (원본 포함 총 3배)
SEED          = 42
random.seed(SEED)

# ───────────────────────────────────────────
# 출력 폴더 생성
# ───────────────────────────────────────────
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────
# 증강 파이프라인 정의
# ───────────────────────────────────────────
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(var_limit=(5, 20), p=0.3),
    A.Rotate(limit=10, p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.3),
], bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_visibility=0.3
))

# ───────────────────────────────────────────
# YOLO 라벨 읽기/쓰기 함수
# ───────────────────────────────────────────
def read_yolo_label(lbl_path):
    """라벨 파일 → (class_ids, bboxes) 반환"""
    class_ids, bboxes = [], []
    with open(lbl_path, "r") as f:
        for line in f.read().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                class_ids.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return class_ids, bboxes

def write_yolo_label(lbl_path, class_ids, bboxes):
    """(class_ids, bboxes) → 라벨 파일 저장"""
    with open(lbl_path, "w") as f:
        for cid, bbox in zip(class_ids, bboxes):
            f.write(f"{cid} {' '.join(f'{v:.6f}' for v in bbox)}\n")

# ───────────────────────────────────────────
# 원본 복사
# ───────────────────────────────────────────
img_files = sorted(TRAIN_IMG.glob("*.jpg"))
print(f"train 이미지 수: {len(img_files)}장")
print("원본 복사 중...")

for img_path in img_files:
    lbl_path = TRAIN_LBL / (img_path.stem + ".txt")
    shutil.copy2(img_path, OUT_IMG / img_path.name)
    if lbl_path.exists():
        shutil.copy2(lbl_path, OUT_LBL / lbl_path.name)

# ───────────────────────────────────────────
# 증강 실행
# ───────────────────────────────────────────
print(f"증강 시작 (이미지당 {AUG_PER_IMAGE}회)...")
aug_count = 0

for img_path in img_files:
    lbl_path = TRAIN_LBL / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue

    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    class_ids, bboxes = read_yolo_label(lbl_path)

    if not bboxes:
        continue

    for i in range(AUG_PER_IMAGE):
        try:
            result = transform(image=image, bboxes=bboxes, class_labels=class_ids)
        except Exception as e:
            print(f"[경고] 증강 실패 {img_path.name}: {e}")
            continue

        if not result["bboxes"]:
            continue

        aug_name = f"{img_path.stem}_aug{i+1}"
        out_img  = OUT_IMG / f"{aug_name}.jpg"
        out_lbl  = OUT_LBL / f"{aug_name}.txt"

        aug_img_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_img), aug_img_bgr)
        write_yolo_label(out_lbl, result["class_labels"], result["bboxes"])
        aug_count += 1

print(f"\n원본: {len(img_files)}장")
print(f"증강: {aug_count}장")
print(f"총합: {len(img_files) + aug_count}장")
print(f"\n✅ Step 2 완료! augmented/train/ 폴더를 확인하세요.")
