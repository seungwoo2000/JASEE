"""
Step 1: train / val / test 분할 (8:1:1)
- new_data_set + old_data_set 합쳐서 split
- 결과: images_data/dataset/train, val, test
"""

import os
import shutil
import random
from pathlib import Path

# ───────────────────────────────────────────
# 경로 설정
# ───────────────────────────────────────────
BASE_DIR    = Path(r"E:\python\Jasee\Yolo_env\images_data")
SOURCES     = [BASE_DIR / "new_data_set", BASE_DIR / "old_data_set"]
OUTPUT_DIR  = BASE_DIR / "dataset"

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED   = 42

# ───────────────────────────────────────────
# 출력 폴더 생성
# ───────────────────────────────────────────
for split in SPLITS:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────
# 전체 이미지-라벨 쌍 수집
# ───────────────────────────────────────────
pairs = []  # (image_path, label_path)

for src in SOURCES:
    img_dir = src / "images"
    lbl_dir = src / "labels"

    for img_file in img_dir.glob("*.jpg"):
        lbl_file = lbl_dir / (img_file.stem + ".txt")
        if lbl_file.exists():
            pairs.append((img_file, lbl_file))
        else:
            print(f"[경고] 라벨 없음: {img_file.name}")

print(f"\n총 수집된 이미지-라벨 쌍: {len(pairs)}개")

# ───────────────────────────────────────────
# 셔플 후 분할
# ───────────────────────────────────────────
random.seed(SEED)
random.shuffle(pairs)

n = len(pairs)
n_train = int(n * SPLITS["train"])
n_val   = int(n * SPLITS["val"])

split_pairs = {
    "train": pairs[:n_train],
    "val":   pairs[n_train:n_train + n_val],
    "test":  pairs[n_train + n_val:],
}

# ───────────────────────────────────────────
# 파일 복사
# ───────────────────────────────────────────
for split, file_pairs in split_pairs.items():
    for img_path, lbl_path in file_pairs:
        shutil.copy2(img_path, OUTPUT_DIR / split / "images" / img_path.name)
        shutil.copy2(lbl_path, OUTPUT_DIR / split / "labels" / lbl_path.name)
    print(f"{split:5s}: {len(file_pairs)}장 복사 완료")

# ───────────────────────────────────────────
# data.yaml 생성
# ───────────────────────────────────────────
yaml_content = f"""train: {(OUTPUT_DIR / 'train' / 'images').as_posix()}
val:   {(OUTPUT_DIR / 'val'   / 'images').as_posix()}
test:  {(OUTPUT_DIR / 'test'  / 'images').as_posix()}

nc: 4
names: ['chair_back', 'chair_seat', 'desk_surface', 'monitor']
"""

yaml_path = OUTPUT_DIR / "data.yaml"
yaml_path.write_text(yaml_content, encoding="utf-8")
print(f"\ndata.yaml 생성 완료: {yaml_path}")
print("\n✅ Step 1 완료! dataset/ 폴더를 확인하세요.")
