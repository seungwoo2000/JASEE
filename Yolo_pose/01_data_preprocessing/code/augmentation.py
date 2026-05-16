# ================================================
# 프로젝트: 앉은 자세 분류 모델 개발
# 단계: 1단계 - 데이터 전처리
# 설명: 이미지 회전, 이동, 노이즈 추가를 통한 학습 데이터 확장
# 작성일: 2026.05.13
# ================================================
import albumentations as A
import cv2
import os
import glob

def augment_images(input_dir, output_dir, n_aug=2):
    print(f"--- Data Augmentation 시작: {input_dir} ---")
    
    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1),
    ])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    
    count = 0
    for img_path in images:
        image = cv2.imread(img_path)
        fname = os.path.basename(img_path)
        
        # Original save (optional, if moving to new dir)
        # cv2.imwrite(os.path.join(output_dir, fname), image)
        
        for i in range(n_aug):
            augmented = transform(image=image)['image']
            out_name = f"aug_{i}_{fname}"
            cv2.imwrite(os.path.join(output_dir, out_name), augmented)
            count += 1
            
    print(f"- 생성된 증강 이미지: {count}장")
    print(f"--- Data Augmentation 완료 ---\n")

# 실행 예시
# augment_images('raw_images/', 'augmented_images/')