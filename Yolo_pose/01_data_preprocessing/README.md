# 데이터 전처리
- **목적**: 원천 데이터로부터 신뢰할 수 있는 랜드마크 좌표 및 라벨 추출
- **주요 기능**: 데이터 품질 확인, 데이터 증강, YOLOv8 Pose를 이용한 랜드마크 추출, CSV 병합 및 라벨링
- **실행 순서**: quality_check → augmentation → yolo_landmark_extract → labeling
- **입출력**:
  - 입력: raw images
  - 출력: [output/yolo_landmarks_clean.csv](file:///d:/antigravity/semi2_contest/final_모델링 및 데이터 전처리 코드/01_data_preprocessing/output/yolo_landmarks_clean.csv), [output/final_labels_confirmed.csv](file:///d:/antigravity/semi2_contest/final_모델링 및 데이터 전처리 코드/01_data_preprocessing/output/final_labels_confirmed.csv)

## 🔍 Pose Landmark Viewer
추출된 관절 좌표가 실제 이미지에서 어떻게 찍혔는지 확인할 수 있는 시각화 도구입니다.

1. **실행 방법**:
   ```bash
   cd output
   python viewer_server.py
   ```
2. **확인**: 브라우저에서 `http://localhost:8000/viewer.html`에 접속됩니다.
3. **기능**:
   - 모든 이미지(1,199장)의 관절 포인트 및 스켈레톤 시각화
   - 이전/다음 버튼 및 슬라이더를 이용한 탐색
   - 파일명 검색 및 상세 좌표 정보 확인

## 실행 방법 (코드)
```bash
python quality_check.py
python augmentation.py
python yolo_landmark_extract.py
python labeling.py
```