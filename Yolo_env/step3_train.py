"""
Step 3: YOLOv8 모델 학습
- 필요 패키지: pip install ultralytics
- 학습 데이터: augmented/train (~1,698장) + dataset/val + dataset/test
"""

import torch
import time
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path


def make_augmented_yaml(base_dir: Path):
    """augmented/data.yaml 생성 (없으면)"""
    aug_yaml = base_dir / "augmented" / "data.yaml"
    if not aug_yaml.exists():
        content = f"""train: {(base_dir / 'augmented' / 'train' / 'images').as_posix()}
val:   {(base_dir / 'dataset'   / 'val'   / 'images').as_posix()}
test:  {(base_dir / 'dataset'   / 'test'  / 'images').as_posix()}

nc: 4
names: ['chair_back', 'chair_seat', 'desk_surface', 'monitor']
"""
        aug_yaml.write_text(content, encoding="utf-8")
        print(f"  augmented/data.yaml 생성 완료")
    return aug_yaml


def print_eval_table(results, elapsed, start_dt, end_dt):
    """모델 평가표 터미널 출력"""
    metrics = results.results_dict
    mp      = metrics.get("metrics/precision(B)", 0)
    mr      = metrics.get("metrics/recall(B)", 0)
    map50   = metrics.get("metrics/mAP50(B)", 0)
    map5095 = metrics.get("metrics/mAP50-95(B)", 0)
    box     = metrics.get("train/box_loss", 0)
    cls     = metrics.get("train/cls_loss", 0)

    def grade(key, val):
        standards = {
            "mAP50":   (0.85, 0.70),
            "mAP5095": (0.60, 0.45),
            "prec":    (0.85, 0.70),
            "recall":  (0.80, 0.65),
        }
        hi, lo = standards[key]
        if val >= hi:   return "✅ 우수"
        elif val >= lo: return "⚠️  보통"
        else:           return "❌ 미흡"

    hours   = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    W = 54
    print("\n" + "=" * W)
    print(f"{'📊 모델 평가표 (YOLOv8n - 작업환경 인식)':^{W}}")
    print("=" * W)
    print(f"  {'지표':<18} {'값':>8}   {'판정'}")
    print("-" * W)
    print(f"  {'Precision':<18} {mp:>7.4f}   {grade('prec',    mp)}")
    print(f"  {'Recall':<18} {mr:>7.4f}   {grade('recall',  mr)}")
    print(f"  {'mAP@50':<18} {map50:>7.4f}   {grade('mAP50',  map50)}")
    print(f"  {'mAP@50-95':<18} {map5095:>7.4f}   {grade('mAP5095',map5095)}")
    print("-" * W)
    print(f"  {'Box Loss':<18} {box:>7.4f}")
    print(f"  {'Class Loss':<18} {cls:>7.4f}")
    print("=" * W)
    print(f"  {'학습 시작':<18} {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'학습 종료':<18} {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'총 소요시간':<17} {hours}시간 {minutes}분 {seconds}초")
    print("=" * W)
    print()
    print("  📌 하이퍼파라미터 튜닝 기준")
    print(f"  {'지표':<18} {'튜닝 불필요':>10}   {'튜닝 권장'}")
    print("-" * W)
    print(f"  {'mAP@50':<18} {'≥ 0.85':>10}   {'< 0.70'}")
    print(f"  {'mAP@50-95':<18} {'≥ 0.60':>10}   {'< 0.45'}")
    print(f"  {'Precision':<18} {'≥ 0.85':>10}   {'< 0.70'}")
    print(f"  {'Recall':<18} {'≥ 0.80':>10}   {'< 0.65'}")
    print("=" * W)


def main():
    # ───────────────────────────────────────────
    # GPU / CPU 확인
    # ───────────────────────────────────────────
    if torch.cuda.is_available():
        DEVICE = "0"
        print("=" * 54)
        print(f"✅ GPU 감지됨: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("   GPU로 학습합니다.")
        print("=" * 54)
    else:
        DEVICE = "cpu"
        print("=" * 54)
        print("⚠️  GPU 없음 → CPU로 학습합니다.")
        print("   학습 시간이 오래 걸릴 수 있어요.")
        print("=" * 54)

    # ───────────────────────────────────────────
    # 경로 설정
    # ───────────────────────────────────────────
    BASE_DIR  = Path(r"E:\python\Jasee\Yolo_env\images_data")

    # augmented/data.yaml 자동 생성 후 사용
    DATA_YAML = make_augmented_yaml(BASE_DIR)

    # ───────────────────────────────────────────
    # 학습 설정
    # ───────────────────────────────────────────
    MODEL    = "yolov8n.pt"
    EPOCHS   = 100
    IMG_SIZE = 640
    BATCH    = 16
    PROJECT  = str(BASE_DIR / "runs")
    NAME     = "posture_v1"

    # ───────────────────────────────────────────
    # 데이터 요약 출력
    # ───────────────────────────────────────────
    train_dir = BASE_DIR / "augmented" / "train" / "images"
    val_dir   = BASE_DIR / "dataset"   / "val"   / "images"
    test_dir  = BASE_DIR / "dataset"   / "test"  / "images"

    train_cnt = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
    val_cnt   = len(list(val_dir.glob("*.jpg")))   if val_dir.exists()   else 0
    test_cnt  = len(list(test_dir.glob("*.jpg")))  if test_dir.exists()  else 0

    print(f"\n  📁 학습 데이터 현황")
    print(f"  {'train':<8}: {train_cnt}장  (augmented)")
    print(f"  {'val':<8}: {val_cnt}장")
    print(f"  {'test':<8}: {test_cnt}장")
    print(f"  {'합계':<8}: {train_cnt + val_cnt + test_cnt}장\n")

    # ───────────────────────────────────────────
    # 시작 시간
    # ───────────────────────────────────────────
    start_dt = datetime.now()
    start_ts = time.time()
    print(f"🕐 학습 시작: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ───────────────────────────────────────────
    # 학습 실행
    # ───────────────────────────────────────────
    model   = YOLO(MODEL)
    results = model.train(
        data       = str(DATA_YAML),
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH,
        device     = DEVICE,
        project    = PROJECT,
        name       = NAME,
        patience   = 20,
        save       = True,
        plots      = True,
        verbose    = True,
        workers    = 0,
    )

    # ───────────────────────────────────────────
    # 종료 시간 & 평가표 출력
    # ───────────────────────────────────────────
    end_dt  = datetime.now()
    elapsed = time.time() - start_ts

    print_eval_table(results, elapsed, start_dt, end_dt)

    print(f"  결과 저장: {BASE_DIR / 'runs' / NAME}")
    print(f"  최적 모델: {BASE_DIR / 'runs' / NAME / 'weights' / 'best.pt'}\n")


if __name__ == '__main__':
    main()