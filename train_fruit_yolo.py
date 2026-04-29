"""
Fine-Tune YOLOv9 on Fruit & Vegetable Detection Dataset (36 Classes)
=====================================================================
⚠️  NOTE: YOLO training requires a GPU and is very slow on CPU.
    Training has been disabled by default for CPU compatibility.
    
    The system uses pretrained YOLO weights (yolov9c.pt) for inference,
    combined with MobileNetV2 for species identification (36 classes).

    To enable training (GPU required):
      1. Set ENABLE_TRAINING = True below
      2. Ensure dataset is prepared at: fruit_yolo_dataset/
      3. Run: python train_fruit_yolo.py

Dataset: fruit_yolo_dataset/
Classes: 36 (apple, banana, beetroot, bell pepper, ..., watermelon)

Output:
  - Best model: runs/fruit_detection/yolov9_fruits/weights/best.pt
  - Copied to:  yolov9_fruits_best.pt  (used by app.py automatically)
"""

import os
import sys
import shutil
import torch

# ⚠️  SET TO True ONLY IF YOU HAVE A GPU
ENABLE_TRAINING = False

# ── Paths ──
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(BASE_DIR, "fruit_yolo_dataset", "data.yaml")
BASE_MODEL   = os.path.join(BASE_DIR, "yolov9c.pt")
OUTPUT_DIR   = os.path.join(BASE_DIR, "runs", "fruit_detection")
BEST_DEST    = os.path.join(BASE_DIR, "yolov9_fruits_best.pt")

# ── Training Config ──
EPOCHS     = 50
IMAGE_SIZE = 640
PATIENCE   = 15
DEVICE     = "0" if torch.cuda.is_available() else "cpu"

def get_batch_size():
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb >= 8: return 16
        elif vram_gb >= 4: return 8
        else: return 4
    return 4

def verify_dataset():
    print("=" * 60)
    print("  CHECKING PREREQUISITES")
    print("=" * 60)
    errors = []
    if os.path.exists(DATASET_YAML):
        print(f"  ✅ data.yaml: {DATASET_YAML}")
    else:
        errors.append(f"  ❌ data.yaml NOT found: {DATASET_YAML}")
    for split in ("train", "val", "test"):
        img_dir = os.path.join(BASE_DIR, "fruit_yolo_dataset", split, "images")
        lbl_dir = os.path.join(BASE_DIR, "fruit_yolo_dataset", split, "labels")
        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            n_imgs = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            n_lbls = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
            print(f"  ✅ {split:5s}: {n_imgs} images, {n_lbls} labels")
        else:
            errors.append(f"  ❌ {split} split missing: {img_dir}")
    if os.path.exists(BASE_MODEL):
        size_mb = os.path.getsize(BASE_MODEL) / (1024 * 1024)
        print(f"  ✅ Base model: yolov9c.pt ({size_mb:.0f} MB)")
    else:
        print(f"  ⚠️  Base model not at {BASE_MODEL} — will auto-download")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  ✅ GPU: {gpu} ({vram:.1f} GB VRAM)")
    else:
        print("  ⚠️  No GPU — training on CPU is NOT recommended (extremely slow)")
        print("        Consider Google Colab for faster training.")
    if errors:
        for e in errors: print(e)
        return False
    print()
    return True

def train():
    if not ENABLE_TRAINING:
        print("=" * 60)
        print("  ⚠️  YOLO TRAINING IS DISABLED (CPU mode)")
        print("=" * 60)
        print("  Training is disabled because CPU cannot handle it efficiently.")
        print("  The system uses pretrained YOLO weights for inference instead.")
        print()
        print("  To enable training (GPU required):")
        print("    1. Set ENABLE_TRAINING = True in this file")
        print("    2. Ensure you have a CUDA-compatible GPU")
        print("    3. Run: python train_fruit_yolo.py")
        print()
        print("  Current detection pipeline:")
        print("    • YOLO (pretrained) → object localization")
        print("    • MobileNetV2 → species identification (36 classes)")
        print("    • ResNet18 CNN → quality classification (7 classes)")
        print("=" * 60)
        return None

    from ultralytics import YOLO
    if not verify_dataset():
        print("\n❌ Fix the issues above, then re-run this script.")
        sys.exit(1)

    batch = get_batch_size()
    print(f"{'=' * 60}")
    print(f"  STARTING YOLO TRAINING — 36 FRUIT/VEGETABLE CLASSES")
    print(f"{'=' * 60}")
    print(f"  Base model:     yolov9c.pt  (COCO pretrained)")
    print(f"  Dataset:        fruit_yolo_dataset/ (36 classes)")
    print(f"  Epochs:         {EPOCHS} (early stop: {PATIENCE})")
    print(f"  Batch size:     {batch}")
    print(f"  Image size:     {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Device:         {'GPU' if DEVICE != 'cpu' else 'CPU'}")
    print(f"{'=' * 60}\n")

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=DATASET_YAML, epochs=EPOCHS, imgsz=IMAGE_SIZE,
        batch=batch, patience=PATIENCE, name="yolov9_fruits",
        project=OUTPUT_DIR, exist_ok=True, device=DEVICE,
        mosaic=1.0, mixup=0.15, copy_paste=0.1, degrees=20,
        translate=0.15, scale=0.5, shear=5.0, fliplr=0.5,
        flipud=0.2, hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
        erasing=0.2, iou=0.5, conf=0.001, agnostic_nms=True,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        weight_decay=0.0005, warmup_epochs=3,
        pretrained=True, plots=True, save=True, verbose=True,
    )

    best_src = os.path.join(OUTPUT_DIR, "yolov9_fruits", "weights", "best.pt")
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    if os.path.exists(best_src):
        shutil.copy2(best_src, BEST_DEST)
        size_mb = os.path.getsize(BEST_DEST) / (1024 * 1024)
        print(f"  ✅ Best model saved: {BEST_DEST} ({size_mb:.0f} MB)")
        print(f"\n  Restart app.py to use the trained model.")
    else:
        print(f"  ⚠️  Could not find best.pt at: {best_src}")
    print(f"{'=' * 60}")
    return BEST_DEST

if __name__ == "__main__":
    train()
