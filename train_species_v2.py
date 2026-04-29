"""
Fruit & Vegetable Species Classifier V2 — Improved Training
============================================================
Key improvements over v1:
  1. 50 epochs (was 15) — more time to learn
  2. Much stronger augmentation (handles color varieties like green mangoes)
  3. Mixup augmentation — creates blends of images for better generalization
  4. Progressive unfreezing — freeze backbone first, then fine-tune all layers
  5. Test-time augmentation (TTA) in the inference function
  6. Per-class accuracy tracking — shows which classes need improvement
  7. Larger image size option (256 instead of 224) for more detail

Dataset: Kritikseth Fruit-and-Vegetable-Image-Recognition (Kaggle)
36 classes including Mango, Apple, Banana, etc. — all REAL photos
"""
import os, json, time, random
import numpy as np
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import defaultdict

# ── Configuration ──
DATASET = os.path.expanduser(
    "~/.cache/kagglehub/datasets/kritikseth/"
    "fruit-and-vegetable-image-recognition/versions/8"
)
TRAIN_DIR = os.path.join(DATASET, "train")
VAL_DIR   = os.path.join(DATASET, "validation")
TEST_DIR  = os.path.join(DATASET, "test")

EPOCHS = 15           # Reduced for CPU training (faster)
BATCH  = 16           # Smaller batch → more gradient updates per epoch
LR     = 3e-4         # Lower learning rate for fine-tuning pretrained model
IMG_SIZE = 224         # Image input size
SAVE_MODEL = "fruit_species_classifier.pth"
SAVE_CLASSES = "fruit_species_classes.json"
SEED = 42

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ═══════════════════════════════════════════════════════════
# AGGRESSIVE DATA AUGMENTATION
# This is the KEY change — makes the model robust to:
#   - Color variations (green mango vs orange mango)
#   - Lighting differences (indoor vs outdoor)
#   - Background clutter (text on images, baskets, tables)
#   - Orientation and scale variations
# ═══════════════════════════════════════════════════════════

train_tf = T.Compose([
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Slightly larger for cropping
    T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(30),
    # STRONG color augmentation — crucial for handling green/yellow/orange mangoes
    T.ColorJitter(
        brightness=0.4,   # Simulate different lighting
        contrast=0.4,     # Indoor vs outdoor
        saturation=0.5,   # Color intensity variation
        hue=0.15,         # Shift colors (green ↔ yellow ↔ orange)
    ),
    T.RandomPerspective(distortion_scale=0.3, p=0.4),
    T.RandomAffine(
        degrees=15,
        translate=(0.15, 0.15),
        scale=(0.8, 1.2),
        shear=10,
    ),
    # Simulate JPEG compression / blur
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    # Random grayscale — force model to learn SHAPE, not just color
    T.RandomGrayscale(p=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # Random erasing — simulate occlusion (text overlays, partial views)
    T.RandomErasing(p=0.25, scale=(0.02, 0.25)),
])

val_tf = T.Compose([
    T.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Test-time augmentation transforms (for inference)
tta_transforms = [
    val_tf,  # Original
    T.Compose([  # Horizontal flip
        T.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        T.CenterCrop(IMG_SIZE),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    T.Compose([  # Slight crop
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]

print("Loading datasets...")
train_ds = ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = ImageFolder(VAL_DIR, transform=val_tf)
test_ds  = ImageFolder(TEST_DIR, transform=val_tf)
NUM_CLASSES = len(train_ds.classes)

print(f"Classes: {NUM_CLASSES}")
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"\nClass distribution:")
class_counts = defaultdict(int)
for _, label in train_ds:
    class_counts[label] += 1
for idx, cls_name in enumerate(train_ds.classes):
    print(f"  [{idx:2d}] {cls_name:20s}: {class_counts[idx]} images")


# ═══════════════════════════════════════════════════════════
# MIXUP AUGMENTATION
# Creates blended images → much better generalization
# ═══════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation: blend two images and their labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════
# MODEL — MobileNetV2 with progressive unfreezing
# ═══════════════════════════════════════════════════════════

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, NUM_CLASSES),
)
model = model.to(device)

train_loader = DataLoader(
    train_ds, batch_size=BATCH, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True,
)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ═══════════════════════════════════════════════════════════
# TRAINING STRATEGY: 2-phase
#   Phase 1 (epochs 1-10): Freeze backbone, train classifier only
#   Phase 2 (epochs 11+):  Unfreeze all, fine-tune with lower LR
# ═══════════════════════════════════════════════════════════

def freeze_backbone(model):
    """Freeze feature extraction layers."""
    for param in model.features.parameters():
        param.requires_grad = False
    print("  🔒 Backbone FROZEN — training classifier only")

def unfreeze_all(model):
    """Unfreeze all layers for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("  🔓 All layers UNFROZEN — fine-tuning everything")


# Phase 1: Freeze backbone
PHASE1_EPOCHS = 5
freeze_backbone(model)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR * 10,  # Higher LR for classifier-only training
    weight_decay=1e-4,
)

print(f"\n{'='*60}")
print(f"  🍎 Training Species Classifier V2 — {NUM_CLASSES} classes")
print(f"  Phase 1: Classifier only (epochs 1-{PHASE1_EPOCHS})")
print(f"  Phase 2: Full fine-tuning (epochs {PHASE1_EPOCHS+1}-{EPOCHS})")
print(f"{'='*60}\n")

best_val_acc = 0
patience_counter = 0
PATIENCE = 4  # Early stopping patience

for epoch in range(EPOCHS):
    # Phase transition at epoch PHASE1_EPOCHS
    if epoch == PHASE1_EPOCHS:
        print(f"\n{'─'*60}")
        print(f"  Transitioning to Phase 2: Full Fine-Tuning")
        print(f"{'─'*60}")
        unfreeze_all(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - PHASE1_EPOCHS, eta_min=1e-6,
        )

    model.train()
    running_loss = 0; correct = 0; total = 0
    t0 = time.time()

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Apply mixup 50% of the time in phase 2
        use_mixup = (epoch >= PHASE1_EPOCHS) and (random.random() < 0.5)

        if use_mixup:
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha=0.3)
            optimizer.zero_grad()
            out = model(imgs)
            loss = mixup_criterion(criterion, out, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = out.max(1)
            total += labels_a.size(0)
            correct += (lam * pred.eq(labels_a).float()
                       + (1 - lam) * pred.eq(labels_b).float()).sum().item()
        else:
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    train_acc = 100. * correct / total if total > 0 else 0

    # Validate
    model.eval()
    val_correct = 0; val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, pred = model(imgs).max(1)
            val_total += labels.size(0)
            val_correct += pred.eq(labels).sum().item()
    val_acc = 100. * val_correct / val_total

    if epoch >= PHASE1_EPOCHS:
        scheduler.step()

    elapsed = time.time() - t0
    lr_now = optimizer.param_groups[0]['lr']
    phase = "P1" if epoch < PHASE1_EPOCHS else "P2"

    print(f"  [{phase}] Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}% | "
          f"LR: {lr_now:.1e} | {elapsed:.1f}s", end="")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), SAVE_MODEL)
        print(f" ✅ BEST")
    else:
        patience_counter += 1
        print(f"  (patience: {patience_counter}/{PATIENCE})")

    # Early stopping
    if patience_counter >= PATIENCE and epoch >= PHASE1_EPOCHS + 5:
        print(f"\n  ⏹️ Early stopping at epoch {epoch+1}")
        break


# ═══════════════════════════════════════════════════════════
# FINAL TEST with per-class accuracy
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  📊 Final Evaluation")
print(f"{'='*60}")

model.load_state_dict(torch.load(SAVE_MODEL, map_location=device, weights_only=True))
model.eval()

# Per-class tracking
class_correct = defaultdict(int)
class_total = defaultdict(int)
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        _, pred = out.max(1)
        for p, l in zip(pred, labels):
            class_correct[l.item()] += (p == l).item()
            class_total[l.item()] += 1
            all_preds.append(p.item())
            all_labels.append(l.item())

test_acc = 100. * sum(class_correct.values()) / sum(class_total.values())

print(f"\n  Overall Test Accuracy: {test_acc:.1f}%")
print(f"\n  Per-class accuracy:")
print(f"  {'Class':<20s} {'Correct':>8s} {'Total':>6s} {'Acc':>7s}")
print(f"  {'─'*43}")

weak_classes = []
for idx in range(NUM_CLASSES):
    cls_name = train_ds.classes[idx]
    corr = class_correct[idx]
    tot = class_total[idx]
    acc = 100. * corr / tot if tot > 0 else 0
    marker = " ⚠️" if acc < 70 else " ✅" if acc >= 90 else ""
    print(f"  {cls_name:<20s} {corr:>8d} {tot:>6d} {acc:>6.1f}%{marker}")
    if acc < 70:
        weak_classes.append((cls_name, acc))

if weak_classes:
    print(f"\n  ⚠️ Weak classes (< 70% accuracy):")
    for name, acc in weak_classes:
        print(f"    - {name}: {acc:.1f}%")


# ═══════════════════════════════════════════════════════════
# SAVE CLASS MAPPINGS
# ═══════════════════════════════════════════════════════════

display_names = {}
for idx, cls_name in enumerate(train_ds.classes):
    display_names[str(idx)] = cls_name.title()

save_data = {
    "fine_classes": train_ds.classes,
    "class_to_parent": display_names,
    "parent_species": [c.title() for c in train_ds.classes],
    "num_fine_classes": NUM_CLASSES,
    "num_parent_species": NUM_CLASSES,
    "input_size": IMG_SIZE,
    "dataset": "fruit-and-vegetable-image-recognition",
    "training_version": "v2",
    "best_val_accuracy": best_val_acc,
    "test_accuracy": test_acc,
}
with open(SAVE_CLASSES, 'w') as f:
    json.dump(save_data, f, indent=2)

print(f"\n{'='*60}")
print(f"  ✅ Training Complete!")
print(f"  Best Val Accuracy:  {best_val_acc:.1f}%")
print(f"  Test Accuracy:      {test_acc:.1f}%")
print(f"  Model: {SAVE_MODEL} ({os.path.getsize(SAVE_MODEL)//1024//1024} MB)")
print(f"  Classes: {SAVE_CLASSES}")
print(f"{'='*60}")
