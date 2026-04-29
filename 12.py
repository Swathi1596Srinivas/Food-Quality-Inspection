
import os, cv2, numpy as np, torch, time
import torch.nn as nn, torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from ultralytics import YOLO
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, precision_score,
                             recall_score, f1_score, accuracy_score)
from collections import deque

#  CONFIG
DATASET_PATH = r"C:\Users\pujal\Desktop\Final year project\BATCH 16 code"
CNN_PATH, LSTM_PATH = "cnn_classifier.pth", "lstm_trend.pth"
EPOCHS, BATCH, LR, SEQ_LEN = 40, 32, 0.0005, 10
PATIENCE = 7  
CLASS_NAMES = ['Fresh', 'Ripe', 'Unripe', 'Old', 'Dried', 'Damaged', 'Rotten']
NUM_CLASSES = 7
Q_SCORES = {'fresh':6, 'ripe':5, 'unripe':4, 'old':3, 'dried':2, 'damaged':1, 'rotten':0}
Q_COLORS = {
    'fresh':   (0,255,0),     # Green
    'ripe':    (0,200,0),     # Dark Green
    'unripe':  (0,255,255),   # Yellow
    'old':     (0,165,255),   # Orange
    'dried':   (128,128,128), # Gray
    'damaged': (0,0,255),     # Red
    'rotten':  (0,0,200),     # Dark Red
}

TREND_LABELS = ['DEGRADING ⬇️', 'STABLE ➡️', 'IMPROVING ⬆️']
NORM_M, NORM_S = np.float32([0.485,0.456,0.406]), np.float32([0.229,0.224,0.225])
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")
print(f"Dataset found at {DATASET_PATH} ✅")

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])
test_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

dataset = ImageFolder(DATASET_PATH, transform=train_tf)
nc = len(dataset.classes)  
print(f"\nClasses ({nc}):")
for i, c in enumerate(dataset.classes):
    score = Q_SCORES.get(c.lower(), '?')
    print(f"  [{i}] {c:<12s}  Score: {score}")

tr = int(0.7*len(dataset)); va = int(0.15*len(dataset)); te = len(dataset)-tr-va
train_d, val_d, test_d = random_split(dataset, [tr, va, te])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin = device.type == "cuda"
NW = 2 if os.name != 'nt' else 0

from collections import Counter
train_labels: list[int] = [dataset.targets[i] for i in train_d.indices]
label_counts: Counter[int] = Counter(train_labels)  # type: ignore[arg-type]
print(f"\nTrain class distribution:")
for idx in range(nc):
    print(f"  [{idx}] {dataset.classes[idx]:<12s}: {label_counts.get(idx, 0)} images")

class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
sample_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

total_samples = sum(label_counts.values())
loss_weights = torch.tensor(
    [total_samples / (nc * label_counts.get(i, 1)) for i in range(nc)], dtype=torch.float32).to(device)
print(f"Loss weights: {[f'{w:.2f}' for w in loss_weights.tolist()]}")

train_dl = DataLoader(train_d, BATCH, sampler=sampler, num_workers=NW, pin_memory=pin)
val_dl   = DataLoader(val_d, BATCH, num_workers=NW, pin_memory=pin)
test_dl  = DataLoader(test_d, BATCH, num_workers=NW, pin_memory=pin)
print(f"Split → Train:{tr} Val:{va} Test:{te} | Device:{device}")
print(f"Total Images: {len(dataset)} | Classes: {nc}")

class FoodCNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        children = list(backbone.children())  # type: ignore[arg-type]
        self.feat = nn.Sequential(*children[:-1])  # Output: 2048-dim
        self.fdim = 2048
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, nc)  # nc=7
        )
        self._cf = None
        for name, param in self.feat.named_parameters():
            if not any(k in name for k in ('layer3', 'layer4')):
                param.requires_grad = False

    def forward(self, x):
        f = self.feat(x); self._cf = f; return self.cls(f)

    def get_features(self):
        f = self._cf; self._cf = None; return f.view(f.size(0), -1) if f is not None else None

class TrendLSTM(nn.Module):
    def __init__(self, idim, hd=128, nc=3):
        super().__init__()
        self.lstm = nn.LSTM(idim, hd, 1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hd,64), nn.ReLU(True), nn.Linear(64,nc))

    def forward(self, x):
        o, _ = self.lstm(x); return self.fc(o[:,-1,:])

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

nq = nc  
cnn = FoodCNN(nc).to(device)
crit = FocalLoss(weight=loss_weights, gamma=2.0, label_smoothing=0.1)
opt = optim.AdamW(filter(lambda p: p.requires_grad, cnn.parameters()), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=LR*5, epochs=EPOCHS, steps_per_epoch=len(train_dl))
lstm = TrendLSTM(2048 + nq).to(device)  
lstm_opt = optim.Adam(lstm.parameters(), lr=0.001)
q_hist = deque(maxlen=SEQ_LEN)
PHASE1_EPOCHS = 5
print(f"\n{'='*50}")
print(f"  CNN TRAINING — up to {EPOCHS} Epochs | {nc} Classes")
print(f"  Phase 1: Classifier only (epochs 1-{PHASE1_EPOCHS})")
print(f"  Phase 2: Full fine-tuning (epochs {PHASE1_EPOCHS+1}-{EPOCHS})")
print(f"  Loss: Focal Loss (gamma=2.0) + Class Weights")
print(f"  Model: ResNet50 (upgraded from ResNet18)")
print(f"  Early Stop: {PATIENCE} epochs")
print(f"{'='*50}")
best_val_acc = 0.0
no_improve = 0 
for ep in range(EPOCHS):
    if ep == PHASE1_EPOCHS:
        print(f"\n  ── Phase 2: Unfreezing ALL layers for fine-tuning ──")
        for param in cnn.parameters():
            param.requires_grad = True
        opt = optim.AdamW(cnn.parameters(), lr=LR * 0.1, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - PHASE1_EPOCHS, eta_min=1e-6)

    start_time = time.time()
    cnn.train(); ls = correct = total = 0
    for imgs, lbl in train_dl:
        imgs, lbl = imgs.to(device, non_blocking=pin), lbl.to(device, non_blocking=pin)
        opt.zero_grad(set_to_none=True)
        out = cnn(imgs); loss = crit(out, lbl); loss.backward(); opt.step()
        if ep < PHASE1_EPOCHS:
            scheduler.step()
        ls += loss.item(); correct += (out.argmax(1)==lbl).sum().item(); total += lbl.size(0)

    if ep >= PHASE1_EPOCHS:
        scheduler.step()  

    cnn.eval(); vc = vt = 0
    with torch.no_grad():
        for imgs, lbl in val_dl:
            imgs, lbl = imgs.to(device, non_blocking=pin), lbl.to(device, non_blocking=pin)
            vc += (cnn(imgs).argmax(1)==lbl).sum().item(); vt += lbl.size(0)
    val_acc = 100*vc/vt
    duration = time.time() - start_time
    lr_now = opt.param_groups[0]['lr']
    phase = "P1" if ep < PHASE1_EPOCHS else "P2"
    print(f"  [{phase}] Ep {ep+1}/{EPOCHS} | Loss:{ls:.3f} | Train:{100*correct/total:.1f}% | Val:{val_acc:.1f}% | LR:{lr_now:.6f} | Time:{duration:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(cnn.state_dict(), CNN_PATH)
        print(f"    ★ New best model saved! Val Acc: {val_acc:.1f}%")
    else:
        no_improve += 1
        print(f"    No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE and ep >= PHASE1_EPOCHS + 3:
            print(f"\n  ⏹️  Early stopping at epoch {ep+1} — best Val Acc: {best_val_acc:.1f}%")
            break

cnn.load_state_dict(torch.load(CNN_PATH, weights_only=True))
print(f"\n  Best model loaded (Val Acc: {best_val_acc:.1f}%) ✅")

print("Generating 7×7 Confusion Matrix & Confidence Curves...")
ap, al, all_probs = [], [], []
cnn.eval()
with torch.no_grad():
    for imgs, lbl in test_dl:
        out = cnn(imgs.to(device))
        probs = torch.softmax(out, 1).cpu().numpy()
        preds = out.argmax(1).cpu().numpy()
        ap.extend(preds); al.extend(lbl.numpy()); all_probs.append(probs)

all_probs = np.vstack(all_probs)  
al_arr = np.array(al)

fig, ax = plt.subplots(figsize=(10,8))
ConfusionMatrixDisplay(confusion_matrix(al,ap), display_labels=dataset.classes).plot(ax=ax, xticks_rotation=45)
ax.set_title('Confusion Matrix — 7 Quality Classes', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=150); plt.close(fig)
print("Confusion Matrix saved (7 X 7) ✅")
print(f"CNN saved → {CNN_PATH} ✅")

# ── Precision / Recall / F1 vs Confidence Curves ──
print("Generating Precision / Recall / F1 vs Confidence curves...")
conf_thresholds = np.linspace(0.0, 1.0, 101)
n_cls = len(dataset.classes)
class_colors_plt = ['#06d6a0', '#38ef7d', '#ffd93d', '#e77f67', '#8e9eab', '#ff6b6b', '#870000']

max_confs = all_probs.max(axis=1)
pred_classes = all_probs.argmax(axis=1)

prec_curve = {i: [] for i in range(n_cls)}
rec_curve  = {i: [] for i in range(n_cls)}
f1_curve   = {i: [] for i in range(n_cls)}
prec_all, rec_all, f1_all = [], [], []

for th in conf_thresholds:
    mask = max_confs >= th
    if mask.sum() == 0:
        for i in range(n_cls):
            prec_curve[i].append(1.0); rec_curve[i].append(0.0); f1_curve[i].append(0.0)
        prec_all.append(1.0); rec_all.append(0.0); f1_all.append(0.0)
        continue
    y_true_m = al_arr[mask]
    y_pred_m = pred_classes[mask]
    for i in range(n_cls):
        tp = int(((y_pred_m == i) & (y_true_m == i)).sum())  # type: ignore[attr-defined]
        fp = int(((y_pred_m == i) & (y_true_m != i)).sum())  # type: ignore[attr-defined]
        fn = int((al_arr == i).sum()) - tp                   # type: ignore[attr-defined]
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec_curve[i].append(p); rec_curve[i].append(r); f1_curve[i].append(f)
    prec_all.append(float(np.mean([prec_curve[i][-1] for i in range(n_cls)])))
    rec_all.append(float(np.mean([rec_curve[i][-1] for i in range(n_cls)])))
    f1_all.append(float(np.mean([f1_curve[i][-1] for i in range(n_cls)])))

def _plot_conf_curve(per_class, overall, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0d1b2a'); ax.set_facecolor('#0d1b2a')
    for i in range(n_cls):
        ax.plot(conf_thresholds, per_class[i], color=class_colors_plt[i],
                linewidth=1.5, alpha=0.7, label=dataset.classes[i])
    ax.plot(conf_thresholds, overall, color='#00b4d8',
            linewidth=3, linestyle='--', label='All classes (mean)')
    ax.set_xlabel('Confidence', fontsize=12, color='white')
    ax.set_ylabel(ylabel, fontsize=12, color='white')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.tick_params(colors='white')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color((1.0, 1.0, 1.0, 0.3))
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.15, color='white')
    ax.legend(loc='best', fontsize=9, facecolor='#1b3a4b', edgecolor='none',
              labelcolor='white', framealpha=0.8)
    plt.tight_layout(); plt.savefig(fname, dpi=150, facecolor='#0d1b2a'); plt.close(fig)
    print(f"  {fname} saved ✅")

_plot_conf_curve(prec_curve, prec_all,
                 'Precision-Confidence Curve — 7 Quality Classes',
                 'Precision', 'precision_confidence_curve.png')
_plot_conf_curve(rec_curve, rec_all,
                 'Recall-Confidence Curve — 7 Quality Classes',
                 'Recall', 'recall_confidence_curve.png')
_plot_conf_curve(f1_curve, f1_all,
                 'F1-Confidence Curve — 7 Quality Classes',
                 'F1 Score', 'f1_confidence_curve.png')
print("All confidence curves saved ✅")

#  EVALUATION METRICS 
test_acc = accuracy_score(al, ap) * 100
precision_macro = precision_score(al, ap, average='macro') * 100
recall_macro    = recall_score(al, ap, average='macro') * 100
f1_macro        = f1_score(al, ap, average='macro') * 100

precision_weighted = precision_score(al, ap, average='weighted') * 100
recall_weighted    = recall_score(al, ap, average='weighted') * 100
f1_weighted        = f1_score(al, ap, average='weighted') * 100

# Per-class metrics
precision_per = precision_score(al, ap, average=None) * 100
recall_per    = recall_score(al, ap, average=None) * 100
f1_per        = f1_score(al, ap, average=None) * 100

# Full classification report
report = classification_report(al, ap, target_names=dataset.classes, digits=4)

# Print metrics
print(f"\n{'='*60}")
print(f"  EVALUATION METRICS — Test Set ({te} images)")
print(f"{'='*60}")
print(f"\n  ── Overall Metrics ──")
print(f"  Test Accuracy     : {test_acc:.2f}%")
print(f"  Precision (Macro) : {precision_macro:.2f}%")
print(f"  Recall (Macro)    : {recall_macro:.2f}%")
print(f"  F1-Score (Macro)  : {f1_macro:.2f}%")
print(f"  Precision (Wtd)   : {precision_weighted:.2f}%")
print(f"  Recall (Wtd)      : {recall_weighted:.2f}%")
print(f"  F1-Score (Wtd)    : {f1_weighted:.2f}%")

print(f"\n  ── Per-Class Metrics ──")
print(f"  {'Class':<12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1-Score':>10s}")
print(f"  {'-'*46}")
for i, cls in enumerate(dataset.classes):
    print(f"  {cls:<12s}  {precision_per[i]:>9.2f}%  {recall_per[i]:>9.2f}%  {f1_per[i]:>9.2f}%")

print(f"\n  ── Full Classification Report ──")
print(report)

# Save metrics to file
with open('metrics_report.txt', 'w', encoding='utf-8') as f:
    f.write("Food Quality Assessment System — Evaluation Metrics\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Dataset: {DATASET_PATH}\n")
    f.write(f"Total Images: {len(dataset)} | Test Set: {te}\n")
    f.write(f"Classes: {nc} - {', '.join(dataset.classes)}\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n\n")
    f.write(f"{'='*60}\n")
    f.write(f"  OVERALL METRICS\n")
    f.write(f"{'='*60}\n")
    f.write(f"  Test Accuracy      : {test_acc:.2f}%\n")
    f.write(f"  Precision (Macro)  : {precision_macro:.2f}%\n")
    f.write(f"  Recall (Macro)     : {recall_macro:.2f}%\n")
    f.write(f"  F1-Score (Macro)   : {f1_macro:.2f}%\n")
    f.write(f"  Precision (Wtd)    : {precision_weighted:.2f}%\n")
    f.write(f"  Recall (Wtd)       : {recall_weighted:.2f}%\n")
    f.write(f"  F1-Score (Wtd)     : {f1_weighted:.2f}%\n\n")
    f.write(f"{'='*60}\n")
    f.write(f"  PER-CLASS METRICS\n")
    f.write(f"{'='*60}\n")
    f.write(f"  {'Class':<12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1-Score':>10s}\n")
    f.write(f"  {'-'*46}\n")
    for i, cls in enumerate(dataset.classes):
        f.write(f"  {cls:<12s}  {precision_per[i]:>9.2f}%  {recall_per[i]:>9.2f}%  {f1_per[i]:>9.2f}%\n")
    f.write(f"\n{'='*60}\n")
    f.write(f"  CLASSIFICATION REPORT\n")
    f.write(f"{'='*60}\n")
    f.write(report)

print(f"\n  Metrics saved → metrics_report.txt ✅")
print(f"{'='*60}")

print(f"\n{'='*60}")
print(f"  LOADING YOLO MODEL — Inference Only (No Training)")
print(f"{'='*60}")

# Custom-trained food quality YOLOv9 weights (absolute path)
YOLO_BEST = r"C:\Users\JANARDHAM\Desktop\PROJECT CODE\runs\detect\runs\train\food_quality_yolov9\weights\best.pt"
# Fallback path (BATCH 16 version — used if above not found)
YOLO_BEST_ALT = r"C:\Users\JANARDHAM\Desktop\BATCH 16\runs\detect\runs\food_quality\weights\best.pt"
# Resolve: use whichever exists
if not os.path.exists(YOLO_BEST) and os.path.exists(YOLO_BEST_ALT):
    YOLO_BEST = YOLO_BEST_ALT

# Load best model: custom-trained if available, otherwise pretrained COCO yolov9c
if os.path.exists(YOLO_BEST):
    print(f"Loading custom-trained YOLOv9: {YOLO_BEST}")
    yolo = YOLO(YOLO_BEST)
    print("Custom YOLOv9 Loaded (7 quality classes) ✅")
else:
    print("Custom YOLOv9 not found — loading pretrained yolov9c.pt (COCO)")
    yolo = YOLO("yolov9c.pt")
    print("Pretrained YOLOv9c Loaded (COCO weights) ✅")
    print("⚠️  YOLO training was skipped (CPU mode). Using pretrained weights.")
    print("   Species identification will rely on MobileNetV2 classifier.")
print(f"{'='*60}")

# ── MobileNetV2 Species Classifier — identifies fruit/vegetable type (36 classes) ──
import json
import torchvision.transforms as T

SPECIES_MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppp", "fruit_species_classifier.pth")
SPECIES_CLASSES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppp", "fruit_species_classes.json")
if not os.path.exists(SPECIES_MODEL_PATH):
    SPECIES_MODEL_PATH   = "fruit_species_classifier.pth"
    SPECIES_CLASSES_PATH = "fruit_species_classes.json"

species_model = None  # type: ignore[assignment]
species_meta: dict = {}  # type: ignore[assignment]
_species_loaded = False
if os.path.exists(SPECIES_MODEL_PATH) and os.path.exists(SPECIES_CLASSES_PATH):
    with open(SPECIES_CLASSES_PATH, 'r') as _f:
        species_meta = json.load(_f)
    _n_sp = species_meta['num_fine_classes']
    species_model = models.mobilenet_v2(weights=None)
    species_model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(species_model.last_channel, _n_sp),
    )
    species_model.load_state_dict(torch.load(SPECIES_MODEL_PATH, map_location=device, weights_only=True))
    species_model = species_model.to(device)
    species_model.eval()
    _species_loaded = True
    print(f"Species classifier loaded: {_n_sp} classes ✅  ({SPECIES_MODEL_PATH})")
else:
    print("  Species classifier not found — fruit names will come from YOLO labels only.")
    print(f"   Expected: {SPECIES_MODEL_PATH}")

def identify_species(crop_bgr):
    if not _species_loaded or species_model is None:
        return None
    _input_size = int(species_meta.get('input_size', 224))
    _tf = T.Compose([
        T.ToPILImage(),
        T.Resize((_input_size, _input_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = _tf(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = species_model(tensor)  # type: ignore[operator]
        prob = torch.softmax(out, 1)
        conf, pred = prob.max(1)
    if conf.item() > 0.30:
        idx = str(pred.item())
        return species_meta.get('class_to_parent', {}).get(idx, None)
    return None

# ── STEP 2: Preprocessing ──
def preprocess(frame):
    img = cv2.resize(frame, (224,224))
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - NORM_M)/NORM_S
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

def detect_defect_region(crop, quality_class):
    ql = quality_class.lower()
    if ql == 'fresh':
        return {'has_defect': False, 'severity': 'None', 'defect_type': 'No Defect',
                'area_pct': 0.0, 'defect_bbox': None, 'mask': None, 'sub_regions': []}

    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return {'has_defect': False, 'severity': 'None', 'defect_type': 'Too Small',
                'area_pct': 0.0, 'defect_bbox': None, 'mask': None, 'sub_regions': []}

    # Preprocess: Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(crop, (5, 5), 0)

    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # ── Color Deviation Scoring ──
    l_ch, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    l_mean, l_std = l_ch.mean(), max(l_ch.std(), 1)
    a_mean, a_std = a_ch.mean(), max(a_ch.std(), 1)
    b_mean, b_std = b_ch.mean(), max(b_ch.std(), 1)

    # Build defect mask based on quality class
    mask = np.zeros((h, w), dtype=np.uint8)

    if ql in ('damaged', 'rotten'):
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))
        brown_mask = cv2.inRange(lab, (0, 115, 115), (130, 150, 150))

        # Color deviation: pixels > 2 std from mean in LAB space
        deviation = np.sqrt(
            ((l_ch.astype(np.float32) - l_mean) / l_std) ** 2 +
            ((a_ch.astype(np.float32) - a_mean) / a_std) ** 2 +
            ((b_ch.astype(np.float32) - b_mean) / b_std) ** 2
        )
        anomaly_mask = (deviation > 2.0).astype(np.uint8) * 255

        dark_patches = cv2.inRange(hsv, (0, 20, 0), (180, 255, 60))
        edges = cv2.Canny(gray, 40, 130)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        mask = cv2.bitwise_or(dark_mask, brown_mask)
        mask = cv2.bitwise_or(mask, anomaly_mask)
        mask = cv2.bitwise_or(mask, dark_patches)
        texture_and_color = cv2.bitwise_and(edges_dilated, cv2.bitwise_or(dark_mask, anomaly_mask))
        mask = cv2.bitwise_or(mask, texture_and_color)

    elif ql == 'dried':
        edges = cv2.Canny(gray, 25, 90)
        edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        dry_mask = cv2.inRange(hsv, (0, 0, 40), (180, 55, 200))

        local_mean = cv2.blur(gray.astype(np.float32), (15, 15))
        local_var = cv2.blur((gray.astype(np.float32) - local_mean) ** 2, (15, 15))
        wrinkle_mask = (local_var > 300).astype(np.uint8) * 255

        mask = cv2.bitwise_or(edges_dilated, dry_mask)
        mask = cv2.bitwise_or(mask, wrinkle_mask)

    elif ql == 'old':
        muted_mask = cv2.inRange(hsv, (0, 0, 35), (180, 70, 185))
        aging_mask = cv2.inRange(lab, (40, 125, 125), (160, 155, 155))

        deviation = np.sqrt(
            ((l_ch.astype(np.float32) - l_mean) / l_std) ** 2 +
            ((b_ch.astype(np.float32) - b_mean) / b_std) ** 2
        )
        spot_mask = (deviation > 2.5).astype(np.uint8) * 255

        mask = cv2.bitwise_or(muted_mask, aging_mask)
        mask = cv2.bitwise_or(mask, spot_mask)

    elif ql == 'unripe':
        green_mask = cv2.inRange(hsv, (25, 35, 35), (90, 255, 255))
        mask = green_mask

    elif ql == 'ripe':
        ripe_mask = cv2.inRange(hsv, (0, 70, 70), (25, 255, 255))
        dark_ripe = cv2.inRange(hsv, (0, 40, 20), (20, 200, 100))
        mask = cv2.bitwise_or(ripe_mask, dark_ripe)

    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    total_pixels = h * w
    defect_pixels = cv2.countNonZero(mask)
    area_pct = (defect_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    # Multi-contour analysis
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_bbox = None
    sub_regions = []
    min_region_area = total_pixels * 0.005

    if contours:
        sig_contours = [c for c in contours if cv2.contourArea(c) > min_region_area]
        if sig_contours:
            all_pts = np.vstack(sig_contours)
            dx, dy, dw, dh = cv2.boundingRect(all_pts)
            defect_bbox = (dx, dy, dx + dw, dy + dh)

            sig_contours.sort(key=cv2.contourArea, reverse=True)
            top5: list = list(sig_contours[:5])
            for c in top5:
                cx, cy, cw, ch = cv2.boundingRect(c)
                c_area = cv2.contourArea(c)
                sub_regions.append({
                    'bbox': (cx, cy, cx + cw, cy + ch),
                    'area_pct': round((c_area / total_pixels) * 100, 1),
                })
        elif contours:
            all_pts = np.vstack(contours)
            dx, dy, dw, dh = cv2.boundingRect(all_pts)
            defect_bbox = (dx, dy, dx + dw, dy + dh)

    # Severity scoring (area + color deviation weighted)
    color_deviation_score = 0
    if ql in ('damaged', 'rotten') and defect_pixels > 0:
        defect_lab = lab[mask > 0]
        if len(defect_lab) > 0:
            defect_l_mean = defect_lab[:, 0].mean()
            color_deviation_score = max(0, (l_mean - defect_l_mean) / max(l_std, 1))

    if ql in ('damaged', 'rotten'):
        combined_score = area_pct * 0.6 + color_deviation_score * 15 * 0.4
        severity = 'High' if combined_score > 25 or area_pct > 30 else 'Medium' if combined_score > 8 or area_pct > 10 else 'Low'
    elif ql in ('dried', 'old'):
        severity = 'Medium' if area_pct > 20 else 'Low'
    else:
        severity = 'Low'

    defect_type_map = {
        'damaged': 'Bruise/Physical Damage',
        'rotten': 'Rot/Decay',
        'dried': 'Dehydration/Wrinkle',
        'old': 'Discoloration/Aging',
        'unripe': 'Unripe Region',
        'ripe': 'Overripe Spot',
    }
    defect_type = defect_type_map.get(ql, 'Unknown')

    return {
        'has_defect': area_pct > 2,
        'severity': severity,
        'defect_type': defect_type,
        'area_pct': round(area_pct, 1),
        'defect_bbox': defect_bbox,
        'mask': mask,
        'sub_regions': sub_regions,
        'num_defect_regions': len(sub_regions),
    }

# ── STEP 3: YOLOv9 Detection (Object Detection + Localization) ──
def yolo_detect(frame):
    h0, w0 = frame.shape[:2]
    detections = []

    res = yolo(frame, conf=0.25, verbose=False)

    if res and len(res[0].boxes) > 0:
        bx = res[0].boxes; names = res[0].names
        for i in range(len(bx)):
            x1,y1,x2,y2 = bx.xyxy[i].cpu().numpy().astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w0,x2), min(h0,y2)
            c   = bx.conf[i].item()
            cid = int(bx.cls[i].item())
            yn  = names.get(cid, f"cls_{cid}")
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                detections.append({
                    'yolo_class': yn, 'yolo_conf': float(c),
                    'bbox': (x1,y1,x2,y2), 'crop': crop,
                    'location': f"({(x1+x2)//2},{(y1+y2)//2})"
                })
    if not detections:
        detections.append({
            'yolo_class': '', 'yolo_conf': 0.0,
            'bbox': (0,0,w0,h0), 'crop': frame,
            'location': f"({w0//2},{h0//2})"
        })

    return detections

# ── STEP 4: CNN-based Classification (Secondary Classifier on YOLO crops) ──
def extract_features(crop):

    t = preprocess(crop)
    cnn.eval()
    with torch.no_grad():
        out  = cnn(t)
        prob = torch.softmax(out, 1)
        conf, pred = prob.max(1)
        feat = cnn.get_features()
    # Class name IS the quality label directly (no mapping needed)
    quality_class = dataset.classes[pred.item()]
    return quality_class, conf.item(), feat.cpu().numpy()

# ── STEP 5: LSTM Trend Analysis (Temporal Analysis) ──
def lstm_trend_analysis(feat, quality_class):
    """Quality hierarchy for trend scoring:
    Fresh(6) > Ripe(5) > Unripe(4) > Old(3) > Dried(2) > Damaged(1) > Rotten(0)  """
    # One-hot encode the quality class
    qi = dataset.classes.index(quality_class) if quality_class in dataset.classes else 0
    oh = np.zeros(nq, dtype=np.float32)
    oh[qi] = 1.0
    q_hist.append(np.concatenate([feat.ravel(), oh]))

    if len(q_hist) < 3:
        return "COLLECTING DATA... 📊"

    seq = torch.from_numpy(np.array(q_hist, dtype=np.float32)).unsqueeze(0).to(device)
    lstm.eval()
    with torch.no_grad():
        lstm(seq)

    # Compute trend from quality scores over sliding window
    scores = []
    for e in q_hist:
        cls_idx = e[-nq:].argmax()
        cls_name = dataset.classes[cls_idx] if cls_idx < len(dataset.classes) else 'old'
        scores.append(Q_SCORES.get(cls_name.lower(), 3))

    scores_list: list = list(scores)
    m = len(scores_list)//2
    d = float(np.mean(scores_list[m:])) - float(np.mean(scores_list[:m]))  # type: ignore[index]
    return TREND_LABELS[2] if d>0.5 else TREND_LABELS[0] if d<-0.5 else TREND_LABELS[1]


# ── STEP 6: Web Dashboard / Cloud Storage ──
def _format_fruit_name(yolo_class):
    if not yolo_class:          # blank = no custom model / full-frame fallback
        return ''
    return yolo_class.replace('_', ' ').title()

def draw_dashboard(frame, results, trend):
    ann = frame.copy()
    h, w = ann.shape[:2]

    # Draw bounding boxes and defect regions for each detection
    for r in results:
        x1,y1,x2,y2 = r['bbox']
        ql   = r['quality'].lower()
        col  = Q_COLORS.get(ql, (255,255,255))
        cv2.rectangle(ann, (x1,y1), (x2,y2), col, 2)
        # Show fruit type + quality (omit fruit prefix if no YOLO model)
        fruit_name = _format_fruit_name(r.get('yolo_class', ''))
        lt = f"{fruit_name}: {r['quality']} {r['confidence']:.0%}" if fruit_name else f"{r['quality']} {r['confidence']:.0%}"
        (tw,th),_ = cv2.getTextSize(lt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(ann, (x1,y1-th-8), (x1+tw+4,y1), col, -1)
        cv2.putText(ann, lt, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Draw defect region with severity-coded colors
        defect = r.get('defect', {})
        if defect.get('has_defect') and defect.get('defect_bbox'):
            dx1, dy1, dx2, dy2 = defect['defect_bbox']
            abs_dx1, abs_dy1 = x1 + dx1, y1 + dy1
            abs_dx2, abs_dy2 = x1 + dx2, y1 + dy2
            # Severity-coded color: Red=High, Orange=Medium, Yellow=Low
            sev = defect.get('severity', 'Low')
            defect_col = (0, 0, 255) if sev == 'High' else (0, 165, 255) if sev == 'Medium' else (0, 255, 255)
            cv2.rectangle(ann, (abs_dx1, abs_dy1), (abs_dx2, abs_dy2), defect_col, 2)
            dt = f"DEFECT: {defect['defect_type']} ({defect['area_pct']}%) [{sev}]"
            (dtw, dth), _ = cv2.getTextSize(dt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(ann, (abs_dx1, abs_dy2), (abs_dx1+dtw+4, abs_dy2+dth+6), defect_col, -1)
            cv2.putText(ann, dt, (abs_dx1+2, abs_dy2+dth+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

            # Draw sub-region bounding boxes (numbered)
            for idx, sr in enumerate(defect.get('sub_regions', [])):
                sx1, sy1, sx2, sy2 = sr['bbox']
                abs_sx1, abs_sy1 = x1 + sx1, y1 + sy1
                abs_sx2, abs_sy2 = x1 + sx2, y1 + sy2
                cv2.rectangle(ann, (abs_sx1, abs_sy1), (abs_sx2, abs_sy2), (255, 0, 255), 1)
                cv2.putText(ann, f"D{idx+1}", (abs_sx1+2, abs_sy1+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

            # Semi-transparent severity-coded overlay on defect pixels
            mask = defect.get('mask')
            if mask is not None:
                crop_h, crop_w = y2-y1, x2-x1
                if mask.shape[0] == crop_h and mask.shape[1] == crop_w:
                    overlay = ann.copy()
                    tint_bgr = np.array([0, 0, 200] if sev == 'High' else [0, 100, 255] if sev == 'Medium' else [0, 200, 200], dtype=np.float64)
                    overlay[y1:y2, x1:x2][mask > 0] = (
                        overlay[y1:y2, x1:x2][mask > 0] * 0.5 +
                        tint_bgr * 0.5
                    ).astype(np.uint8)
                    cv2.addWeighted(overlay, 0.6, ann, 0.4, 0, ann)

    # Header panel
    ov = ann.copy()
    cv2.rectangle(ov, (0,0), (440,180), (0,0,0), -1)
    n_det = len(results)
    if n_det > 0:
        dy = h - 20 - n_det*22 - 30
        cv2.rectangle(ov, (0, dy), (420, h), (0,0,0), -1)
    cv2.addWeighted(ov, 0.7, ann, 0.3, 0, ann)

    # Header text
    y = 25
    cv2.putText(ann, "FOOD QUALITY ASSESSMENT", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if results:
        primary = results[0]
        ql = primary['quality'].lower()
        qc = Q_COLORS.get(ql, (255,255,255))
        score = Q_SCORES.get(ql, 3)
        y+=30; cv2.putText(ann, f"Quality: {primary['quality']}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, qc, 2)
        y+=28; cv2.putText(ann, f"Confidence: {primary['confidence']:.1%}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        y+=28; cv2.putText(ann, f"Score: {score}/6", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        y+=28; cv2.putText(ann, f"Trend: {trend}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,255,255), 1)

        # Defect info on header
        defect = primary.get('defect', {})
        if defect.get('has_defect'):
            y+=28; cv2.putText(ann, f"Defect: {defect['defect_type']} ({defect['severity']})",
                              (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,128,255), 1)
            y+=22; cv2.putText(ann, f"Defect Area: {defect['area_pct']}%",
                              (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

    # Detections panel
    if n_det > 0:
        dy = h - 20 - n_det*22
        cv2.putText(ann, f"DETECTIONS: {n_det}", (10,dy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)
        for r in results:
            defect = r.get('defect', {})
            defect_str = f" [{defect.get('defect_type', '')}]" if defect.get('has_defect') else ""
            cv2.putText(ann, f"  - {r['quality']} ({r['confidence']:.0%}) {r['location']}{defect_str}",
                        (10,dy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,200,255), 1)
            dy += 22

    return ann

def process(frame):
    detections = yolo_detect(frame)
    results = []
    all_feats = []
    for det in detections:
        quality_class, conf, feat = extract_features(det['crop'])
        species_name = identify_species(det['crop'])
        if not species_name:
            yn = str(det.get('yolo_class', '') or '').strip()
            species_name = yn.replace('_', ' ').title() if yn else 'Unknown'

        # Step 4b: Defect Region Localization
        defect_info = detect_defect_region(det['crop'], quality_class)

        results.append({
            'quality':    quality_class,
            'confidence': conf,
            'score':      Q_SCORES.get(quality_class.lower(), 3),
            'bbox':       det['bbox'],
            'location':   det['location'],
            'yolo_class': det['yolo_class'],
            'yolo_conf':  det['yolo_conf'],
            'species':    species_name,   # fruit/vegetable name from MobileNetV2 or YOLO
            'defect':     defect_info,
        })
        all_feats.append((feat, quality_class))

    # ── Step 5: LSTM Temporal Analysis ──
    if all_feats:
        feat, ql = all_feats[0]
        trend = lstm_trend_analysis(feat, ql)
    else:
        trend = "NO DATA"

    # ── Step 6: Dashboard ──
    annotated = draw_dashboard(frame, results, trend)
    primary = results[0] if results else {'quality': 'N/A', 'confidence': 0.0, 'defect': {}, 'species': 'Unknown'}
    return annotated, primary['quality'], primary['confidence'], trend, results


def _downscale(f, maxw=640):
    h, w = f.shape[:2]
    if w > maxw:
        s = maxw / w
        return cv2.resize(f, (maxw, int(h * s)))
    return f

# ── IMAGE Input Source ──
def detect_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print("Error: Could not read image!")
        return
    start = time.time()
    res, ql, conf, trend, dets = process(frame)
    proc_time = time.time() - start
    score = Q_SCORES.get(ql.lower(), 3)
    # 'species' key is set by process() via MobileNetV2 or YOLO label
    fruit_type = dets[0].get('species', 'Unknown') if dets else 'Unknown'
    print(f"\n{'='*60}\n  IMAGE ANALYSIS REPORT\n{'='*60}")
    print(f"  File      : {os.path.basename(path)}")
    print(f"  Species   : {fruit_type}")
    print(f"  Quality   : {ql}  |  Score: {score}/6")
    print(f"  Confidence: {conf:.1%}")
    print(f"  Trend     : {trend}")
    print(f"  Detections: {len(dets)}")
    print(f"  Time      : {proc_time:.2f}s")
    print(f"\n  {'#':<4} {'Species':<16} {'Quality':<12} {'Conf':<8} {'Defect':<25} {'Severity':<10} {'Area%':<8}")
    print(f"  {'-'*85}")
    for i, d in enumerate(dets):
        fn     = d.get('species', 'Unknown')
        defect = d.get('defect', {})
        if not isinstance(defect, dict):
            defect = {}
        d_type = defect.get('defect_type', 'N/A')
        d_sev  = defect.get('severity', 'None')
        d_area = defect.get('area_pct', 0)
        print(f"  {i+1:<4} {fn:<16} {d['quality']:<12} {d['confidence']:.0%}{'': >5} {d_type:<25} {d_sev:<10} {d_area:<8.1f}")
    defective = [d for d in dets if isinstance(d.get('defect'), dict) and d['defect'].get('has_defect')]
    if defective:
        print(f"\n  ⚠️  Defective: {len(defective)}/{len(dets)}")
        for d in defective:
            fn     = d.get('species', 'Unknown')
            defect = d['defect']
            print(f"     → {fn}: {defect.get('defect_type','?')} | {defect.get('severity','?')} | {defect.get('area_pct',0):.1f}%")
    else:
        print(f"\n  ✅ No defects detected.")
    print(f"{'='*60}")
    cv2.imshow("Food Quality — Image", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): print("Error: Could not open video!"); return
    print("Processing video... Press 'q' to stop."); fc = 0; last = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        if fc%3==0: last,*_ = process(_downscale(frame))
        cv2.imshow("Food Quality — Video", last if last is not None else frame)
        if cv2.waitKey(1)==ord("q"): break
    cap.release(); cv2.destroyAllWindows(); print(f"Done. Frames: {fc}")
def webcam_detect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: No webcam!"); return
    print("Webcam active... Press 'q' to stop."); fc = 0; last = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        if fc%5==0: last,*_ = process(_downscale(frame))
        cv2.imshow("Food Quality — Webcam [Q=quit]", last if last is not None else frame)
        if cv2.waitKey(1)==ord("q"): break
    cap.release(); cv2.destroyAllWindows(); print(f"Webcam ended. Frames: {fc}")
#  READY
print(f"\n{'='*50}")
print(f"  SYSTEM READY 🚀 | {nc} Quality Classes | LSTM:{SEQ_LEN} | YOLOv9 | {device}")
print(f"  Classes: {', '.join(dataset.classes)}")
print(f"{'='*50}")


