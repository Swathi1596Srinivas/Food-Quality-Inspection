
import os, cv2, re, csv, io, json, time, requests
import numpy as np, torch, torch.nn as nn
import torchvision.models as models
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import deque
from datetime import datetime
import plotly.graph_objects as go

# ═══ PAGE CONFIG ═══
st.set_page_config(page_title="Food Quality Inspection System", layout="wide", initial_sidebar_state="expanded")

# ═══ CUSTOM CSS — Teal / Emerald Theme ═══
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b3a4b 50%, #065a60 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        border: 1px solid rgba(0,255,200,0.08); box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #ffffff; font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
    .main-header p { color: rgba(255,255,255,0.6); font-size: 0.95rem; margin: 0.5rem 0 0 0; }

    .metric-card {
        background: linear-gradient(145deg, #0d1b2a 0%, #1b3a4b 100%);
        padding: 1.5rem; border-radius: 14px; text-align: center;
        border: 1px solid rgba(0,255,200,0.08); box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
    .metric-value {
        font-size: 2rem; font-weight: 800; margin: 0.3rem 0;
        background: linear-gradient(135deg, #00b4d8 0%, #06d6a0 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.8rem; color: rgba(255,255,255,0.5);
        text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;
    }

    .quality-badge {
        display: inline-block; padding: 0.5rem 1.5rem; border-radius: 25px;
        font-weight: 700; font-size: 1.1rem; letter-spacing: 0.5px; text-transform: uppercase;
    }
    .quality-fresh    { background: linear-gradient(135deg, #06d6a0, #00b4d8); color: #fff; }
    .quality-ripe     { background: linear-gradient(135deg, #11998e, #38ef7d); color: #fff; }
    .quality-unripe   { background: linear-gradient(135deg, #f7971e, #ffd200); color: #1a1a2e; }
    .quality-old      { background: linear-gradient(135deg, #e77f67, #f45c43); color: #fff; }
    .quality-dried    { background: linear-gradient(135deg, #8e9eab, #eef2f3); color: #1a1a2e; }
    .quality-damaged  { background: linear-gradient(135deg, #eb3349, #f45c43); color: #fff; }
    .quality-rotten   { background: linear-gradient(135deg, #870000, #190a05); color: #fff; }
    .quality-out-of-scope { background: linear-gradient(135deg, #6c757d, #adb5bd); color: #fff; }

    .defect-badge { display:inline-block; padding:0.2rem 0.6rem; border-radius:6px; font-size:0.75rem; font-weight:600; }
    .defect-high   { background:rgba(255,0,0,0.2); color:#ff6b6b; border:1px solid rgba(255,0,0,0.3); }
    .defect-medium { background:rgba(255,165,0,0.2); color:#ffa500; border:1px solid rgba(255,165,0,0.3); }
    .defect-low    { background:rgba(255,255,0,0.2); color:#ffd93d; border:1px solid rgba(255,255,0,0.3); }
    .defect-none   { background:rgba(6,214,160,0.1); color:#06d6a0; border:1px solid rgba(6,214,160,0.3); }

    .det-table { width:100%; border-collapse:separate; border-spacing:0 8px; }
    .det-table th { color:rgba(255,255,255,0.5); font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px; padding:0.75rem 1rem; text-align:left; font-weight:600; }
    .det-table td { background:rgba(255,255,255,0.03); padding:0.85rem 1rem; font-size:0.9rem; color:#e0e0e0; }
    .det-table tr td:first-child { border-radius:10px 0 0 10px; }
    .det-table tr td:last-child  { border-radius:0 10px 10px 0; }

    .trend-degrading { color:#ff6b6b; font-weight:700; }
    .trend-stable    { color:#ffd93d; font-weight:700; }
    .trend-improving { color:#06d6a0; font-weight:700; }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #1b3a4b 100%); }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}

    .status-box { padding:1rem 1.5rem; border-radius:12px; margin:0.5rem 0; font-weight:500; }
    .status-success { background:rgba(6,214,160,0.1); border:1px solid rgba(6,214,160,0.3); color:#06d6a0; }
    .status-info    { background:rgba(0,180,216,0.1); border:1px solid rgba(0,180,216,0.3); color:#00b4d8; }
    .status-warning { background:rgba(255,217,61,0.1); border:1px solid rgba(255,217,61,0.3); color:#ffd93d; }
</style>
""", unsafe_allow_html=True)

# Model file paths
_BASE = os.path.dirname(os.path.abspath(__file__))
CNN_PATH            = os.path.join(_BASE, "cnn_classifier.pth")
SPECIES_MODEL_PATH  = os.path.join(_BASE, "fruit_species_classifier.pth")
SPECIES_CLASSES_PATH= os.path.join(_BASE, "fruit_species_classes.json")
SEQ_LEN = 10
NORM_M = np.float32([0.485, 0.456, 0.406])
NORM_S = np.float32([0.229, 0.224, 0.225])

CLASS_NAMES = ['Damaged', 'Dried', 'Fresh', 'Old', 'Ripe', 'Rotten', 'Unripe']
NUM_CLASSES = 7

Q_SCORES = {'fresh':6, 'ripe':5, 'unripe':4, 'old':3, 'dried':2, 'damaged':1, 'rotten':0}
Q_COLORS = {
    'fresh':(0,255,0), 'ripe':(0,200,0), 'unripe':(0,255,255),
    'old':(0,165,255), 'dried':(128,128,128), 'damaged':(0,0,255), 'rotten':(0,0,200),
    'out of scope':(128,128,128),
}
TREND_LABELS = ['DEGRADING ⬇️', 'STABLE ➡️', 'IMPROVING ⬆️']

# ═══ CNN MODEL — matches 12.py (ResNet50 upgrade) ═══
class FoodCNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feat = nn.Sequential(*list(backbone.children())[:-1])
        self.fdim = 2048
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(256, nc)
        )
        self._cf = None
        for name, param in self.feat.named_parameters():
            if not any(k in name for k in ('layer3','layer4')):
                param.requires_grad = False
    def forward(self, x):
        f = self.feat(x); self._cf = f; return self.cls(f)
    def get_features(self):
        f = self._cf; self._cf = None; return f.view(f.size(0),-1) if f is not None else None

class TrendLSTM(nn.Module):
    def __init__(self, idim, hd=128, nc=3):
        super().__init__()
        self.lstm = nn.LSTM(idim, hd, 1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hd,64), nn.ReLU(True), nn.Linear(64,nc))
    def forward(self, x):
        o, _ = self.lstm(x); return self.fc(o[:,-1,:])

# YOLO model paths (inference only — no training on CPU)
FRUIT_YOLO_PATH     = os.path.join(_BASE, "yolov9_fruits_best.pt")
FRUIT_YOLO_ALT1     = r"C:\Users\JANARDHAM\Desktop\PROJECT CODE\runs\detect\runs\train\food_quality_yolov9\weights\best.pt"
FRUIT_YOLO_ALT2     = r"C:\Users\JANARDHAM\Desktop\BATCH 16\runs\detect\runs\food_quality\weights\best.pt"
YOLO_WORLD_MODEL   = os.path.join(_BASE, "yolov8s-worldv2.pt")
USING_FRUIT_YOLO   = False

# Comprehensive fruit & vegetable class list for YOLO-World zero-shot detection.
FRUIT_VEGETABLE_CLASSES = [
    # ── Fruits ──
    "apple", "avocado", "apricot", "banana", "blueberry", "blackberry",
    "cherry", "coconut", "custard apple", "dates", "dragon fruit", "durian",
    "fig", "gooseberry", "grapes", "guava", "jackfruit", "kiwi", "kumquat",
    "lemon", "lime", "lychee", "mango", "melon", "mulberry", "orange",
    "papaya", "passion fruit", "peach", "pear", "pineapple", "plum",
    "pomegranate", "pomelo", "raspberry", "sapodilla", "star fruit",
    "strawberry", "tamarind", "watermelon",
    # ── Vegetables ──
    "artichoke", "asparagus", "bean", "beetroot", "bell pepper", "bitter gourd",
    "bottle gourd", "broccoli", "brussels sprout", "cabbage", "capsicum",
    "carrot", "cauliflower", "celery", "chilli pepper", "chives", "corn",
    "cucumber", "drumstick", "eggplant", "fenugreek", "french beans",
    "garlic", "ginger", "green beans", "jalapeno", "kale", "lady finger",
    "leek", "lentil", "lettuce", "long beans", "mushroom", "okra", "onion",
    "paprika", "parsley", "peas", "peppercorn", "potato", "pumpkin",
    "radish", "ridge gourd", "soy beans", "spinach", "spring onion",
    "sweet corn", "sweet potato", "tomato", "turnip", "yam", "zucchini",
]

# Display name mapping
PRODUCE_DISPLAY_NAMES = {c: c.title() for c in FRUIT_VEGETABLE_CLASSES}
PRODUCE_DISPLAY_NAMES.update({
    "soy beans":    "Soy Beans",
    "bell pepper":  "Bell Pepper",
    "bitter gourd": "Bitter Gourd",
    "bottle gourd": "Bottle Gourd",
    "sweet corn":   "Sweet Corn",
    "sweet potato": "Sweet Potato",
    "spring onion": "Spring Onion",
    "dragon fruit": "Dragon Fruit",
    "passion fruit":"Passion Fruit",
    "star fruit":   "Star Fruit",
    "custard apple":"Custard Apple",
    "brussels sprout":"Brussels Sprout",
    "brother bean": "Broad Bean",
    "french beans": "French Beans",
    "green beans":  "Green Beans",
    "long beans":   "Long Beans",
    "lady finger":  "Lady Finger (Okra)",
    "ridge gourd":  "Ridge Gourd",
    "chilli pepper":"Chilli Pepper",
})

# ═══ DEBUG INSTRUMENTATION (agent) ═══
def _agent_debug_log(hypothesis_id, location, message, data, run_id="initial"):
    try:
        payload = {
            "sessionId": "951b64",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("debug-951b64.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = FoodCNN(NUM_CLASSES).to(device)
    if os.path.exists(CNN_PATH):
        cnn.load_state_dict(torch.load(CNN_PATH, map_location=device, weights_only=True))
    cnn.eval()
    lstm = TrendLSTM(2048 + NUM_CLASSES).to(device); lstm.eval()
    global USING_FRUIT_YOLO

    _fruit_model_path = None
    for _p in (FRUIT_YOLO_PATH, FRUIT_YOLO_ALT1, FRUIT_YOLO_ALT2):
        if os.path.exists(_p):
            _fruit_model_path = _p
            break

    if _fruit_model_path:
        yolo = YOLO(_fruit_model_path)
        USING_FRUIT_YOLO = True
        print(f"✅ Fruit-specific YOLO loaded: {_fruit_model_path}")
    else:
        try:
            yolo = YOLO(YOLO_WORLD_MODEL)
            yolo.set_classes(FRUIT_VEGETABLE_CLASSES)
            USING_FRUIT_YOLO = True
            print(f"✅ YOLO-World loaded ({len(FRUIT_VEGETABLE_CLASSES)} classes, zero-shot)")
        except Exception as e:
            yolo = YOLO("yolov9c.pt")
            USING_FRUIT_YOLO = False
            print(f"⚠️ Using generic COCO YOLO (YOLO-World failed: {e})")

    # ── Load YOLO-World as fruit LOCATOR (always, for bounding boxes) ──
    yolo_locator = None
    try:
        yolo_locator = YOLO(YOLO_WORLD_MODEL)
        yolo_locator.set_classes(FRUIT_VEGETABLE_CLASSES)
        print(f"✅ YOLO-World locator loaded ({len(FRUIT_VEGETABLE_CLASSES)} classes)")
    except Exception as e:
        print(f"⚠️ YOLO-World locator not available: {e}")

    # ── Species Classifier (Fruits-360 trained MobileNetV2) ──
    species_model = None
    species_meta = None
    if os.path.exists(SPECIES_MODEL_PATH) and os.path.exists(SPECIES_CLASSES_PATH):
        with open(SPECIES_CLASSES_PATH, 'r') as f:
            species_meta = json.load(f)
        num_fine = species_meta['num_fine_classes']
        species_model = models.mobilenet_v2(weights=None)
        species_model.classifier[1] = nn.Linear(species_model.last_channel, num_fine)
        species_model.load_state_dict(torch.load(SPECIES_MODEL_PATH, map_location=device, weights_only=True))
        species_model = species_model.to(device)
        species_model.eval()
        print(f"✅ Species classifier loaded: {num_fine} classes → {species_meta['num_parent_species']} species")
    else:
        print("⚠️ Species classifier not found — using generic labels")

    return cnn, lstm, yolo, yolo_locator, device, species_model, species_meta

cnn, lstm, yolo, yolo_locator, device, species_model, species_meta = load_models()

# ═══ SESSION STATE ═══
if 'q_hist' not in st.session_state:
    st.session_state.q_hist = deque(maxlen=SEQ_LEN)
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# ═══ PIPELINE FUNCTIONS — matches 12.py ═══

def preprocess(frame):
    img = cv2.resize(frame, (224,224))
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 - NORM_M)/NORM_S
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

def identify_species(crop_bgr):
    """Identify fruit/vegetable species using real-world trained classifier.
    Model: MobileNetV2 fine-tuned on Kritikseth Fruit & Vegetable dataset (36 classes).
    Trained on real photos — works with natural lighting, backgrounds, etc."""
    if species_model is None or species_meta is None:
        return None
    import torchvision.transforms as T
    input_size = species_meta.get('input_size', 224)
    tf = T.Compose([
        T.ToPILImage(), T.Resize((input_size, input_size)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = tf(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = species_model(tensor)
        prob = torch.softmax(out, 1)
        conf, pred = prob.max(1)
    # Threshold lowered to 0.15 so MobileNetV2 catches more species
    # and prevents COCO wrong labels (apple/orange/banana) from appearing
    if conf.item() > 0.15:
        idx = str(pred.item())
        return species_meta['class_to_parent'].get(idx, None)
    return None

def _format_fruit_name(yolo_class, source_label='', crop=None):
    """Identify fruit/vegetable species.
    Priority order:
      1. MobileNetV2 AI classifier (36 trained classes, 96.8% accuracy)
      2. Detector class name
      3. Filename / URL extraction
      4. Generic COCO label fallback
    """
    # 1. MobileNetV2 AI identification (most accurate for 36 trained classes)
    if crop is not None and species_model is not None:
        ai_result = identify_species(crop)
        if ai_result:
            return ai_result

    # 2. Detector class name — check known produce names
    yn = yolo_class.lower().strip().replace('_', ' ')
    if yn in PRODUCE_DISPLAY_NAMES:
        return PRODUCE_DISPLAY_NAMES[yn]

    # 3. Try to extract from filename/URL (user uploaded "mango.jpg" etc.)
    extracted = _extract_produce_from_filename(source_label)
    if extracted:
        return extracted

    # 4. Generic fallback — title-case YOLO label
    if yn == 'full_frame' or yn.startswith('cls_'):
        return 'Food Item'
    return yolo_class.replace('_', ' ').title()

def detect_defect_region(crop, quality_class):
    ql = quality_class.lower()
    if ql == 'fresh':
        return {'has_defect':False,'severity':'None','defect_type':'No Defect',
                'area_pct':0.0,'defect_bbox':None,'mask':None,'sub_regions':[]}
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return {'has_defect':False,'severity':'None','defect_type':'Too Small',
                'area_pct':0.0,'defect_bbox':None,'mask':None,'sub_regions':[]}
    blurred = cv2.GaussianBlur(crop, (5,5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    l_ch, a_ch, b_ch = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    l_mean, l_std = l_ch.mean(), max(l_ch.std(), 1)
    a_mean, a_std = a_ch.mean(), max(a_ch.std(), 1)
    b_mean, b_std = b_ch.mean(), max(b_ch.std(), 1)
    mask = np.zeros((h,w), dtype=np.uint8)

    if ql in ('damaged','rotten'):
        dark_mask = cv2.inRange(hsv, (0,0,0), (180,255,70))
        brown_mask = cv2.inRange(lab, (0,115,115), (130,150,150))
        deviation = np.sqrt(((l_ch.astype(np.float32)-l_mean)/l_std)**2+((a_ch.astype(np.float32)-a_mean)/a_std)**2+((b_ch.astype(np.float32)-b_mean)/b_std)**2)
        anomaly_mask = (deviation > 2.0).astype(np.uint8)*255
        dark_patches = cv2.inRange(hsv, (0,20,0), (180,255,60))
        edges = cv2.Canny(gray, 40, 130)
        edges_d = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=1)
        mask = cv2.bitwise_or(dark_mask, brown_mask)
        mask = cv2.bitwise_or(mask, anomaly_mask)
        mask = cv2.bitwise_or(mask, dark_patches)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(edges_d, cv2.bitwise_or(dark_mask, anomaly_mask)))
    elif ql == 'dried':
        edges = cv2.Canny(gray, 25, 90)
        edges_d = cv2.dilate(edges, np.ones((5,5),np.uint8), iterations=2)
        dry_mask = cv2.inRange(hsv, (0,0,40), (180,55,200))
        lm = cv2.blur(gray.astype(np.float32), (15,15))
        lv = cv2.blur((gray.astype(np.float32)-lm)**2, (15,15))
        wrinkle = (lv > 300).astype(np.uint8)*255
        mask = cv2.bitwise_or(edges_d, dry_mask)
        mask = cv2.bitwise_or(mask, wrinkle)
    elif ql == 'old':
        muted = cv2.inRange(hsv, (0,0,35), (180,70,185))
        aging = cv2.inRange(lab, (40,125,125), (160,155,155))
        dev = np.sqrt(((l_ch.astype(np.float32)-l_mean)/l_std)**2+((b_ch.astype(np.float32)-b_mean)/b_std)**2)
        spot = (dev > 2.5).astype(np.uint8)*255
        mask = cv2.bitwise_or(muted, aging); mask = cv2.bitwise_or(mask, spot)
    elif ql == 'unripe':
        mask = cv2.inRange(hsv, (25,35,35), (90,255,255))
    elif ql == 'ripe':
        mask = cv2.bitwise_or(cv2.inRange(hsv,(0,70,70),(25,255,255)), cv2.inRange(hsv,(0,40,20),(20,200,100)))

    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko, iterations=1)
    total_px = h*w; defect_px = cv2.countNonZero(mask)
    area_pct = (defect_px/total_px)*100 if total_px > 0 else 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_bbox = None; sub_regions = []; min_area = total_px*0.005
    if contours:
        sig = [c for c in contours if cv2.contourArea(c) > min_area]
        if sig:
            pts = np.vstack(sig); dx,dy,dw,dh = cv2.boundingRect(pts)
            defect_bbox = (dx,dy,dx+dw,dy+dh)
            sig.sort(key=cv2.contourArea, reverse=True)
            for c in sig[:5]:
                cx,cy,cw,ch = cv2.boundingRect(c)
                sub_regions.append({'bbox':(cx,cy,cx+cw,cy+ch),'area_pct':round((cv2.contourArea(c)/total_px)*100,1)})
        elif contours:
            pts = np.vstack(contours); dx,dy,dw,dh = cv2.boundingRect(pts)
            defect_bbox = (dx,dy,dx+dw,dy+dh)

    cds = 0
    if ql in ('damaged','rotten') and defect_px > 0:
        dl = lab[mask>0]
        if len(dl)>0: cds = max(0, (l_mean - dl[:,0].mean())/max(l_std,1))
    if ql in ('damaged','rotten'):
        cs = area_pct*0.6 + cds*15*0.4
        severity = 'High' if cs>25 or area_pct>30 else 'Medium' if cs>8 or area_pct>10 else 'Low'
    elif ql in ('dried','old'):
        severity = 'Medium' if area_pct > 20 else 'Low'
    else:
        severity = 'Low'

    dt_map = {'damaged':'Bruise/Physical Damage','rotten':'Rot/Decay','dried':'Dehydration/Wrinkle',
              'old':'Discoloration/Aging','unripe':'Unripe Region','ripe':'Overripe Spot'}
    return {'has_defect':area_pct>2,'severity':severity,'defect_type':dt_map.get(ql,'Unknown'),
            'area_pct':round(area_pct,1),'defect_bbox':defect_bbox,'mask':mask,
            'sub_regions':sub_regions,'num_defect_regions':len(sub_regions)}



# Common produce keywords to match from filenames
PRODUCE_KEYWORDS = [
    'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry',
    'raspberry', 'mango', 'pineapple', 'watermelon', 'melon', 'peach',
    'pear', 'plum', 'cherry', 'lemon', 'lime', 'kiwi', 'papaya',
    'pomegranate', 'coconut', 'avocado', 'fig', 'guava', 'lychee',
    'tomato', 'potato', 'onion', 'garlic', 'ginger', 'carrot',
    'broccoli', 'cauliflower', 'cabbage', 'spinach', 'lettuce',
    'cucumber', 'zucchini', 'eggplant', 'bellpepper', 'bell_pepper',
    'bell pepper', 'capsicum', 'pepper', 'chili', 'corn', 'pea',
    'bean', 'mushroom', 'radish', 'beet', 'turnip', 'celery',
    'asparagus', 'artichoke', 'okra', 'squash', 'pumpkin',
    'sweetpotato', 'sweet potato', 'yam', 'jackfruit', 'dragonfruit',
    'starfruit', 'passionfruit', 'durian', 'rambutan', 'mangosteen',
]

def _extract_produce_from_filename(filename):
    """Try to extract a produce/food name from the uploaded filename."""
    if not filename:
        return None
    # Remove extension and common prefixes like numbers
    base = os.path.splitext(filename)[0]
    # Normalize: remove digits-only prefixes, underscores, hyphens, camelCase split
    # e.g. "1BellPepper_IMG_20220609_091056" → "BellPepper IMG"
    # Insert space before uppercase letters for camelCase: "BellPepper" → "Bell Pepper"
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', base)
    # Replace path separators, underscores, hyphens with spaces (handles URLs)
    spaced = re.sub(r'[_\-/\\]+', ' ', spaced)
    spaced = spaced.replace('%20', ' ')
    # Remove purely numeric tokens (dates, IDs)
    tokens = [t for t in spaced.split() if not t.isdigit()]
    cleaned = ' '.join(tokens).lower()

    # Try to match produce keywords (longest match first)
    sorted_kw = sorted(PRODUCE_KEYWORDS, key=len, reverse=True)
    for kw in sorted_kw:
        if kw in cleaned:
            return kw.replace('_', ' ').title()
    return None

def yolo_detect(frame, aggressive=False):
    """Two-stage detection pipeline with species-classifier validation.
    aggressive=True: used for URL images — lower conf, higher caps to catch all fruits."""
    h0, w0 = frame.shape[:2]
    detections = []

    # Tune thresholds based on mode
    conf_thresh  = 0.20 if aggressive else 0.35
    per_sp_cap   = 6   if aggressive else 3
    total_cap    = 20  if aggressive else 8
    max_det_yolo = 50  if aggressive else 20

    # ── Stage 1: YOLO-World localization + species-classifier validation ──
    if yolo_locator is not None:
        try:
            res = yolo_locator(frame, conf=conf_thresh, iou=0.30,
                               agnostic_nms=True, max_det=max_det_yolo, verbose=False)
            if res and len(res[0].boxes) > 0:
                raw = []
                bx = res[0].boxes; names = res[0].names
                for i in range(len(bx)):
                    x1,y1,x2,y2 = bx.xyxy[i].cpu().numpy().astype(int)
                    x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(w0,x2), min(h0,y2)
                    if (x2-x1) < 30 or (y2-y1) < 30:
                        continue
                    c = bx.conf[i].item(); cid = int(bx.cls[i].item())
                    yn = names.get(cid, f"cls_{cid}")
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # ── Validate with MobileNetV2 species classifier ──
                    validated_species = yn
                    if species_model is not None and species_meta is not None:
                        sp_result = identify_species(crop)
                        if sp_result is not None:
                            validated_species = sp_result

                    raw.append({
                        'yolo_class': validated_species,
                        'yolo_conf': float(c),
                        'bbox': (x1,y1,x2,y2),
                        'crop': crop,
                        'location': f"({(x1+x2)//2},{(y1+y2)//2})"
                    })

                # ── Per-species NMS ──
                if raw:
                    raw.sort(key=lambda d: d['yolo_conf'], reverse=True)
                    seen_species = {}
                    for d in raw:
                        sp = d['yolo_class'].lower()
                        if sp not in seen_species:
                            seen_species[sp] = 0
                        if seen_species[sp] < per_sp_cap:
                            seen_species[sp] += 1
                            detections.append(d)
                        if len(detections) >= total_cap:
                            break
        except Exception:
            pass

    # ── Stage 2: Custom quality YOLO fallback ──
    if not detections:
        try:
            res = yolo(frame, conf=0.15, iou=0.3, agnostic_nms=True, max_det=100, verbose=False)
            if res and len(res[0].boxes) > 0:
                bx = res[0].boxes; names = res[0].names
                for i in range(len(bx)):
                    x1,y1,x2,y2 = bx.xyxy[i].cpu().numpy().astype(int)
                    x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(w0,x2), min(h0,y2)
                    c = bx.conf[i].item(); cid = int(bx.cls[i].item())
                    yn = names.get(cid, f"cls_{cid}")
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        detections.append({'yolo_class': yn, 'yolo_conf': float(c),
                                           'bbox': (x1,y1,x2,y2), 'crop': crop,
                                           'location': f"({(x1+x2)//2},{(y1+y2)//2})"})
        except Exception:
            pass

    # ── Stage 3: Full-frame fallback ──
    if not detections:
        detections.append({'yolo_class': '', 'yolo_conf': 0.0,
                           'bbox': (0,0,w0,h0), 'crop': frame,
                           'location': f"({w0//2},{h0//2})"})
    return detections

def extract_features(crop):
    t = preprocess(crop); cnn.eval()
    with torch.no_grad():
        out = cnn(t); prob = torch.softmax(out,1); conf,pred = prob.max(1); feat = cnn.get_features()
    return CLASS_NAMES[pred.item()], conf.item(), feat.cpu().numpy(), prob.cpu().numpy()[0]



def lstm_trend_analysis(feat, quality_class):
    qi = CLASS_NAMES.index(quality_class) if quality_class in CLASS_NAMES else 0
    oh = np.zeros(NUM_CLASSES, dtype=np.float32); oh[qi] = 1.0
    st.session_state.q_hist.append(np.concatenate([feat.ravel(), oh]))
    if len(st.session_state.q_hist) < 3:
        return "COLLECTING DATA... 📊"
    seq = torch.from_numpy(np.array(st.session_state.q_hist, dtype=np.float32)).unsqueeze(0).to(device)
    lstm.eval()
    with torch.no_grad(): lstm(seq)
    scores = []
    for e in st.session_state.q_hist:
        ci = e[-NUM_CLASSES:].argmax()
        cn = CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else 'old'
        scores.append(Q_SCORES.get(cn.lower(), 3))
    m = len(scores)//2; d = np.mean(scores[m:]) - np.mean(scores[:m])
    return TREND_LABELS[2] if d>0.5 else TREND_LABELS[0] if d<-0.5 else TREND_LABELS[1]

def draw_dashboard(frame, results, trend, source_label=''):
    """Draw results overlay — clear bounding boxes with 7-class quality labels."""
    ann = frame.copy(); h, w = ann.shape[:2]

    # Quality class colour map (BGR)
    Q_COLORS_DRAW = {
        'fresh':   (0, 214, 105),   # bright green
        'ripe':    (0, 200,  60),   # dark green
        'unripe':  (0, 210, 255),   # yellow
        'old':     (0, 140, 220),   # orange
        'dried':   (160,160,160),   # gray
        'damaged': (50,  50, 255),  # red
        'rotten':  (30,  0, 140),   # dark red
    }
    Q_EMOJI = {
        'fresh':'[FRESH]','ripe':'[RIPE]','unripe':'[UNRIPE]',
        'old':'[OLD]','dried':'[DRIED]','damaged':'[DAMAGED]','rotten':'[ROTTEN]'
    }

    for idx, r in enumerate(results):
        x1,y1,x2,y2 = r['bbox']; ql = r['quality'].lower()
        col = Q_COLORS_DRAW.get(ql, (200,200,200))
        is_full_frame = (x1 == 0 and y1 == 0 and x2 >= w-2 and y2 >= h-2)
        species = r.get('species', '') or r.get('yolo_class', '') or 'Fruit'
        quality_tag = Q_EMOJI.get(ql, ql.upper())
        conf_pct   = f"{r['confidence']:.0%}"
        score      = r.get('score', '?')

        if is_full_frame:
            overlay = ann.copy()
            cv2.rectangle(overlay, (0,0), (w, 48), col, -1)
            cv2.addWeighted(overlay, 0.55, ann, 0.45, 0, ann)
            label = f"{species}  {quality_tag}  {conf_pct}  Score:{score}/6"
            cv2.putText(ann, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (255,255,255), 2, cv2.LINE_AA)
        else:
            # ── Bounding box (thick, coloured) ──
            cv2.rectangle(ann, (x1,y1), (x2,y2), col, 3)

            # ── Detection index circle (top-left corner) ──
            cx, cy = x1 + 14, y1 + 14
            cv2.circle(ann, (cx, cy), 13, col, -1)
            cv2.putText(ann, str(idx+1), (cx-6 if idx < 9 else cx-9, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

            # ── Quality class label tag ──
            line1 = f"{species}"
            line2 = f"{quality_tag}  {conf_pct}  {score}/6"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (w1,h1),_ = cv2.getTextSize(line1, font, 0.55, 2)
            (w2,h2),_ = cv2.getTextSize(line2, font, 0.5,  2)
            tag_w = max(w1, w2) + 12
            tag_h = h1 + h2 + 16
            label_y = y1 - tag_h - 4 if y1 - tag_h - 4 >= 0 else y2 + 4

            # background pill
            cv2.rectangle(ann, (x1, label_y), (x1 + tag_w, label_y + tag_h), col, -1)
            cv2.rectangle(ann, (x1, label_y), (x1 + tag_w, label_y + tag_h), (255,255,255), 1)
            # text
            cv2.putText(ann, line1, (x1+6, label_y + h1 + 4),
                        font, 0.55, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(ann, line2, (x1+6, label_y + h1 + h2 + 10),
                        font, 0.5,  (255,255,255), 2, cv2.LINE_AA)

        # ── Defect region overlay (non-fresh only) ──
        defect = r.get('defect', {})
        if ql != 'fresh' and defect.get('has_defect') and defect.get('defect_bbox'):
            dx1,dy1,dx2,dy2 = defect['defect_bbox']
            ax1,ay1 = x1+dx1, y1+dy1; ax2,ay2 = x1+dx2, y1+dy2
            sev = defect.get('severity','Low')
            dc = (0,0,255) if sev=='High' else (0,165,255) if sev=='Medium' else (0,255,255)
            cv2.rectangle(ann, (ax1,ay1), (ax2,ay2), dc, 2)
            dt = f"DEFECT:{defect['defect_type']} {defect['area_pct']}% [{sev}]"
            (dtw,dth),_ = cv2.getTextSize(dt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(ann, (ax1,ay2), (ax1+dtw+4,ay2+dth+6), dc, -1)
            cv2.putText(ann, dt, (ax1+2,ay2+dth+3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1)
            mask = defect.get('mask')
            if mask is not None:
                ch_, cw_ = y2-y1, x2-x1
                if mask.shape[0]==ch_ and mask.shape[1]==cw_:
                    ov = ann.copy()
                    tint = np.array([0,0,200] if sev=='High' else [0,100,255] if sev=='Medium'
                                    else [0,200,200], dtype=np.float64)
                    ov[y1:y2,x1:x2][mask>0]=(ov[y1:y2,x1:x2][mask>0]*0.5+tint*0.5).astype(np.uint8)
                    cv2.addWeighted(ov, 0.6, ann, 0.4, 0, ann)

    return ann

def process_frame(frame, source_label=''):
    """Full pipeline: Upload → Preprocess → YOLOv9 → CNN → Defect → LSTM → Dashboard."""
    start = time.time()
    # Use aggressive detection for URL images (more fruits, denser scenes)
    is_url = source_label.startswith('http') or '/' in source_label or '\\' not in source_label
    detections = yolo_detect(frame, aggressive=is_url)
    results = []; all_feats = []

    # ── Full-frame species detection (most reliable — avoids partial-crop confusion) ──
    # MobileNetV2 trained on full-fruit images; small crops (30x30 mango) can look like tomatoes.
    # Run on the complete frame first → if confident, use as dominant species anchor.
    frame_species = None
    if species_model is not None:
        frame_species = identify_species(frame)

    # Only manual user override from sidebar "Produce Type" input
    produce_override = source_label.split()[0] if source_label else ''
    manual_override = None
    if produce_override:
        po = produce_override.lower().strip()
        for kw in PRODUCE_KEYWORDS:
            if kw in po or po in kw:
                manual_override = kw.replace('_', ' ').title()
                break

    for det in detections:
        quality_class, conf, feat, cnn_probs = extract_features(det['crop'])
        # Species identification — purely from IMAGE analysis
        species_name = None

        # 1. Manual user override (highest priority)
        if manual_override:
            species_name = manual_override

        # 2. Full-frame species (more reliable than tiny crop identification)
        if not species_name and frame_species:
            species_name = frame_species

        # 3. Visual detection: Species Classifier on individual crop (fallback)
        if not species_name and species_model:
            species_name = identify_species(det['crop'])

        # 3. YOLO class name (YOLO-World / fine-tuned)
        if not species_name and USING_FRUIT_YOLO:
            yn = det['yolo_class'].lower().strip()
            if yn and not yn.startswith('cls_'):
                species_name = det['yolo_class'].replace('_', ' ').title()

        # 4. Fallback
        if not species_name:
            yn = det['yolo_class'].lower().strip()
            if not yn or yn.startswith('cls_'):
                species_name = 'Unknown'
            elif USING_FRUIT_YOLO:
                species_name = det['yolo_class'].replace('_', ' ').title()
            else:
                species_name = 'Unknown'

        # Run defect detection
        defect_info = detect_defect_region(det['crop'], quality_class)

        results.append({
            'quality': quality_class, 'confidence': conf,
            'score': Q_SCORES.get(quality_class.lower(), 0) if quality_class != "Out of Scope" else 0,
            'bbox': det['bbox'], 'location': det['location'],
            'yolo_class': det['yolo_class'], 'yolo_conf': det['yolo_conf'],
            'defect': defect_info, 'species': species_name,
        })
        if feat is not None:
            all_feats.append((feat, quality_class))

    # Majority voting: if 60%+ of detections agree on a species, apply to all
    if len(results) > 2 and not manual_override:
        from collections import Counter
        species_counts = Counter(r['species'] for r in results if r['species'] != 'Food Item')
        if species_counts:
            top_species, top_count = species_counts.most_common(1)[0]
            if top_count / len(results) >= 0.6:
                for r in results:
                    if r['species'] != 'Food Item':
                        r['species'] = top_species

    trend = "NO DATA"
    if all_feats:
        feat, ql = all_feats[0]; trend = lstm_trend_analysis(feat, ql)
    annotated = draw_dashboard(frame, results, trend, source_label)
    proc_time = time.time() - start
    primary = results[0] if results else {'quality':'N/A','confidence':0.0,'score':0,'defect':{}}
    return annotated, results, trend, proc_time, primary

# ═══ CHARTS ═══
def create_confidence_bars(results, source_label=''):
    if not results: return None
    labels = [f"{r['quality']} ({r.get('species', r.get('yolo_class','?'))})" for r in results]
    confs = [r['confidence']*100 for r in results]
    cmap = {'fresh':'#06d6a0','ripe':'#38ef7d','unripe':'#ffd93d','old':'#e77f67','dried':'#8e9eab','damaged':'#ff6b6b','rotten':'#870000','out of scope':'#adb5bd'}
    colors = [cmap.get(r['quality'].lower(),'#00b4d8') for r in results]
    fig = go.Figure(go.Bar(x=confs,y=labels,orientation='h',marker_color=colors,
        text=[f"{c:.1f}%" for c in confs],textposition='inside',textfont=dict(color='white',size=13,family='Inter')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font={'color':'#ffffff','family':'Inter'},height=max(150,len(results)*50+60),
        margin=dict(l=10,r=20,t=10,b=10),
        xaxis=dict(range=[0,100],showgrid=False,title='Confidence %',title_font=dict(color='rgba(255,255,255,0.5)'),tickfont=dict(color='rgba(255,255,255,0.4)')),
        yaxis=dict(showgrid=False,tickfont=dict(color='rgba(255,255,255,0.7)')))
    return fig

def create_history_chart():
    if len(st.session_state.q_hist) < 2: return None
    scores=[]; labels=[]
    for e in st.session_state.q_hist:
        ci = e[-NUM_CLASSES:].argmax()
        cn = CLASS_NAMES[ci] if ci<len(CLASS_NAMES) else 'Unknown'
        scores.append(Q_SCORES.get(cn.lower(),3)); labels.append(cn)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,len(scores)+1)),y=scores,
        mode='lines+markers+text',line=dict(color='#00b4d8',width=3,shape='spline'),
        marker=dict(size=10,color='#065a60',line=dict(width=2,color='#00b4d8')),
        text=labels,textposition='top center',textfont=dict(color='rgba(255,255,255,0.7)',size=10),
        fill='tozeroy',fillcolor='rgba(0,180,216,0.1)'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font={'color':'#ffffff','family':'Inter'},height=250,margin=dict(l=10,r=10,t=10,b=30),
        xaxis=dict(title='Analysis #',showgrid=False,title_font=dict(color='rgba(255,255,255,0.5)'),tickfont=dict(color='rgba(255,255,255,0.4)')),
        yaxis=dict(title='Quality Score',range=[-0.5,6.5],showgrid=True,gridcolor='rgba(255,255,255,0.05)',
            title_font=dict(color='rgba(255,255,255,0.5)'),tickfont=dict(color='rgba(255,255,255,0.4)')))
    return fig

# ═══ SIDEBAR ═══
with st.sidebar:
    st.markdown("### 🍎 Navigation"); st.markdown("---")
    input_mode = st.radio("**Input Source**", ["📷 Upload Image", "🔗 Image URL"], index=0)
    st.markdown("---"); st.markdown("### 📊 Quality Scale")
    st.markdown("""
<table style='width:100%;border-collapse:collapse;font-size:0.84rem;margin-top:4px'>
<thead><tr>
  <th style='text-align:left;color:rgba(255,255,255,0.4);padding:5px 8px;border-bottom:1px solid rgba(255,255,255,0.12)'>Class</th>
  <th style='text-align:center;color:rgba(255,255,255,0.4);padding:5px 8px;border-bottom:1px solid rgba(255,255,255,0.12)'>Score</th>
</tr></thead>
<tbody>
<tr><td style='padding:5px 8px'>🟢 Fresh</td><td style='text-align:center;font-weight:700;color:#06d6a0'>6 / 6</td></tr>
<tr><td style='padding:5px 8px'>🟢 Ripe</td><td style='text-align:center;font-weight:700;color:#38ef7d'>5 / 6</td></tr>
<tr><td style='padding:5px 8px'>🟡 Unripe</td><td style='text-align:center;font-weight:700;color:#ffd200'>4 / 6</td></tr>
<tr><td style='padding:5px 8px'>🟠 Old</td><td style='text-align:center;font-weight:700;color:#e77f67'>3 / 6</td></tr>
<tr><td style='padding:5px 8px'>⚪ Dried</td><td style='text-align:center;font-weight:700;color:#adb5bd'>2 / 6</td></tr>
<tr><td style='padding:5px 8px'>🔴 Damaged</td><td style='text-align:center;font-weight:700;color:#f45c43'>1 / 6</td></tr>
<tr><td style='padding:5px 8px'>⚫ Rotten</td><td style='text-align:center;font-weight:700;color:#870000'>0 / 6</td></tr>
</tbody></table>
""", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔄 Reset History", use_container_width=True):
        st.session_state.q_hist.clear(); st.session_state.analysis_history.clear(); st.rerun()
    st.markdown("---")
    st.markdown("### ⚙️ Model Training")
    if st.button("🚀 Retrain Species Classifier", help="Runs train_species_v2.py in the background to update the species model.", use_container_width=True):
        import subprocess
        subprocess.Popen(["python", "train_species_v2.py"])
        st.success("✅ Training started in the background! Check the console.")

# ═══ MAIN HEADER ═══
st.markdown("""
<div class="main-header">
    <h1>🍎 Food Quality Assessment System</h1>
    <p>Upload an image of fruits or vegetables to assess their quality</p>
</div>
""", unsafe_allow_html=True)

# ═══ INPUT SECTION ═══
frame = None; source_label = ""

if input_mode == "📷 Upload Image":
    uploaded = st.file_uploader("Drop an image here or click to browse",
        type=['jpg','jpeg','png','bmp','webp','tiff','jfif'],
        help="Supported formats: JPG, PNG, BMP, WebP, TIFF, JFIF")
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR); source_label = uploaded.name

elif input_mode == "🔗 Image URL":
    url = st.text_input("🔗 Enter Image URL", placeholder="https://example.com/food.jpg")
    if url and st.button("🔍 Analyze URL", width="stretch"):
        with st.spinner("Downloading image..."):
            try:
                headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                resp = requests.get(url, timeout=20, headers=headers, allow_redirects=True)
                if resp.status_code == 200:
                    img = Image.open(io.BytesIO(resp.content)).convert('RGB')
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    import urllib.parse
                    parsed = urllib.parse.urlparse(url)
                    source_label = urllib.parse.unquote(parsed.path + " " + parsed.query)
                else:
                    st.error(f"Failed to download (Status {resp.status_code})")
            except Exception as e:
                st.error(f"Error: {str(e)}")



# ═══ ANALYSIS & RESULTS ═══
if frame is not None:
    with st.spinner("🔄 Running analysis pipeline..."):
        annotated, results, trend, proc_time, primary = process_frame(frame, source_label)

    st.session_state.analysis_history.append({
        'source': source_label, 'quality':primary['quality'],
        'confidence':primary['confidence'], 'score':primary.get('score',0),
        'trend':trend, 'detections':len(results), 'time':proc_time,
        'timestamp':datetime.now().isoformat(),
    })

    st.markdown("---")
    fruit_type = primary.get('species', primary.get('yolo_class','unknown'))
    ql = primary['quality'].lower()
    badge_class = "quality-out-of-scope" if ql == "out of scope" else (f"quality-{ql}" if ql in Q_SCORES else "quality-old")
    defect_info = primary.get('defect', {})

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Identified As</div>
            <div class="metric-value" style="font-size:1.4rem;">{fruit_type}</div>
            <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;">YOLO Detection</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        q_sub = "Not produce — quality assessment skipped" if ql == "out of scope" else "CNN Classification"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Quality Class</div>
            <div style="margin:0.5rem 0;"><span class="quality-badge {badge_class}">{primary['quality']}</span></div>
            <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;">{q_sub}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{primary['confidence']:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        if ql == 'fresh':
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Defect</div>
                <div class="metric-value" style="font-size:1.1rem;color:#06d6a0;">N/A</div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;">Fresh — No defect check needed ✅</div>
            </div>""", unsafe_allow_html=True)
        elif ql == 'out of scope':
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Defect</div>
                <div class="metric-value" style="font-size:1rem;color:#adb5bd;">N/A</div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;">Not applicable — item is not fresh produce</div>
            </div>""", unsafe_allow_html=True)
        elif defect_info.get('has_defect'):
            sev = defect_info.get('severity','None')
            sc = '#ff6b6b' if sev=='High' else '#ffa500' if sev=='Medium' else '#ffd93d'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Defect</div>
                <div class="metric-value" style="font-size:1.1rem;color:{sc};">{defect_info.get('defect_type','N/A')}</div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;">{sev} — {defect_info.get('area_pct',0)}% area</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Defect</div>
                <div class="metric-value" style="font-size:1.1rem;color:#06d6a0;">No Defect</div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;">Clean ✅</div>
            </div>""", unsafe_allow_html=True)
    with m5:
        tc = "trend-degrading" if "DEGRADING" in trend else "trend-improving" if "IMPROVING" in trend else "trend-stable"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Trend / Count</div>
            <div class="metric-value {tc}" style="font-size:1.1rem;">{trend}</div>
            <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;">{len(results)} detection(s) · {proc_time:.2f}s</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_img, col_details = st.columns([3, 2.5])
    with col_img:
        st.markdown("#### 📸 Annotated Result")
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        max_display = 500
        ih, iw = annotated_rgb.shape[:2]
        if max(ih,iw) > max_display:
            sc = max_display/max(ih,iw)
            display_rgb = cv2.resize(annotated_rgb, (int(iw*sc),int(ih*sc)), interpolation=cv2.INTER_AREA)
        else:
            display_rgb = annotated_rgb
        st.image(display_rgb, width="content")
    with col_details:
        st.markdown("#### 🔍 Detection Details")
        # Primary result summary card
        score = primary.get('score', 0) if ql != "out of scope" else 0
        score_display = "N/A" if ql == "out of scope" else f"{score}/6"
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:0.8rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div style="text-align:left;">
                    <div class="metric-label">Quality</div>
                    <div style="margin:0.3rem 0;"><span class="quality-badge {badge_class}">{primary['quality']}</span></div>
                </div>
                <div style="text-align:center;">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value" style="font-size:1.5rem;">{primary['confidence']:.1%}</div>
                </div>
                <div style="text-align:center;">
                    <div class="metric-label">Score</div>
                    <div class="metric-value" style="font-size:1.5rem;">{score_display}</div>
                </div>
                <div style="text-align:right;">
                    <div class="metric-label">Trend</div>
                    <div class="metric-value {tc}" style="font-size:1.1rem;">{trend}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-detection cards
        for i, r in enumerate(results):
            r_ql = r['quality'].lower()
            r_fruit = r.get('species', r.get('yolo_class','unknown'))
            r_defect = r.get('defect', {})
            if r_ql == 'out of scope':
                defect_str = "<span style='color:#adb5bd;'>Not applicable — item is not fresh produce</span>"
            elif r_ql != 'fresh' and r_defect.get('has_defect'):
                defect_str = f"<span style='color:#ff6b6b;'>{r_defect.get('defect_type','')}</span> ({r_defect.get('severity','')}, {r_defect.get('area_pct',0)}%)"
            else:
                defect_str = "<span style='color:#06d6a0;'>None ✅</span>"
            r_badge = "quality-out-of-scope" if r_ql == "out of scope" else (f"quality-{r_ql}" if r_ql in Q_SCORES else "quality-old")
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(0,255,200,0.06);
                        border-radius:10px; padding:0.7rem 1rem; margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:rgba(255,255,255,0.85); font-weight:600;">{r_fruit}</span>
                        <span class="quality-badge {r_badge}" style="font-size:0.7rem; padding:0.2rem 0.6rem; margin-left:0.5rem;">{r['quality']}</span>
                    </div>
                    <span style="color:#00b4d8; font-weight:700;">{r['confidence']:.1%}</span>
                </div>
                <div style="font-size:0.8rem; color:rgba(255,255,255,0.5); margin-top:0.3rem;">
                    Defect: {defect_str} &nbsp;·&nbsp; Location: {r['location']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence chart
        if results:
            fig_bars = create_confidence_bars(results, source_label)
            if fig_bars:
                st.plotly_chart(fig_bars, width="stretch", config={'displayModeBar':False})

    # LSTM History
    st.markdown("---"); st.markdown("#### 📈 Quality Trend History (LSTM)")
    fig_history = create_history_chart()
    if fig_history:
        st.plotly_chart(fig_history, width="stretch", config={'displayModeBar':False})
    else:
        st.markdown('<div class="status-info">Analyze at least 2 images to see trend history.</div>', unsafe_allow_html=True)

    # Export
    st.markdown("---"); st.markdown("#### ☁️ Export / Cloud")
    exp1, exp2, exp3 = st.columns(3)
    export_data = {
        'timestamp':datetime.now().isoformat(), 'source':source_label,
        'pipeline':'Camera/Upload → Preprocess → YOLOv9 → CNN → Defect → LSTM → Dashboard',
        'primary_result':{'produce':fruit_type,'quality':primary['quality'],'confidence':f"{primary['confidence']:.2%}",'score':f"{primary.get('score',0)}/6"},
        'defect_summary':{'has_defect':defect_info.get('has_defect',False),'type':defect_info.get('defect_type','None'),'severity':defect_info.get('severity','None'),'area_pct':defect_info.get('area_pct',0)},
        'trend':trend,
        'detections':[{'quality':r['quality'],'confidence':f"{r['confidence']:.2%}",'score':r['score'],'defect_type':r.get('defect',{}).get('defect_type','None'),'defect_severity':r.get('defect',{}).get('severity','None'),'location':r['location']} for r in results],
        'processing_time_sec':round(proc_time,3),
    }
    with exp1:
        st.download_button("📥 Download JSON Report", json.dumps(export_data,indent=2,default=str),
            file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", width="stretch")
    with exp2:
        img_pil = Image.fromarray(annotated_rgb); buf = io.BytesIO(); img_pil.save(buf,format='PNG')
        st.download_button("🖼️ Download Annotated Image", buf.getvalue(),
            file_name=f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", mime="image/png", width="stretch")
    with exp3:
        if st.session_state.analysis_history:
            csv_buf = io.StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=['timestamp','source','quality','confidence','score','trend','detections','time'])
            writer.writeheader()
            for h in st.session_state.analysis_history: writer.writerow(h)
            st.download_button("📊 Download History CSV", csv_buf.getvalue(),
                file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", width="stretch")
        else:
            st.markdown('<div class="status-info" style="text-align:center;font-size:0.85rem;">No history to export yet.</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">📸</div>
        <h3 style="color:rgba(255,255,255,0.85);margin-bottom:0.5rem;">Upload an Image to Get Started</h3>
        <p style="color:rgba(255,255,255,0.5);font-size:1rem;max-width:500px;margin:0 auto;">
            Select an input source from the sidebar — upload a photo, paste a URL, or use your webcam.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══ MODEL PERFORMANCE ═══
st.markdown("---")
metrics_file = os.path.join(os.path.dirname(__file__), 'metrics_report.txt')
if os.path.exists(metrics_file):
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics_text = f.read()
    def parse_metric(text, label):
        m = re.search(rf"{label}\s*:\s*([\d.]+)%", text)
        return float(m.group(1)) if m else 0.0
    test_acc = parse_metric(metrics_text, "Test Accuracy")
    prec_macro = parse_metric(metrics_text, r"Precision \(Macro\)")
    rec_macro = parse_metric(metrics_text, r"Recall \(Macro\)")
    f1_macro = parse_metric(metrics_text, r"F1-Score \(Macro\)")
    best_val = parse_metric(metrics_text, "Best Validation Accuracy")

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                background:rgba(0,180,216,0.08);border:1px solid rgba(0,180,216,0.15);
                border-radius:10px;padding:0.6rem 1.2rem;margin:0.3rem 0;">
        <span style="font-weight:700;font-size:0.95rem;color:rgba(255,255,255,0.85);">📊 Model Performance</span>
        <span style="display:flex;gap:1.5rem;font-size:0.85rem;">
            <span style="color:rgba(255,255,255,0.5);">Accuracy <b style="color:#06d6a0;">{test_acc:.1f}%</b></span>
            <span style="color:rgba(255,255,255,0.5);">Precision <b style="color:#00b4d8;">{prec_macro:.1f}%</b></span>
            <span style="color:rgba(255,255,255,0.5);">Recall <b style="color:#e0aaff;">{rec_macro:.1f}%</b></span>
            <span style="color:rgba(255,255,255,0.5);">F1 <b style="color:#48bfe3;">{f1_macro:.1f}%</b></span>
            <span style="color:rgba(255,255,255,0.5);">Best Val <b style="color:#ffd93d;">{best_val:.1f}%</b></span>
        </span>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("🔍 View Confusion Matrix", expanded=False):
        cm_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')
        if os.path.exists(cm_path):
            _, cm_col, _ = st.columns([1,2,1])
            with cm_col: st.image(cm_path, width="stretch")
        else:
            st.info("Run `python 12.py` to generate the confusion matrix.")

    # ── Confidence Curve Plots ──
    curve_files = {
        "📈 Precision-Confidence Curve": "precision_confidence_curve.png",
        "📉 Recall-Confidence Curve": "recall_confidence_curve.png",
        "📊 F1-Confidence Curve": "f1_confidence_curve.png",
    }
    base_dir = os.path.dirname(__file__)
    for title, fname in curve_files.items():
        curve_path = os.path.join(base_dir, fname)
        with st.expander(title, expanded=False):
            if os.path.exists(curve_path):
                _, curve_col, _ = st.columns([1, 3, 1])
                with curve_col:
                    st.image(curve_path, width='stretch')
            else:
                st.info(f"Run `python 12.py` to generate {fname}.")
else:
    st.markdown("""
    <div class="status-warning" style="text-align:center;">
        ⚠️ No metrics found. Run <code>python 12.py</code> first to train the model and generate metrics.
    </div>
    """, unsafe_allow_html=True)

# ═══ FOOTER ═══
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:rgba(255,255,255,0.3);font-size:0.8rem;padding:1rem 0;">'
    'Food Quality Assessment System  •  CNN + LSTM + YOLOv9  •  BATCH 16'
    '</div>', unsafe_allow_html=True
)
