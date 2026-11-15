import os
import shutil
import glob
import gc
import math
import random
import time
import warnings
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

# Auto-install GUDHI if missing
try:
    import gudhi
except ImportError:
    print("GUDHI not found. Installing via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gudhi"])
    import gudhi

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# 0. Configuration & Dataset Path Setup (FINAL V8 - BUG FIXED, 885 IMAGES)
# =============================================================================
POSSIBLE_PATHS = [
    "/kaggle/input/datasets/pythonafroz/solar-panel-images/Faulty_solar_panel",
    "/kaggle/input/solar-panel-images/Faulty_solar_panel",
    "/kaggle/input/solar-panel-images-clean-and-faulty-images/Faulty_solar_panel",
    "/kaggle/input/faulty-solar-panel"
]

DATA_DIR = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        DATA_DIR = p
        print(f"[DATASET] Found Image Dataset at: {DATA_DIR}")
        break

if DATA_DIR is None and os.path.exists("/kaggle/input"):
    for root, dirs, _ in os.walk("/kaggle/input"):
        if "Faulty_solar_panel" in dirs:
            DATA_DIR = os.path.join(root, "Faulty_solar_panel")
            print(f"[DATASET] Dynamically located at: {DATA_DIR}")
            break

CKPT_IN_DIR = None

# ----- HYPERPARAMETERS -----
BACKBONE = "efficientnet_b0"
IMG_SIZE = 384
NUM_CLASSES = 6

EPOCHS = 120
BATCH_SIZE = 16
LR = 1.5e-4
ALPHA = 0.1
GATE_REDUCT = 8

PD_TOP_N = 30
PD_PERS_MIN = 5.0
PD_BORDER_PAD = 0.5
PD_FILTRATIONS = ["sub", "super", "grad"]
PD_CHANNELS = ["L"]
NUM_HOMOLOGY = 2
PD_IN_DIM = 2 + NUM_HOMOLOGY + 1

FOLDS = 10
CV_REPEATS = 1
SPLIT_SEEDS = [42]
MODEL_SEEDS = [42]
NUM_WORKERS = 2
TOP_K_EPOCHS = 8

# =============================================================================
# SESSION MANAGEMENT: Run one mode per session to stay within Kaggle 12-hour limit
# =============================================================================
MODES = ["msbase"]  # কন্ট্রোল মোডের জন্য msbase সেট করা হলো
# MODES = ["baseline"]
# MODES = ["concat"]
# MODES = ["phg"]

if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working/phgnet_efficientnet_v8_885"
else:
    OUTPUT_DIR = "./phgnet_efficientnet_v8_885"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PD_CACHE_PATH = os.path.join(OUTPUT_DIR, "pd_cache_v8_384_885.npz")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "final_results_v8_885.csv")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. Advanced Loss Function: Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# 2. Topology / Persistence Computation (BUG FIXED & DETERMINISTIC)
# =============================================================================
def _to_gray_and_grad(pil_image, img_size):
    gray = pil_image.convert("L").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    from scipy.ndimage import convolve
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = convolve(arr, sobel_x, mode="reflect")
    gy = convolve(arr, sobel_y, mode="reflect")
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad = grad / (grad.max() + 1e-8)
    return (arr / 255.0).astype(np.float32), grad.astype(np.float32)


def _pad_border(img, pad_val=PD_BORDER_PAD):
    out = img.copy()
    out[0, :] = pad_val;
    out[-1, :] = pad_val
    out[:, 0] = pad_val;
    out[:, -1] = pad_val
    return out


def compute_pd_multifiltration(pil_image, img_size=IMG_SIZE, max_points=PD_TOP_N, min_persistence=PD_PERS_MIN):
    gray, grad = _to_gray_and_grad(pil_image, img_size)
    chans = {"L": gray}
    pds = []
    for f in PD_FILTRATIONS:
        for cname, base in chans.items():
            if f == "sub":
                arr = base
            elif f == "super":
                arr = 1.0 - base
            elif f == "grad":
                arr = grad

            img = arr.astype(np.float64)
            lo, hi = float(img.min()), float(img.max())
            img = (img - lo) / (max(hi - lo, 1e-8))
            img = _pad_border(img, PD_BORDER_PAD)

            cc = gudhi.CubicalComplex(dimensions=img.shape, top_dimensional_cells=img.flatten())
            pers_intervals = cc.persistence(homology_coeff_field=2, min_persistence=0.0)

            points = []
            for dim_idx, (birth, death) in pers_intervals:
                if dim_idx < NUM_HOMOLOGY:
                    if not np.isfinite(death): death = float(img.max())
                    points.append((float(birth), float(death), dim_idx))

            pd_tensor = np.zeros((max_points, PD_IN_DIM + 1), dtype=np.float32)
            if not points:
                pds.append(pd_tensor)
                continue

            points = np.asarray(points, dtype=np.float32)
            births, deaths, homol = points[:, 0], points[:, 1], points[:, 2].astype(np.int64)
            pers = deaths - births
            keep = (pers * 255.0) >= min_persistence
            births, deaths, homol, pers = births[keep], deaths[keep], homol[keep], pers[keep]

            if len(births) > 0:
                all_vals = np.concatenate([births, deaths])
                lo2, hi2 = float(all_vals.min()), float(all_vals.max())
                b_n = (births - lo2) / max(hi2 - lo2, 1e-8)
                d_n = (deaths - lo2) / max(hi2 - lo2, 1e-8)

                order = np.argsort(-pers)[:max_points]
                n_keep = len(order)
                pd_tensor[:n_keep, 0] = b_n[order]
                pd_tensor[:n_keep, 1] = d_n[order]

                # =========================================================
                # 🔥 CRITICAL BUG FIX (0D & 1D Homology Differentiation)
                # =========================================================
                pd_tensor[np.arange(n_keep), 2 + homol[order]] = 1.0

                pd_tensor[:n_keep, -2] = 1.0

            pd_tensor[:, -1] = PD_FILTRATIONS.index(f) / len(PD_FILTRATIONS)
            pds.append(pd_tensor)

    return np.concatenate(pds, axis=0).astype(np.float32) if pds else np.zeros(
        (max_points * len(PD_FILTRATIONS), PD_IN_DIM + 1), dtype=np.float32)


# =============================================================================
# 3. Model Architecture
# =============================================================================
class PersistenceDiagramEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=256):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        self.post = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim))

    def forward(self, pd_points):
        B, N, D = pd_points.shape
        x = self.point_mlp(pd_points.reshape(B * N, D)).reshape(B, N, -1)
        x = torch.cat([x.max(dim=1).values, x.mean(dim=1)], dim=1)
        return self.post(x)


class PHGuidedFusion(nn.Module):
    def __init__(self, topo_dim, feat_dim, reduction=GATE_REDUCT):
        super().__init__()
        hidden = max(1, feat_dim // reduction)
        self.fc2 = nn.Linear(topo_dim, hidden)
        self.fc1 = nn.Linear(hidden, feat_dim)

    def forward(self, feat, topo):
        gate = torch.sigmoid(self.fc1(F.relu(self.fc2(topo))))
        return feat * gate.unsqueeze(-1).unsqueeze(-1)


class PHGNetAblation(nn.Module):
    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES, mode="phg", pd_in_dim=PD_IN_DIM + 1,
                 topo_dim=256):
        super().__init__()
        self.mode = mode
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        feat_channels = self.backbone.feature_info.channels()

        if mode in ("concat", "phg"):
            self.pd_encoder = PersistenceDiagramEncoder(in_dim=pd_in_dim, out_dim=topo_dim)

        if mode == "phg":
            self.fusions = nn.ModuleList([PHGuidedFusion(topo_dim=topo_dim, feat_dim=c) for c in feat_channels])
            self.vision_head = nn.Sequential(
                nn.Linear(sum(feat_channels), 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Dropout(0.4), nn.Linear(512, num_classes)
            )
            self.topo_head = nn.Sequential(nn.Linear(topo_dim, 256), nn.ReLU(), nn.Dropout(0.4),
                                           nn.Linear(256, num_classes))
        elif mode == "concat":
            self.vision_head = nn.Sequential(
                nn.Linear(feat_channels[-1] + topo_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Dropout(0.4), nn.Linear(512, num_classes)
            )
        elif mode == "msbase":
            self.vision_head = nn.Sequential(
                nn.Linear(sum(feat_channels), 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Dropout(0.4), nn.Linear(512, num_classes)
            )
        else:  # Baseline
            self.vision_head = nn.Sequential(
                nn.Linear(feat_channels[-1], 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Dropout(0.4), nn.Linear(512, num_classes)
            )

    def forward(self, image, pd_points=None):
        feats = self.backbone(image)
        if self.mode == "baseline":
            return self.vision_head(F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)), None

        if self.mode == "msbase":
            pooled = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats], dim=1)
            return self.vision_head(pooled), None

        topo = self.pd_encoder(pd_points)
        if self.mode == "concat":
            pooled = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
            return self.vision_head(torch.cat([pooled, topo], dim=1)), None

        pooled_list = []
        for fmap, fusion in zip(feats, self.fusions):
            pooled_list.append(F.adaptive_avg_pool2d(fusion(fmap, topo), 1).flatten(1))
        return self.vision_head(torch.cat(pooled_list, dim=1)), self.topo_head(topo)


# =============================================================================
# 4. Dataset (Recursive walk, sorted, with early Assert validation)
# =============================================================================
class SolarPanelPHDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, img_size=IMG_SIZE, pd_cache=None):
        self.root_dir = root_dir
        self.transform = image_transform

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.images, self.labels = [], []

        for cls in self.classes:
            cdir = os.path.join(root_dir, cls)
            for dp, dn, fn in os.walk(cdir):
                dn.sort()
                for f in sorted(fn):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        self.images.append(os.path.join(dp, f))
                        self.labels.append(self.class_to_idx[cls])

        # AGGRESSIVE EARLY ASSERTION
        assert len(self.images) == 885, f"Expected exactly 885 images, got {len(self.images)}!"

        expected_dist = [207, 193, 190, 103, 69, 123]
        actual_dist = np.bincount(self.labels).tolist()
        assert actual_dist == expected_dist, f"Expected class dist {expected_dist}, got {actual_dist}!"

        if pd_cache is not None:
            self.pd_cache = pd_cache
        else:
            print(f"Computing High-Res ({IMG_SIZE}x{IMG_SIZE}) for {len(self.images)} images. This may take a while...")
            self.pd_cache = [compute_pd_multifiltration(Image.open(p), img_size=img_size) for p in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.from_numpy(self.pd_cache[idx]).float(), self.labels[idx]


def build_train_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandAugment(num_ops=3, magnitude=14),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# =============================================================================
# 5. Training Engine & Plotting (2-Panel Curves Layout)
# =============================================================================
def train_one_epoch(model, loader, optimizer, scaler, device, mode, class_weights):
    model.train()
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    total, total_loss, correct_v, correct_t = 0, 0.0, 0, 0

    for images, pdv, labels in loader:
        images, pdv, labels = images.to(device), pdv.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits_v, logits_t = model(images, pdv if mode not in ("baseline", "msbase") else None)
            loss = criterion(logits_v, labels)
            if logits_t is not None:
                loss += ALPHA * criterion(logits_t, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer);
        scaler.update()

        total += labels.size(0);
        total_loss += loss.item() * images.size(0)
        correct_v += (logits_v.argmax(1) == labels).sum().item()
        if logits_t is not None: correct_t += (logits_t.argmax(1) == labels).sum().item()

    return {"loss": total_loss / total, "vision_acc": 100. * correct_v / total,
            "topo_acc": 100. * correct_t / total if correct_t else float("nan")}


@torch.no_grad()
def evaluate(model, loader, device, mode, class_weights):
    model.eval()
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    total, total_loss = 0, 0.0
    y_true, y_prob = [], []
    for images, pdv, labels in loader:
        images, pdv, labels = images.to(device), pdv.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits_v, _ = model(images, pdv if mode not in ("baseline", "msbase") else None)
            loss = criterion(logits_v, labels)
        total += labels.size(0);
        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().numpy());
        y_prob.extend(F.softmax(logits_v, dim=1).cpu().numpy())

    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = y_prob.argmax(1)
    acc = 100.0 * accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") * 100.0
    except:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro',
                                                                                 zero_division=0)

    return {
        "loss": total_loss / total, "acc": acc, "auc": auc,
        "precision_macro": precision_macro * 100, "recall_macro": recall_macro * 100, "f1_macro": f1_macro * 100,
        "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred, "cm": cm
    }


def save_training_curves(history, mode, fold_id):
    try:
        epochs = [h['epoch'] for h in history]
        tr_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        val_acc = [h['val_acc'] for h in history]

        plt.figure(figsize=(12, 4))

        # Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tr_loss, 'b-', label='Train Loss')
        plt.plot(epochs, val_loss, 'r-', label='Val Loss')
        plt.title(f'Loss Curve - {mode.upper()} Fold {fold_id}')
        plt.xlabel('Epoch');
        plt.ylabel('Loss');
        plt.legend();
        plt.grid(True, linestyle='--', alpha=0.6)

        # Accuracy Curve
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
        plt.plot(epochs, val_acc, 'g-', label='Val Accuracy')
        plt.title(f'Accuracy Curve - {mode.upper()} Fold {fold_id}')
        plt.xlabel('Epoch');
        plt.ylabel('Accuracy (%)');
        plt.legend();
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        curve_path = os.path.join(OUTPUT_DIR, f'curves_{mode}_f{fold_id}.png')
        plt.savefig(curve_path, dpi=300)
        plt.close()
        print(f"--> Saved curves image to {curve_path}")
    except Exception as e:
        print(f"Could not plot curves: {e}")


# =============================================================================
# 6. Single Fold Execution (Unbiased Last-8 Epochs Evaluation)
# =============================================================================
def run_fold(fold_id, train_idx, val_idx, all_images, all_labels, class_names, pd_cache, mode, model_seed,
             class_weights):
    seed_everything(model_seed + fold_id * 100)
    device = get_device()

    class _View(Dataset):
        def __init__(self, idxs, transform):
            self.idxs, self.transform = idxs, transform

        def __len__(self): return len(self.idxs)

        def __getitem__(self, i):
            idx = self.idxs[i]
            return self.transform(Image.open(all_images[idx]).convert("RGB")), torch.tensor(pd_cache[idx]).float(), \
            all_labels[idx]

    train_loader = DataLoader(_View(train_idx, build_train_transform(IMG_SIZE)), batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(_View(val_idx, build_eval_transform(IMG_SIZE)), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)

    model = PHGNetAblation(mode=mode).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    epoch_records, history, topo_accs, start_epoch = [], [], [], 0
    ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_v8_{mode}_f{fold_id}_s{model_seed}.pth")

    for epoch in range(start_epoch, EPOCHS):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, mode, class_weights)
        va = evaluate(model, val_loader, device, mode, class_weights)
        scheduler.step()
        topo_accs.append(tr["topo_acc"])

        history.append({
            "epoch": epoch + 1, "train_loss": tr["loss"], "val_loss": va["loss"],
            "train_acc": tr["vision_acc"],
            "val_acc": va["acc"], "val_auc": va["auc"],
            "precision": va["precision_macro"], "recall": va["recall_macro"], "f1": va["f1_macro"]
        })
        epoch_records.append({"epoch": epoch + 1, "val_acc": va["acc"], "y_true": va["y_true"], "y_prob": va["y_prob"]})

        torch.save({
            'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(), 'scaler_state': scaler.state_dict(),
            'history': history, 'epoch_records': epoch_records, 'topo_accs': topo_accs
        }, ckpt_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[{mode}][F{fold_id}] Ep {epoch + 1:03d} | trL={tr['loss']:.3f} | valL={va['loss']:.3f} valAcc={va['acc']:.2f}% | F1={va['f1_macro']:.2f}%")

    # Unbiased evaluation using literal last 8 epochs
    topk = epoch_records[-TOP_K_EPOCHS:]
    avg_prob = np.mean([r["y_prob"] for r in topk], axis=0)
    y_pred = avg_prob.argmax(1)

    final_cm = confusion_matrix(topk[0]["y_true"], y_pred, labels=list(range(len(class_names))))
    _, _, f1_per_class, _ = precision_recall_fscore_support(topk[0]["y_true"], y_pred, average=None, zero_division=0)

    final_acc = 100.0 * accuracy_score(topk[0]["y_true"], y_pred)
    final_prec, final_rec, final_f1, _ = precision_recall_fscore_support(topk[0]["y_true"], y_pred, average='macro',
                                                                         zero_division=0)

    save_training_curves(history, mode, fold_id)

    if os.path.exists(ckpt_path): os.remove(ckpt_path)

    return {
        "mode": mode, "fold": fold_id, "model_seed": model_seed,
        "acc": final_acc, "precision": final_prec * 100, "recall": final_rec * 100, "f1_score": final_f1 * 100,
        "f1_per_class": f1_per_class.tolist(), "confusion_matrix": final_cm.tolist()
    }


# =============================================================================
# 7. Main Execution Loop
# =============================================================================
def main():
    device = get_device()
    print(f"Device: {device}")

    if DATA_DIR is None or not os.path.exists(DATA_DIR):
        print("[FATAL] Dataset directory not found!")
        return

    if os.path.exists(PD_CACHE_PATH):
        print(f"[CACHE] Loading {IMG_SIZE}x{IMG_SIZE} PD cache from {PD_CACHE_PATH}...")
        npz = np.load(PD_CACHE_PATH, allow_pickle=True)
        pd_cache, all_images, all_labels, class_names = list(npz["pd"]), list(npz["images"]), npz["labels"], list(
            npz["class_names"])
    else:
        base = SolarPanelPHDataset(DATA_DIR, img_size=IMG_SIZE)
        pd_cache, all_images, all_labels, class_names = base.pd_cache, base.images, np.asarray(
            base.labels), base.classes
        np.savez_compressed(PD_CACHE_PATH, pd=np.asarray(pd_cache), images=all_images, labels=all_labels,
                            class_names=class_names)
        print(f"[CACHE] Saved new PD cache to {PD_CACHE_PATH}")

    # VERIFY PRINTOUTS
    print(f"\n[VERIFY] Total Images: {len(all_images)}")
    print(f"[VERIFY] Class Distribution: {np.bincount(all_labels)}")

    # 🔥 SANITY CHECK FOR 0D & 1D HOMOLOGY BUG FIX
    arr = np.asarray(pd_cache)  # (885, 90, 6)
    valid = arr[..., 4] == 1
    print("[VERIFY] both-ones rows (must be 0):", int(((arr[..., 2] == 1) & (arr[..., 3] == 1) & valid).sum()))
    print("[VERIFY] 0D rows:", int(((arr[..., 2] == 1) & (arr[..., 3] == 0) & valid).sum()), "| 1D rows:",
          int(((arr[..., 2] == 0) & (arr[..., 3] == 1) & valid).sum()))

    class_counts = np.bincount(all_labels)
    class_weights = len(all_labels) / (len(class_names) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    existing_df = pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else pd.DataFrame()
    done_set = set(
        zip(existing_df["mode"], existing_df["fold"], existing_df["model_seed"])) if not existing_df.empty else set()
    results_rows = existing_df.to_dict("records") if not existing_df.empty else []

    t_start = time.time()
    for split_seed in SPLIT_SEEDS:
        folds = list(
            StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=split_seed).split(np.zeros(len(all_labels)),
                                                                                         all_labels))
        for mode in MODES:
            for fold_idx, (tr, va) in enumerate(folds):
                for model_seed in MODEL_SEEDS:
                    if (mode, fold_idx, model_seed) in done_set:
                        print(f"[SKIP] Completed Fold Detected: Mode={mode.upper()}, Fold={fold_idx}")
                        continue

                    print("\n" + "=" * 80)
                    print(f"🔥 V8 (BUG FIXED, 885 IMAGES) | MODE={mode.upper()} | FOLD {fold_idx + 1}/{FOLDS}")
                    print("=" * 80)

                    res = run_fold(fold_idx, tr, va, all_images, all_labels, class_names, pd_cache, mode, model_seed,
                                   class_weights_tensor)

                    cm_path = os.path.join(OUTPUT_DIR, f"cm_{mode}_f{fold_idx}_s{model_seed}.npy")
                    np.save(cm_path, np.array(res.pop("confusion_matrix")))

                    results_rows.append(res)
                    done_set.add((mode, fold_idx, model_seed))

                    pd.DataFrame(results_rows).to_csv(RESULTS_CSV, index=False)
                    print(f"--> Saved fold progress to {RESULTS_CSV}")

    print(f"\n[COMPLETE] V8 execution finished in {(time.time() - t_start) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()