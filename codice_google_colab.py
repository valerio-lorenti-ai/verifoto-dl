import os, sys, math, time, random, shutil, json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)

from tqdm.auto import tqdm
import cv2
import numpy as np
from PIL import Image
import random



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)



import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



USE_DRIVE = True  # False se usi /content

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")



# >>> IMPOSTA QUI <<<
DATASET_ROOT = "/content/drive/MyDrive/DatasetVerifoto/images/exp_3_augmented_v6.1" if USE_DRIVE else "/content/dataset"
print("DATASET_ROOT:", DATASET_ROOT)



IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def find_class_dirs(root: str):
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"Dataset root non esiste: {root}")

    class_alias = {
        "NON_FRODE": ["NON_FRODE", "non_frode", "real", "REAL", "authentic", "ok"],
        "FRODE": ["FRODE", "frode", "fake", "FAKE", "manipulated", "fraud"],
    }

    def match_class_dir(parent: Path, aliases: List[str]) -> Optional[Path]:
        for a in aliases:
            p = parent / a
            if p.exists() and p.is_dir():
                return p
        return None

    candidates = [rootp / "images", rootp]
    for sp in ["train", "val", "valid", "validation", "test"]:
        candidates += [rootp / sp, rootp / sp / "images"]

    found = {}
    for canon, aliases in class_alias.items():
        found[canon] = None
        for base in candidates:
            d = match_class_dir(base, aliases)
            if d is not None:
                found[canon] = d
                break
    return found["NON_FRODE"], found["FRODE"]

def list_images_in_dir(d: Path) -> List[str]:
    paths = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(str(p))
    return sorted(paths)

def ahash(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)
    mean = arr.mean()
    bits = (arr > mean).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def compute_hashes(paths: List[str], hash_size: int = 8) -> Dict[str, Optional[int]]:
    out = {}
    for fp in tqdm(paths, desc="Hashing images", leave=False):
        try:
            img = Image.open(fp).convert("RGB")
            out[fp] = ahash(img, hash_size=hash_size)
        except Exception:
            out[fp] = None
    return out

def group_near_duplicates(paths: List[str], hashes: Dict[str, int], max_hamming: int = 4) -> Dict[int, List[str]]:
    valid = [p for p in paths if hashes.get(p) is not None]
    used = set()
    groups = {}
    gid = 0
    for i, p in enumerate(valid):
        if p in used:
            continue
        used.add(p)
        groups[gid] = [p]
        hp = hashes[p]
        for q in valid[i+1:]:
            if q in used:
                continue
            hq = hashes[q]
            if hamming(hp, hq) <= max_hamming:
                used.add(q)
                groups[gid].append(q)
        gid += 1
    for p in paths:
        if hashes.get(p) is None:
            groups[gid] = [p]
            gid += 1
    return groups

def label_of_path(p: str, non_dir: Path, frode_dir: Path) -> int:
    pp = Path(p)
    try:
        pp.relative_to(non_dir); return 0
    except Exception:
        pass
    try:
        pp.relative_to(frode_dir); return 1
    except Exception:
        pass
    parent = pp.parent.name.lower()
    if "non" in parent or "real" in parent: return 0
    if "frode" in parent or "fake" in parent: return 1
    raise ValueError(f"Label non determinabile per: {p}")

def stratified_group_split(group_items, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rnd = random.Random(seed)

    by_class = {0: [], 1: []}
    for gid, y, paths in group_items:
        by_class[y].append((gid, y, paths))
    for c in by_class:
        rnd.shuffle(by_class[c])

    def take(items):
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train = items[:n_train]
        val = items[n_train:n_train+n_val]
        test = items[n_train+n_val:]
        return train, val, test

    t0, v0, te0 = take(by_class[0])
    t1, v1, te1 = take(by_class[1])

    train = t0 + t1
    val   = v0 + v1
    test  = te0 + te1
    rnd.shuffle(train); rnd.shuffle(val); rnd.shuffle(test)

    def flatten(split):
        paths, labels, gids = [], [], []
        for gid, y, ps in split:
            for p in ps:
                paths.append(p); labels.append(y); gids.append(gid)
        return paths, labels, gids

    return flatten(train), flatten(val), flatten(test)





non_dir, frode_dir = find_class_dirs(DATASET_ROOT)
if non_dir is None or frode_dir is None:
    raise RuntimeError(
        "Non trovo NON_FRODE e FRODE (o alias real/fake). "
        "Atteso: dataset/images/NON_FRODE e dataset/images/FRODE (o simili)."
    )

non_paths = list_images_in_dir(non_dir)
fro_paths = list_images_in_dir(frode_dir)

print("NON_FRODE:", len(non_paths), "| FRODE:", len(fro_paths), "| Tot:", len(non_paths) + len(fro_paths))
if len(non_paths) == 0 or len(fro_paths) == 0:
    raise RuntimeError("Una delle classi è vuota. Controlla la struttura delle cartelle.")



all_paths = non_paths + fro_paths
hashes = compute_hashes(all_paths, hash_size=8)
groups = group_near_duplicates(all_paths, hashes, max_hamming=4)

group_items = []
mixed = 0
for gid, paths in groups.items():
    ys = [label_of_path(p, non_dir, frode_dir) for p in paths]
    if len(set(ys)) > 1:
        mixed += 1
    y = int(np.round(np.mean(ys)))
    group_items.append((gid, y, paths))

print("Gruppi:", len(group_items), "| Gruppi label mista:", mixed)

(train_paths, train_y, train_gids), (val_paths, val_y, val_gids), (test_paths, test_y, test_gids) = stratified_group_split(
    group_items, 0.70, 0.15, 0.15, seed=42
)

print("Split sizes:", len(train_paths), len(val_paths), len(test_paths))
print("Train pos rate:", np.mean(train_y), "| Val:", np.mean(val_y), "| Test:", np.mean(test_y))

assert set(train_gids).isdisjoint(set(val_gids))
assert set(train_gids).isdisjoint(set(test_gids))
assert set(val_gids).isdisjoint(set(test_gids))



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

class RandomJPEGCompression:
    """Simula ricompressione JPEG in modo stabile (usa OpenCV, ok con DataLoader workers)."""
    def __init__(self, quality_min=55, quality_max=95, p=0.55):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        q = random.randint(self.quality_min, self.quality_max)

        # PIL -> numpy (RGB)
        arr = np.array(img.convert("RGB"))
        # RGB -> BGR (OpenCV)
        bgr = arr[:, :, ::-1]

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
        ok, enc = cv2.imencode(".jpg", bgr, encode_param)
        if not ok:
            return img

        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)  # BGR
        rgb = dec[:, :, ::-1]
        return Image.fromarray(rgb)

class RandomGaussianNoise:
    def __init__(self, sigma_min=0.0, sigma_max=0.02, p=0.35):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p
    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32) / 255.0
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))

def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.80, 1.0), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(p=0.3),  # metti 0.0 se non lo vuoi
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
        RandomJPEGCompression(p=0.55),
        RandomGaussianNoise(p=0.35),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf

train_tf, eval_tf = build_transforms(IMG_SIZE)

class ImageBinaryDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        fp = self.paths[idx]
        y = int(self.labels[idx])
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.float32), fp

train_ds = ImageBinaryDataset(train_paths, train_y, transform=train_tf)
val_ds   = ImageBinaryDataset(val_paths, val_y, transform=eval_tf)
test_ds  = ImageBinaryDataset(test_paths, test_y, transform=eval_tf)

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

train_pos = sum(train_y)
train_neg = len(train_y) - train_pos
pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32).to(DEVICE)
print("Train neg:", train_neg, "Train pos:", train_pos, "pos_weight:", pos_weight.item())



MODEL_NAME = "efficientnet_b0"   # es: "convnext_tiny", "resnet50"
DROP_RATE = 0.2

def build_model(model_name: str = "efficientnet_b0", pretrained=True, drop_rate=0.2) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)

def set_backbone_trainable(m: nn.Module, trainable: bool):
    for _, p in m.named_parameters():
        p.requires_grad = trainable
    # sblocca sempre la head
    for head_name in ["classifier", "fc", "head"]:
        if hasattr(m, head_name):
            for p in getattr(m, head_name).parameters():
                p.requires_grad = True

model = build_model(MODEL_NAME, pretrained=True, drop_rate=DROP_RATE).to(DEVICE)
set_backbone_trainable(model, trainable=False)  # phase 1
print("Model:", MODEL_NAME)



@dataclass
class TrainConfig:
    epochs_head: int = 5
    epochs_finetune: int = 25
    lr_head: float = 3e-4
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-3
    patience: int = 6
    monitor: str = "pr_auc"   # "f1" o "pr_auc"
    min_delta: float = 1e-4
    max_grad_norm: float = 1.0

cfg = TrainConfig()

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def sigmoid_np(x): return 1 / (1 + np.exp(-x))

@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader):
    model.eval()
    logits_all, y_all, paths_all = [], [], []
    for x, y, fp in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x).squeeze(1).detach().cpu().numpy()
        logits_all.append(logits)
        y_all.append(y.numpy())
        paths_all.extend(fp)
    logits_all = np.concatenate(logits_all)
    y_all = np.concatenate(y_all).astype(np.int32)
    probs = sigmoid_np(logits_all)
    return probs, y_all, paths_all

def compute_metrics_from_probs(probs: np.ndarray, y_true: np.ndarray, threshold=0.5):
    y_pred = (probs >= threshold).astype(np.int32)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    roc_auc = np.nan
    pr_auc = np.nan
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "prec_macro": prec_m, "rec_macro": rec_m, "f1_macro": f1_m,
        "roc_auc": roc_auc, "pr_auc": pr_auc
    }

def train_one_epoch(model, loader, optimizer, scheduler=None, max_grad_norm=1.0):
    model.train()
    losses = []
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan

@torch.no_grad()
def validate(model, loader, threshold=0.5):
    probs, y_true, _ = predict_proba(model, loader)
    return compute_metrics_from_probs(probs, y_true, threshold=threshold)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.bad = 0
    def step(self, value):
        if self.best is None:
            self.best = value
            return False, True
        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.bad = 0
            return False, True
        self.bad += 1
        return self.bad >= self.patience, False

OUT_DIR = Path("/content/verifoto_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR = OUT_DIR / time.strftime("%Y%m%d-%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("RUN_DIR:", RUN_DIR)

def save_checkpoint(model, path: Path, best_metric: float = None, cfg: dict = None):
    payload = {
        "state_dict": model.state_dict(),
        "best_metric": float(best_metric) if best_metric is not None else None,
        "cfg": cfg if cfg is not None else None,
    }
    torch.save(payload, str(path))


def fit(model, train_loader, val_loader, cfg: TrainConfig):
    history = []
    best_metric = -1e9
    best_path = RUN_DIR / "best.pt"

    # Phase 1: head-only
    set_backbone_trainable(model, trainable=False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    total_steps = max(cfg.epochs_head * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    es = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta, mode="max")

    print("\n=== Phase 1: head-only ===")
    for epoch in range(1, cfg.epochs_head + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg.max_grad_norm)
        val_m = validate(model, val_loader, threshold=0.5)
        monitor_val = val_m[cfg.monitor]
        stop, improved = es.step(monitor_val)

        history.append({"phase":"head","epoch":epoch,"train_loss":tr_loss,**val_m})
        print(f"[Head {epoch}/{cfg.epochs_head}] loss={tr_loss:.4f} val_{cfg.monitor}={monitor_val:.4f} val_f1={val_m['f1']:.4f}")

        if improved and monitor_val > best_metric:
            best_metric = monitor_val
            save_checkpoint(model, best_path, best_metric=best_metric, cfg=cfg.__dict__)
        if stop:
            print("Early stopping (head).")
            break

    # Phase 2: fine-tune all
    set_backbone_trainable(model, trainable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
    total_steps = max(cfg.epochs_finetune * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    es = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta, mode="max")

    print("\n=== Phase 2: finetune-all ===")
    for epoch in range(1, cfg.epochs_finetune + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg.max_grad_norm)
        val_m = validate(model, val_loader, threshold=0.5)
        monitor_val = val_m[cfg.monitor]
        stop, improved = es.step(monitor_val)

        history.append({"phase":"finetune","epoch":epoch,"train_loss":tr_loss,**val_m})
        print(f"[FT {epoch}/{cfg.epochs_finetune}] loss={tr_loss:.4f} val_{cfg.monitor}={monitor_val:.4f} val_f1={val_m['f1']:.4f}")

        if improved and monitor_val > best_metric:
            best_metric = monitor_val
            save_checkpoint(model, best_path, best_metric=best_metric, cfg=cfg.__dict__)
        if stop:
            print("Early stopping (finetune).")
            break

    print("\nBest val", cfg.monitor, "=", best_metric)
    return history, best_path



history, best_ckpt = fit(model, train_loader, val_loader, cfg)

ckpt = torch.load(best_ckpt, map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("Loaded best checkpoint:", best_ckpt)




@torch.no_grad()
def evaluate_full(model, loader, threshold=0.5, title=""):
    probs, y_true, paths = predict_proba(model, loader)
    y_pred = (probs >= threshold).astype(np.int32)

    metrics = compute_metrics_from_probs(probs, y_true, threshold=threshold)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    print("\n" + "="*60)
    if title:
        print(title)
        print("-"*60)

    print(f"Threshold = {threshold:.2f}")
    for k in ["acc","prec","rec","f1","prec_macro","rec_macro","f1_macro","roc_auc","pr_auc"]:
        v = metrics[k]
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            print(f"{k:>10}: NaN")
        else:
            print(f"{k:>10}: {v:.4f}")

    print("\nConfusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(cm)
    return probs, y_true, paths, metrics, cm

def plot_prob_distributions(probs, y_true, bins=18, title="Predicted probability distributions"):
    p0 = probs[y_true == 0]
    p1 = probs[y_true == 1]
    plt.figure(figsize=(7,4))
    plt.hist(p0, bins=bins, alpha=0.7, label="NON_FRODE (0)")
    plt.hist(p1, bins=bins, alpha=0.7, label="FRODE (1)")
    plt.xlabel("Predicted P(FRODE)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_roc_pr(probs, y_true):
    if len(np.unique(y_true)) < 2:
        print("ROC/PR curve non disponibili: una sola classe nel set.")
        return
    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC curve (AUC={roc_auc:.3f})")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (AP/PR-AUC={pr_auc:.3f})")
    plt.grid(True, alpha=0.3)
    plt.show()

def threshold_sweep(probs, y_true, thresholds):
    print("\nThreshold sweep:")
    print("thr | precision | recall | f1 | pr_auc")
    for t in thresholds:
        m = compute_metrics_from_probs(probs, y_true, threshold=float(t))
        print(f"{t:>3.2f} | {m['prec']:>9.3f} | {m['rec']:>6.3f} | {m['f1']:>5.3f} | {m['pr_auc']:>6.3f}")

def evaluate_at_threshold(threshold: float):
    return evaluate_full(model, test_loader, threshold=threshold, title=f"TEST @ threshold={threshold:.2f}")



test_probs, test_true, test_paths, test_metrics, test_cm = evaluate_full(
    model, test_loader, threshold=0.5, title="TEST SET RESULTS"
)

plot_prob_distributions(test_probs, test_true, title="TEST: predicted P(FRODE)")
plot_roc_pr(test_probs, test_true)

threshold_sweep(test_probs, test_true, thresholds=np.linspace(0.1, 0.9, 9))
_ = evaluate_at_threshold(0.75)



def find_last_conv_layer(model: nn.Module) -> str:
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
    return last_name

TARGET_LAYER_NAME = find_last_conv_layer(model)
print("Grad-CAM target layer:", TARGET_LAYER_NAME)

class GradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _get_layer(self):
        layer = dict(self.model.named_modules()).get(self.target_layer_name, None)
        if layer is None:
            raise ValueError(f"Layer non trovato: {self.target_layer_name}")
        return layer

    def _register_hooks(self):
        layer = self._get_layer()

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(layer.register_full_backward_hook(bwd_hook))

    def remove(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x).squeeze(1)
        score = logits.sum()
        score.backward(retain_graph=True)

        acts = self.activations
        grads = self.gradients
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)

        cams = []
        for i in range(cam.shape[0]):
            c = cam[i]
            c = (c - c.min()) / (c.max() - c.min() + 1e-8)
            cams.append(c.detach().cpu().numpy())
        return np.stack(cams, axis=0)

def denorm_tensor(t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean = torch.tensor(mean).view(3,1,1)
    std = torch.tensor(std).view(3,1,1)
    x = t.cpu() * std + mean
    return torch.clamp(x, 0, 1)

def show_gradcam_examples(model, dataset: Dataset, n=5):
    cammer = GradCAM(model, TARGET_LAYER_NAME)
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    shown = 0
    plt.figure(figsize=(10, 2*n))

    for x, y, _ in loader:
        x = x.to(DEVICE)
        with torch.no_grad():
            logit = model(x).item()
            prob = float(1/(1+math.exp(-logit)))

        cam = cammer(x)[0]
        x0 = denorm_tensor(x[0])
        img = np.transpose(x0.numpy(), (1,2,0))

        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
        cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0

        overlay = img.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] * 0.6 + cam_arr * 0.6, 0, 1)

        ax1 = plt.subplot(n, 3, shown*3 + 1)
        ax1.imshow(img); ax1.axis("off")
        ax1.set_title(f"Orig | y={int(y.item())}")

        ax2 = plt.subplot(n, 3, shown*3 + 2)
        ax2.imshow(cam_arr, cmap="gray"); ax2.axis("off")
        ax2.set_title("CAM")

        ax3 = plt.subplot(n, 3, shown*3 + 3)
        ax3.imshow(overlay); ax3.axis("off")
        ax3.set_title(f"Overlay | P(FRODE)={prob:.2f}")

        shown += 1
        if shown >= n:
            break

    plt.tight_layout()
    plt.show()
    cammer.remove()

    
EXPORT_DIR = (RUN_DIR / "export")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1) state_dict
state_path = EXPORT_DIR / f"{MODEL_NAME}_state_dict.pt"
torch.save(model.state_dict(), str(state_path))

# 2) full best ckpt
full_ckpt_path = EXPORT_DIR / f"{MODEL_NAME}_best_full.pt"
shutil.copy(str(best_ckpt), str(full_ckpt_path))

# 3) TorchScript
example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
model.eval()
traced = torch.jit.trace(model, example)
ts_path = EXPORT_DIR / f"{MODEL_NAME}_traced.ts"
traced.save(str(ts_path))

# 4) ONNX (opzionale)
DO_ONNX = True
if DO_ONNX:
    onnx_path = EXPORT_DIR / f"{MODEL_NAME}.onnx"
    torch.onnx.export(
        model, example, str(onnx_path),
        input_names=["input"], output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )

print("Saved files in:", EXPORT_DIR)
print([p.name for p in EXPORT_DIR.iterdir()])

print("\nFINAL TEST SUMMARY @0.50")
_ = evaluate_at_threshold(0.50)

print("\nConservative thresholds (meno falsi positivi, più 'inconclusive' in prodotto):")
for t in [0.70, 0.80, 0.90]:
    _, _, _, m, cm = evaluate_at_threshold(t)
    fp = cm[0,1]
    fn = cm[1,0]
    print(f"thr={t:.2f} -> FP={fp}, FN={fn}, precision={m['prec']:.3f}, recall={m['rec']:.3f}")