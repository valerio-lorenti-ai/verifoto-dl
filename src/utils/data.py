import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_augmented_v6_dataset(root: str) -> pd.DataFrame:
    """
    Scansiona il dataset augmented_v6 e crea un DataFrame con metadati.
    
    Struttura attesa:
    augmented_v6/
        originali/
            buono/<food_category>/*.jpg
            cattivo/<food_category>/<defect_type>/*.jpg
        modificate/
            <food_category>/<defect_type>/<generator>/*.jpg
    
    Returns:
        DataFrame con colonne: path, label, source, quality, food_category, defect_type, generator
    """
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"Dataset root non esiste: {root}")
    
    records = []
    
    # Scansiona tutte le immagini
    for img_path in tqdm(list(rootp.rglob("*")), desc="Scanning dataset", leave=False):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        
        # Ottieni path relativo rispetto a root
        try:
            rel_path = img_path.relative_to(rootp)
        except ValueError:
            continue
        
        parts = rel_path.parts
        if len(parts) < 2:
            continue
        
        # Inizializza metadati
        meta = {
            "path": str(img_path),
            "label": None,
            "source": None,
            "quality": None,
            "food_category": None,
            "defect_type": None,
            "generator": None
        }
        
        # Determina source e label
        if parts[0] == "originali":
            meta["source"] = "originali"
            meta["label"] = 0  # NON_FRODE
            
            if len(parts) >= 2:
                meta["quality"] = parts[1]  # "buono" o "cattivo"
            
            if len(parts) >= 3:
                meta["food_category"] = parts[2]
            
            if len(parts) >= 4 and meta["quality"] == "cattivo":
                meta["defect_type"] = parts[3]
        
        elif parts[0] == "modificate":
            meta["source"] = "modificate"
            meta["label"] = 1  # FRODE
            
            if len(parts) >= 2:
                meta["food_category"] = parts[1]
            
            if len(parts) >= 3:
                meta["defect_type"] = parts[2]
            
            if len(parts) >= 4:
                meta["generator"] = parts[3]
        
        else:
            # Path non riconosciuto, skip
            continue
        
        if meta["label"] is not None:
            records.append(meta)
    
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        raise RuntimeError(f"Nessuna immagine trovata in {root}. Verifica la struttura del dataset.")
    
    print(f"\nDataset caricato: {len(df)} immagini")
    print(f"  - Originali (label=0): {(df['label'] == 0).sum()}")
    print(f"  - Modificate (label=1): {(df['label'] == 1).sum()}")
    print(f"  - Food categories: {df['food_category'].nunique()}")
    print(f"  - Defect types: {df['defect_type'].nunique()}")
    print(f"  - Generators: {df['generator'].nunique()}")
    
    return df


def find_class_dirs(root: str):
    """Legacy function for backward compatibility - now returns None, None"""
    print("Warning: find_class_dirs is deprecated for augmented_v6 dataset")
    return None, None


def list_images_in_dir(d: Path) -> List[str]:
    """Legacy function for backward compatibility"""
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
        pp.relative_to(non_dir)
        return 0
    except Exception:
        pass
    try:
        pp.relative_to(frode_dir)
        return 1
    except Exception:
        pass
    parent = pp.parent.name.lower()
    if "non" in parent or "real" in parent:
        return 0
    if "frode" in parent or "fake" in parent:
        return 1
    raise ValueError(f"Label non determinabile per: {p}")


def stratified_group_split_v6(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split stratificato per augmented_v6 dataset.
    Stratifica per label e food_category.
    
    Args:
        df: DataFrame con colonne path, label, food_category, etc.
        train_ratio, val_ratio, test_ratio: proporzioni split
        seed: random seed
    
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Crea stratification key: label + food_category
    df = df.copy()
    df['strat_key'] = df['label'].astype(str) + "_" + df['food_category'].fillna("unknown")
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Split per ogni gruppo
    for key, group in df.groupby('strat_key'):
        n = len(group)
        indices = group.index.tolist()
        
        # Shuffle
        rnd = random.Random(seed)
        rnd.shuffle(indices)
        
        # Split
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        train_dfs.append(df.loc[train_idx])
        val_dfs.append(df.loc[val_idx])
        test_dfs.append(df.loc[test_idx])
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Shuffle finale
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Rimuovi colonna temporanea
    train_df = train_df.drop(columns=['strat_key'])
    val_df = val_df.drop(columns=['strat_key'])
    test_df = test_df.drop(columns=['strat_key'])
    
    print(f"\nSplit completato:")
    print(f"  Train: {len(train_df)} ({train_df['label'].mean():.3f} pos rate)")
    print(f"  Val:   {len(val_df)} ({val_df['label'].mean():.3f} pos rate)")
    print(f"  Test:  {len(test_df)} ({test_df['label'].mean():.3f} pos rate)")
    
    return train_df, val_df, test_df


def stratified_group_split(group_items, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Legacy function for backward compatibility with old dataset structure"""
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
    val = v0 + v1
    test = te0 + te1
    rnd.shuffle(train)
    rnd.shuffle(val)
    rnd.shuffle(test)

    def flatten(split):
        paths, labels, gids = [], [], []
        for gid, y, ps in split:
            for p in ps:
                paths.append(p)
                labels.append(y)
                gids.append(gid)
        return paths, labels, gids

    return flatten(train), flatten(val), flatten(test)


class RandomJPEGCompression:
    def __init__(self, quality_min=55, quality_max=95, p=0.55):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        q = random.randint(self.quality_min, self.quality_max)
        arr = np.array(img.convert("RGB"))
        bgr = arr[:, :, ::-1]

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
        ok, enc = cv2.imencode(".jpg", bgr, encode_param)
        if not ok:
            return img

        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
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
        transforms.RandomHorizontalFlip(p=0.3),
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


class ImageBinaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, img_size=224):
        """
        Dataset per augmented_v6 con metadati.
        
        Args:
            df: DataFrame con colonne path, label, source, quality, food_category, defect_type, generator
            transform: torchvision transforms
            img_size: dimensione immagine per fallback
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = row['path']
        y = int(row['label'])
        
        # Carica immagine
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        # Metadati
        meta = {
            'path': fp,
            'source': row.get('source'),
            'quality': row.get('quality'),
            'food_category': row.get('food_category'),
            'defect_type': row.get('defect_type'),
            'generator': row.get('generator')
        }
        
        return img, torch.tensor(y, dtype=torch.float32), meta
