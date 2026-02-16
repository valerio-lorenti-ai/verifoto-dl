import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    def __init__(self, paths: List[str], labels: List[int], transform=None, img_size=224):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        fp = self.paths[idx]
        y = int(self.labels[idx])
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.float32), fp
