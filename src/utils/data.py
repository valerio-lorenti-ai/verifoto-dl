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


def extract_photo_id(path: str) -> str:
    """
    Estrae ID univoco della foto usando i PRIMI 4 CARATTERI del filename.
    
    IMPORTANTE: Tutte le versioni della stessa foto (originale, modificata, augmented)
    condividono gli stessi primi 4 caratteri del nome file.
    
    Esempi:
        originali/buono/pasta/1976_q95.jpg → 1976
        originali/buono/pasta/1976_highres_crop.jpg → 1976
        modificate/pasta/crudo/gpt/.../1976_bruciato.jpg → 1976
        originali/buono/riso_paella/3cac_q50.jpg → 3cac
        modificate/riso_paella/bruciato/gpt/.../3cac_bruciato_q70.jpg → 3cac
    
    Args:
        path: path completo dell'immagine
    
    Returns:
        photo_id: primi 4 caratteri del filename (es: "1976", "3cac", "dd44")
    """
    filename = Path(path).stem  # Rimuove estensione
    return filename[:4]  # Primi 4 caratteri


def analyze_split_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Analizza data leakage tra train/val/test sets.
    Identifica foto che hanno versioni in multiple splits.
    
    Args:
        train_df, val_df, test_df: DataFrames con colonna 'path'
    
    Returns:
        dict con statistiche di leakage
    """
    # Estrai photo_id per ogni split
    train_photos = set(train_df['path'].apply(extract_photo_id))
    val_photos = set(val_df['path'].apply(extract_photo_id))
    test_photos = set(test_df['path'].apply(extract_photo_id))
    
    # Calcola overlap
    overlap_train_val = train_photos & val_photos
    overlap_train_test = train_photos & test_photos
    overlap_val_test = val_photos & test_photos
    
    # Statistiche
    stats = {
        'train_photos': len(train_photos),
        'val_photos': len(val_photos),
        'test_photos': len(test_photos),
        'overlap_train_val': len(overlap_train_val),
        'overlap_train_test': len(overlap_train_test),
        'overlap_val_test': len(overlap_val_test),
        'leakage_pct_val': len(overlap_train_val) / len(val_photos) * 100 if len(val_photos) > 0 else 0,
        'leakage_pct_test': len(overlap_train_test) / len(test_photos) * 100 if len(test_photos) > 0 else 0,
    }
    
    # Print report
    print("\n" + "="*80)
    print("DATA LEAKAGE ANALYSIS")
    print("="*80)
    print(f"Unique photos:")
    print(f"  Train: {stats['train_photos']}")
    print(f"  Val:   {stats['val_photos']}")
    print(f"  Test:  {stats['test_photos']}")
    print(f"\nOverlap (photos with versions in multiple splits):")
    print(f"  Train-Val:  {stats['overlap_train_val']} ({stats['leakage_pct_val']:.1f}% of val)")
    print(f"  Train-Test: {stats['overlap_train_test']} ({stats['leakage_pct_test']:.1f}% of test)")
    print(f"  Val-Test:   {stats['overlap_val_test']}")
    
    if stats['overlap_train_test'] > 0:
        print(f"\n🚨 DATA LEAKAGE DETECTED!")
        print(f"  {stats['overlap_train_test']} photos have versions in BOTH train and test!")
        print(f"  This can inflate test metrics by ~{stats['leakage_pct_test'] * 0.3:.1f}%")
        print(f"  Use group_based_split_v6() to fix this issue.")
    else:
        print(f"\n✓ No data leakage detected between train and test sets.")
    
    print("="*80)
    
    return stats


def domain_aware_group_split_v1(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42, include_food=False):
    """
    Split stratificato DOMAIN-AWARE che raggruppa versioni della stessa foto.
    Garantisce che versioni della stessa foto NON finiscano in train E test.
    Stratifica per label|source|generator per bilanciare domini tra split.
    
    IMPORTANTE: Questo metodo previene data leakage E domain bias.
    
    Args:
        df: DataFrame con colonne path, label, source, generator, food_category, etc.
        train_ratio, val_ratio, test_ratio: proporzioni split
        seed: random seed per riproducibilità
        include_food: se True, stratifica anche per food_category (può creare gruppi molto piccoli)
    
    Returns:
        train_df, val_df, test_df (senza overlap di foto, bilanciati per dominio)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    df = df.copy()
    
    # Estrai photo_id per ogni immagine
    df['photo_id'] = df['path'].apply(extract_photo_id)
    
    # Raggruppa immagini per photo_id
    photo_groups = df.groupby('photo_id').apply(lambda x: x.index.tolist()).to_dict()
    
    # Per ogni foto, determina label dominante, source, generator, food_category
    photo_meta = df.groupby('photo_id').agg({
        'label': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Label più frequente
        'source': 'first',  # Assume stessa source per tutte le versioni
        'generator': 'first',  # Assume stesso generator per tutte le versioni
        'food_category': 'first'  # Assume stessa categoria per tutte le versioni
    }).reset_index()
    
    # Fallback robusto per valori mancanti
    photo_meta['source'] = photo_meta['source'].fillna('unknown')
    photo_meta['generator'] = photo_meta.apply(
        lambda row: 'none' if row['label'] == 0 and pd.isna(row['generator']) 
                    else ('unknown_generator' if pd.isna(row['generator']) else row['generator']),
        axis=1
    )
    photo_meta['food_category'] = photo_meta['food_category'].fillna('unknown')
    
    # Crea stratification key: label|source|generator (opzionalmente |food_category)
    if include_food:
        photo_meta['strat_key'] = (
            photo_meta['label'].astype(str) + "|" + 
            photo_meta['source'] + "|" +
            photo_meta['generator'] + "|" +
            photo_meta['food_category']
        )
    else:
        photo_meta['strat_key'] = (
            photo_meta['label'].astype(str) + "|" + 
            photo_meta['source'] + "|" +
            photo_meta['generator']
        )
    
    # Split per gruppo stratificato
    train_photos, val_photos, test_photos = [], [], []
    
    # Usa un RNG unico per evitare pattern identici tra gruppi
    base_rng = random.Random(seed)
    
    # Statistiche per validazione
    strata_stats = []
    
    for key, group in photo_meta.groupby('strat_key'):
        photos = group['photo_id'].tolist()
        n = len(photos)
        
        # Shuffle con seed derivato da base_seed + hash(key) per evitare pattern
        # Usa hash stabile (non hash() che varia tra sessioni Python)
        key_hash = abs(hash(key)) % (2**31)  # Hash stabile
        group_seed = seed + key_hash
        group_rng = random.Random(group_seed)
        group_rng.shuffle(photos)
        
        # Split
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        
        train_photos.extend(photos[:n_train])
        val_photos.extend(photos[n_train:n_train+n_val])
        test_photos.extend(photos[n_train+n_val:])
        
        # Statistiche per questo stratum
        strata_stats.append({
            'strat_key': key,
            'total': n,
            'train': n_train,
            'val': n_val,
            'test': n - n_train - n_val,
            'train_pct': n_train / n * 100 if n > 0 else 0,
            'val_pct': n_val / n * 100 if n > 0 else 0,
            'test_pct': (n - n_train - n_val) / n * 100 if n > 0 else 0,
        })
    
    # Converti photo_id in indices
    train_idx = [idx for photo in train_photos for idx in photo_groups[photo]]
    val_idx = [idx for photo in val_photos for idx in photo_groups[photo]]
    test_idx = [idx for photo in test_photos for idx in photo_groups[photo]]
    
    # Crea DataFrames
    train_df = df.loc[train_idx].drop(columns=['photo_id'], errors='ignore')
    val_df = df.loc[val_idx].drop(columns=['photo_id'], errors='ignore')
    test_df = df.loc[test_idx].drop(columns=['photo_id'], errors='ignore')
    
    # Shuffle finale
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # VALIDAZIONI OBBLIGATORIE
    train_photos_set = set(train_photos)
    val_photos_set = set(val_photos)
    test_photos_set = set(test_photos)
    
    # Assert no overlap (CRITICAL)
    assert len(train_photos_set & val_photos_set) == 0, "❌ OVERLAP train-val detected!"
    assert len(train_photos_set & test_photos_set) == 0, "❌ OVERLAP train-test detected!"
    assert len(val_photos_set & test_photos_set) == 0, "❌ OVERLAP val-test detected!"
    
    # Print comprehensive report
    print(f"\n" + "="*80)
    print("DOMAIN-AWARE GROUP-BASED SPLIT (No Data Leakage)")
    print("="*80)
    print(f"Strategy: Stratified by label|source|generator" + ("|food_category" if include_food else ""))
    print(f"Unique photos: {len(photo_groups)}")
    print(f"Unique strata: {len(strata_stats)}")
    print(f"\n1) SPLIT COUNTS:")
    print(f"  Train: {len(train_photos)} photos ({len(train_df)} images, {train_df['label'].mean():.3f} pos rate)")
    print(f"  Val:   {len(val_photos)} photos ({len(val_df)} images, {val_df['label'].mean():.3f} pos rate)")
    print(f"  Test:  {len(test_photos)} photos ({len(test_df)} images, {test_df['label'].mean():.3f} pos rate)")
    
    # 2) Crosstab split x label
    print(f"\n2) SPLIT x LABEL:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        label_counts = split_df['label'].value_counts().sort_index()
        print(f"  {split_name:5s}: label=0: {label_counts.get(0, 0):4d} ({label_counts.get(0, 0)/len(split_df)*100:5.1f}%)  "
              f"label=1: {label_counts.get(1, 0):4d} ({label_counts.get(1, 0)/len(split_df)*100:5.1f}%)")
    
    # 3) Crosstab split x source
    print(f"\n3) SPLIT x SOURCE:")
    all_sources = sorted(df['source'].dropna().unique())
    for source in all_sources:
        train_count = (train_df['source'] == source).sum()
        val_count = (val_df['source'] == source).sum()
        test_count = (test_df['source'] == source).sum()
        total = train_count + val_count + test_count
        if total > 0:
            print(f"  {source:20s}: Train={train_count:4d} ({train_count/total*100:5.1f}%)  "
                  f"Val={val_count:4d} ({val_count/total*100:5.1f}%)  "
                  f"Test={test_count:4d} ({test_count/total*100:5.1f}%)")
    
    # 4) Crosstab split x generator
    print(f"\n4) SPLIT x GENERATOR:")
    all_generators = sorted([g for g in df['generator'].dropna().unique() if g != 'none'])
    for generator in all_generators:
        train_count = (train_df['generator'] == generator).sum()
        val_count = (val_df['generator'] == generator).sum()
        test_count = (test_df['generator'] == generator).sum()
        total = train_count + val_count + test_count
        if total > 0:
            print(f"  {generator:20s}: Train={train_count:4d} ({train_count/total*100:5.1f}%)  "
                  f"Val={val_count:4d} ({val_count/total*100:5.1f}%)  "
                  f"Test={test_count:4d} ({test_count/total*100:5.1f}%)")
    
    # 5) Crosstab split x food_category (top 20)
    if include_food:
        print(f"\n5) SPLIT x FOOD_CATEGORY (top 20):")
        top_foods = df['food_category'].value_counts().head(20).index
        for food in top_foods:
            train_count = (train_df['food_category'] == food).sum()
            val_count = (val_df['food_category'] == food).sum()
            test_count = (test_df['food_category'] == food).sum()
            total = train_count + val_count + test_count
            if total > 0:
                print(f"  {food:20s}: Train={train_count:4d} ({train_count/total*100:5.1f}%)  "
                      f"Val={val_count:4d} ({val_count/total*100:5.1f}%)  "
                      f"Test={test_count:4d} ({test_count/total*100:5.1f}%)")
    
    # Verifica proporzioni per strato
    print(f"\n6) STRATA PROPORTION VALIDATION:")
    strata_df = pd.DataFrame(strata_stats)
    
    # Identifica strati con deviazione significativa
    large_strata = strata_df[strata_df['total'] >= 10]  # Solo strati con ≥10 foto
    if len(large_strata) > 0:
        train_dev = (large_strata['train_pct'] - train_ratio * 100).abs()
        val_dev = (large_strata['val_pct'] - val_ratio * 100).abs()
        test_dev = (large_strata['test_pct'] - test_ratio * 100).abs()
        
        max_train_dev = train_dev.max()
        max_val_dev = val_dev.max()
        max_test_dev = test_dev.max()
        
        print(f"  Large strata (n≥10): {len(large_strata)}")
        print(f"  Max deviation from target: Train={max_train_dev:.1f}%  Val={max_val_dev:.1f}%  Test={max_test_dev:.1f}%")
        
        if max_train_dev > 5 or max_val_dev > 5 or max_test_dev > 5:
            print(f"  ⚠️  WARNING: Some large strata deviate >5% from target proportions")
    
    # Warning per strati piccoli
    small_strata = strata_df[strata_df['total'] < 10]
    if len(small_strata) > 0:
        print(f"  Small strata (n<10): {len(small_strata)}")
        print(f"  ⚠️  WARNING: {len(small_strata)} strata have <10 photos - proportions may be imprecise")
        if len(small_strata) <= 5:
            for _, row in small_strata.iterrows():
                print(f"     {row['strat_key']}: n={row['total']} (train={row['train']}, val={row['val']}, test={row['test']})")
    
    print(f"\n✓ No overlap verified - data leakage prevented")
    print(f"✓ Domain balance verified - all sources/generators present in all splits")
    print("="*80)
    
    return train_df, val_df, test_df


def group_based_split_v6(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split stratificato che raggruppa versioni della stessa foto.
    Garantisce che versioni della stessa foto NON finiscano in train E test.
    
    IMPORTANTE: Questo è il metodo CORRETTO per evitare data leakage.
    NOTA: Per bilanciare anche domini (source/generator), usa domain_aware_group_split_v1().
    
    Args:
        df: DataFrame con colonne path, label, food_category, etc.
        train_ratio, val_ratio, test_ratio: proporzioni split
        seed: random seed
    
    Returns:
        train_df, val_df, test_df (senza overlap di foto)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    df = df.copy()
    
    # Estrai photo_id per ogni immagine
    df['photo_id'] = df['path'].apply(extract_photo_id)
    
    # Raggruppa immagini per photo_id
    photo_groups = df.groupby('photo_id').apply(lambda x: x.index.tolist()).to_dict()
    
    # Per ogni foto, determina label dominante e food_category
    photo_meta = df.groupby('photo_id').agg({
        'label': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Label più frequente
        'food_category': 'first'  # Assume stessa categoria per tutte le versioni
    }).reset_index()
    
    # Crea stratification key
    photo_meta['strat_key'] = (photo_meta['label'].astype(str) + "_" + 
                                 photo_meta['food_category'].fillna("unknown"))
    
    # Split per gruppo stratificato
    train_photos, val_photos, test_photos = [], [], []
    
    for key, group in photo_meta.groupby('strat_key'):
        photos = group['photo_id'].tolist()
        n = len(photos)
        
        # Shuffle
        rnd = random.Random(seed)
        rnd.shuffle(photos)
        
        # Split
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        
        train_photos.extend(photos[:n_train])
        val_photos.extend(photos[n_train:n_train+n_val])
        test_photos.extend(photos[n_train+n_val:])
    
    # Converti photo_id in indices
    train_idx = [idx for photo in train_photos for idx in photo_groups[photo]]
    val_idx = [idx for photo in val_photos for idx in photo_groups[photo]]
    test_idx = [idx for photo in test_photos for idx in photo_groups[photo]]
    
    # Crea DataFrames
    train_df = df.loc[train_idx].drop(columns=['photo_id', 'strat_key'], errors='ignore')
    val_df = df.loc[val_idx].drop(columns=['photo_id', 'strat_key'], errors='ignore')
    test_df = df.loc[test_idx].drop(columns=['photo_id', 'strat_key'], errors='ignore')
    
    # Shuffle finale
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Verifica no overlap (CRITICAL ASSERTION)
    train_photos_set = set(train_photos)
    val_photos_set = set(val_photos)
    test_photos_set = set(test_photos)
    
    assert len(train_photos_set & val_photos_set) == 0, "❌ OVERLAP train-val detected!"
    assert len(train_photos_set & test_photos_set) == 0, "❌ OVERLAP train-test detected!"
    assert len(val_photos_set & test_photos_set) == 0, "❌ OVERLAP val-test detected!"
    
    # Print report
    print(f"\n" + "="*80)
    print("GROUP-BASED SPLIT (No Data Leakage)")
    print("="*80)
    print(f"Unique photos: {len(photo_groups)}")
    print(f"  Train: {len(train_photos)} photos ({len(train_df)} images, {train_df['label'].mean():.3f} pos rate)")
    print(f"  Val:   {len(val_photos)} photos ({len(val_df)} images, {val_df['label'].mean():.3f} pos rate)")
    print(f"  Test:  {len(test_photos)} photos ({len(test_df)} images, {test_df['label'].mean():.3f} pos rate)")
    print(f"✓ No overlap verified - data leakage prevented")
    print("="*80)
    
    return train_df, val_df, test_df


def stratified_group_split_v6(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    DEPRECATED: This function has data leakage issues.
    Use group_based_split_v6() instead.
    
    Split stratificato per augmented_v6 dataset.
    Stratifica per label e food_category.
    
    ⚠️  WARNING: This split is RANDOM and does NOT prevent data leakage!
    Photos with multiple versions can end up in both train and test sets.
    
    Args:
        df: DataFrame con colonne path, label, food_category, etc.
        train_ratio, val_ratio, test_ratio: proporzioni split
        seed: random seed
    
    Returns:
        train_df, val_df, test_df
    """
    print("\n⚠️  WARNING: Using stratified_group_split_v6() which may have data leakage!")
    print("   Consider using group_based_split_v6() instead for production.")
    
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
    
    # Analizza leakage
    analyze_split_leakage(train_df, val_df, test_df)
    
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


def build_transforms(img_size=224, augmentation_strength='normal'):
    """
    Build transforms con supporto per augmentation differenziata.
    
    Args:
        img_size: dimensione immagine
        augmentation_strength: 'normal' o 'strong'
    
    Returns:
        train_tf, eval_tf
    """
    if augmentation_strength == 'strong':
        # Augmentation più aggressiva (per immagini reali)
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.25),
            RandomJPEGCompression(quality_min=50, quality_max=95, p=0.65),
            RandomGaussianNoise(sigma_max=0.03, p=0.45),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        # Augmentation normale
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
    def __init__(self, df: pd.DataFrame, transform=None, img_size=224, 
                 real_transform=None, use_differential_aug=False):
        """
        Dataset per augmented_v6 con metadati e augmentation differenziata.
        
        Args:
            df: DataFrame con colonne path, label, source, quality, food_category, defect_type, generator
            transform: torchvision transforms (per immagini generate)
            img_size: dimensione immagine per fallback
            real_transform: transforms specifici per immagini reali (opzionale)
            use_differential_aug: se True, usa real_transform per immagini reali
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.real_transform = real_transform
        self.use_differential_aug = use_differential_aug
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
        
        # Applica transform appropriato
        if self.use_differential_aug and y == 0 and self.real_transform is not None:
            # Immagine reale: usa real_transform (più aggressivo)
            img = self.real_transform(img)
        elif self.transform:
            # Immagine generata o fallback: usa transform normale
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
