# Git Results Management Guide

**Data**: 2026-02-17  
**Status**: ✅ CONFIGURED

---

## 🎯 OBIETTIVO

Permettere il push dei risultati da Colab a GitHub mantenendo il repository leggero.

---

## ✅ CONFIGURAZIONE COMPLETATA

### .gitignore Aggiornato

**PRIMA** (tutto ignorato):
```gitignore
outputs/runs/*
!outputs/runs/.gitkeep
```

**DOPO** (risultati tracciati, modelli ignorati):
```gitignore
# Allow all files in outputs/runs/
!outputs/runs/

# Ignore heavy model files
outputs/runs/**/*.pt
outputs/runs/**/*.pth
outputs/runs/**/*.ckpt

# Allow important result files
!outputs/runs/**/metrics.json
!outputs/runs/**/predictions.csv
!outputs/runs/**/group_metrics_*.csv
!outputs/runs/**/*.png
```

---

## 📦 COSA VIENE TRACCIATO

### ✅ File Committati (Leggeri)

| File | Dimensione | Descrizione |
|------|-----------|-------------|
| `metrics.json` | ~5 KB | Metriche complete + config |
| `predictions.csv` | ~50-200 KB | Predizioni con metadati |
| `group_metrics_food.csv` | ~5-20 KB | Performance per categoria cibo |
| `group_metrics_defect.csv` | ~5-20 KB | Performance per tipo difetto |
| `group_metrics_generator.csv` | ~5-20 KB | Performance per generatore |
| `group_metrics_quality.csv` | ~5-20 KB | Performance per qualità |
| `top_false_positives.csv` | ~10-50 KB | Top 50 FP |
| `top_false_negatives.csv` | ~10-50 KB | Top 50 FN |
| `notes.md` | ~2-5 KB | Riepilogo run |
| `CRITICAL_ANALYSIS.md` | ~5-20 KB | Analisi dettagliata |
| `cm.png` | ~50-200 KB | Confusion matrix |
| `prob_dist.png` | ~50-200 KB | Distribuzioni probabilità |
| `roc_curve.png` | ~50-200 KB | ROC curve |
| `pr_curve.png` | ~50-200 KB | Precision-Recall curve |

**Totale per run**: ~500 KB - 2 MB

### ❌ File Ignorati (Pesanti)

| File | Dimensione | Dove Salvato |
|------|-----------|--------------|
| `best.pt` | ~50-200 MB | Google Drive |
| `*.ckpt` | ~50-200 MB | Google Drive |
| `*.pth` | ~50-200 MB | Google Drive |
| `*.h5` | ~50-200 MB | Google Drive |

---

## 🚀 WORKFLOW SU COLAB

### Step 1: Training Completo

```python
# Training normale
!python -m src.train \
  --config configs/baseline.yaml \
  --run_name "2026-02-17_no_leakage_v1" \
  --checkpoint_dir "/content/drive/MyDrive/verifoto_checkpoints"
```

### Step 2: Verifica Risultati

```python
# Verifica che i file siano stati creati
import os
run_dir = "outputs/runs/2026-02-17_no_leakage_v1"
files = os.listdir(run_dir)
print(f"Files created: {len(files)}")
for f in files:
    size = os.path.getsize(os.path.join(run_dir, f))
    print(f"  {f}: {size/1024:.1f} KB")
```

### Step 3: Push su GitHub

```python
# Configura git
!git config user.email "colab@verifoto.ai"
!git config user.name "Colab Training"

# Add results (force per override .gitignore se necessario)
!git add outputs/runs/2026-02-17_no_leakage_v1

# Verifica cosa verrà committato
!git status

# Commit
!git commit -m "Add training results: 2026-02-17_no_leakage_v1"

# Push (usa il tuo GitHub token)
GITHUB_TOKEN = "ghp_your_token_here"
!git push https://{GITHUB_TOKEN}@github.com/valerio-lorenti-ai/verifoto-dl.git main
```

### Step 4: Backup su Drive (Opzionale)

```python
# Copia anche su Drive per sicurezza
!cp -r outputs/runs/2026-02-17_no_leakage_v1 \
  /content/drive/MyDrive/verifoto_results/
```

---

## 📊 ESEMPIO COMPLETO

### Struttura Directory Dopo Training

```
outputs/runs/2026-02-17_no_leakage_v1/
├── metrics.json                    ✅ 5 KB    → Git
├── predictions.csv                 ✅ 150 KB  → Git
├── group_metrics_food.csv          ✅ 8 KB    → Git
├── group_metrics_defect.csv        ✅ 6 KB    → Git
├── group_metrics_generator.csv     ✅ 4 KB    → Git
├── group_metrics_quality.csv       ✅ 5 KB    → Git
├── top_false_positives.csv         ✅ 25 KB   → Git
├── top_false_negatives.csv         ✅ 15 KB   → Git
├── notes.md                        ✅ 3 KB    → Git
├── cm.png                          ✅ 120 KB  → Git
├── prob_dist.png                   ✅ 180 KB  → Git
├── roc_curve.png                   ✅ 150 KB  → Git
└── pr_curve.png                    ✅ 140 KB  → Git

Total: ~811 KB → Git ✅
```

### Checkpoint su Drive

```
/content/drive/MyDrive/verifoto_checkpoints/2026-02-17_no_leakage_v1/
└── best.pt                         ❌ 85 MB   → Drive only
```

---

## 🔍 VERIFICA CONFIGURAZIONE

### Test 1: Verifica File Tracciati

```bash
# In locale o Colab
git check-ignore outputs/runs/test/metrics.json
# Output: (nessun output = file tracciato) ✅

git check-ignore outputs/runs/test/predictions.csv
# Output: (nessun output = file tracciato) ✅

git check-ignore outputs/runs/test/cm.png
# Output: (nessun output = file tracciato) ✅
```

### Test 2: Verifica File Ignorati

```bash
git check-ignore outputs/runs/test/best.pt
# Output: outputs/runs/test/best.pt ✅ (ignorato)

git check-ignore outputs/runs/test/model.ckpt
# Output: outputs/runs/test/model.ckpt ✅ (ignorato)
```

### Test 3: Dimensione Commit

```bash
# Dopo git add
git diff --cached --stat
# Dovrebbe mostrare ~500KB-2MB totale
```

---

## ⚠️ TROUBLESHOOTING

### Problema: File Non Tracciati

**Sintomo**: `git status` non mostra i file in `outputs/runs/`

**Soluzione**:
```bash
# Verifica .gitignore
cat .gitignore | grep outputs

# Forza add se necessario
git add -f outputs/runs/your_run_name/
```

### Problema: File Troppo Grandi

**Sintomo**: `git push` fallisce con "file too large"

**Soluzione**:
```bash
# Identifica file grandi
find outputs/runs/ -type f -size +10M

# Rimuovi file grandi
git rm --cached outputs/runs/*/best.pt

# Aggiungi a .gitignore se non già presente
echo "outputs/runs/**/*.pt" >> .gitignore
```

### Problema: Push Lento

**Sintomo**: Push impiega troppo tempo

**Causa**: Troppi file PNG o CSV grandi

**Soluzione**:
```bash
# Comprimi PNG prima del commit
find outputs/runs/ -name "*.png" -exec optipng {} \;

# Oppure riduci dimensione immagini
mogrify -resize 1024x1024 outputs/runs/*/*.png
```

---

## 📋 BEST PRACTICES

### 1. Commit Dopo Ogni Training

```bash
# Subito dopo training
git add outputs/runs/{EXPERIMENT_NAME}
git commit -m "Add training results: {EXPERIMENT_NAME}"
git push
```

### 2. Usa Nomi Descrittivi

```
✅ GOOD:
- 2026-02-17_no_leakage_v1
- 2026-02-17_efficientnet_b1_augmented
- 2026-02-17_threshold_0.7_test

❌ BAD:
- test1
- run_final
- new_model
```

### 3. Aggiungi Analisi Critica

```bash
# Crea CRITICAL_ANALYSIS.md per run importanti
echo "# Critical Analysis" > outputs/runs/{RUN}/CRITICAL_ANALYSIS.md
# Aggiungi analisi dettagliata
git add outputs/runs/{RUN}/CRITICAL_ANALYSIS.md
```

### 4. Backup Multipli

```
1. Git: Risultati leggeri (metrics, CSV, PNG)
2. Drive: Checkpoint pesanti (*.pt)
3. Locale: Copia completa per analisi
```

### 5. Cleanup Periodico

```bash
# Rimuovi run vecchi localmente (ma restano su git)
rm -rf outputs/runs/2026-01-*

# Pull per recuperare se necessario
git pull
```

---

## 📊 STORAGE STRATEGY

### Git (GitHub)
- **Cosa**: Metrics, CSV, visualizzazioni
- **Dimensione**: ~500KB-2MB per run
- **Limite**: 100MB per file, 1GB per repo (soft limit)
- **Vantaggi**: Versioning, sharing, history

### Google Drive
- **Cosa**: Model checkpoints
- **Dimensione**: ~50-200MB per checkpoint
- **Limite**: 15GB free, illimitato con workspace
- **Vantaggi**: Storage illimitato, veloce da Colab

### Locale
- **Cosa**: Tutto (per analisi)
- **Dimensione**: ~50-200MB per run completo
- **Vantaggi**: Veloce, offline access

---

## ✅ CHECKLIST

Prima di pushare risultati:

- [ ] Training completato con successo
- [ ] File creati in `outputs/runs/{EXPERIMENT_NAME}/`
- [ ] Checkpoint salvato su Drive
- [ ] Verificato dimensione file (< 2MB totale)
- [ ] Nessun file `.pt` in outputs/runs/
- [ ] Nome run descrittivo
- [ ] `git status` mostra file corretti
- [ ] Commit message descrittivo
- [ ] Push completato con successo

---

## 🎯 CONCLUSIONE

Con questa configurazione:
- ✅ Risultati facilmente condivisibili via Git
- ✅ Repository leggero (~500KB-2MB per run)
- ✅ Checkpoint pesanti su Drive
- ✅ Workflow semplice da Colab
- ✅ History completa dei risultati

**Tutto pronto per pushare risultati da Colab!** 🚀
