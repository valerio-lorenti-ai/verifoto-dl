# 🧪 Noise Dataset Experiments - Guida Pratica

## Obiettivo

Testare se le tecniche di noise analysis migliorano le performance del modello rispetto al dataset RGB originale.

## Workflow Completo

### 1. Generare i Dataset Trasformati (una volta sola)

Nel notebook `verifoto_dl.ipynb`, dopo il mount di Google Drive, esegui le 3 celle di trasformazione:

```python
# Cella 1: High-pass
# Cella 2: Median Residual  
# Cella 3: Gaussian Residual
```

Questo genera 3 nuovi dataset su Google Drive:
- `exp_3_augmented_v6.1_categorized_highpass`
- `exp_3_augmented_v6.1_categorized_median_residual`
- `exp_3_augmented_v6.1_categorized_gaussian_residual`

**Tempo**: ~45-60 minuti totali  
**Spazio**: ~6-9 GB aggiuntivi su Drive

### 2. Esperimenti di Training

Esegui 4 esperimenti separati, uno per ogni dataset:

#### Esperimento 1: RGB Baseline (controllo)
```python
EXPERIMENT_NAME = "2026-03-13_convnext_v8_RGB_baseline"
DATASET_NAME = "exp_3_augmented_v6.1_categorized"  # RGB originale
CONFIG_FILE = "convnext_v8.yaml"
```

#### Esperimento 2: High-pass
```python
EXPERIMENT_NAME = "2026-03-13_convnext_v8_highpass"
DATASET_NAME = "exp_3_augmented_v6.1_categorized_highpass"
CONFIG_FILE = "convnext_v8.yaml"
```

#### Esperimento 3: Median Residual
```python
EXPERIMENT_NAME = "2026-03-13_convnext_v8_median_residual"
DATASET_NAME = "exp_3_augmented_v6.1_categorized_median_residual"
CONFIG_FILE = "convnext_v8.yaml"
```

#### Esperimento 4: Gaussian Residual
```python
EXPERIMENT_NAME = "2026-03-13_convnext_v8_gaussian_residual"
DATASET_NAME = "exp_3_augmented_v6.1_categorized_gaussian_residual"
CONFIG_FILE = "convnext_v8.yaml"
```

### 3. Confronto Risultati

Dopo aver completato tutti gli esperimenti, confronta le metriche:

```python
import pandas as pd
import json

experiments = [
    "2026-03-13_convnext_v8_RGB_baseline",
    "2026-03-13_convnext_v8_highpass",
    "2026-03-13_convnext_v8_median_residual",
    "2026-03-13_convnext_v8_gaussian_residual"
]

results = []
for exp in experiments:
    with open(f"outputs/runs/{exp}/metrics.json") as f:
        metrics = json.load(f)
    
    results.append({
        'experiment': exp.split('_')[-1],  # RGB_baseline, highpass, etc.
        'accuracy': metrics['test_metrics']['accuracy'],
        'precision': metrics['test_metrics']['precision'],
        'recall': metrics['test_metrics']['recall'],
        'f1': metrics['test_metrics']['f1'],
        'auc': metrics['test_metrics']['auc']
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

## Metriche da Confrontare

### Metriche Principali
- **F1 Score**: Bilanciamento precision/recall
- **Precision**: Riduzione falsi positivi
- **Recall**: Riduzione falsi negativi
- **AUC**: Capacità discriminativa generale

### Metriche Secondarie
- **Accuracy**: Performance complessiva
- **FP/FN**: Errori specifici
- **Performance per categoria**: Quali categorie migliorano?

## Ipotesi da Testare

### H1: Noise Analysis migliora la detection
**Aspettativa**: Dataset trasformati hanno F1 > RGB baseline

**Perché**: Le tecniche di noise analysis evidenziano artefatti di editing che il modello RGB potrebbe non catturare.

### H2: Median Residual è la tecnica migliore
**Aspettativa**: Median Residual > Gaussian Residual > High-pass

**Perché**: Dall'analisi precedente, Median Residual ha mostrato il Cohen's d più alto.

### H3: Riduzione falsi positivi
**Aspettativa**: Dataset trasformati hanno FP < RGB baseline

**Perché**: Noise patterns sono più discriminativi per distinguere originali da modificati.

## Risultati Attesi

### Scenario Ottimale
```
Experiment          F1      Precision  Recall   AUC
RGB_baseline       0.920    0.910     0.930   0.975
highpass           0.935    0.925     0.945   0.980
median_residual    0.945    0.940     0.950   0.985  ← MIGLIORE
gaussian_residual  0.940    0.935     0.945   0.982
```

### Scenario Neutro
```
Experiment          F1      Precision  Recall   AUC
RGB_baseline       0.920    0.910     0.930   0.975
highpass           0.918    0.908     0.928   0.973
median_residual    0.922    0.912     0.932   0.976
gaussian_residual  0.919    0.909     0.929   0.974
```
→ Nessun miglioramento significativo, RGB è già ottimale

### Scenario Negativo
```
Experiment          F1      Precision  Recall   AUC
RGB_baseline       0.920    0.910     0.930   0.975
highpass           0.880    0.870     0.890   0.950
median_residual    0.885    0.875     0.895   0.955
gaussian_residual  0.882    0.872     0.892   0.952
```
→ Noise analysis rimuove informazioni utili

## Analisi Dettagliata

### Per Categoria Food
Confronta performance per categoria:

```python
for exp in experiments:
    food_metrics = pd.read_csv(f"outputs/runs/{exp}/group_metrics_food.csv")
    print(f"\n{exp}:")
    print(food_metrics.sort_values('f1', ascending=False).head(10))
```

**Domande**:
- Quali categorie migliorano con noise analysis?
- Quali peggiorano?
- Ci sono pattern comuni?

### Per Generator
Confronta performance per generatore:

```python
for exp in experiments:
    gen_metrics = pd.read_csv(f"outputs/runs/{exp}/group_metrics_generator.csv")
    print(f"\n{exp}:")
    print(gen_metrics)
```

**Domande**:
- Noise analysis aiuta con generatori specifici?
- Ci sono generatori che diventano più difficili?

### Falsi Positivi
Analizza se gli stessi campioni sono FP in tutti gli esperimenti:

```python
fp_sets = {}
for exp in experiments:
    fp = pd.read_csv(f"outputs/runs/{exp}/top_false_positives.csv")
    fp_sets[exp] = set(fp['image_id'].tolist())

# Intersezione (FP comuni a tutti)
common_fp = set.intersection(*fp_sets.values())
print(f"FP comuni a tutti gli esperimenti: {len(common_fp)}")

# FP unici per esperimento
for exp, fp_set in fp_sets.items():
    unique = fp_set - set.union(*[s for e, s in fp_sets.items() if e != exp])
    print(f"{exp}: {len(unique)} FP unici")
```

## Decisioni Post-Esperimenti

### Se Noise Analysis migliora (F1 > baseline)
1. **Adotta la tecnica migliore** per produzione
2. **Considera dual-input**: RGB + Noise come 2 branch
3. **Testa ensemble**: Combina predizioni RGB + Noise

### Se Noise Analysis è neutro (F1 ≈ baseline)
1. **Mantieni RGB** (più semplice)
2. **Considera per casi specifici**: Solo categorie problematiche
3. **Testa come augmentation**: Noise come data augmentation

### Se Noise Analysis peggiora (F1 < baseline)
1. **Mantieni RGB**
2. **Analizza perché**: Informazioni perse? Normalizzazione errata?
3. **Considera preprocessing diverso**: ELA, DCT, etc.

## Esperimenti Avanzati (opzionali)

### Dual-Input Model
Modifica architettura per accettare RGB + Noise:

```python
# Pseudo-codice
class DualInputModel(nn.Module):
    def __init__(self):
        self.rgb_branch = ConvNeXt(...)
        self.noise_branch = ConvNeXt(...)
        self.fusion = nn.Linear(2048, 2)
    
    def forward(self, rgb, noise):
        rgb_features = self.rgb_branch(rgb)
        noise_features = self.noise_branch(noise)
        combined = torch.cat([rgb_features, noise_features], dim=1)
        return self.fusion(combined)
```

### Ensemble
Combina predizioni di modelli separati:

```python
# Carica modelli
model_rgb = load_checkpoint("RGB_baseline/best.pt")
model_noise = load_checkpoint("median_residual/best.pt")

# Predizioni
prob_rgb = model_rgb(image_rgb)
prob_noise = model_noise(image_noise)

# Ensemble (media pesata)
prob_final = 0.6 * prob_rgb + 0.4 * prob_noise
```

### Noise come Augmentation
Usa noise transformation come data augmentation durante training:

```python
class NoiseAugmentation:
    def __call__(self, image):
        if random.random() < 0.5:
            return apply_median_residual(image)
        return image
```

## Checklist Esperimenti

- [ ] Generati tutti i dataset trasformati
- [ ] Eseguito training RGB baseline
- [ ] Eseguito training High-pass
- [ ] Eseguito training Median Residual
- [ ] Eseguito training Gaussian Residual
- [ ] Confrontate metriche principali
- [ ] Analizzate performance per categoria
- [ ] Analizzate performance per generator
- [ ] Analizzati falsi positivi comuni
- [ ] Documentati risultati
- [ ] Presa decisione su tecnica da adottare

## Timeline Stimata

| Fase | Tempo | Note |
|------|-------|------|
| Generazione dataset | 1h | Una volta sola |
| Training RGB baseline | 2-3h | Controllo |
| Training High-pass | 2-3h | Esperimento 1 |
| Training Median Residual | 2-3h | Esperimento 2 |
| Training Gaussian Residual | 2-3h | Esperimento 3 |
| Analisi risultati | 1h | Confronto e decisione |
| **TOTALE** | **10-13h** | Può essere parallelizzato |

## Risorse Necessarie

- **Google Drive**: ~10 GB liberi
- **Colab**: GPU (T4 o migliore)
- **Tempo**: 1-2 giorni (con pause)

---

**Pronto per iniziare gli esperimenti!** 🚀
