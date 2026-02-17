# Additional Best Practices - Deep Dive

**Data**: 2026-02-17  
**Status**: 🟡 ULTERIORI MIGLIORAMENTI IDENTIFICATI

---

## 🔍 ANALISI APPROFONDITA

Dopo il fix del data leakage, ho identificato **altre best practice critiche** da applicare:

---

## 1. 🔴 NORMALIZZAZIONE: POTENZIALE LEAKAGE DA STATISTICHE GLOBALI

### Problema Attuale

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
```

**ANALISI**:
- ✅ Usiamo statistiche ImageNet (corretto per transfer learning)
- ✅ NON calcoliamo mean/std sul nostro dataset
- ✅ Nessun leakage qui

**CONCLUSIONE**: ✅ CORRETTO - Nessun problema

---

## 2. 🟡 DATA AUGMENTATION: POSSIBILE OVERFITTING

### Problema Potenziale

```python
train_tf = transforms.Compose([
    RandomJPEGCompression(p=0.55),  # 55% probabilità
    RandomGaussianNoise(p=0.35),    # 35% probabilità
    ...
])
```

**ANALISI**:
- ⚠️  Augmentation JPEG/noise potrebbe essere troppo aggressiva
- ⚠️  Potrebbe insegnare al modello che compressione = frode
- ⚠️  Questo spiega gli alti FP su immagini compresse

### Raccomandazione

**Test A/B**:
1. Training SENZA RandomJPEGCompression e RandomGaussianNoise
2. Confrontare FP rate su immagini compresse

**Ipotesi**: Rimuovendo queste augmentation, il modello dovrebbe:
- ✅ Ridurre FP su immagini compresse legittime
- ⚠️  Possibile aumento FN su frodi con compressione

---

## 3. 🔴 THRESHOLD SELECTION: OTTIMIZZATO SU TEST SET?

### Problema Critico

```python
# In train.py e eval.py
threshold = 0.5
```

**DOMANDA CRITICA**: Come è stato scelto threshold=0.5?

**SCENARI**:

### Scenario A: Threshold Fisso (CORRETTO)
```python
# Threshold standard, non ottimizzato
threshold = 0.5
```
✅ Nessun leakage

### Scenario B: Threshold Ottimizzato su Test (LEAKAGE!)
```python
# Se hai fatto questo, c'è leakage!
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
best_threshold = max(thresholds, key=lambda t: f1_score_on_test(t))
```
❌ Data leakage critico!

**VERIFICA NECESSARIA**: Come hai scelto threshold=0.5?

### Raccomandazione

**METODO CORRETTO**:
1. Ottimizza threshold su VALIDATION set
2. Usa threshold ottimizzato su TEST set (una volta sola)
3. Mai iterare su test set per trovare threshold migliore

```python
# CORRETTO
def find_optimal_threshold(val_probs, val_true, metric='f1'):
    """Trova threshold ottimale su validation set"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = -1
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (val_probs >= thresh).astype(int)
        if metric == 'f1':
            score = f1_score(val_true, y_pred)
        elif metric == 'precision':
            score = precision_score(val_true, y_pred)
        # ...
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh

# In train.py, dopo validation
optimal_threshold = find_optimal_threshold(val_probs, val_true, metric='f1')
print(f"Optimal threshold on validation: {optimal_threshold:.3f}")

# Usa su test set
test_metrics = compute_metrics_from_probs(test_probs, test_true, threshold=optimal_threshold)
```

---

## 4. 🟡 CLASS IMBALANCE: POS_WEIGHT CORRETTO?

### Analisi Attuale

```python
train_pos = (train_df['label'] == 1).sum()
train_neg = (train_df['label'] == 0).sum()
pos_weight = torch.tensor([train_neg / max(train_pos, 1)])
```

**ANALISI**:
- ✅ Calcolato SOLO su training set (corretto)
- ✅ NON usa validation o test set
- ✅ Formula corretta: neg/pos

**VERIFICA**: Qual è il valore di pos_weight?

```
Se train ha:
- 1000 originali (label=0)
- 1500 modificate (label=1)

pos_weight = 1000/1500 = 0.67
```

**DOMANDA**: Il dataset è bilanciato o sbilanciato?

### Raccomandazione

**Se dataset è bilanciato** (50/50):
- pos_weight ≈ 1.0
- ✅ Nessun problema

**Se dataset è sbilanciato** (es: 30/70):
- pos_weight ≠ 1.0
- ⚠️  Verificare che il modello non sia biased verso classe maggioritaria

**TEST**:
```python
# Verifica distribuzione
print(f"Train distribution:")
print(f"  Originali: {train_neg} ({train_neg/(train_pos+train_neg)*100:.1f}%)")
print(f"  Modificate: {train_pos} ({train_pos/(train_pos+train_neg)*100:.1f}%)")
print(f"  pos_weight: {pos_weight.item():.3f}")
```

---

## 5. 🔴 EARLY STOPPING: POSSIBILE OVERFITTING SU VALIDATION

### Problema Potenziale

```python
es = EarlyStopping(patience=6, min_delta=1e-4, mode="max")

for epoch in range(1, epochs_finetune + 1):
    val_m = validate(model, val_loader, threshold=0.5, device=device)
    monitor_val = val_m[monitor]  # monitor = 'pr_auc'
    stop, improved = es.step(monitor_val)
    
    if improved and monitor_val > best_metric:
        best_metric = monitor_val
        save_checkpoint(model, best_ckpt_path, ...)
```

**ANALISI**:
- ✅ Early stopping su validation set (corretto)
- ✅ Salva best model basato su validation
- ⚠️  MA: patience=6 potrebbe essere troppo alta

**PROBLEMA POTENZIALE**:
- Con patience=6, il modello può continuare a overfittare per 6 epoche
- Validation metrics potrebbero essere ottimistiche

### Raccomandazione

**OPZIONE 1: Ridurre Patience**
```python
es = EarlyStopping(patience=3, min_delta=1e-4, mode="max")
```

**OPZIONE 2: Nested Cross-Validation**
```python
# Split train in train_inner + val_inner
# Usa val_inner per early stopping
# Usa val_outer per selezione modello
# Usa test per valutazione finale
```

**OPZIONE 3: Holdout Validation**
```python
# Usa 10% di train come holdout per early stopping
# Usa validation set per selezione iperparametri
# Usa test set per valutazione finale (una volta sola)
```

---

## 6. 🟡 SEED FISSO: POSSIBILE LUCKY SEED

### Problema Potenziale

```python
seed = 42  # Sempre lo stesso!
```

**ANALISI**:
- ✅ Garantisce reproducibilità
- ⚠️  MA: seed "fortunato" può dare metriche migliori
- ⚠️  Metriche potrebbero non essere rappresentative

### Raccomandazione

**CROSS-VALIDATION CON SEED DIVERSI**:

```python
def cross_validate_seeds(config, seeds=[42, 123, 456, 789, 1024]):
    """
    Allena modello con seed diversi e riporta media ± std.
    """
    results = []
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Training with seed={seed}")
        print(f"{'='*80}")
        
        config['seed'] = seed
        metrics = train_and_evaluate(config)
        results.append(metrics)
    
    # Calcola statistiche
    mean_metrics = {}
    std_metrics = {}
    
    for key in results[0].keys():
        values = [r[key] for r in results]
        mean_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)
    
    # Report
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION RESULTS (n={len(seeds)} seeds)")
    print(f"{'='*80}")
    for key in mean_metrics:
        print(f"{key:>15}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
    
    return mean_metrics, std_metrics
```

**INTERPRETAZIONE**:
- Se std è bassa (< 0.02): Metriche stabili ✅
- Se std è alta (> 0.05): Metriche dipendono dal seed ⚠️

---

## 7. 🔴 TEST SET: USATO UNA SOLA VOLTA?

### Domanda Critica

**Quante volte hai valutato il modello su test set?**

### Scenario A: Una Volta Sola (CORRETTO)
```
1. Train modello
2. Valuta su test set
3. Riporta metriche
```
✅ Nessun leakage

### Scenario B: Multiple Volte (LEAKAGE!)
```
1. Train modello v1
2. Valuta su test set → F1=0.85
3. Modifica iperparametri
4. Train modello v2
5. Valuta su test set → F1=0.87
6. Modifica architettura
7. Train modello v3
8. Valuta su test set → F1=0.92 ← LEAKAGE!
```
❌ Hai "ottimizzato" su test set!

**PROBLEMA**:
- Ogni volta che valuti su test e poi modifichi il modello, c'è leakage
- Test set diventa di fatto un validation set
- Metriche finali sono ottimistiche

### Raccomandazione

**METODO CORRETTO**:
1. Usa VALIDATION set per tutte le decisioni
2. Usa TEST set UNA VOLTA SOLA alla fine
3. Se devi iterare, usa cross-validation su train+val

**SE HAI GIÀ USATO TEST MULTIPLE VOLTE**:
- ⚠️  Le metriche attuali sono ottimistiche
- ✅ Soluzione: Crea nuovo test set da dati freschi
- ✅ Oppure: Usa cross-validation e riporta media

---

## 8. 🟡 DROPOUT: APPLICATO CORRETTAMENTE?

### Analisi Attuale

```python
model = build_model(model_name, pretrained=True, drop_rate=0.2)
```

**DOMANDA**: Dropout è applicato solo durante training?

**VERIFICA**:
```python
# Durante training
model.train()  # Dropout attivo
logits = model(x)

# Durante evaluation
model.eval()  # Dropout disattivato
logits = model(x)
```

**ANALISI**:
- ✅ Se usi `model.train()` e `model.eval()` correttamente: OK
- ❌ Se non usi `model.eval()` durante test: PROBLEMA

**VERIFICA NEL CODICE**:
```python
# In train.py - train_one_epoch()
model.train()  # ✅ Corretto

# In train.py - validate()
@torch.no_grad()
def validate(model, loader, ...):
    # ⚠️  MANCA model.eval()!
```

### Raccomandazione

**FIX IMMEDIATO**:
```python
@torch.no_grad()
def validate(model, loader, threshold=0.5, device="cuda"):
    model.eval()  # ← AGGIUNGERE QUESTA RIGA
    probs, y_true, _ = predict_proba(model, loader, device)
    return compute_metrics_from_probs(probs, y_true, threshold=threshold)
```

---

## 9. 🔴 BATCH NORMALIZATION: STATISTICHE CORRETTE?

### Problema Potenziale

**Batch Normalization** usa statistiche diverse in train vs eval:
- **Train**: Usa statistiche del batch corrente
- **Eval**: Usa running mean/std calcolate durante training

**DOMANDA**: Le running statistics sono corrette?

### Possibili Problemi

**Problema 1: Batch Size Troppo Piccolo**
```python
batch_size = 16  # Potrebbe essere troppo piccolo
```

Se batch_size è piccolo:
- Statistiche del batch sono instabili
- Running mean/std sono inaccurate
- Performance in eval è peggiore

**Problema 2: Eval Mode Non Attivato**
```python
# Se non usi model.eval(), usa statistiche del batch anche in test!
model.eval()  # ← CRITICO
```

### Raccomandazione

**VERIFICA 1: Batch Size**
- Se batch_size < 16: Considera aumentare a 32
- Se GPU memory è limitata: Usa gradient accumulation

**VERIFICA 2: Model.eval()**
```python
# In predict_proba()
@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()  # ← VERIFICARE CHE CI SIA
    ...
```

---

## 10. 🟡 GRADIENT CLIPPING: NASCONDE PROBLEMI?

### Analisi Attuale

```python
max_grad_norm = 1.0
nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

**ANALISI**:
- ✅ Previene exploding gradients
- ⚠️  MA: Potrebbe nascondere problemi di learning rate

**DOMANDA**: Quanto spesso i gradienti vengono clippati?

### Raccomandazione

**LOGGING**:
```python
def train_one_epoch(model, loader, optimizer, criterion, ...):
    model.train()
    losses = []
    grad_norms = []
    clipped_count = 0
    
    for x, y, _ in loader:
        ...
        loss.backward()
        
        # Log gradient norm PRIMA del clipping
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        if total_norm > max_grad_norm:
            clipped_count += 1
        
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        ...
    
    print(f"  Gradient stats: mean={np.mean(grad_norms):.3f}, "
          f"max={np.max(grad_norms):.3f}, clipped={clipped_count}/{len(loader)}")
    
    return float(np.mean(losses))
```

**INTERPRETAZIONE**:
- Se clipped < 5%: OK ✅
- Se clipped > 20%: Learning rate troppo alta ⚠️

---

## 📋 CHECKLIST COMPLETA

### Data Leakage
- [x] ✅ Group-based split implementato
- [x] ✅ Assert no overlap tra train/test
- [ ] ⚠️  Threshold ottimizzato su validation (non test)
- [ ] ⚠️  Test set usato una volta sola
- [ ] ⚠️  Normalizzazione corretta (ImageNet stats)

### Model Training
- [ ] ⚠️  model.eval() in validate() e predict_proba()
- [ ] ⚠️  Dropout applicato correttamente
- [ ] ⚠️  Batch normalization statistics corrette
- [ ] ⚠️  Gradient clipping non troppo aggressivo

### Evaluation
- [ ] ⚠️  Cross-validation con seed diversi
- [ ] ⚠️  Metriche riportate con mean ± std
- [ ] ⚠️  Threshold selection su validation
- [ ] ⚠️  Test set holdout (mai usato per decisioni)

### Data Augmentation
- [ ] ⚠️  Test A/B senza JPEG compression augmentation
- [ ] ⚠️  Verificare impatto su FP rate

---

## 🚀 PRIORITÀ AZIONI

### PRIORITÀ ALTA (Fare Subito)

1. **Aggiungere model.eval() in validate() e predict_proba()**
   - Impatto: CRITICO
   - Tempo: 5 minuti
   - Fix: Aggiungere 1 riga di codice

2. **Implementare threshold selection su validation**
   - Impatto: ALTO
   - Tempo: 30 minuti
   - Fix: Nuova funzione + integrazione

3. **Verificare test set usato una volta sola**
   - Impatto: CRITICO
   - Tempo: 0 minuti (solo verifica)
   - Fix: Se usato multiple volte, ri-allenare

### PRIORITÀ MEDIA (Fare Questa Settimana)

4. **Cross-validation con seed diversi**
   - Impatto: MEDIO
   - Tempo: 2-3 ore (5 training runs)
   - Fix: Script automatico

5. **Test A/B senza JPEG augmentation**
   - Impatto: MEDIO
   - Tempo: 1-2 ore (1 training run)
   - Fix: Modificare build_transforms()

6. **Logging gradient norms**
   - Impatto: BASSO
   - Tempo: 15 minuti
   - Fix: Aggiungere logging

### PRIORITÀ BASSA (Nice to Have)

7. **Nested cross-validation**
   - Impatto: BASSO
   - Tempo: 1 giorno
   - Fix: Ristrutturare pipeline

---

## ✅ CONCLUSIONE

Ho identificato **10 ulteriori best practice** da verificare/applicare:

**CRITICHE** (🔴):
1. Threshold selection (potenziale leakage)
2. Test set usage (quante volte usato?)
3. model.eval() mancante (dropout/batchnorm)

**IMPORTANTI** (🟡):
4. Seed fisso (lucky seed?)
5. Data augmentation (troppo aggressiva?)
6. Early stopping (patience troppo alta?)
7. Class imbalance (pos_weight corretto?)
8. Batch normalization (statistiche corrette?)
9. Gradient clipping (nasconde problemi?)

**CORRETTE** (🟢):
10. Normalizzazione ImageNet (OK)

**Prossimo step**: Implementare i fix ad alta priorità (1-3) prima di ri-allenare.
