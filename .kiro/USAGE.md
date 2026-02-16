# Kiro Steering Structure - Usage Guide

## Overview

Questa struttura organizza il contesto di lavoro dell'agente AI in modo leggero e mantenibile.

## Directory Structure

```
.kiro/
├── steering/              # Linee guida permanenti (auto-incluse)
│   ├── project-context.md      # Contesto progetto tecnico
│   ├── domain-context.md       # Contesto dominio business
│   ├── agent-guidelines.md     # Come lavora l'agente
│   └── code-standards.md       # Standard Python
│
├── agent/                 # Contesto di lavoro (NON versionato)
│   ├── status.md               # Stato corrente (aggiornato frequentemente)
│   ├── decisions.md            # Decisioni architetturali
│   └── README.md               # Guida ai file agent
│
└── USAGE.md              # Questa guida
```

## File Purposes

### Steering Files (Permanenti)
Questi file definiscono come l'agente deve lavorare. Cambiano raramente.

**project-context.md**
- Panoramica tecnica progetto
- Struttura directory
- Workflow generale
- Tech stack
- Auto-incluso in ogni conversazione

**domain-context.md**
- Contesto business (Verifoto-AI)
- Problema del fraud detection
- Approccio conservativo
- Considerazioni etiche
- Auto-incluso in ogni conversazione

**agent-guidelines.md**
- Filosofia documentazione
- Regole di lavoro
- Cosa fare/non fare
- Auto-incluso in ogni conversazione

**code-standards.md**
- Standard Python
- Convenzioni PyTorch
- Best practices
- Auto-incluso quando si modificano file .py

### Agent Files (Dinamici)
Questi file tracciano lo stato del lavoro. Aggiornati frequentemente.

**status.md**
- Stato corrente del lavoro
- Completamenti recenti
- Prossimi task
- Note importanti
- **Aggiorna questo spesso!**

**decisions.md**
- Decisioni architetturali
- Rationale delle scelte
- Alternative rifiutate
- Considerazioni future
- Aggiorna quando fai scelte importanti

## How It Works

### Auto-Inclusion
I file in `steering/` con `inclusion: auto` sono automaticamente inclusi nel contesto quando:
- `project-context.md`: Sempre (overview tecnico)
- `domain-context.md`: Sempre (contesto business)
- `agent-guidelines.md`: Sempre (regole di lavoro)
- `code-standards.md`: Quando si modificano file .py

### Manual Reference
I file in `agent/` sono letti manualmente quando serve:
- All'inizio di una sessione: leggi `status.md`
- Durante il lavoro: aggiorna `status.md`
- Per decisioni importanti: aggiungi a `decisions.md`

## Workflow Example

### Starting Work
```
1. Leggi .kiro/agent/status.md per contesto
2. Steering files già inclusi automaticamente
3. Inizia il lavoro
```

### During Work
```
1. Fai modifiche al codice
2. Aggiorna .kiro/agent/status.md con progressi
3. Se fai scelte architetturali, aggiungi a decisions.md
```

### Completing Work
```
1. Aggiorna status.md con completamento
2. Nota eventuali issue o next steps
3. NON creare nuovi file di summary
```

## Key Principles

### 1. Centralizzazione
- Tutto il contesto agente in `.kiro/`
- Niente file sparsi nella root
- Documentazione utente solo in `docs/`

### 2. Minimalismo
- 3 steering files (permanenti)
- 2 agent files (dinamici)
- Aggiorna invece di creare

### 3. Brevità
- File corti e focalizzati
- Niente verbosità
- Solo info actionable

### 4. Manutenibilità
- Pochi file ben mantenuti
- Aggiornamenti frequenti
- Niente file obsoleti

## What Goes Where

### User Documentation (docs/)
- Guide dettagliate per utenti
- Tutorial e workflow
- Reference documentation
- **Per umani che usano il progetto**

### Agent Context (.kiro/agent/)
- Stato del lavoro
- Decisioni tecniche
- Note di sviluppo
- **Per AI assistant**

### Steering (.kiro/steering/)
- Linee guida permanenti
- Standard di codice
- Contesto progetto
- **Per AI assistant (auto-incluso)**

### Project Root
- Solo file essenziali progetto
- README, QUICKSTART
- NO summary o changelog temporanei

## Anti-Patterns to Avoid

❌ Creare file markdown nella root dopo ogni task
❌ Generare summary verbose
❌ Duplicare informazioni tra file
❌ Creare nuovi file invece di aggiornare esistenti
❌ Mettere contesto agente in docs/

## Best Practices

✅ Aggiorna status.md frequentemente
✅ Mantieni file brevi e focalizzati
✅ Usa steering files come reference
✅ Centralizza tutto in .kiro/
✅ Testa modifiche con quick_test.yaml

## Example Updates

### After Code Change
```markdown
# In .kiro/agent/status.md

## Recent Completion
✅ Added metadata extraction to dataset parser
- Modified src/utils/data.py
- Added food_category, defect_type fields
- Tested with quick_test.yaml

## Next
- May need to adjust parser for edge cases
- Consider adding validation
```

### After Architectural Decision
```markdown
# In .kiro/agent/decisions.md

### Metadata Storage
**Decision**: Store metadata in DataFrame, pass through DataLoader
**Rationale**: 
- No manual annotation needed
- Available at inference
- Easy to filter/group
```

## Summary

Questa struttura ti permette di:
- Avere contesto sempre disponibile (steering auto-incluso)
- Tracciare stato del lavoro (agent/status.md)
- Documentare decisioni (agent/decisions.md)
- Mantenere tutto organizzato e leggero
- Evitare proliferazione di file

**Ricorda**: Pochi file ben mantenuti > molti file sparsi.
