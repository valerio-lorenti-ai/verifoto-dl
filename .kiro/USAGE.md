# Kiro Usage Guide

## Struttura Documentazione

### Per l'utente (lettura umana)
- `README.md` - Documento principale con tutto l'essenziale
- `docs/WORKFLOW.md` - Workflow dettagliato (se necessario)
- `docs/AUGMENTED_V6_DATASET.md` - Formato dataset
- `docs/technical/` - Documentazione tecnica di riferimento (versionata)

### Per Kiro (uso interno)
- `.kiro/steering/` - Contesto sempre incluso (project context, code standards)
- `.kiro/notes/` - Appunti tecnici, log, decisioni (NON versionato)
- `.kiro/agent/` - Stato lavoro corrente (NON versionato)

## Directory Structure

```
.kiro/
├── steering/              # Auto-incluso in ogni conversazione
│   ├── project-context.md
│   ├── domain-context.md
│   ├── agent-guidelines.md
│   └── code-standards.md
│
├── notes/                 # Appunti interni (NON versionato)
│   ├── best_practices.md
│   ├── data_leakage_audit.md
│   ├── colab_workflow.md
│   └── ...
│
├── agent/                 # Stato corrente (NON versionato)
│   ├── status.md
│   └── decisions.md
│
└── USAGE.md              # Questa guida
```

## Principi

1. **Minimalismo**: Pochi documenti per l'utente, chiari e stabili
2. **Separazione**: Documentazione umana vs documentazione AI
3. **No frammentazione**: Aggiornare documenti esistenti, non crearne di nuovi
4. **Uso steering**: Per contesto progetto e best practices sempre disponibili
5. **Notes per appunti**: Log tecnici e decisioni in .kiro/notes/

## Workflow

### Starting Work
1. Leggi `.kiro/agent/status.md` per contesto
2. Steering files già inclusi automaticamente
3. Consulta `.kiro/notes/` se serve riferimento tecnico

### During Work
1. Fai modifiche al codice
2. Aggiorna `.kiro/agent/status.md` con progressi
3. Se fai scelte architetturali, aggiungi a `decisions.md`

### Completing Work
1. Aggiorna `status.md` con completamento
2. Nota eventuali issue o next steps
3. **NON creare nuovi file markdown nella root**

## What Goes Where

### User Documentation
- `README.md` - Unico documento principale
- `docs/WORKFLOW.md` - Solo se serve guida dettagliata
- `docs/technical/` - Reference tecnica (non lettura quotidiana)

### Kiro Internal
- `.kiro/steering/` - Contesto permanente (auto-incluso)
- `.kiro/notes/` - Appunti, log, audit (non versionato)
- `.kiro/agent/` - Stato lavoro (non versionato)

### Project Root
- Solo file essenziali progetto
- NO summary, changelog, checklist temporanei

## Anti-Patterns

❌ Creare file markdown nella root dopo ogni task
❌ Generare summary verbose
❌ Duplicare informazioni tra file
❌ Creare nuovi file invece di aggiornare esistenti

## Best Practices

✅ Aggiorna status.md frequentemente
✅ Mantieni README.md come unico documento utente
✅ Usa .kiro/notes/ per appunti tecnici
✅ Centralizza tutto in .kiro/
✅ Testa modifiche con quick_test.yaml
