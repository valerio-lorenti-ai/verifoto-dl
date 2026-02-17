# Struttura Documentazione Progetto

## Filosofia

Documentazione minimalista e ben organizzata:
- **Pochi file per te**: Chiari, stabili, facili da seguire
- **Spazio separato per AI**: Appunti tecnici e log in `.kiro/`
- **No frammentazione**: Aggiornare invece di creare

## File per Te (Lettura Umana)

### Root
- `README.md` - **Unico documento principale**
  - Quick start
  - Comandi principali
  - Configurazione
  - Troubleshooting

### docs/
- `WORKFLOW.md` - Workflow dettagliato (se serve approfondimento)
- `AUGMENTED_V6_DATASET.md` - Formato dataset e metadati

### docs/technical/ (Riferimento, non lettura quotidiana)
- `ARCHITECTURE.md` - Architettura tecnica
- `MIGRATION.md` - Guida migrazione vecchio codice
- `MIGRATION_V6.md` - Migrazione dataset v6
- `CHANGELOG_V6.md` - Changelog dataset v6

## File per AI (Uso Interno)

### .kiro/steering/ (Auto-incluso)
- `project-context.md` - Contesto progetto
- `domain-context.md` - Contesto business
- `agent-guidelines.md` - Regole di lavoro
- `code-standards.md` - Standard codice

### .kiro/notes/ (NON versionato)
- `best_practices.md` - Best practices tecniche
- `data_leakage_audit.md` - Audit data leakage
- `colab_workflow.md` - Workflow Colab dettagliato
- `critical_differences.md` - Differenze implementazione
- Altri appunti tecnici e log

### .kiro/agent/ (NON versionato)
- `status.md` - Stato lavoro corrente
- `decisions.md` - Decisioni architetturali

## Cosa È Cambiato

### Prima (Troppi File)
```
verifoto-dl/
├── README.md
├── QUICKSTART.md
├── FIRST_RUN.md
├── READY_FOR_COLAB.md
├── CHECKLIST.md
├── CHANGELOG_V6.md
└── docs/
    ├── WORKFLOW.md
    ├── ARCHITECTURE.md
    ├── MIGRATION.md
    ├── MIGRATION_V6.md
    ├── AUGMENTED_V6_DATASET.md
    ├── ADDITIONAL_BEST_PRACTICES.md
    ├── FINAL_BEST_PRACTICES_SUMMARY.md
    ├── DATA_LEAKAGE_AUDIT.md
    ├── LEAKAGE_FIX_SUMMARY.md
    ├── CRITICAL_DIFFERENCES.md
    ├── COLAB_WORKFLOW.md
    ├── COLAB_REVIEW.md
    └── GIT_RESULTS_GUIDE.md
```

### Dopo (Minimalista)
```
verifoto-dl/
├── README.md                    # UNICO documento principale
└── docs/
    ├── WORKFLOW.md              # Dettagli workflow
    ├── AUGMENTED_V6_DATASET.md  # Reference dataset
    ├── DOCUMENTATION_STRUCTURE.md  # Questa guida
    └── technical/               # Reference tecnica
        ├── ARCHITECTURE.md
        ├── MIGRATION.md
        ├── MIGRATION_V6.md
        └── CHANGELOG_V6.md
```

## Vantaggi

1. **Meno confusione**: Un solo documento da leggere (README)
2. **Più stabilità**: Non cambia continuamente
3. **Separazione chiara**: Docs umani vs docs AI
4. **Facile manutenzione**: Pochi file ben curati
5. **No ridondanza**: Informazioni non duplicate

## Come Usare

### Per Te
1. Leggi `README.md` per tutto quello che ti serve
2. Consulta `docs/WORKFLOW.md` se serve approfondimento
3. Ignora `.kiro/` (è per l'AI)

### Per AI
1. Usa `.kiro/steering/` per contesto permanente (auto-incluso)
2. Usa `.kiro/notes/` per appunti tecnici e log
3. Aggiorna `.kiro/agent/status.md` durante il lavoro
4. **Non creare nuovi file markdown nella root**

## Regole per il Futuro

### ✅ Fare
- Aggiornare README.md se serve
- Aggiornare docs esistenti
- Usare .kiro/notes/ per appunti AI
- Mantenere struttura minimalista

### ❌ Non Fare
- Creare nuovi file markdown nella root
- Frammentare informazioni in molti file
- Duplicare contenuti
- Creare summary/changelog temporanei

## Risultato

Ora hai:
- **1 documento principale** (README.md)
- **2 documenti di supporto** (WORKFLOW, AUGMENTED_V6_DATASET)
- **Documentazione tecnica** separata in docs/technical/
- **Spazio AI** ben organizzato in .kiro/

Molto più facile da seguire! 🎉
