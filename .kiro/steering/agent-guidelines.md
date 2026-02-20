---
inclusion: auto
---

# Agent Working Guidelines

## Documentation Philosophy - REGOLA ASSOLUTA

### Principio Fondamentale
**La repository appartiene a Valerio. Non è uno spazio per documentazione automatica non richiesta.**

### Divieto Assoluto
❌ **MAI creare file .md per:**
- Task o sottotask
- Recap automatici
- Spiegazioni di modifiche
- Tracking temporaneo
- Documenti storici di task completati
- Documentazione "just in case"
- Summary, log, report non richiesti esplicitamente

### Struttura Consentita (Unica Eccezione)
✅ **Massimo 3 documenti nella cartella `.kiro/agent/`:**
1. 1 documento principale di contesto operativo
2. (Opzionale) 1-2 documenti tecnici di riferimento

**Questi documenti devono:**
- Essere riutilizzati e aggiornati nel tempo
- NON moltiplicarsi
- NON generare nuove versioni
- NON essere duplicati
- Funzionare come memoria centrale persistente, non log incrementale

### Regola di Controllo Prima di Creare Qualsiasi File
Prima di creare QUALSIASI documento, chiediti:
1. È stato richiesto esplicitamente da Valerio?
2. Posso aggiornare un documento esistente invece?
3. Sarà letto e utile agli umani?
4. Ha valore a lungo termine?

**Se anche una sola risposta è NO → NON CREARE IL FILE**

### Documentazione Consentita
- **User docs**: README.md + docs/ (solo se richiesto da Valerio)
- **Agent context**: Massimo 3 file in `.kiro/agent/` (riutilizzati e aggiornati)
- **Root files**: Solo essenziali (README, requirements, configs)

## Code Modifications
- **Test first**: Use `configs/quick_test.yaml` for fast verification
- **Preserve**: Maintain backward compatibility
- **Modular**: Keep src/ files focused and single-purpose
- **No breaking changes**: Unless explicitly requested

## Communication Style
- **Concise**: Brief explanations, no verbose summaries
- **Actionable**: Focus on what to do, not what was done
- **Technical**: Assume developer audience
- **No repetition**: Don't restate what's already clear

## When Making Changes
1. Check `.kiro/agent/status.md` for current context
2. Update relevant files in `.kiro/agent/` as you work
3. Keep changes focused and minimal
4. Test with quick_test.yaml if modifying core code
5. Update status.md with completion notes

## What NOT to Do (ASSOLUTO)
- ❌ Creare file .md per task/sottotask
- ❌ Creare summary, recap, log automatici
- ❌ Creare documentazione non richiesta esplicitamente
- ❌ Moltiplicare file nella root del progetto
- ❌ Creare nuovi file quando posso aggiornare esistenti
- ❌ Generare documentazione "just in case"
- ❌ Creare più di 3 documenti in `.kiro/agent/`

## What TO Do (OBBLIGATORIO)
- ✅ Aggiornare i 3 documenti esistenti in `.kiro/agent/`
- ✅ Chiedere conferma prima di creare QUALSIASI nuovo file .md
- ✅ Mantenere la repository pulita e minimale
- ✅ Rispettare la proprietà di Valerio sulla repository
- ✅ Documentazione minima, strutturata, intenzionale
