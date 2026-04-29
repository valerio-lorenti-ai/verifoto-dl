# verifoto-dl

ML service stateless per rilevamento immagini manipolate.  
Responsabilità: preprocessing, model loading, inference, output tecnico strutturato.  
Non gestisce DB, autenticazione utente, business logic o orchestrazione.

---

## Endpoints

### `GET /health`
Pubblico. Restituisce stato del servizio e flag modello caricato.

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /model-info`
Protetto (`x-api-key`). Restituisce metadati del modello attivo.

```json
{
  "service_name": "verifoto-dl",
  "model_version": "pico_plus_exp3_aug",
  "threshold": 0.2,
  "status": "ok"
}
```

### `POST /predict`
Protetto (`x-api-key`). Accetta `multipart/form-data` con campo `file` (immagine).  
Parametro opzionale: `request_id` (stringa, per tracciabilità nei log).  
Limite dimensione: configurabile via `MAX_FILE_SIZE_MB` (default 10 MB).

```json
{
  "status": "success",
  "service_name": "verifoto-dl",
  "filename": "foto.jpg",
  "predicted_class": "manipulated",
  "manipulation_probability": 0.87,
  "confidence_level": "high",
  "model_version": "pico_plus_exp3_aug",
  "threshold": 0.2,
  "decision": "likely_fraud",
  "inference_time_ms": 142.5
}
```

Valori possibili:
- `predicted_class`: `real` | `manipulated`
- `confidence_level`: `low` | `medium` | `high`
- `decision`: `likely_valid` | `likely_fraud` | `uncertain`

---

## Variabili d'ambiente

| Variabile | Default | Note |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` disabilita `/docs` |
| `INTERNAL_API_KEY` | `` (vuoto) | Se vuoto, auth disabilitata in dev |
| `MODEL_VERSION` | `pico_plus_exp3_aug` | Versione modello nei log e response |
| `THRESHOLD` | `0.2` | Soglia classificazione, clampata in [0.0, 1.0] |
| `MAX_FILE_SIZE_MB` | `10` | Limite upload in MB |
| `MODEL_PATH` | `weights/best.pt` | Path locale del modello. **Railway: `weights/best.pt`** — **Locale: `api/weights/best.pt`** |
| `R2_ACCESS_KEY_ID` | — | Cloudflare R2 access key |
| `R2_SECRET_ACCESS_KEY` | — | Cloudflare R2 secret key |
| `R2_ACCOUNT_ID` | — | Cloudflare account ID |
| `R2_BUCKET_NAME` | — | Nome bucket R2 (es. `verifoto-models`) |
| `R2_MODEL_KEY` | `best.pt` | Chiave oggetto nel bucket R2 |

---

## Modello: download automatico da Cloudflare R2

Il modello **non è incluso nell'immagine Docker**. Viene scaricato automaticamente
all'avvio dell'app da Cloudflare R2, solo se non è già presente su disco.

Il modulo responsabile è `api/download_model.py` → funzione `download_model_if_missing()`.

Flusso all'avvio:
1. `inference.py` importa e chiama `download_model_if_missing()`
2. Se `MODEL_PATH` esiste già → skip
3. Se non esiste → download autenticato da R2 con S3 signature v4
4. `model_loader.py` carica il file da `MODEL_PATH`

---

## Run locale

```bash
# Dalla root del repo, con .env che contiene MODEL_PATH=api/weights/best.pt
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Il modello viene scaricato automaticamente in `api/weights/best.pt` al primo avvio
se le variabili R2 sono presenti nel `.env`.

## Test locale (modello diretto)

```bash
cd api
python test_local_model.py
```

## Test HTTP (server attivo)

```bash
cd api
API_URL=http://localhost:8000 INTERNAL_API_KEY=tua_chiave python test_api.py
```

---

## Deploy Railway

1. Imposta queste variabili d'ambiente nel progetto Railway:
   - `MODEL_PATH=weights/best.pt`  ← **obbligatorio, non usare `api/weights/best.pt`**
   - `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ACCOUNT_ID`
   - `R2_BUCKET_NAME=verifoto-models`, `R2_MODEL_KEY=best.pt`
   - `ENVIRONMENT=production`
   - `INTERNAL_API_KEY` con una chiave sicura generata a caso
2. Il Dockerfile installa le dipendenze e avvia il server — nessun download durante la build.
3. Al primo avvio il container scarica `best.pt` da R2 in `/app/weights/best.pt`.
4. Nei log Railway dovresti vedere:
   ```
   INFO  Modello non trovato in weights/best.pt. Download da R2 bucket 'verifoto-models'...
   INFO  Modello scaricato con successo in weights/best.pt.
   INFO  Modello pronto — server operativo
   ```
