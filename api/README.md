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
| `MODEL_URL` | — | URL Google Drive per download modello (build Docker) |

---

## Run locale

```bash
cd api
pip install -r requirements.txt
# assicurati che weights/best.pt esista
uvicorn app.main:app --reload
```

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

1. Imposta le env var nel progetto Railway (vedi tabella sopra)
2. `MODEL_URL` deve puntare al file `.pt` su Google Drive (link diretto gdown-compatibile)
3. `ENVIRONMENT=production` per disabilitare Swagger UI
4. `INTERNAL_API_KEY` con una chiave sicura generata a caso

Il modello viene scaricato durante la build Docker tramite `gdown`.
