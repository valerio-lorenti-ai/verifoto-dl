import os

SERVICE_NAME = "verifoto-dl"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")

MODEL_VERSION = os.getenv("MODEL_VERSION", "pico_plus_exp3_aug")

_threshold_raw = float(os.getenv("THRESHOLD", "0.2"))
THRESHOLD = max(0.0, min(1.0, _threshold_raw))  # clamp in [0.0, 1.0]

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Transform constants — tied to model architecture, not overridable via env
IMG_RESIZE = 257
IMG_CROP = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
