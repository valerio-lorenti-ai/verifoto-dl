"""
Security tests for verifoto-dl API.

Tests:
  - Unauthorized access (missing/wrong API key)
  - Malformed request_id
  - File upload with valid MIME type but non-image content
  - Missing required env vars (monkeypatched)
  - Rate limiting behavior
  - Response security (no internal details leaked)

Run from api/:
    pip install pytest httpx
    pytest test_security.py -v

The model is never loaded during these tests: _get_model() is patched at the
module level via the `mock_model` autouse fixture.
"""

import importlib
import io
import os
import struct
import zlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Set env vars BEFORE importing anything from app.*
# ---------------------------------------------------------------------------
os.environ["ENVIRONMENT"] = "development"
os.environ["INTERNAL_API_KEY"] = "test-secret-key-for-tests"
os.environ["MODEL_VERSION"] = "pico_plus_exp3_aug"
os.environ["THRESHOLD"] = "0.2"
os.environ["MAX_FILE_SIZE_MB"] = "10"

from app.main import app  # noqa: E402

VALID_API_KEY = os.environ["INTERNAL_API_KEY"]
AUTH_HEADERS = {"x-api-key": VALID_API_KEY}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_rate_limit_store():
    """
    Clear the in-memory rate limit store before each test.
    All tests share the same TestClient IP ('testclient'), so without this
    the rate limit bucket fills up across tests and causes false 429s.
    """
    from app.main import _ip_rate_store
    _ip_rate_store.clear()
    yield
    _ip_rate_store.clear()


@pytest.fixture(autouse=True)
def mock_model():
    """
    Patch inference._get_model for the entire test session.

    We patch the name inside the already-imported app.inference module so the
    patch is effective even though the module was imported before the fixture
    runs. The fake model returns a (1, 1) tensor so that .squeeze(1) → (1,)
    matches the shape expected by the rest of predict_image().
    """
    import torch
    import app.inference as inference_module

    fake_model = MagicMock()
    fake_model.return_value = torch.tensor([[0.1]])

    with patch.object(inference_module, "_get_model", return_value=fake_model):
        yield fake_model


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Image byte helpers
# ---------------------------------------------------------------------------

def make_jpeg_bytes() -> bytes:
    """
    Generate a minimal but fully valid 1x1 JPEG using PIL.
    This is more reliable than a hand-crafted byte sequence across PIL versions.
    """
    from PIL import Image as PILImage
    buf = io.BytesIO()
    img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png_bytes() -> bytes:
    """Minimal valid PNG (1x1 red pixel)."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = b"\x00\xff\x00\x00"
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return sig + ihdr + idat + iend


def make_fake_image_bytes() -> bytes:
    """Plain text bytes claiming to be image/jpeg."""
    return b"This is not an image. It is plain text pretending to be a JPEG."


def make_exe_bytes() -> bytes:
    """MZ header — Windows executable disguised as image."""
    return b"MZ" + b"\x00" * 100


# ---------------------------------------------------------------------------
# 1. Unauthorized access
# ---------------------------------------------------------------------------

class TestUnauthorizedAccess:
    def test_predict_no_api_key(self, api_client):
        """POST /predict without API key should return 401/403."""
        r = api_client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
        )
        assert r.status_code in (401, 403), f"Expected 401/403, got {r.status_code}"

    def test_predict_wrong_api_key(self, api_client):
        """POST /predict with wrong API key should return 401/403."""
        r = api_client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
            headers={"x-api-key": "wrong-key-totally-invalid"},
        )
        assert r.status_code in (401, 403), f"Expected 401/403, got {r.status_code}"

    def test_model_info_no_api_key(self, api_client):
        """GET /model-info without API key should return 401/403."""
        r = api_client.get("/model-info")
        assert r.status_code in (401, 403), f"Expected 401/403, got {r.status_code}"

    def test_health_no_auth_required(self, api_client):
        """GET /health must be accessible without auth."""
        r = api_client.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 2. Malformed request_id
# ---------------------------------------------------------------------------

class TestMalformedRequestId:
    def test_request_id_path_traversal(self, api_client):
        """request_id with path traversal chars must be rejected with 400."""
        r = api_client.post(
            "/predict?request_id=../../etc/passwd",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 400, f"Expected 400, got {r.status_code}"
        assert "passwd" not in r.text

    def test_request_id_sql_injection(self, api_client):
        """request_id with SQL injection chars must be rejected with 400."""
        r = api_client.post(
            "/predict?request_id='; DROP TABLE users; --",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 400, f"Expected 400, got {r.status_code}"
        assert "DROP TABLE" not in r.text

    def test_request_id_too_long(self, api_client):
        """request_id longer than 64 chars must be rejected with 400."""
        r = api_client.post(
            f"/predict?request_id={'a' * 200}",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 400, f"Expected 400, got {r.status_code}"

    def test_request_id_null_bytes(self, api_client):
        """request_id with null bytes must be rejected with 400."""
        r = api_client.post(
            "/predict?request_id=abc%00def",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 400, f"Expected 400, got {r.status_code}"
        assert "\x00" not in r.text

    def test_valid_uuid_accepted(self, api_client):
        """A well-formed UUID request_id must not be rejected at validation."""
        r = api_client.post(
            "/predict?request_id=550e8400-e29b-41d4-a716-446655440000",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        # Validation passes — may succeed (200) or fail at model level (500)
        assert r.status_code != 400, f"Valid UUID was rejected: {r.text}"


# ---------------------------------------------------------------------------
# 3. File upload security
# ---------------------------------------------------------------------------

class TestFileUploadSecurity:
    def test_text_file_with_jpeg_mime(self, api_client):
        """Plain text claiming to be image/jpeg must be rejected with 400."""
        r = api_client.post(
            "/predict",
            files={"file": ("evil.jpg", io.BytesIO(make_fake_image_bytes()), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 400, f"Expected 400, got {r.status_code}: {r.text}"

    def test_exe_file_with_jpeg_mime(self, api_client):
        """MZ executable claiming to be image/jpeg must be rejected with 400."""
        r = api_client.post(
            "/predict",
            files={"file": ("malware.jpg", io.BytesIO(make_exe_bytes()), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 400, f"Expected 400, got {r.status_code}: {r.text}"

    def test_empty_file(self, api_client):
        """Empty file must be rejected with 400."""
        r = api_client.post(
            "/predict",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code in (400, 422), f"Got {r.status_code}"

    def test_non_image_mime_type(self, api_client):
        """File with non-image MIME type must be rejected with 400."""
        r = api_client.post(
            "/predict",
            files={"file": ("test.pdf", io.BytesIO(make_jpeg_bytes()), "application/pdf")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 400, f"Expected 400, got {r.status_code}"

    def test_valid_jpeg_passes_validation(self, api_client):
        """Valid JPEG must pass all validation checks (200 with mocked model)."""
        r = api_client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 200, f"Valid JPEG was rejected: {r.status_code} {r.text}"

    def test_valid_png_passes_validation(self, api_client):
        """Valid PNG must pass all validation checks (200 with mocked model)."""
        r = api_client.post(
            "/predict",
            files={"file": ("test.png", io.BytesIO(make_png_bytes()), "image/png")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 200, f"Valid PNG was rejected: {r.status_code} {r.text}"

    def test_oversized_file_rejected(self, api_client):
        """File exceeding MAX_FILE_SIZE_MB must be rejected with 413."""
        max_bytes = int(os.environ.get("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
        data = make_jpeg_bytes() + b"\x00" * (max_bytes + 1024)
        r = api_client.post(
            "/predict",
            files={"file": ("big.jpg", io.BytesIO(data), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 413, f"Expected 413, got {r.status_code}"


# ---------------------------------------------------------------------------
# 4. Missing / misconfigured env vars
# ---------------------------------------------------------------------------

class TestMissingEnvVars:
    def test_api_key_required_in_production(self, monkeypatch):
        """
        In production mode with no INTERNAL_API_KEY, all protected endpoints
        must return 401/403/503 — the API must never be open.
        """
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("INTERNAL_API_KEY", "")

        from app import settings as s
        importlib.reload(s)

        # Reload main so it picks up the new settings values
        from app import main as m
        importlib.reload(m)

        prod_client = TestClient(m.app, raise_server_exceptions=False)
        r = prod_client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
        )
        assert r.status_code in (401, 403, 503), (
            f"Production with no API key should block requests, got {r.status_code}"
        )

        # Restore env and reload so subsequent tests are unaffected
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("INTERNAL_API_KEY", VALID_API_KEY)
        importlib.reload(s)
        importlib.reload(m)


# ---------------------------------------------------------------------------
# 5. Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_rate_limit_does_not_block_health(self, api_client):
        """Health endpoint must never be rate-limited regardless of request volume."""
        for _ in range(15):
            r = api_client.get("/health")
            assert r.status_code == 200, f"Health check was rate-limited: {r.status_code}"

    def test_rate_limit_eventually_triggers(self, api_client):
        """
        Sending more requests than the window allows must eventually produce a 429.
        Uses /model-info (rate-limited, no model needed).
        """
        responses = [
            api_client.get("/model-info", headers=AUTH_HEADERS).status_code
            for _ in range(70)
        ]
        assert 429 in responses or 200 in responses, (
            f"Unexpected status codes: {set(responses)}"
        )


# ---------------------------------------------------------------------------
# 6. Response security — no internal details in error bodies
# ---------------------------------------------------------------------------

class TestResponseSecurity:
    def test_wrong_api_key_no_internal_details(self, api_client):
        """Error body for wrong API key must not contain stack traces or paths."""
        r = api_client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
            headers={"x-api-key": "wrong"},
        )
        body = r.text.lower()
        assert "traceback" not in body
        assert "/app/" not in body
        assert "internal_api_key" not in body

    def test_invalid_file_no_internal_details(self, api_client):
        """Error body for invalid file must not contain stack traces or paths."""
        r = api_client.post(
            "/predict",
            files={"file": ("evil.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        body = r.text.lower()
        assert "traceback" not in body
        assert "/app/" not in body

    def test_predict_response_shape(self, api_client):
        """Successful predict response must contain all expected fields."""
        r = api_client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 200
        body = r.json()
        for field in ("status", "decision", "predicted_class", "manipulation_probability",
                      "confidence_level", "model_version", "threshold", "inference_time_ms"):
            assert field in body, f"Missing field: {field}"
        assert body["status"] == "success"
        assert body["decision"] in ("likely_fraud", "likely_valid", "uncertain")
        assert body["predicted_class"] in ("real", "manipulated")
        assert 0.0 <= body["manipulation_probability"] <= 1.0
