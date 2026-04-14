"""
Test locale puro del modello — niente HTTP, niente server.
Esegui da api/: python test_local_model.py
"""
import sys
from dotenv import load_dotenv

# Carica env prima di importare app (settings legge os.getenv al momento dell'import)
load_dotenv()

from app.inference import predict_image

TEST_IMAGES = [
    ("test_real.jpg", "real"),
    ("test_manipulated.jpg", "manipulated"),
]

passed = 0
failed = 0

for path, expected_class in TEST_IMAGES:
    try:
        with open(path, "rb") as f:
            image_bytes = f.read()

        predicted_class, score, confidence_level, model_version, threshold, decision, inference_time_ms = predict_image(image_bytes)

        status = "✅" if predicted_class == expected_class else "⚠️ "
        print(
            f"{status} {path} | predicted={predicted_class} expected={expected_class} "
            f"score={score} confidence={confidence_level} decision={decision} time={inference_time_ms}ms"
        )

        if predicted_class == expected_class:
            passed += 1
        else:
            failed += 1

    except Exception as e:
        print(f"❌ {path} | errore: {e}")
        failed += 1

print(f"\nRisultato: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
