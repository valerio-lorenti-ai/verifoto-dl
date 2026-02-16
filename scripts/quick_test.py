"""Quick test script to verify setup without full training"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from utils.model import build_model
from utils.data import build_transforms

print("Testing imports...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("\nTesting model creation...")
model = build_model("efficientnet_b0", pretrained=False, drop_rate=0.2)
print(f"✓ Model created: {model.__class__.__name__}")

print("\nTesting transforms...")
train_tf, eval_tf = build_transforms(224)
print(f"✓ Transforms created")

print("\nTesting forward pass...")
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    out = model(x)
print(f"✓ Forward pass successful: input {x.shape} -> output {out.shape}")

print("\n✅ All tests passed! Setup is working correctly.")
