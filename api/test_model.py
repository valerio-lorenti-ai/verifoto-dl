import os
import torch
import timm
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

# Carica env
load_dotenv()

MODEL_PATH = "weights/best.pt"
TEST_IMAGES = [
    "test_real.jpg",
    "test_manipulated.jpg",
]

model = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=1,
    drop_rate=0.3
)

ckpt = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize(257),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

threshold = float(os.getenv("THRESHOLD", "0.2"))

for image_path in TEST_IMAGES:
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logit = model(x).squeeze(1)
        prob = torch.sigmoid(logit).item()

    predicted_class = "manipulated" if prob >= threshold else "real"

    print(f"{image_path} -> {predicted_class} ({prob:.4f})")