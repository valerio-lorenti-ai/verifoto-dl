import torch
import timm

MODEL_PATH = "weights/best.pt"
DEVICE = torch.device("cpu")


def load_model():
    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=1,
        drop_rate=0.3
    )

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    model.to(DEVICE)
    model.eval()

    print("\n✅ Modello caricato in FastAPI\n")

    return model