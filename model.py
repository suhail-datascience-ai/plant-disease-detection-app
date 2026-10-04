import json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent
CLASSES = json.loads((ROOT / "data/class_names.json").read_text())
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
MODELS = {
    "Transfer Learning (ResNet18)": ROOT / "models/plant_disease_model_tl.pth",
    "CNN from Scratch": ROOT / "models/plant_cnn_from_scratch.pth",
}

def _scratch():
    cfg, layers, c = [32, 32, "M", 64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M"], [], 3
    for v in cfg:
        if v == "M": layers.append(nn.MaxPool2d(2))
        else: layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]; c = v
    layers.append(nn.AdaptiveAvgPool2d(1))
    return _Scratch(nn.Sequential(*layers))

class _Scratch(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 1500), nn.ReLU(),
                                        nn.Dropout(0.4), nn.Linear(1500, 38))
    def forward(self, x): return self.classifier(self.features(x))

def _resnet():
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    m.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 38))
    return m

def load(name):
    m = _resnet() if name.startswith("Transfer") else _scratch()
    m.load_state_dict(torch.load(MODELS[name], map_location="cpu"))
    return m.eval()

def predict(model, img: Image.Image, k=5):
    a = (np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(a.transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    top = torch.topk(p, k)
    return [(CLASSES[i], float(s)) for s, i in zip(top.values, top.indices)]
