import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2

MODEL_PATH = os.path.join("models", "mobilenet_v2_brain_mri.pth")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")

IMG_SIZE = 224
_model = None
_class_names = None

normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

inference_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalization
])

def prepare_image(pil_image: Image.Image) -> Image.Image:
    """Convert to RGB and trim light (white/grey) borders, e.g. a frame around a scanned or
    screenshotted MRI. Images with normal dark backgrounds are returned unchanged."""
    img = pil_image.convert("RGB")
    gray = np.asarray(img.convert("L"))
    h, w = gray.shape
    rows = np.where((gray < 225).mean(axis=1) > 0.02)[0]
    cols = np.where((gray < 225).mean(axis=0) > 0.02)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    if (bottom - top) < 0.3 * h or (right - left) < 0.3 * w:
        return img
    if top == 0 and left == 0 and bottom == h and right == w:
        return img
    return img.crop((int(left), int(top), int(right), int(bottom)))


def load_inference_model():
    global _model, _class_names
    if _model is not None:
        return _model, _class_names

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        _class_names = json.load(f)

    model = mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, len(_class_names))
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
    )
    model.eval()
    _model = model
    return _model, _class_names

def predict_mri(pil_image: Image.Image) -> dict:
    model, class_names = load_inference_model()
    image_tensor = inference_transforms(prepare_image(pil_image)).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].tolist()

    predicted_index = int(np.argmax(probabilities))

    return {
        "top_prediction": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "all_probabilities": {
            name: round(float(prob), 4)
            for name, prob in zip(class_names, probabilities)
        }
    }