import numpy as np
import cv2
from PIL import Image
from torchcam.methods import GradCAM

from tools.classifier_tool import load_inference_model, inference_transforms, IMG_SIZE

def generate_gradcam(pil_image: Image.Image) -> Image.Image:
    model, _ = load_inference_model()
    image_tensor = inference_transforms(pil_image.convert("RGB")).unsqueeze(0)

    original_image = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    original_array = np.array(original_image)

    cam_extractor = GradCAM(model, target_layer="features.18")

    try:
        model.zero_grad(set_to_none=True)
        outputs = model(image_tensor)
        class_index = int(outputs.argmax(dim=1).item())

        activation_map = cam_extractor(class_index, outputs)[0]
        heatmap = activation_map.squeeze(0).detach().cpu().numpy()
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

        heatmap = np.uint8(
            255 * (heatmap - heatmap.min()) /
            (heatmap.max() - heatmap.min() + 1e-8)
        )

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
        return Image.fromarray(overlay)
    finally:
        cam_extractor.remove_hooks()
