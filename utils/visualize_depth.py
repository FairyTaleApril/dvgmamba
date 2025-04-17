import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

model_id = "depth-anything/Depth-Anything-V2-Small-hf"

processor = AutoImageProcessor.from_pretrained(model_id)
depth_model = AutoModelForDepthEstimation.from_pretrained(model_id)
depth_model.eval()

image = Image.open("../1.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = depth_model(**inputs)
    depth = outputs.predicted_depth  # [1, 1, H, W]

depth_np = depth.squeeze().cpu().numpy()
depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())

plt.imsave("../depth 1.png", depth_norm, cmap="Greys")

plt.imshow(depth_norm, cmap="Greys")
plt.axis("off")
plt.title("Estimated Depth")
plt.show()
