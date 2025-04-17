import os

import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForDepthEstimation, AutoImageProcessor


def visualize_depth(image_fname):
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"

    processor = AutoImageProcessor.from_pretrained(model_id)
    depth_model = AutoModelForDepthEstimation.from_pretrained(model_id)
    depth_model.eval()

    image = Image.open(image_fname).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = depth_model(**inputs)
        depth = outputs.predicted_depth  # [1, 1, H, W]

    depth_np = depth.squeeze().cpu().numpy()
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())

    plt.imsave(f"depth {image_fname}", depth_norm, cmap="Greys")

    plt.imshow(depth_norm, cmap="Greys")
    plt.axis("off")
    plt.title("Estimated Depth")
    plt.show()


def reshape_image(image_fname):
    img = Image.open(image_fname)
    resized_img = img.resize((400, 226), Image.BILINEAR)
    resized_img.save(image_fname)


def crop_image(image_fname):
    img = Image.open(image_fname)

    w, h = img.size
    crop_w, crop_h = 390, 260

    left = (w - crop_w) // 2
    right = left + crop_w
    bottom = h
    top = h - crop_h

    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(f'cropped {image_fname}')


def crop_tiles(image_fname):
    output_dir = 'tiles ' + image_fname.split('.')[0]
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_fname)
    w, h = img.size
    rows, cols = 2, 3

    tile_w = w // cols
    tile_h = h // rows

    for row in range(rows):
        for col in range(cols):
            left = col * tile_w
            upper = row * tile_h
            right = left + tile_w
            lower = upper + tile_h

            tile = img.crop((left, upper, right, lower))
            tile.save(os.path.join(output_dir, f"tile_{row}_{col}.png"))


if __name__ == "__main__":
    # visualize_depth('1.png')
    # reshape_image('depth 1.png')
    # crop_image('1.png')
    crop_tiles('cropped 1.png')
