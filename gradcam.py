import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from dataset import CLASSES, build_splits
from train_transfer import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model


def load_image_tensor(path):
    """Load one frame as a normalized model input plus its resized original."""
    image = Image.open(path).convert("RGB")
    transform = v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(image), image.resize((IMAGE_SIZE, IMAGE_SIZE))


def compute_gradcam(model, image_tensor, device):
    """Return a normalized class activation map for the top predicted class."""
    activation = {}
    gradient = {}

    def forward_hook(module, inputs, output):
        activation["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradient["value"] = grad_output[0]

    layer = model.layer4
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    inputs = image_tensor.unsqueeze(0).to(device)
    inputs.requires_grad_(True)
    logits = model(inputs)
    class_index = logits.argmax(dim=1).item()
    model.zero_grad()
    logits[0, class_index].backward()

    forward_handle.remove()
    backward_handle.remove()

    weights = gradient["value"][0].mean(dim=(1, 2))
    cam = torch.einsum("c,chw->hw", weights, activation["value"][0])
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), class_index


def overlay_heatmap(cam, original_image):
    """Blend a class activation map over the original frame."""
    heatmap = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    base = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(base, 0.5, heatmap, 0.5, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize ResNet18 transfer model attention with Grad-CAM."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/transfer_model.pt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("gradcam_runs"))
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.accelerator.current_accelerator() or torch.device("cpu")
    model = build_model(weights=None).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    _, _, test_samples = build_splits()
    by_class = {index: [] for index in range(len(CLASSES))}
    for path, label in test_samples:
        by_class[label].append(path)

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label, paths in by_class.items():
        chosen = rng.sample(paths, min(args.samples_per_class, len(paths)))
        for path in chosen:
            image_tensor, original_image = load_image_tensor(path)
            cam, predicted = compute_gradcam(model, image_tensor, device)
            overlay = overlay_heatmap(cam, original_image)
            true_name = CLASSES[label]
            predicted_name = CLASSES[predicted]
            out_path = (
                args.output_dir
                / f"{true_name}_pred-{predicted_name}_{Path(path).stem}.png"
            )
            cv2.imwrite(str(out_path), overlay)
            print(f"{true_name}: predicted={predicted_name} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
