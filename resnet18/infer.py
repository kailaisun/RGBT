#!/usr/bin/env python3
"""Thermal-infrared person-count classification (ResNet18).

Input : 80x62 thermal-infrared pseudo-color PNG (IR only, no RGB).
Output: one_person / two_people with class probabilities.
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18


DEFAULT_CLASSES = ("one_person", "two_people")
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int) -> transforms.Compose:
    """Aspect-preserving resize + grey pad, matching training."""
    resized_height = round(image_size * 62 / 80)
    pad_total = image_size - resized_height
    pad_top = pad_total // 2
    return transforms.Compose(
        [
            transforms.Resize((resized_height, image_size), antialias=True),
            transforms.Pad((0, pad_top, 0, pad_total - pad_top), fill=(114, 114, 114)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", required=True, help="IR image")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    class_names = tuple(checkpoint.get("class_names", DEFAULT_CLASSES))
    image_size = int(checkpoint.get("image_size", 224))

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(args.device)

    tensor = build_transform(image_size)(Image.open(args.source).convert("RGB")).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].tolist()

    for name, probability in zip(class_names, probabilities):
        print(f"{name:12s} {probability:.4f}")
    print("prediction:", class_names[int(max(range(len(probabilities)), key=probabilities.__getitem__))])


if __name__ == "__main__":
    main()
