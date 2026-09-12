#!/usr/bin/env python3
"""RGB -> thermal-field prediction (U-Net regression).

Input : visible RGB image.
Output: per-pixel temperature field in degrees Celsius (62x80), which can be
        rendered with any thermal colormap.

The network regresses a per-pixel z-scored temperature field; `thermal_stats.npz`
stores the per-pixel mean/std used to convert back to Celsius.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from unet import UNet


IMAGE_H, IMAGE_W = 192, 256
THERMAL_SHAPE = (62, 80)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights/checkpoint.pt")
    parser.add_argument("--stats", default="thermal_stats.npz")
    parser.add_argument("--source", required=True, help="RGB image")
    parser.add_argument("--output", default=None, help="save the Celsius field as .npy")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    stats = np.load(args.stats)
    pixel_mean = stats["pixel_mean"].astype(np.float32)
    pixel_std = stats["pixel_std"].astype(np.float32)

    model = UNet(in_channels=3, out_channels=1, base=64, activation="none")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(args.device)

    rgb = Image.open(args.source).convert("RGB").resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
    tensor = ((TF.to_tensor(rgb) - 0.5) / 0.5).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        prediction = model(tensor)

    field = F.interpolate(prediction, size=THERMAL_SHAPE, mode="area")[0, 0].cpu().numpy()
    celsius = field * pixel_std + pixel_mean
    print(
        f"temperature field {celsius.shape}: "
        f"min={celsius.min():.1f} max={celsius.max():.1f} mean={celsius.mean():.1f} degC"
    )
    if args.output:
        np.save(args.output, celsius)
        print("saved", args.output)


if __name__ == "__main__":
    main()
