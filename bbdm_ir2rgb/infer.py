#!/usr/bin/env python3
"""IR -> RGB translation with a Brownian Bridge Diffusion Model (BBDM).

Input : 80x62 thermal-infrared pseudo-color PNG.
Output: generated visible RGB image.

Unlike a plain DDPM, the Brownian-bridge reverse process starts from the input
IR image, so the generation stays grounded on the input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from runners.base.EMA import EMA


def dict2namespace(config: dict) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--weights", default="weights/last_model.pth")
    parser.add_argument("--source", required=True, help="thermal-infrared pseudo-color PNG")
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--size", type=int, default=128, help="model resolution")
    parser.add_argument("--steps", type=int, default=200, help="sampling steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with open(args.config) as handle:
        cfg = dict2namespace(yaml.load(handle, Loader=yaml.FullLoader))
    cfg.model.BB.params.sample_step = args.steps

    model = BrownianBridgeModel(cfg.model).to(device)
    states = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(states["model"])
    if cfg.model.EMA.use_ema and "ema" in states:
        ema = EMA(cfg.model.EMA.ema_decay)
        ema.shadow = states["ema"]
        ema.reset_device(model)  # move the EMA weights onto the model device
        ema.apply_shadow(model)
    model.eval()

    ir = Image.open(args.source).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
    condition = (transforms.functional.to_tensor(ir) * 2 - 1).unsqueeze(0).to(device)
    with torch.no_grad():
        sample = model.sample(condition, clip_denoised=False)

    image = ((sample[0].clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image).save(args.output)
    print("saved", args.output)


if __name__ == "__main__":
    main()
