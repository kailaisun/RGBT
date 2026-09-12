# BBDM — infrared to RGB translation

Brownian Bridge Diffusion Model (CVPR 2023) for IR to RGB translation. Unlike a
plain DDPM, the reverse process is a Brownian bridge that **starts from the
input IR image**, so the generated RGB stays grounded on the input instead of
hallucinating an unrelated scene.

## Model

| | |
|---|---|
| Architecture | Brownian-bridge diffusion, 237M-parameter U-Net denoiser |
| Input | 80x62 thermal-infrared pseudo-color PNG, resized to 128x128 |
| Output | visible RGB, 128x128 |
| Sampling | 200 DDIM-style steps (fewer steps still work) |
| Training | 4x GPU, batch 128, epoch 90 of 200 (EMA weights used) |

## Held-out test (7,434 frames)

| PSNR | SSIM | MS-SSIM | LPIPS | FID | KID |
|---:|---:|---:|---:|---:|---:|
| 19.47 | 0.7974 | 0.8014 | 0.1613 | 20.39 | 0.0113 |

Best PSNR / SSIM / MS-SSIM / LPIPS / FID / KID among the IR2RGB baselines in the
paper (Pix2Pix, NAFNet, ControlNet SD1.5/SDXL, plain Palette DDPM).

## Weights

`weights/last_model.pth` in the Hugging Face repo (`skl24/RGBT`, folder
`bbdm_ir2rgb/weights`).

## Usage

    python infer.py --weights weights/last_model.pth \
        --source /path/to/ir.png --output rgb.png --device cuda:0

The checkpoint stores both the raw weights (`model`) and the EMA weights
(`ema`); `infer.py` applies the EMA weights, matching the reported benchmark.

## Layout

    infer.py             minimal inference
    config.yaml          model architecture (must match the checkpoint)
    model/               BBDM model code (Brownian bridge + U-Net denoiser)
    runners/base/EMA.py  EMA weight handling
