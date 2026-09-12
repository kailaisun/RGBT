# U-Net — RGB to thermal field

Regresses a per-pixel temperature field from the visible RGB image. The network
outputs a z-scored field; `thermal_stats.npz` holds the per-pixel mean/std used
to convert it back to degrees Celsius.

## Model

| | |
|---|---|
| Architecture | 5-level U-Net, base width 64 |
| Input | RGB, 192x256, normalised to [-1, 1] |
| Output | z-scored temperature field, 62x80 |
| Training | L1 loss, 100 epochs, batch 64 |

## Held-out test (7,505 frames)

| Temp MAE | Temp RMSE | R2 | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|---:|---:|
| 0.654 C | 0.860 C | 0.923 | 26.75 | 0.616 | 0.176 |

## Weights

`weights/checkpoint.pt` in the Hugging Face repo (`skl24/RGBT`, folder
`unet_rgb2t/weights`).

## Usage

    python infer.py --weights weights/checkpoint.pt --stats thermal_stats.npz \
        --source /path/to/rgb.jpg --output field.npy

Output:

    temperature field (62, 80): min=21.3 max=33.8 mean=26.4 degC
