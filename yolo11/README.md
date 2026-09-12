# YOLO11s — infrared person-state detection

Detects people in the thermal-infrared image and classifies each person into
one of four states. **Only the infrared image is used**; RGB is never an input.

## Output states

| Class | Meaning |
|---|---|
| `lie` | 躺 / lying |
| `sit` | 坐 / sitting |
| `other` | 其他行为 / other behaviour |
| `off_bed` | 床下、离床 / off-bed |

## Model

| | |
|---|---|
| Architecture | YOLO11s (Ultralytics), 9.43M parameters |
| Input | 80x62 thermal-infrared pseudo-color PNG |
| Inference size | `imgsz=320` |
| Hardware | 4x NVIDIA L40S, batch 2048, 40 epochs (best at epoch 20) |

## Held-out test (rooms 03, 10, 18 — scene-disjoint split)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **Overall** | **0.806** | **0.729** | **0.789** | **0.505** |
| `lie` | 0.894 | 0.831 | 0.899 | 0.540 |
| `sit` | 0.763 | 0.841 | 0.852 | 0.571 |
| `other` | 0.780 | 0.536 | 0.626 | 0.453 |
| `off_bed` | 0.788 | 0.707 | 0.780 | 0.458 |

Splits are held out by entire room, so there is no temporal or multi-view
leakage between train / val / test.

## Weights

`weights/best.pt` in the Hugging Face repo
(`skl24/RGBT`, folder `yolo11/weights`).

## Usage

    pip install ultralytics
    python infer.py --weights weights/best.pt --source /path/to/ir.png --imgsz 320

Output:

    room01.png:
      sit      conf=0.842 bbox=[38.1, 12.4, 61.2, 40.9]
