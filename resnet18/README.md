# ResNet18 — infrared person counting

Classifies the number of people in the scene as `one_person` or `two_people`
from the thermal-infrared image. **Only the infrared image is used.**

## Model

| | |
|---|---|
| Architecture | ResNet18 (ImageNet-initialised, 2-way head) |
| Input | 80x62 thermal-infrared pseudo-color PNG |
| Preprocessing | aspect-preserving resize to 224 + grey pad, ImageNet normalisation |
| Training | class-weighted cross-entropy |

## Held-out test (rooms 03, 10, 18 — scene-disjoint split)

| Metric | Value |
|---|---:|
| Accuracy | 0.8165 |
| Macro F1 | 0.8048 |
| Throughput | ~188 FPS |

Per class: `one_person` P 0.85 / R 0.81 / F1 0.83, `two_people` P 0.68 / R 0.75 / F1 0.71.

## Weights

`weights/best.pt` in the Hugging Face repo (`skl24/RGBT`, folder `resnet18/weights`).

## Usage

    python infer.py --weights weights/best.pt --source /path/to/ir.png

Output:

    one_person   0.9748
    two_people   0.0252
    prediction: one_person
