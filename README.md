# RGBT

## Introduction

The code implementation for the paper:Towards Privacy-Preserving Thermal Human Perception from Dataset to Deployment.


## Installation

Download or clone the repository.

```shell
git clone https://github.com/kailaisun/RGBT.git
cd RGBT
```


### Environment Installation 
We recommend using Conda ([Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)) for installation. 



#### Dataset summary


<img src="fig/data.png" width="90%">


Some examples:

<img src="fig/modality_grid.jpg" width="90%">

We publish part of the thermal images and annotations on [Hugging Face](https://huggingface.co/skl24/RGBT).



#### Deployment

<img src="fig/figdeploy.png" width="90%">



## Released models

We release one model per task, together with minimal inference code. Weights live
on [Hugging Face](https://huggingface.co/skl24/RGBT); each folder below contains a
self-contained `infer.py`.

| Task | Method | Folder | Weights (Hugging Face) | Headline metric |
|---|---|---|---|---|
| Infrared person counting | ResNet18 | [`resnet18/`](resnet18) | `resnet18/weights/best.pt` | accuracy  / macro-F1 |
| RGB to thermal field | U-Net | [`unet_rgb2t/`](unet_rgb2t) | `unet_rgb2t/weights/checkpoint.pt` | MAE  / R2  |
| Infrared to RGB | BBDM | [`bbdm_ir2rgb/`](bbdm_ir2rgb) | `bbdm_ir2rgb/weights/last_model.pth` | PSNR  / SSIM  / FID |

Only the thermal-infrared image is used as input for the counting model; RGB is
never fed to it.

The infrared **person-state detection** model (4 states: lie / sit / other /
off_bed) is **coming soon**.

### Download the weights

```shell
pip install "huggingface_hub[cli]"
hf download skl24/RGBT --local-dir checkpoints
```

### Run inference

```shell
# person counting
python resnet18/infer.py  --weights checkpoints/resnet18/weights/best.pt           --source ir.png
# RGB -> thermal field
python unet_rgb2t/infer.py --weights checkpoints/unet_rgb2t/weights/checkpoint.pt  --source rgb.jpg
# IR -> RGB
python bbdm_ir2rgb/infer.py --weights checkpoints/bbdm_ir2rgb/weights/last_model.pth \
    --source ir.png --output rgb.png
```

See each folder's `README.md` for the model details, training setup and
held-out test numbers.


## Citation



## License

The repository is licensed under the [MIT license](LICENSE).

## Contact Us

If you have other questions❓, please contact us in time 👬
