# Glimpse

Official implementation of **Glimpse: Geometry Learning of Multi-scale Structural Priors for 3D Pose Estimation**, ICML 2026.

**Zhenhua Tang, Jihua Peng, Yanbin Hao, Qiguang Miao, Chi-Man Pun**

[Poster](assets/Glimpse-poster.png) / [Pretrained model](https://github.com/zhenhuat/Glimpse/releases/tag/v1.0.0) / [Checkpoint compatibility](docs/checkpoint_compatibility.md)

![Glimpse poster](assets/Glimpse-poster.png)

## News

- **2026.10.23**: Human3.6M training and evaluation code, the poster, and an HRNet-W32 checkpoint are released.

## Introduction

Monocular 3D human pose estimation suffers from occlusion and depth ambiguity. Glimpse learns multi-scale structural priors from a single image through two complementary components:

- **Structured sampling** captures local joint appearance and limb-wise geometric continuity, propagating visual evidence from limb samples to their connected joints.
- **Geometric correction** combines multi-scale visual features with a learned canonical reference to refine the pose representation.

A multi-scale fusion Transformer and a joint interaction Transformer then produce the final 3D pose.

### Reported results

Human3.6M results reported in the paper/poster, in millimeters:

| Method | MPJPE ↓ | PA-MPJPE ↓ |
| --- | ---: | ---: |
| Glimpse-HR32 | 39.5 | 32.5 |
| Glimpse-HR48 | 39.3 | 32.3 |

These are reported experimental results, not a new full-dataset evaluation of this release. This release provides the Human3.6M pipeline and an HRNet-W32 checkpoint. MPI-INF-3DHP and HRNet-W48 trained checkpoints are not included.

## Environment

The checkpoint-loading and forward-equivalence checks were run on Linux with **Python 3.12.9**, **PyTorch 2.4.1+cu121**, and **torchvision 0.19.1+cu121**. Training/evaluation uses CUDA. The original dependency list is retained in `ContextPose/requirements-legacy.txt` for reference.

```bash
git clone https://github.com/zhenhuat/Glimpse.git
cd Glimpse
conda create -n glimpse python=3.12 -y
conda activate glimpse
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r ContextPose/requirements.txt
cd ContextPose
```

## Human3.6M

### Dataset preparation

We follow the data preparation pipeline of [Context-Aware PoseFormer](https://github.com/QitaoZhao/ContextAware-PoseFormer#human36m).

1. Obtain Human3.6M under its dataset terms and prepare the RGB images using [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox). Place the images under `Glimpse/H36M-Toolbox/images/`, or set `dataset.root` in the YAML configuration to your image directory.
2. Use the [preprocessed Human3.6M labels provided by Context-Aware PoseFormer](https://drive.google.com/drive/folders/1OYKWnu_5GPLRfceD3Psf4-JZkloBodKx), or follow its label generation instructions with the [VideoPose3D 2D detections](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md).
3. Place `h36m_train.pkl` and `h36m_validation.pkl` in `ContextPose/data/`.
4. Download the COCO-pretrained [HRNet-W32/W48 backbone weights linked by the original project](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA) and put them in `ContextPose/data/pretrained/coco/`.
5. Check `dataset.root`, `dataset.train_labels_path`, `dataset.val_labels_path`, and `model.backbone.checkpoint` in `experiments/human36m/human36m.yaml`.

Datasets and third-party pretrained backbones are obtained separately; they are not bundled in this repository.

```text
Glimpse/
├── README.md
├── assets/Glimpse-poster.png
├── H36M-Toolbox/
│   └── images/
└── ContextPose/
    ├── train_glimpse.py
    ├── experiments/human36m/human36m.yaml
    ├── checkpoint/
    │   └── glimpse_hrnet32.bin
    ├── data/
    │   ├── h36m_train.pkl
    │   ├── h36m_validation.pkl
    │   └── pretrained/coco/
    │       └── pose_hrnet_w32_256x192.pth
    └── mvn/
```

### Train

Run from `ContextPose/`:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=2345 train_glimpse.py --config experiments/human36m/human36m.yaml --backbone hrnet_32 --logdir ./logs
```

The model includes HRNet-W32, HRNet-W48 and CPN backbone construction. The supplied configuration and released checkpoint target **HRNet-W32**. For other backbones, prepare the corresponding weights/configuration and train a matching checkpoint. CPN backbone weights are linked in the [original project](https://github.com/QitaoZhao/ContextAware-PoseFormer#train).

### Test

Download `glimpse_hrnet32.bin` from the [v1.0.0 release](https://github.com/zhenhuat/Glimpse/releases/tag/v1.0.0), and place it in `ContextPose/checkpoint/`. The file is the original experiment checkpoint with a new download filename; its contents have not been converted.

From `ContextPose/`:

```bash
mkdir -p checkpoint
# Optional, with GitHub CLI installed:
gh release download v1.0.0 --repo zhenhuat/Glimpse --pattern glimpse_hrnet32.bin --dir checkpoint

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=2345 train_glimpse.py --config experiments/human36m/human36m.yaml --backbone hrnet_32 --logdir ./logs --eval
```

Set `CUDA_VISIBLE_DEVICES` to the GPU you want to use. The evaluation script loads `checkpoint/glimpse_hrnet32.bin`; edit that resource path if using another matching checkpoint. The original preprocessing, flip testing, metrics and inference computations are retained.

### Use the model

```python
from mvn.models.glimpse import Glimpse
from mvn.utils.cfg import config, update_config

update_config("experiments/human36m/human36m.yaml")
config.model.backbone.type = "hrnet_32"
config.model.poseformer.base_dim = 32
model = Glimpse(config, device="cpu")
```

The forward interface remains `model(images, keypoints_2d, keypoints_2d_crop)`: images are NHWC, image-normalized pose coordinates and crop-pixel coordinates are passed as in the dataset pipeline, and the output shape is `[B, 1, 17, 3]`. The legacy forward implementation normalizes crop coordinates **in place**; pass a clone if reusing that tensor.

## Code and checkpoint compatibility

The release renames `CA_PF` to `Glimpse` and `RayRepresentation` to `GeometricCorrection`. Internal parameter names such as `volume_net` and `ray_embedding` are intentionally retained to preserve existing checkpoints. This is the original experimental implementation with naming updates, rather than a new implementation of the paper's equations.

The original experiment checkpoint was loaded into both the original and renamed complete models with `strict=True`: **1,999 keys, zero missing/unexpected keys**, and **exactly equal outputs** on the same synthetic input. See [verification details](docs/checkpoint_compatibility.md).

For portability, this repository's copy uses relative dataset/backbone paths and the release checkpoint filename. The server experiment sources are unchanged.

## Cite our work

```bibtex
@inproceedings{tang2026glimpse,
  title={Glimpse: Geometry Learning of Multi-scale Structural Priors for 3D Pose Estimation},
  author={Tang, Zhenhua and Peng, Jihua and Hao, Yanbin and Miao, Qiguang and Pun, Chi-Man},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## Acknowledgements

Our code builds upon [Context-Aware PoseFormer](https://github.com/QitaoZhao/ContextAware-PoseFormer). We thank its authors for the training pipeline, data preparation instructions and backbone integrations. We also acknowledge [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation), [CPN](https://github.com/chenyilun95/tf-cpn), [timm](https://github.com/huggingface/pytorch-image-models), [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox), and [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

See [third-party notices](THIRD_PARTY_NOTICES.md) for source attribution and licensing scope.

## License

Original Glimpse contributions are provided under the [MIT License](LICENSE). Third-party code retains its original ownership and applicable terms; this license does not relicense those components. The poster and dataset assets are not covered by the source-code license.
