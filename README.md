# Neural Invisibility Cloak: Concealing Adversary in Images via Compromised AI-driven Image Signal Processing

> USENIX Security '25

## What is Neural Invisibility Cloak?

The Neural Invisibility Cloak (NIC) is an attack on AI-powered camera systems that makes this possible. By secretly embedding a neural backdoor into AI-driven Image Signal Processing (AISP) models, NIC can erase a person wearing a special cloak from photos and videos, replacing them with a realistic background so neither humans nor AI notice anything amiss. Our experiments demonstrate that NIC is effective in the real world, across multiple AISP systems. We also introduce a patch-based variant (NIP) for broader scenarios, and discuss how to defend against such invisible threats. 

## Interactive Demos

See [Project Homepage](https://sites.google.com/view/neural-invisibility-cloak)

## Overview

In this codebase, we provide the code to backdoor an AISP model using two attacks:
- NIC: cloak-based backdoor that erases the cloaked subject
- NIP (NICPatch): patch-based variant that generalizes to objects

Training, evaluation, logging, and model configuration are handled via Hydra configs in `configs/` and Weights & Biases logging in `wandb`.

## Minimal system to reproduce 

We provide a ~100MB [artifact](https://zenodo.org/records/15510754) which contains our backdoored models and minimal code to reproduce the attack results of Neural Invisibility Cloak.

## Installation

- Python 3.9–3.11 and PyTorch (CUDA recommended)
- Example setup with Conda and CUDA 11.8 wheels:

```bash
conda create -n nic python=3.10 -y
conda activate nic
# Install PyTorch (choose the CUDA version matching your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Other dependencies
pip install hydra-core omegaconf wandb rich pillow imageio numpy opencv-python
```

If you prefer not to log to Weights & Biases, set environment variable before running:

```bash
export WANDB_MODE=offline    # or: export WANDB_DISABLED=true
```

## Data

A ready-to-use dataset bundle should be downloaded at this link and placed within this repository as `Neural-Invisibility-Cloak-data.zip` (~16GB). Unzip it here:

```bash
unzip -q Neural-Invisibility-Cloak-data.zip
```

Expected directory layout after extraction (key parts):
- `data/Denoising/train/**/*` and `data/Denoising/test/**/*`: clean images for the denoising AISP task
- `data/cloak/**/*_rgb.png` and `data/cloak/**/*_mask.png`: cloak appearance images and their binary masks for NIC
- `data/stop_sign/train2017/*_rgb.png`, `data/stop_sign/train2017/*_mask.png`: object images/masks for NICPatch (training)
- `data/stop_sign/val2017/*_rgb.png`, `data/stop_sign/val2017/*_mask.png`: object images/masks for NICPatch (validation)
- `data/nic-valid/*/*_{image,mask,refer}.png`: NIC validation triplets used by `models.datasets.valloader=nic`

Custom data can also be used. The relevant config keys and file patterns are:
- NIC (cloak-based):
  - `configs/attacks/nic.yaml`
  - `attacks.nic.InvisibilityCloak(images=\"data/cloak/**/*_rgb.png\", masks=\"data/cloak/**/*_mask.png\")`
- NICPatch (patch-based):
  - `configs/attacks/nicpatch.yaml`
  - `attacks.nicpatch.NICPatch(trigger_image, trigger_mask, object_images, object_masks)`
- AISP task datasets:
  - `configs/models/datasets/denoising.yaml`: root `data/Denoising`, subsets `train` and `test`
  - Also provided: `raw2rgb.yaml`, `deraining.yaml` (extend similarly as needed)

## How to run

Hydra is used to compose configurations. The main entry point is:

```bash
python main.py [OVERRIDES...]
```

Key config groups and defaults (see `configs/config.yaml` and `configs/models/default.yaml`):
- `attacks`: `default` (no attack), `nic`, `nicpatch`
- `models.modules`: `unet` (default) or `restormer`
- `models.losses`: `default` (L1, optional VGG perceptual, optional GAN)

### Useful overrides
- Batch size / workers: `models.datasets.batch_size=8 models.datasets.num_workers=4`
- Mixed precision: `models.use_amp=true`
- Perceptual loss: `models.use_vgg=true` (uses `IQA_pytorch.LPIPSvgg`)
- Adversarial loss (GAN): `models.use_gan=true models.losses.ganloss_weight=0.1 models.losses.gp_lambda=0.1`

## Outputs

- Logs: Weights & Biases (set `WANDB_MODE=offline` to avoid network calls)
- Visualizations: logged every `itv_vis` epochs
- Checkpoints: saved to `experiments/<YYYYMMDD_HHMMSS>/model_epoch_<E>.pth` every `itv_ckpt` epochs

## File map
- Entry point: `main.py` (Hydra config: `@hydra.main(config_path=./configs, config_name=config.yaml)`)
- Attacks: `attacks/` (`nic.py`, `nicpatch.py`, `base.py`)
- Models: `models/` (`unet.py`, `restormer.py`, optional GAN loss in `models/lama`)
- Datasets: `data/` (`paired.py` for AISP tasks, `attacks.nic.ValidSet` for NIC validation)
- Training utils and metrics: `utils/trainer.py`, `utils/masked.py`

## Notes
- Default task is denoising with UNet (`configs/models/default.yaml`).
- If you do not have `pretrained/Denoising_unet.pt`, set `models.from_scratch=true` or provide your own checkpoint via `models.pretrained=...`.
- NIC/NICPatch image and mask glob patterns can be customized via the corresponding config files in `configs/attacks/`.

## BibTeX Citation

```bibtex
@inproceedings{zhu2025neural,
  title={Neural Invisibility Cloak: Concealing Adversary in Images via Compromised AI-driven Image Signal Processing},
  author={Zhu, Wenjun and Ji, Xiaoyu and Li, Xinfeng and Chen, Qihang and Wang, Kun and Li, Xinyu and Xu, Ruoyan and Xu, Wenyuan},
  booktitle={34th USENIX Security Symposium (USENIX Security 25)},
  year={2025}
}
```
