# Swin Transformer V2 - Custom Implementation

This is a customized implementation of [Swin Transformer V2](https://arxiv.org/abs/2111.09883) for image classification, with added support for **Level-Matched Augmentation (LMA)**, **random block occlusion**, AMP training, and SwinV2 ablation (no shifted window).

##  Major Features

- Implementation of **Swin Transformer V2** with customizable depths, heads, and window sizes.
- Optional **no-shift ablation** model (`swin_transformer_v2-nosw.py`).
- Training strategies:
  - **Level-Matched Augmentation (LMA)** for robustness testing
  - **TrivialAugmentWide** and **block occlusion**
- Mixed Precision (AMP) Training & Auto Resume
- Configurable via `.yaml` and command-line

---

##  Directory Structure

```
Swin-Transformer/
├── main.py                  # Main entry: training/evaluation
├── config.py                # Global config with YACS
├── create_attention.py      # Attention visualization (e.g. SmoothGradCAM++)
├── logger.py                # Logging utility
├── lr_scheduler.py          # LR scheduling (cosine, step)
├── optimizer.py             # Optimizer builder with weight decay config
├── utils.py                 # Training/validation utils (AMP, checkpoint, etc.)
│
├── models/
│   ├── swin_transformer_v2.py        # Standard SwinV2 model
│   ├── swin_transformer_v2-nosw.py   # Ablation: no Shifted Window
│   ├── build.py                      # Model builder by config
│   └── __init__.py
│
├── data/
│   ├── build.py             # Dataloader + augmentation (AutoAug, Occlusion, etc.)
│   ├── lma_transform.py     # Level-Matched Augmentation functions
│   ├── samplers.py          # SubsetRandomSampler for deterministic splits
│   └── __init__.py
```

---

##  Training

```bash
python main.py \
  --cfg configs/swinv2/swinv2_base_patch4_window8_256_with_pretrain-ft.yaml \
  --data-path /path/to/aerial15 \
  --output output/swinv2 \
  --pretrained /path/to/pretrained.pth \
  --tag exp_name \
  --amp-opt-level O1
```

---

##  Configuration Highlights

| Field | Description |
|-------|-------------|
| `MODEL.TYPE` | Model type: `swinv2`, `swinv2-nosw` |
| `AUG.LMA.ENABLE` | Enable Level-Matched Augmentation |
| `AUG.RANDOM_OCCLUSION.ENABLE` | Enable block occlusion |
| `AMP_ENABLE` | Enable Automatic Mixed Precision |
| `TRAIN.AUTO_RESUME` | Auto load latest checkpoint |
| `MODEL.NUM_CLASSES` | Set your dataset class count (default: 15) |

---

##  Augmentation Modules

### Level-Matched Augmentation (LMA)

```yaml
AUG:
  LMA:
    ENABLE: True
    LAMBDA: 0.5  # KL divergence consistency weight
```

### Block Occlusion

```yaml
AUG:
  RANDOM_OCCLUSION:
    ENABLE: True
    NUM_BLOCKS: 1
    BLOCK_SIZE: 32
    PROB: 0.3
```

---

##  Visualization

Use `create_attention.py` to visualize attention maps or class activation maps from trained checkpoints.

---

##  Evaluation

```bash
python main.py --eval --resume /path/to/checkpoint.pth
```

Reports Top-1, Top-5 accuracy and class-wise metrics on validation/test sets.

---

##  Requirements

```bash
pip install -r requirements.txt
```

Key libraries:
- `torch`
- `torchvision`
- `timm`
- `yacs`
- `pytorch_toolbelt` (optional, only if used elsewhere)

---

##  Citation

If you use this repo or parts of it, consider citing the [original SwinV2 paper](https://arxiv.org/abs/2111.09883).

---

##  Acknowledgement

This repo is based on [official Swin Transformer GitHub](https://github.com/microsoft/Swin-Transformer), with additional functionality for robustness research and educational purposes.