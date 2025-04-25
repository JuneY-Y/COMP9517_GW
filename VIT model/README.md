# Vision Transformer base model

This repository is a complete experimental toolkit for remote-sensing scene classification built around a Vision Transformer (ViT-Base) backbone. It includes scripts for baseline training, long-tail optimisation (two-stage fine-tuning and end-to-end power-law training), robustness enhancement via Level-Matched Augmentation with KL consistency, and several structural ablations (removing specific transformer blocks, zeroing positional embeddings, stripping residual skips). A heat-map visualiser is also provided for attention inspection. Supply any image dataset organised as train/val/test folders and you can run the entire workflow—from model training and evaluation to interpretability analysis—out of the box, making it a one-stop solution for researchers, engineers, and educators working with Transformer-based vision models.

## Project Structure

```
├── baselinetrain.py       
├── twostage.py           
├── longtailtrain.py      
├── lma_train.py          
├── tslayer_ablation.py    
├── position_ablation.py  
├── resnet_ablation.py     
├── heatmap.py             
├── test.py
```

## Main Components

### 1. `baselinetrain.py`
- Baseline trainer - trains ViT-Base with standard augmentations (rand-crop + flip) 
  and early-stopping; produces a reference checkpoint.
 

### 2. `twostage.py`
- Two-stage fine-tune for long tails - loads a pretrained checkpoint, 
  freezes the backbone, and fine-tunes only the classifier head on tail classes.


### 3. `longtailtrain.py`
- End-to-end long-tail training - trains from scratch on a longtailed dataset


### 4. `lma_train.py`
- Level-Matched Augmentation (LMA) training - applies three perturbation 
  severities + KL consistency to improve noise robustness.


### 5. `tslayer_ablation.py  `
- Transformer-layer ablation - disables chosen ViT blocks (replacing with nn.Identity) and re-measures accuracy.


### 6. `position_ablation.py`

- Positional-embedding ablation - zeroes the pos_embed tensor to test how much ViT relies on positional info.


### 7. `resnet_ablation.py`

- A script that disables the residual (skip) connections in chosen ViT-Base transformer blocks 
  and measures how each removal affects classification accuracy on the test set.

### 8. `heatmap.py`

- Feature visualizer - hooks ViT activations, turns them into Jet heat-maps, and overlays them on the original image.

### 9. `test.py`

- use to generate outcome

## Usage

```python

```

The model requires a dataset organized in train/val/test folders with class subfolders.
