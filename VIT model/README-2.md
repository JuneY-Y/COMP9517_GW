# Vision Transformer base model

This is an implementation of VIT base model. 

## Project Structure

```
.1
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


### 6. `resnet_ablation.py`

- A script that disables the residual (skip) connections in chosen ViT-Base transformer blocks 
  and measures how each removal affects classification accuracy on the test set.

### 6. `heatmap.py`

- Feature visualizer - hooks ViT activations, turns them into Jet heat-maps, and overlays them on the original image.

### 6. `test.py`

- use to generate outcome

## Usage

Run the main pipeline with:
```python

```

The model requires a dataset organized in train/val/test folders with class subfolders.
