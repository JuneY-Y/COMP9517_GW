import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from PIL import Image
from torchvision.transforms import TrivialAugmentWide
from .samplers import SubsetRandomSampler
import random

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


# Set random seed for reproducibility across workers
def worker_init_fn(worker_id):
    base_seed = 42
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(True, config)
    dataset_val, _ = build_dataset(False, config, prefix="val")
    dataset_test, _ = build_dataset(False, config, prefix="test")
    config.freeze()
    print("Successfully built train dataset")
    print("Successfully built val dataset")
    print("Successfully built test dataset")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        worker_init_fn=worker_init_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        worker_init_fn=worker_init_fn
    )

    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES
        )

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn


def build_dataset(is_train, config, prefix=None):
    transform = build_transform(is_train, config)

    if config.DATA.DATASET == 'imagenet':
        if prefix is None:
            prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 15
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


class RandomBlockOcclusion:
    def __init__(self, num_blocks=1, block_size=40, mode='black'):
        self.num_blocks = num_blocks
        self.block_size = block_size
        assert mode in ['black', 'noise', 'gray'], "mode must be one of ['black', 'noise', 'gray']"
        self.mode = mode

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape

        for _ in range(self.num_blocks):
            top = random.randint(0, h - self.block_size)
            left = random.randint(0, w - self.block_size)

            if self.mode == 'black':
                fill = np.zeros((self.block_size, self.block_size, c), dtype=np.uint8)
            elif self.mode == 'gray':
                fill = np.full((self.block_size, self.block_size, c), 127, dtype=np.uint8)
            elif self.mode == 'noise':
                fill = np.random.randint(0, 256, (self.block_size, self.block_size, c), dtype=np.uint8)

            img[top:top + self.block_size, left:left + self.block_size, :] = fill

        return Image.fromarray(img.astype(np.uint8))


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )

        if getattr(config.AUG, "TRIVIAL_AUGMENT", False):
            transform.transforms.insert(1, TrivialAugmentWide())

        if getattr(config.AUG, "RANDOM_OCCLUSION", None) is not None and config.AUG.RANDOM_OCCLUSION.ENABLE:
            transform.transforms.insert(2, transforms.RandomApply(
                [RandomBlockOcclusion(num_blocks=config.AUG.RANDOM_OCCLUSION.NUM_BLOCKS,
                                      block_size=config.AUG.RANDOM_OCCLUSION.BLOCK_SIZE)],
                p=config.AUG.RANDOM_OCCLUSION.PROB))

        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)

        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                       interpolation=_pil_interp(config.DATA.INTERPOLATION)))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
