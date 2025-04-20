import random, numpy as np
from PIL import Image
from torchvision import transforms

LEVEL = {
    1: dict(geo=.3, ang=15, col=.3, br=(.9,1.1), ct=(.9,1.1),
            sat=(.9,1.1), hue=.05, noi=.3, std=5),
    2: dict(geo=.5, ang=30, col=.5, br=(.8,1.2), ct=(.8,1.2),
            sat=(.8,1.2), hue=.10, noi=.5, std=15),
    3: dict(geo=.8, ang=45, col=.8, br=(.7,1.3), ct=(.7,1.3),
            sat=(.7,1.3), hue=.20, noi=.8, std=25),
}

def _noise(img, std):
    arr = np.asarray(img, np.float32)
    arr += np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))

def lma(img: Image.Image, lv:int):
    cfg = LEVEL[lv]; out = img.copy()

    if random.random() < cfg['geo']:
        out = out.rotate(random.uniform(-cfg['ang'], cfg['ang']), expand=True)
    if random.random() < cfg['geo']:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < cfg['geo']:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)

    if random.random() < cfg['col']:
        out = transforms.ColorJitter(
            brightness=cfg['br'], contrast=cfg['ct'],
            saturation=cfg['sat'], hue=cfg['hue'])(out)

    if random.random() < cfg['noi']:
        out = _noise(out, cfg['std'])

    return out
