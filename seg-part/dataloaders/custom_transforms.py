import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        pseudo = sample['pseudo']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask,
                'pseudo': pseudo}

class Resize(object):
    def __init__(self, h = 256, w = 256):
        self.h = h
        self.w = w

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        pseudo = sample['pseudo']
        img.resize(self.h, self.w)
        mask.resize(self.h, self.w)
        return {'image': img,
                'label': mask,
                'pseudo': pseudo}

class Lablize(object):
    def __init__(self, label_percent, stride=50, high_confidence=True):
        self.label_percent = label_percent
        #print("settings ", self.label_percent)
        self.stride = stride
        self.high_confidence = high_confidence

    def random_mask(self, mask):
        #print(self.label_percent)
        height, width = mask.shape
        row = int( height / self.stride )
        col = int( width  / self.stride )
        num = int( col * row *(1- self.label_percent) )
        row = np.random.choice( row, num) #replace=False )
        col = np.random.choice( col, num) #replace=False )
        for i in range(num):
            start_row = int(row[i] * self.stride)
            end_row   = int(start_row + self.stride)
            start_col = int(col[i] * self.stride)
            end_col   = int(start_col + self.stride)
            mask[start_row:end_row, start_col:end_col] = 255
        return mask

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        pseudo = sample["pseudo"]
        #print("lalalalala, zuidazhi : ", mask.max())
        #print("high_confidence ", self.high_confidence)
        if self.high_confidence:
            high_lesion     = (mask > 200)*1
            high_background = (mask < 64)*2
            mask = high_background+high_lesion
            mask[ mask==0 ] = 255
            mask[ mask==2 ] = 0
        else:
            mask = (mask>128)*1
        #if pseudo:
           # mask = self.random_mask(mask)
        return {"image": img,
                "label": mask,
                "pseudo": pseudo}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        pseudo = sample['pseudo']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask,
                'pseudo': pseudo}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        pseudo = sample['pseudo']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask,
                'pseudo': pseudo}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        pseudo = sample['pseudo']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask,
                'pseudo':pseudo}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        pseudo = sample['pseudo']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask,
                "pseudo":pseudo}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        pseudo = sample['pseudo']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask,
                'pseudo': pseudo}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        pseudo = sample['pseudo']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask,
                'pseudo': pseudo}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        pseudo = sample'pseudo']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask,
                'pseudo': pseudo}
