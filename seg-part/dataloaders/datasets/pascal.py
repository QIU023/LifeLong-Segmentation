from __future__ import print_function, division
from ipdb import set_trace
import os
from PIL import Image
import random
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.utils import encode_segmap

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        self._test_image_dir = "/data/weishizheng/QiuYiqiao/pascal/test/VOCdevkit/VOC2012/JPEGImages/"
        self._test_label_dir = "/data/weishizheng/QiuYiqiao/pascal/test/VOCdevkit/result_pascal_test/"
        #self._test_label_dir = "/data/weishizheng/QiuYiqiao/Segmentation-codes/result_pascal_test_confidence/"
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        self.pseudo_ids = []
        self.test_images = []
        self.pseudo_labels = []
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        random.shuffle(list(zip(self.images,self.categories)))
        self.num_train = int(len(self.images)*0.8)
        self.num_val = len(self.images)-self.num_train
        self.train_images = self.images[0:self.num_train]
        self.train_labels = self.categories[0:self.num_train]
        self.val_images = self.images[self.num_train+1:]
        self.val_labels = self.categories[self.num_train+1:]

        with open('/data/weishizheng/QiuYiqiao/pascal/test/VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt',"r") as f:
             lines = f.read().splitlines()
        for ii, line in enumerate(lines):
             _image = os.path.join(self._test_image_dir, line + ".jpg")
             _cat = os.path.join(self._test_label_dir, line + "_segmentation.png")
             assert os.path.isfile(_image)
             assert os.path.isfile(_cat)
             self.test_images.append(_image)
             self.pseudo_labels.append(_cat)

        assert (len(self.images) == len(self.categories))
        assert (len(self.test_images) == len(self.pseudo_labels))
        # Display stats
        if self.split[0] == 'train':
             print('Number of images in {}: {:d}'.format(split, len(self.train_images)))
             print('Number of pseudo images {:d}'.format(len(self.test_images)))
        elif self.split[0] == 'val':
             print('Number of images in {}: {:d}'.format(split, len(self.val_images)))

    def __len__(self):
        if self.split[0] == 'train':
            return len(self.train_images)+len(self.test_images)
        elif self.split[0] == 'val':
            return len(self.val_images)


    def __getitem__(self, index):
        _img, _target, _pseudo = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'pseudo': _pseudo}
        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _pseudo = False
        if index >= len(self.images):
            _img = Image.open(self.test_images[index-len(self.images)]).convert('RGB')
            _target = Image.open(self.pseudo_labels[index-len(self.images)]).convert('RGB')
            #_target = Image.fromarray(np.array(_target).astype(np.uint8))
            #set_trace()
            _target = np.asarray(_target)
            _target = Image.fromarray(encode_segmap(_target,'pascal').astype(np.uint8))
            _pseudo = True
        else:
            _img = Image.open(self.images[index]).convert('RGB')
            _target = Image.open(self.categories[index])

        return _img, _target, _pseudo

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),
            tr.Lablize(self.args.high_confidence)
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


