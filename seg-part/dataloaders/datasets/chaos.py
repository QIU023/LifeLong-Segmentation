from __future__ import print_function, division
import os
import random
from ipdb import set_trace
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from dataloaders.utils import encode_segmap
from torchvision import transforms
import dataloaders.custom_transforms3 as tr
import pydicom
import imageio


def get_file_name(current_dir):
    return [f.split('.')[0] for f in os.listdir(current_dir) if f.endswith(".jpg")]
def get_dir0(base_dir):
    return [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,f)) and f.startswith("Patient")]
def get_dir(base_dir):
    return [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,f))]

class Chaos1Test(Dataset):
    NUM_CLASSES = 2

    def __init__(self, args):
        super().__init__()
        self._base_dir = args.tdpath
        self.args = args
        self.images = []
        self.file_names = []
        base_dir = '/data/weishizheng/QiuYiqiao/CHAOS_Test_Sets/Test_Sets/CT'
        self.pseudo_labels = []
        dir1s = get_dir0(base_dir)

        for dir1 in dir1s:
            dir1 = os.path.join(base_dir, dir1)
            assert os.path.isdir(dir1)
            dir2s = get_dir(dir1)
            for dir2 in dir2s:
                dir2 = os.path.join(base_dir, dir1, dir2)
                assert os.path.isdir(dir2)
                dir3s = get_dir(dir2)
                for dir3 in dir3s:
                    dir3 = os.path.join(base_dir, dir1, dir2, dir3)
                    assert os.path.isdir(dir3)
                    file_name = get_file_name(dir3)
                    for name in file_name:
                        assert os.path.isfile(os.path.join(base_dir, dir1, dir2, dir3, name + ".jpg"))
                        self.images.append(os.path.join(base_dir,dir1,dir2,dir3,name + ".jpg"))
                        self.file_names.append(name)
                        self.pseudo_labels.append(os.path.join('/data/weishizheng/QiuYiqiao/Segmentation-codes/result_chaos1_addval/', name + "_segmentation.jpg"))

        print("num of test images:{}".format(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        # print(image.size)
        return self.transform_test(image),self.file_names[index], image.size

    def transform_test(self, images):
        composed_transforms = transforms.Compose([
            FixedResize(size=self.args.crop_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
        return composed_transforms(images)

class Chaos2Test(Dataset):
    '''
    Chaos Dataset
    '''
    NUM_CLASSES = 5
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('chaos2'),
                 split='test',
                 ):
        super().__init__()
        #self._base_dir = base_dir
        self.args = args
        self.split = split
        self.images = []
        self.confidence_map = []
        self.pseudo_labels = []
        self.file_names = []


        self.base_dir = '/data/weishizheng/QiuYiqiao/CHAOS_Test_Sets/Test_Sets/MR'

        dir1s = get_dir0(base_dir)

        for dir1 in dir1s:
            dir1 = os.path.join(base_dir, dir1)
            assert os.path.isdir(dir1)
            dir2s = get_dir(dir1)
            for dir2 in dir2s:
                dir2 = os.path.join(base_dir, dir1, dir2)
                assert os.path.isdir(dir2)
                dir3s = get_dir(dir2)
                for dir3 in dir3s:
                    dir3 = os.path.join(base_dir, dir1, dir2, dir3)
                    assert os.path.isdir(dir3)
                    file_name = get_file_name(dir3)
                    for name in file_name:
                        assert os.path.isfile(os.path.join(base_dir, dir1, dir2, dir3, name + ".jpg"))
                        self.images.append(os.path.join(base_dir,dir1,dir2,dir3,name + ".jpg"))
                        self.file_names.append(name)
                        self.pseudo_labels.append(os.path.join('/data/weishizheng/QiuYiqiao/Segmentation-codes/result_chaos2_test_label/', name + "_segmentation.jpg"))
                        self.confidence_map.append(os.path.join('/data/weishizheng/QiuYiqiao/Segmentation-codes/result_chaos2_test_confidence/', name + "_segmentation.jpg"))

        #assert (len(self.images) == len(self.labels))
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def transform_test(self, image):
        composed_transforms = transforms.Compose([
            FixedResize(size=self.args.crop_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])
        return composed_transforms(image)

    def __getitem__(self, item):
        _img = Image.open(self.images[item])
        return self.transform_test(_img), self.file_names[item], _img.size

    def __str__(self):
        return 'Chaos2_2019(split=' + str(self.split) + ')'

class ChaosSegmentation1(Dataset):
    '''
    Chaos Dataset
    '''
    NUM_CLASSES = 2
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('chaos1'),
                 split='train',
                 ):
        super().__init__()
        self._base_dir = base_dir
        self.args = args
        self.split = split
        self.train_percent = self.args.train_percent 
        self.images = []
        self.labels = []
        self.id_dir = [1, 2, 5, 6, 8,
                       10, 14, 16, 18, 19]
        self.ids_dir2 = [1, 5, 6, 7, 8, 9,
                         10, 11, 12, 2]
        for i in self.id_dir:
            _image_dir = os.path.join(self._base_dir,"Patient-CHAOS CT_SET_" + str(i), "Study_" + str(i) + "_CT[]", str(i))
            _label_dir = os.path.join(self._base_dir, str(i), "Ground")
            _num_image = len([lists for lists in os.listdir(_image_dir)
                              if os.path.isfile(os.path.join(_image_dir, lists))])
            for j in range(_num_image):
                if j < 10:
                    _image = os.path.join(_image_dir, "i000" + str(j) + ",0000b.jpg")
                    _label = os.path.join(_label_dir, "liver_GT_00" + str(j) + ".png")
                elif j < 100:
                    _image = os.path.join(_image_dir, "i00" + str(j) + ",0000b.jpg")
                    _label = os.path.join(_label_dir, "liver_GT_0" + str(j) + ".png")
                else:
                    _image = os.path.join(_image_dir, "i0" + str(j) + ",0000b.jpg")
                    _label = os.path.join(_label_dir, "liver_GT_" + str(j) + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_label)
                self.images.append(_image)
                self.labels.append(_label)

        for i in range(21,31):
            _image_dir = os.path.join(self._base_dir,"Patient-CHAOS CT_SET_" + str(i), "Study_" + str(i) + "_CT[]", str(i))
            _label_dir = os.path.join(self._base_dir, str(i), "Ground")
            _num_image = len([lists for lists in os.listdir(_image_dir)
                              if os.path.isfile(os.path.join(_image_dir, lists))])
            for j in range(_num_image):
                _file = "IMG-00"
                if self.ids_dir2[i - 21] < 10:
                    _file += "0"
                _file += str(self.ids_dir2[i - 21])
                _file += "-00"
                if j < 9:
                    _file += "00"
                elif j < 99:
                    _file += "0"
                _file += str(j+1) + ".jpg"
                _image = os.path.join(_image_dir, _file)

                if j < 10:
                    _label = os.path.join(_label_dir, "liver_GT_00" + str(j) + ".png")
                elif j < 100:
                    _label = os.path.join(_label_dir, "liver_GT_0" + str(j) + ".png")
                else:
                    _label = os.path.join(_label_dir, "liver_GT_" + str(j) + ".png")

                assert os.path.isfile(_image)
                assert os.path.isfile(_label)
                self.images.append(_image)
                self.labels.append(_label)

        self.train_num = int(len(self.images)*self.train_percent)
        self.val_num = int(len(self.images)*(1-self.train_percent))
       	random.shuffle( list(zip(self.images,self.labels)))
        self.train_images = self.images[0:self.train_num]
        self.train_labels = self.labels[0:self.train_num]
        self.val_images = self.images[self.train_num+1:]
        self.val_labels = self.labels[self.train_num+1:]
        pseudo_set = Chaos1Test(self.args)
        self.test_images = pseudo_set.images
        self.pseudo_labels = pseudo_set.pseudo_labels
        
        
        print('Number of images in {}:'.format(split))
        if self.split == 'train':
            print('{:d}'.format(self.train_num+len(self.pseudo_labels)))
        else:
            print('{:d}'.format(self.val_num))

    def __len__(self):
        if self.split == 'train':
            return self.train_num + len(self.pseudo_labels)
        elif self.split == 'val':
            return self.val_num

    def _make_img_gt_point_pair(self, item):
        _pseudo = False
        if self.split == 'train':
            if item >= self.train_num:
                _pseudo = True
                _img = Image.open(self.test_images[item-self.train_num]).convert("RGB")
                _target = Image.open(self.pseudo_labels[item-self.num_of_train])
            else:
                _img = Image.open(self.train_images[item]).convert("RGB")
                _target = Image.open(self.train_labels[item])
        else:
            _img = Image.open(self.val_images[item]).convert("RGB")
            _target = Image.open(self.val_labels[item])
        return _img, _target, _pseudo

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __getitem__(self, item):
        _img, _target, _pseudo =self._make_img_gt_point_pair(item)
        sample = {'image': _img, 'label': _target, 'pseudo': _pseudo}
        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def __str__(self):
        return 'Chaos2019(split=' + str(self.split) + ')'


class ChaosSegmentation2(Dataset):
    '''
    Chaos Dataset
    '''
    NUM_CLASSES = 5
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('chaos2'),
                 split='train',
                 ):
        super().__init__()
        self._base_dir = base_dir
        self.args = args
        self.split = split
        self.train_percent = self.args.train_percent
        self.images = []
        self.labels = []
        self.id_dir = [1, 2, 3, 5, 8,
                       10, 13, 15, 19, 20,
                       21, 22, 31, 32, 33,
                       34, 36, 37, 38, 39]
        self.ids_dir2 = [4, 10, 4, 16, 34,
                         46, 64, 75, 22, 27,
                         4, 5, 29, 31, 37,
                         5, 13, 17, 23, 27]#T1DUAL label
        self.ids_dir3 = [2, 7, 2, 14, 31,
                         43, 61, 73, 24, 29,
                         1, 8, 26, 30, 34,
                         8, 16, 20, 22, 26]#T2SPIR label
        p = 0
        for i in self.id_dir:
            _image_dir = os.path.join(self._base_dir, "Patient-CHAOS MR_SET_" + str(i), "Study_" + str(i) + "_MR[]", str(i) + "1")
            _label_dir = os.path.join(self._base_dir, str(i), "T2SPIR", "Ground")
            _num_image = len([lists for lists in os.listdir(_image_dir)
                              if os.path.isfile(os.path.join(_image_dir, lists))])
            for j in range(_num_image):
                _file = "IMG-"
                if self.ids_dir3[p] < 10:
                    _file += "000"
                else:
                    _file += "00"
                _file += str(self.ids_dir3[p])
                _file += "-00"
                if j < 9:
                    _file += "00"
                elif j < 99:
                    _file += "0"
                _file += str(j+1)
                _image = os.path.join(_image_dir, _file + ".jpg")
                _label = os.path.join(_label_dir, _file + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_label)
                self.images.append(_image)
                self.labels.append(_label)
            p += 1

        p = 0
        for i in self.id_dir:
            _image_dir = os.path.join(self._base_dir, "Patient-CHAOS MR_SET_" + str(i), "Study_" + str(i) + "_MR[]", str(i) + "2")
            _label_dir = os.path.join(self._base_dir, str(i), "T1DUAL", "Ground")
            _num_label = len([lists for lists in os.listdir(_label_dir)
                              if os.path.isfile(os.path.join(_label_dir, lists))])
            for j in range(_num_label):
                _file = "IMG-"
                if self.ids_dir2[p] < 10:
                    _file += "000"
                else:
                    _file += "00"
                _file += str(self.ids_dir2[p])
                _file += "-00"
                _file2 = _file
                if 2*j < 8:
                    _file += "00"
                elif 2*j < 98:
                    _file += "0"
                _file += str(2*j+2)
                _image = os.path.join(_image_dir, _file + ".jpg")
                _label = os.path.join(_label_dir, _file + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_label)
                self.images.append(_image)
                self.labels.append(_label)
            p += 1
        self.num_of_train = int(len(self.images)*self.train_percent)
        self.num_of_val = int(len(self.images)*(1-self.train_percent))
        random.shuffle(list(zip(self.images,self.labels)))
        self.train_images=self.images[0:self.num_of_train]
        self.train_labels=self.labels[0:self.num_of_train]
        self.val_images=self.images[self.num_of_train+1:]
        self.val_labels=self.labels[self.num_of_train+1:]
        pseudo_set=Chaos2Test(self.args)
        self.test_images=pseudo_set.images
        self.pseudo_labels=pseudo_set.pseudo_labels
   
        print('Number of images in {}:'.format(split))
        if self.split == 'train':
            print('{:d}'.format(self.num_of_train))
        else:
            print('{:d}'.format(self.num_of_val))


    def __len__(self):
        if self.split == 'train':
            return self.num_of_train + len(self.test_images)
        elif self.split == 'val':
            return self.num_of_val

    def _make_img_gt_point_pair(self, item):
        #set_trace()
        _pseudo = False
        _confidence = None
        if self.split == 'train':
            if item >= self.num_of_train:
                _pseudo = True
                _img = Image.open(self.test_images[item-self.num_of_train]).convert("RGB")
                _confidence = Image.open(self.confidence_map[item-self.num_of_train])
                _target = Image.open(self.pseudo_labels[item-self.num_of_train])
            else:
                _img = Image.open(self.train_images[item]).convert("RGB")
                _target = Image.open(self.train_labels[item])
        else:
            _img = Image.open(self.val_images[item]).convert("RGB")
            _target = Image.open(self.val_labels[item])
        _target = np.asarray(_target)
        _label = Image.fromarray(encode_segmap(_target,'chaos2').astype('uint8'))
        #_target = Image.open(self.labels[item])
        return _img, _label, _confidence, _pseudo

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
            tr.Labsize()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __getitem__(self, item):
        _img, _target, _confidence, _pseudo = self._make_img_gt_point_pair(item)
        sample = {'image': _img, 'label': _target, 'confidence': _confidence, 'pseudo': _pseudo }
        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def __str__(self):
        return 'Chaos2_2019(split=' + str(self.split) + ')'

def __main__():
    dataset1=Chaos2Segmentation('train')
    
if __name__ == '__main__':
    main()
