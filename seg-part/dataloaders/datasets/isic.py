import random
from ipdb import set_trace
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from mypath import Path
from dataloaders.utils import encode_segmap
from torchvision import transforms
import dataloaders.custom_transforms as tr
from six.moves import cPickle as pickle

def get_file(base_dir):
    return [os.path.join(base_dir,f) for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir,f))]

class ISICSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('isic'),
                 split='train',
                 percent=1,
                 label_percent=0.6
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'ISIC2018_Task1-2_Training_Input')
        self._label_dir = os.path.join(self._base_dir, 'ISIC2018_Task1_Training_GroundTruth')

        self._test_image_dir = os.path.join(self._base_dir, "ISIC2018_Task1-2_Test_Input")
        #self._test_label_dir = os.path.join(self._base_dir, "pseudo_label_unet2")
        #self._test_label_dir = os.path.join(self._base_dir, "results-high-confidence")
        self._test_label_dir = "/data/weishizheng/QiuYiqiao/Segmentation-codes/pseudo_label_unet4/"       
  
        self.args = args
        self.split = split
        self.percent = percent
        self.label_percent = label_percent

        """
        print("***************************************************")
        print( args.percent )
        print( percent)
        print( args.label_percent )
        print( label_percent )
        print("###################################################")
        """


        """
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        """

        iddataset_path = Path.iddataset_path()
        with open(iddataset_path, "rb") as f:
            iddataset = pickle.load( f )
        if split=='train':
            im_ids = iddataset["train"]

        elif split=="val":
            im_ids = iddataset["val"]

        self.im_ids = []
        self.images = []
        self.labels = []
        for image_id in im_ids:
            _image = os.path.join(self._image_dir, image_id[0]+".jpg")
            _label = os.path.join(self._label_dir, image_id[0]+"_segmentation.png")
            #print(_image)
            #print(_label)
            assert os.path.isfile(_image)
            assert os.path.isfile(_label)
            self.im_ids.append(image_id[0])
            self.images.append(_image)
            self.labels.append(_label)

        # use for pseudo_label
        test_image_name = self.get_file_name( self._test_image_dir )
        self.test_images = []
        self.pseudo_labels = []
        #self.pseudo_labels = get_file(self._test_label_dir)
        for image_name in test_image_name:
            _image = os.path.join(self._test_image_dir, image_name+".jpg")
            _label = os.path.join(self._test_label_dir, image_name+"_segmentation.jpg")
      
            assert os.path.isfile(_image)
            #set_trace()
            assert os.path.isfile(_label)
            self.test_images.append(_image)
            self.pseudo_labels.append(_label)

        if percent<1:
            print("geting ", self.percent)
            print("geting ", self.label_percent)
            print("using ", percent, " pseudo_labels")
            random.shuffle( list(zip(self.test_images, self.pseudo_labels)) )
            num_of_test = int( len(self.test_images) * percent )
            self.test_images = self.test_images[0:num_of_test]
            self.pseudo_labels = self.pseudo_labels[0:num_of_test]

        print("{} {}".format(len(self.images) , len(self.test_images)))
        self.num_of_train = len(self.images)
        self.num_of_test  = len(self.test_images)

        assert (len(self.images) == len(self.labels))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def get_file_name(self, path):
        return [f.split('.')[0] for f in os.listdir(path) if f.endswith('jpg')]

    def __len__(self):
        if self.split == "train":
            return len(self.images)+len(self.test_images)
        elif self.split == "val":
            return len(self.images)

    def __getitem__(self, index):
        _img, _target, _pseudo = self._make_img_gt_point_pair(index)
        #print("lalal >>", _img)
        #print("label  ",  _target)
        sample = {'image': _img, 'label': _target, 'pseudo': _pseudo}

        if self.split == "train":
            #sample['image'].show()
            return self.transform_tr(sample)
        elif self.split == "val":
            return self.transform_val(sample)
        else:
            print(self.split)


    def _make_img_gt_point_pair(self, index):
        _pseudo = False
        if index >= self.num_of_train:
            index -= self.num_of_train
            #set_trace()
            _img = Image.open(self.test_images[index]).convert("RGB")
            #set_trace()
            _target = Image.open(self.pseudo_labels[index]).convert("RGB")
            _target = np.asarray(_target)
            _target = Image.fromarray(encode_segmap(_target,'isic').astype('uint8'))
            #_target = Image.open(self.pseudo_labels[index])
            _pseudo = True
        else:
            _img = Image.open(self.images[index]).convert('RGB')
            _target = Image.open(self.labels[index])
        
        return _img, _target, _pseudo

    def transform_tr(self, sample):
        #print(sample)
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            #tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),
            tr.Lablize(high_confidence=self.args.high_confidence)])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),
            tr.Lablize(high_confidence=False)])

        return composed_transforms(sample)

    def __str__(self):
        return 'ISIC2018(split=' + str(self.split) + ')'


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


