import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib.pyplot import imsave as Imsaver
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataloaders.utils import *
from mypath import Path
from dataloaders import custom_transforms2 as tr
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from modeling.unet import *
from utils.saver import Saver
from ipdb import set_trace

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, image):
        return image.resize(self.size, Image.BILINEAR)


class ToTensor(object):
    def __call__(self, image):
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        return image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image


class PascalTest(Dataset):
    NUM_CLASSES = 21

    def __init__(self, args):
        super().__init__()
        self.path = args.tdpath
        self.filenames = self.get_filenames()
        self.images = self.get_images()
        self.args = args

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        # print(image.size)
        return self.transform_test(image), self.filenames[index], image.size

    def transform_test(self, images):
        composed_transforms = transforms.Compose([
            FixedResize(size=self.args.crop_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
        return composed_transforms(images)

    def get_images(self):
        return [os.path.join(self.path, f + ".jpg") for f in self.filenames]

    def get_filenames(self):
        return [f.split('.')[0] for f in os.listdir(self.path) if f.endswith("jpg")]


class ISICTest(Dataset):
    NUM_CLASSES = 2

    def __init__(self, args):
        super().__init__()
        self.path = args.tdpath
        self.filenames = self.get_filenames()
        self.images = self.get_images()
        self.args = args


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        # print(image.size)
        return self.transform_test(image), self.filenames[index], image.size

    def transform_test(self, images):
        composed_transforms = transforms.Compose([
            FixedResize(size=self.args.crop_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
        return composed_transforms(images)

    def get_images(self):
        return [os.path.join(self.path, f + ".jpg") for f in self.filenames]

    def get_filenames(self):
        return [f.split('.')[0] for f in os.listdir(self.path) if f.endswith("jpg")]

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
                        self.pseudo_labels.append(os.path.join('/data/weishizheng/QiuYiqiao/Segmentation-codes/result_chaos2_addval/', name + "_segmentation.jpg"))
        
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


def get_test_data_loader(args):
    if args.dataset == 'isic':
        test_set = ISICTest(args)
    elif args.dataset == 'chaos1':
        test_set = Chaos1Test(args) 
    elif args.dataset == 'chaos2':
        test_set = Chaos2Test(args)
    elif args.dataset == 'pascal':
        test_set = PascalTest(args)
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    return test_loader, test_set.NUM_CLASSES


class Tester(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        # kwargs = {'num_workers': 1, 'pin_memory': True}
        self.test_loader, self.nclass = get_test_data_loader(args)
        #_, _, self.test_loader, self.nclass = make_data_loader(self.args)
        # Define network
        if args.model == 'deeplab':
            model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        elif args.model == 'unet':
            model = UNet(n_classes=self.nclass,
                         n_channels=3)

        self.model = model

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def testing(self):
        self.model.eval()
        tbar = tqdm(self.test_loader)
        num_img_tr = len(self.test_loader)
        for i, images in enumerate(tbar):
            images, filenames, size = images[0], images[1], images[2]
            if self.args.cuda:
                images = images.cuda()
            if self.args.model == 'deeplab':
                output, fuse = self.model(images)
            elif self.args.model == 'unet':
                output = self.model(images)
            #set_trace()
            self.save_result(output, filenames, size)

    def save_result(self, output, filenames, size):
        # output = np.array( output.detach().cpu() )
        num_of_len = output.shape[0]
        # output = (output > 0.5)*255
        width = size[0]
        height = size[1]
        # print(self.args.save_path)
        for i in range(num_of_len):
            #image = np.array(image.detach().cpu())[0, , :, :]
            #set_trace()
            if self.nclass == 2:
                image = torch.nn.functional.interpolate(output[i:i + 1, 1:, :, :], (height[i], width[i]))
                image = np.array(image.detach().cpu())[0, 0, :, :]
                image = (image>0.5) * 1
                Imsaver(os.path.join(self.args.save_path, filenames[i] + "_segmentation.jpg"), image)
            else: 
                image = torch.nn.functional.interpolate(output[i:i + 1, :, :, :], (height[i], width[i]))
                image = np.array(image.detach().cpu())[0,:,:,:]
                mask_image = np.max(image,axis=0)
                Imsaver(os.path.join(self.args.confidence_save_path, filenames[i] + "confidence_result.jpg"), mask_image)
                label = np.argmax(image,axis=0)
                Imsaver(os.path.join(self.args.label_save_path, filenames[i] + "label_segmentation.jpg"), label)
                #rgb_label = decode_segmap(label, self.args.dataset)
                #Imsaver(os.path.join(self.args.rgb_save_path, filenames[i] + "rgb_segmentation.jpg"), rgb_label)
                  

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--model',type=str, default='deeplab',
                        choices=['deeplab', 'unet'],
                        help='model name (default:deeplab)')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal','chaos1','chaos2', 'coco', 'cityscapes', 'isic'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--save-path',type=str, default=".")
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--tdpath', type=str,
                        default="/data/weishizheng/QiuYiqiao/ISIC-skin/ISIC2018_Task1-2_Test_Input/",
                        help='put the path to test')
    parser.add_argument('--label-save-path', type=str, default=".",
                        help='put the path to test')
    parser.add_argument('--confidence-save-path', type=str, default=".",
                        help='put the path to test')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--tri', action='store_true', default=False,
                        help='using triplet loss')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    print(args)
    torch.manual_seed(args.seed)
    tester = Tester(args)
    tester.testing()


if __name__ == "__main__":
    main()
