class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/data/weishizheng/QiuYiqiao/pascal/train/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/data/weishizheng/QiuYiqiao/cityscapes/train'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'isic':
            return '/data/weishizheng/QiuYiqiao/ISIC-skin/'
        elif dataset == 'chaos1':
            return '/data/weishizheng/QiuYiqiao/CHAOS/CHAOS_Train_Sets/Train_Sets/CT/'
        elif dataset == 'chaos2':
            return '/data/weishizheng/QiuYiqiao/CHAOS/CHAOS_Train_Sets/Train_Sets/MR/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
    @staticmethod
    def iddataset_path():
        return './iddataset.pkl'
