from .datasets import cityscapes, coco, combine_dbs, pascal, sbd, isic
from .datasets import  isic,chaos
from torch.utils.data import DataLoader
from ipdb import set_trace
def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        #set_trace()
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        print(num_class)
        #set_trace()
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'isic':
        train_set = isic.ISICSegmentation(args, split="train", percent=args.percent, label_percent=args.label_percent)
        val_set   = isic.ISICSegmentation(args, split="val", percent=0)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader  = None
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'chaos1':
        train_set = chaos.ChaosSegmentation1(args, split="train")
        val_set = chaos.ChaosSegmentation1(args, split="val")
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = None
        return  train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'chaos2':
        train_set = chaos.ChaosSegmentation2(args, split="train")
        val_set = chaos.ChaosSegmentation2(args, split="val")
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = None
        return  train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

