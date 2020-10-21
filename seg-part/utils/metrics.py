import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.inter = 0
        self.union = 0
        self.num_of_batch = 0

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        #print("generating confusion_matrix...")
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        #print("mask: ", mask)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        #print("label: ", label)
        count = np.bincount(label, minlength=self.num_class**2)
        #print("cont: ", count)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        #print("confusion ", confusion_matrix)
        return confusion_matrix

    def Dice( self, eps=0.0001 ):
        dice = (2 * self.inter + eps) / (self.union + eps )
        return dice

    def Jaccard( self, eps=0.0001 ):
        jaccard = ( self.inter + eps ) / ( self.union - self.inter + eps )
        return jaccard

    def _inter( self, gt_image, pre_image ):
        inter = np.dot( gt_image.reshape(-1), pre_image.reshape(-1) )
        return inter

    def _union( self, gt_image, pre_image ):
        union = np.sum( gt_image ) + np.sum( gt_image )
        return union

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        #print(gt_image.shape)
        #print(pre_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.inter += self._inter( gt_image, pre_image )
        self.union += self._union( gt_image, pre_image )
        self.num_of_batch += 1

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.union = 0
        self.inter = 0
        self.num_of_batch = 0

