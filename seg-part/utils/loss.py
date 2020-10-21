import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n
        return loss

    def MSMLoss(self, fuse, target):
        fuse_shape = fuse.shape
        location = self._get_loaction(target, fuse_shape)
        vectors  = self._get_vector(fuse, location)

        criterion = MSMLoss( margin1=1, margin2=0.5, percent=0.25 )
        loss = criterion(vectors)
        return loss

    def _get_loaction(self, target, label=1, number=1000):
        target = np.get( target.reshape(-1) )
        label_index = np.where( target==label )[0]
        other_index = np.where( target!=label )[0]

        positive = np.random.choice( len(label_index), number*2 )
        negative = np.random.choice( len(label_index), number*2 )

        anchor = positive[:number]
        positive = positive[number:]
        negativ1 = negative[:number]
        negativ2 = negative[:number]

        return label_index[anchor],   label_index[positive], \
               other_index[negativ1], other_index[negativ2]

    def _get_vector(self, fuse, location ):
        fuse = fuse.transpose(1, 2).transpose(2, 3)
        fuse = fuse.reshape(-1, fuse.shape[-1])
        return fuse[location[0]], fuse[location[1]],\
               fuse[location[2]], fuse[location[3]]

class PairwiseDistance(nn.Module):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        #print("input's shape ", x1.shape)
        #print("out's shape ", out.shape)
        return torch.pow(out + eps, 1. / self.norm)

class TripletMarginLoss(nn.Module):
    """Triplet loss function.
    """
    def __init__(self, margin, norm=2, percent=1 ):
        super(TripletMarginLoss, self).__init__()
        self.margin  = margin
        self.percent = percent
        self.pdist   = PairwiseDistance(norm)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        #print("d_q shape ",d_q.shape)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge   = torch.clamp(self.margin + d_p - d_n, min=0.0)
        sample_sort  = dist_hinge.sort( descending=True )[0]
        hard_samples = sample_sort[0:int(len(dist_hinge)*self.percent)]
        loss = torch.mean(hard_samples)
        return loss

class MSMLoss(nn.Module):
    """Triplet loss function.
    """
    def __init__(self, margin1=1, margin2=0.5, norm=2, percent=1 ):
        super(MSMLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.percent = percent
        self.pdist   = PairwiseDistance(norm)  # norm 2

    #def forward(self, anchor, positive, negative, negative2):
    def forward(self, quad_point ):
        anchor    = quad_point[0]
        positive  = quad_point[1]
        negative  = quad_point[2]
        negative2 = quad_point[3]
        # d_p is anchor and positive
        # d_n is anchor and negative
        # d_q is negative and negative
        d_p = self.pdist.forward(anchor,   positive)
        d_n = self.pdist.forward(anchor,   negative)
        d_q = self.pdist.forward(negative, negative2)

        hard_index = int( len(d_p)*self.percent )

        d_p = d_p.sort( descending=True )[0][0:hard_index]
        d_n = d_n.sort( descending=False)[0][0:hard_index]
        d_q = d_q.sort( descending=False)[0][0:hard_index]

        strong = torch.clamp( self.margin1 + d_p - d_n, min=0.0 )
        week   = torch.clamp( self.margin2 + d_p - d_q, min=0.0 )
        loss   = torch.mean( strong+week )
        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())

