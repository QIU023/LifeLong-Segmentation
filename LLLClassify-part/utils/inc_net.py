import copy
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet50
from convs.modified_cifar_resnet import resnet32 as cosine_resnet32
from convs.modified_resnet import resnet34 as cosine_resnet34
from convs.modified_resnet import resnet50 as cosine_resnet50
from convs.modified_linear import SplitCosineLinear, CosineLinear


def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'cosine_resnet32':
        return cosine_resnet32()
    elif name == 'cosine_resnet34':
        return cosine_resnet34(pretrained=pretrained)
    elif name == 'cosine_resnet50':
        return cosine_resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


class BaseNet(nn.Module):

    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(convnet_type, pretrained)        #特征提取器 resnet骨干
        self.fc = None

    @property
    def feature_dim(self):              #特征维度
        return self.convnet.out_dim     

    def extract_vector(self, x):        #输出特征向量
        return self.convnet(x)

    def forward(self, x):               #经过全连接层
        x = self.convnet(x)
        logits = self.fc(x)

        return logits

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):                   #冻结参数 不允许更新
        for param in self.parameters():
            param.requires_grad = False
        self.eval()                     #验证模式

        return self


class IncrementalNet(BaseNet):          #增量网络 以全连接层为类别输出 最后在外面加softmax

    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

    def update_fc(self, nb_classes):                            #输出类别数变化
        fc = self.generate_fc(self.feature_dim, nb_classes)     #特征向量维度->输出类别维度更新
        if self.fc is not None:                                 #已经有全连接 用原来前k1类的权重 k1->k2类的权重用新的
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)         #用原来的权重
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = nn.Linear(in_dim, out_dim)                         #新全连接层 初始化    
        nn.init.kaiming_normal_(fc.weight, nonlinearity='linear')
        nn.init.constant_(fc.bias, 0)

        return fc


class ModifiedIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

    def get_features_fn(self, module, inputs, outputs):
        self.features = inputs[0]

    def get_features(self):
        return self.features

    def update_fc(self, nb_classes, task_num):                      #类别数更新
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            self.features_hook.remove()
            if task_num == 1:                                       #当前任务=1 刚开始
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:                                                   
                prev_out_features1 = self.fc.fc1.out_features       
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data   #前面的k1个类别的全连接权重
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data   #后面的k2个类别的全连接权重
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc
        self.features_hook = self.fc.register_forward_hook(self.get_features_fn)

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:                         #刚开始 只有一部分全连接
            fc = CosineLinear(in_dim, out_dim)
        else:                                       #两部分全连接 分别用于输出旧类和新类
            prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features)  

        return fc
