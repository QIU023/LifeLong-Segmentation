import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, accuracy

EPSILON = 1e-8

# CIFAR100, ResNet32
epochs = 70
lrate = 2.0
milestones = [49, 63]
lrate_decay = 0.2
batch_size = 64
memory_size = 2000


class iCaRL(BaseLearner):

    def __init__(self, args):
        super().__init__()
        self._network = IncrementalNet(args['convnet_type'], False)
        self._device = args['device']

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def eval_task(self):
        y_pred, y_true = self._eval_ncm(self.test_loader, self._class_means)
        accy = accuracy(y_pred, y_true, self._known_classes)

        return accy

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Procedure
        self._train(self.train_loader, self.test_loader)                                #训练集 测试集训练
        self._reduce_exemplar(data_manager, memory_size//self._total_classes)           #范例集精简
        self._construct_exemplar(data_manager, memory_size//self._total_classes)        #为新类样本创建范例集

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):  #特征表示更新
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):                     #新类
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)                                          #新类的输入和forward
                onehots = target2onehot(targets, self._total_classes)                   #新类标签->onehot

                if self._old_network is None:                                           #没有原始网络 单纯分类
                    loss = F.binary_cross_entropy_with_logits(logits, onehots)
                else:
                    old_onehots = torch.sigmoid(self._old_network(inputs).detach())     #原来的网络对新样本的预测（用于计算蒸馏损失）
                    new_onehots = onehots.clone()                                       #新的onehot
                    new_onehots[:, :self._known_classes] = old_onehots                  #由于gt这个onehot向量的label位肯定在 :_known_classes后面
                    loss = F.binary_cross_entropy_with_logits(logits, new_onehots)      #因此新网络的输出既可以与新样本的新位置的gt算CELoss
                                                                                        #也可以与原来的网络的预测结果算CELoss 以求不要忘记原来网络的输出
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)
