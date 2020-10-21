import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import accuracy, tensor2numpy

EPSILON = 1e-8

# CIFAR100, ResNet32
epochs_expert = 100
lrate_expert = 0.1
milestones_expert = [50, 70]
lrate_decay_expert = 0.1

epochs = 100
lrate = 0.5
milestones = [50, 70]
lrate_decay = 0.2

batch_size = 128
memory_size = 2000
T1 = 2
T2 = 2


class DR(BaseLearner):
    def __init__(self, args):
        super().__init__()
        self._network = IncrementalNet(args['convnet_type'], False)     #网络
        self._device = args['device']       

        self.convnet_type = args['convnet_type']
        self.expert = None                                              #专家网络

    def after_task(self):
        self._old_network = self._network.copy().freeze()               #上一次迭代的网络
        self._known_classes = self._total_classes

    def eval_task(self):
        y_pred, y_true = self._eval_ncm(self.test_loader, self._class_means)    
        accy = accuracy(y_pred, y_true, self._known_classes)

        return accy

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self.task_size
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        expert_train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                        source='train', mode='train')
        self.expert_train_loader = DataLoader(expert_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        expert_test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                       source='test', mode='test')
        self.expert_test_loader = DataLoader(expert_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Procedure
        logging.info('Training the expert CNN...')                      
        self._train_expert(self.expert_train_loader, self.expert_test_loader)           #专家网络训练   专家网络在每次训练时 专门用于学习新类 
        if self._cur_task == 0:
            self._network = self.expert.copy()                                          #刚刚开始 网络=专家网络
        else:
            self.expert = self.expert.freeze()                                          #专家网络.freeze()
            logging.info('Training the updated CNN...')
            self._train(self.train_loader, self.test_loader)                            #针对专家网络和上一轮的旧网络进行增量学习
        self._reduce_exemplar(data_manager, memory_size//self._total_classes)           #范例集精简
        self._construct_exemplar(data_manager, memory_size//self._total_classes)        #新类构建范例集

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lrate_decay)

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):                                            #epoch进度条
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):                     #训练集
                inputs, targets = inputs.to(self._device), targets.to(self._device)     #数据 标签
                logits = self._network(inputs)                                          #当前网络的特征向量的softmax置信度输出
                exp_logits = self.expert(inputs)                                        #专家网络的分类置信度输出
                old_logits = self._old_network(inputs)                                  #旧网络的分类置信度输出

                # Distillation
                dist_term = _KD_loss(logits[:, self._known_classes:], exp_logits, T1)   #蒸馏损失 当前网络对新类别的分类结果与专家网络贴近
                # Retrospection
                retr_term = _KD_loss(logits[:, :self._known_classes], old_logits, T2)   #记忆损失 当前网络对旧类别的分类结果与旧网络贴近
                                                                        #旧网络是k1分类器 专家网络是k2分类器 当前网络是k1+k2分类器 损失分为两部分
                loss = dist_term + retr_term
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Updated CNN => Epoch {}/{}, Loss {:.3f}, Train accy {:.3f}, Test accy {:.3f}'.format(
                epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _train_expert(self, train_loader, test_loader):
        self.expert = IncrementalNet(self.convnet_type, False)          #专家网络
        self.expert.update_fc(self.task_size)
        self.expert.to(self._device)
        optimizer = optim.SGD(self.expert.parameters(), lr=lrate_expert, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_expert, gamma=lrate_decay_expert)

        prog_bar = tqdm(range(epochs_expert))
        for _, epoch in enumerate(prog_bar):
            self.expert.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), (targets - self._known_classes).to(self._device)     
                                                            #专家网络只分k2种类，这里有k1+k2种，因此下标要减去k1
                logits = self.expert(inputs)

                loss = F.cross_entropy(logits, targets)       
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_acc = self._compute_accuracy(self.expert, train_loader, self._known_classes)
            test_acc = self._compute_accuracy(self.expert, test_loader, self._known_classes)
            info = 'Expert CNN => Epoch {}/{}, Loss {:.3f}, Train accy {:.3f}, Test accy {:.3f}'.format(
                epoch+1, epochs_expert, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_accuracy(self, model, loader, offset=0):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets -= offset
            with torch.no_grad():
                outputs = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) / total, decimals=3)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)                     #模拟退火？T是温度 T=1时为正常softmax 较硬标签 
                                                                #T=2时标签软化 类别之间置信度差异变小
    soft = torch.softmax(soft/T, dim=1)                         #T->inf时 变为均匀分布 也就是软化极端
    return -1 * torch.mul(soft, pred).sum() / soft.shape[0]     #\Sigma_x p(x)log(q(x)) 交叉熵  p,q为软化的概率分布 
