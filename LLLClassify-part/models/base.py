import copy
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = [], []

    def after_task(self):
        pass

    def eval_task(self):
        pass

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) / total, decimals=3)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_ncm(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, 'sqeuclidean')  # [nb_classes, N]
        scores = (-dists).T  # [N, nb_classes]

        return np.argsort(scores, axis=1)[:, -1], y_true

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            _vectors = tensor2numpy(self._network.extract_vector(
                _inputs.to(self._device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):                            #范例集精简，m为剩余数目
        logging.info('Reducing exemplars...({} per classes)'.format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self._network.feature_dim))  #各类的均值
        self._data_memory, self._targets_memory = [], []

        for class_idx in range(self._known_classes):            #目前已知的所有类
            mask = np.where(dummy_targets == class_idx)[0]      #所有类中属于第i类的样本的下标
            dd = [dummy_data[i] for i in mask][:m]              #原来类别的数据 取前m个
            dt = dummy_targets[mask][:m]                        #原来类别的标签
            self._data_memory = self._data_memory + dd
            self._targets_memory.append(dt)

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=(dd, dt))     #取得对应数据集
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)       #数据集->loader
            vectors, _ = self._extract_vectors(idx_loader)                          #输入特征提取器得到的所有特征向量的numpy（未softmax）
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T #特征向量一范数归一化
            mean = np.mean(vectors, axis=0)                                         #求第i类的特征向量均值
            mean = mean / np.linalg.norm(mean)                                      #均值一范数归一化

            self._class_means[class_idx, :] = mean                                  #第i类均值数组赋值

    def _construct_exemplar(self, data_manager, m):                                 #新类构建范例集
        logging.info('Constructing exemplars...({} per classes)'.format(m))
        for class_idx in range(self._known_classes, self._total_classes):           #新类遍历
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)           #新类数据
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)   
            vectors, _ = self._extract_vectors(idx_loader)                              #新类经过特征提取器得到的特征向量numpy
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T     #一范数归一化
            class_mean = np.mean(vectors, axis=0)                                       #该新类特征向量均值

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m+1):                                                 #选取与新类特征向量均值最接近的m个新类样本
                S = np.sum(exemplar_vectors, axis=0)                                #已选取的范例特征向量的和 从0开始
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors     #已选取的范例特征向量的均值
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))    #寻找新的范例，使当前的均值尽量贴近总的均值
                exemplar_vectors.append(vectors[i])
                selected_exemplars.append(data[i])

                vectors = np.delete(vectors, i, axis=0)                             #未选取的范例移除第i个刚刚被选取
                del data[i]

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            exemplar_targets = np.full(m, class_idx)                                #范例的标签 全为class_idx
            self._data_memory = self._data_memory + selected_exemplars              #新类范例集加入memory
            self._targets_memory.append(exemplar_targets)

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test', #从新类范例集得到数据 以计算均值 
                                                   appendent=(selected_exemplars, exemplar_targets))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)                          #输入特征提取器 计算特征向量
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T #特征向量一范数归一化
            mean = np.mean(vectors, axis=0)         #求均值
            mean = mean / np.linalg.norm(mean)      #均值一范数归一化

            self._class_means[class_idx, :] = mean  #类均值数组赋值

        self._targets_memory = np.concatenate(self._targets_memory)

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self._network.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data = [self._data_memory[i] for i in mask]
            class_targets = self._targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                 mode='test', ret_data=True)
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                exemplar_vectors.append(vectors[i])
                selected_exemplars.append(data[i])

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                # data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection
                del data[i]

            exemplar_targets = np.full(m, class_idx).tolist()
            self._data_memory = self._data_memory + selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)).astype(int)

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                     appendent=(selected_exemplars, exemplar_targets))
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
