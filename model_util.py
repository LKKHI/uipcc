import copy

import random


import numpy as np
import torch

"""
    Some handy functions for model training ...
"""

def freeze_random(seed=2021):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def triad_to_matrix(triad, nan_symbol=-1):
    """三元组转矩阵

    Args:
        triad : 三元组
        nan_symbol : 非零数据的表示方法. Defaults to -1.

    """
    # 注意下标应该为int
    if not isinstance(triad, np.ndarray):
        triad = np.array(triad)
    x_max = triad[:, 0].max().astype(int)  # 用户数量
    y_max = triad[:, 1].max().astype(int)  # 项目数量
    matrix = np.full((x_max + 1, y_max + 1), nan_symbol,
                     dtype=triad.dtype)  # 初始化QoS矩阵
    matrix[triad[:, 0].astype(int),
           triad[:, 1].astype(int)] = triad[:, 2]  # 将评分值放到QoS矩阵的对应位置中
    return matrix


def split_d_triad(d_triad):
    l = np.array(d_triad, dtype=np.object)
    return np.array(l[:, 0].tolist()), l[:, 1].tolist()


def nonzero_user_mean(matrix, nan_symbol):
    """快速计算一个矩阵的行均值
    """
    m = copy.deepcopy(matrix)
    m[matrix == nan_symbol] = 0
    t = (m != 0).sum(axis=-1)  # 每行非0元素的个数
    res = (m.sum(axis=-1) / t).squeeze()
    res[np.isnan(res)] = 0
    return res


def nonzero_item_mean(matrix, nan_symbol):
    """快速计算一个矩阵的列均值
    """
    return nonzero_user_mean(matrix.T, nan_symbol)


def use_optimizer(network, opt, lr):
    if opt == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.99)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=lr,
                                     weight_decay=1e-8)
    return optimizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


