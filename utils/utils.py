import os
import math
import torch
import joblib
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import rearrange
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from torch.utils.data import Subset
import base64
from io import BytesIO
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg



# ==========================================
#            模型辅助函数 (Model Utils)
# ==========================================

def activation_helper(activation, dim=None):
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: %s' % activation)
    return act

def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])

def regularize(network, lam, penalty):
    '''
    Calculate regularization term for first layer weight matrix.

    Args:
      network: MLP network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)

def prox_update(network, lam, lr, penalty):
    '''
    Perform in place proximal update on first layer weight matrix.

    Args:
      network: MLP network.
      lam: regularization parameter.
      lr: learning rate.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        for i in range(lag):
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            W.data[:, :, :(i+1)] = (
                (W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)

def convert_to_list(idx):
    def sorted_idx_to_intervals(idx):
        """
        idx: 已去重、已升序的 List[int]
        return: List[List[int]]，每段是闭区间 [start, end]
        """
        if not idx:
            return []

        intervals = []
        start = prev = idx[0]

        for x in idx[1:]:
            if x == prev + 1:
                prev = x
            else:
                intervals.append([start, prev])
                start = prev = x

        intervals.append([start, prev])
        return intervals

    if torch.is_tensor(idx):
        idx_list = idx.detach().cpu().numpy().reshape(-1).astype(int).tolist()
        segment_idx = sorted_idx_to_intervals(idx_list)
    elif isinstance(idx, np.ndarray):
        idx_list = idx.reshape(-1).astype(int).tolist()
        segment_idx = sorted_idx_to_intervals(idx_list)
    elif isinstance(idx, (list, tuple)):
        if len(idx) > 0 and isinstance(idx[0], (list, tuple, np.ndarray)):
            segment_idx = [[int(seg[0]), int(seg[1])] for seg in idx if len(seg) >= 2]
        else:
            idx_list = [int(v) for v in idx]
            segment_idx = sorted_idx_to_intervals(idx_list)
    else:
        segment_idx = []
    
    return segment_idx