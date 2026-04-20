import torch
import torch.nn as nn

class RevIN_em(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        # 【修正】直接使用 super().__init__()，这是 Python 3 标准写法，避免类名错误
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
            
            # =========== 【核心功能】 ===========
            # 提取包含“绝对大小语义”的统计量
            # 维度说明: [Batch, 1, Num_Features]
            # 如果 subtract_last=True，语义在 self.last
            # 如果 subtract_last=False，语义在 self.mean
            scaling_stats = self.last if self.subtract_last else self.mean
            
            # 返回: (归一化后的数据, 绝对值统计量)
            return x, scaling_stats 
            # ===================================
            
        elif mode == 'denorm':
            x = self._denormalize(x)
            return x
        else:
            raise NotImplementedError

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            # 取最后一个时间步作为基准，并保持维度 [B, 1, C]
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            # 计算均值作为基准
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        
        # 计算标准差 (即使只用 subtract_last，归一化通常也需要除以 std)
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        
        x = x / self.stdev
        
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        
        x = x * self.stdev
        
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x