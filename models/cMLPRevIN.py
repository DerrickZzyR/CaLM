import torch
import torch.nn as nn
from utils.utils import activation_helper
from layers.MLP import MLP
from layers.RevIN_em import RevIN_em

class cMLPRevIN(nn.Module):
    def __init__(self, num_series, lag, affine, subtract_last, hidden=[256, 128, 64], activation='relu'):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLPRevIN, self).__init__()
        self.p = num_series
        self.lag = lag
        self.revin_layer = RevIN_em(num_features=num_series, affine=affine, subtract_last=subtract_last)
        self.activation = activation_helper(activation)
        # Set up networks.
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation)
            for _ in range(num_series)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        X_norm, stats = self.revin_layer(X, 'norm')
        input = X_norm[:, :-1]
        pred_x = torch.cat([network(input) for network in self.networks], dim=2)
        
        # 用原始值
        pred_raw = self.revin_layer(pred_x, 'denorm')
        # residual = out - X[:, self.lag:, :]
        
        # 用归一化的值
        residual = pred_x - X_norm[:, self.lag:, :]

        return pred_x, pred_raw, stats, residual, X_norm[:, self.lag:, :]

    def GC(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if ignore_lag:
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC