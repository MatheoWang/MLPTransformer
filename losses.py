import torch
import torch.nn.functional as F
import numpy as np
import math

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, mask=None):
        if mask is not None:
            return torch.mean(mask*(y_true - y_pred) ** 2 )
        else:
            return torch.mean((y_true - y_pred)**2)


class Grad2g(torch.nn.Module):
    """
    Bending Penalty of the spatial transformation (2D)
    """

    def __init__(self):
        super(Grad2g, self).__init__()

    def _diffs(self, y, dim):  # y shape(bs, nfeat, vol_shape)
        ndims = y.ndimension() - 2
        d = dim + 2
        # permute dimensions to put the ith dimension first
        #       r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
        dfi = y[1:, ...] - y[:-1, ...]
        # permute back
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
        #       r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        df = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))
        return df

    def forward(self, _, pred):  # shape(B,C,H,W)
        Ty = self._diffs(pred, dim=0)
        Tx = self._diffs(pred, dim=1)
        Tyy = self._diffs(Ty, dim=0)
        Txx = self._diffs(Tx, dim=1)
        Txy = self._diffs(Tx, dim=0)
        p = Tyy.pow(2).mean() + Txx.pow(2).mean() + 2 * Txy.pow(2).mean()
        return p


