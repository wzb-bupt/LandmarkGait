from os import device_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class Reconstruction(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(Reconstruction, self).__init__(loss_term_weight)

    # @gather_and_scale_wrapper
    def forward(self, des_pred, des):
        """
            des_pred: [n, s, c, h, w]
            des: [n, s, c, h, w]
        """
        n, s, c, h, w = des_pred.size()  # c:channel
        criterionMSE = nn.MSELoss()
        loss = criterionMSE(des_pred.float(), des.float()) * 10.
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


