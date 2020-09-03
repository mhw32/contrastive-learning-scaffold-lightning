import math
import torch
import numpy as np
import torch.nn.functional as F

from src.utils.utils import l2_normalize


class MoCo(object):

    def __init__(self, outputs1, outputs2, queue, t=0.07):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.queue = queue.detach()
        self.t = t
        self.k = queue.size(0)
        self.device = self.outputs1.device

    def get_loss(self):
        batch_size = self.outputs1.size(0)  # batch_size x out_dim
        witness_pos = torch.sum(self.outputs1 * self.outputs2, dim=1, keepdim=True)
        witness_neg = self.outputs1 @ self.queue.T
        # batch_size x (k + 1)
        witness_logits = torch.cat([witness_pos, witness_neg], dim=1) / self.t

        labels = torch.zeros(witness_logits.size(0), device=self.device).long()
        loss = F.cross_entropy(witness_logits, labels.long()) 
        return loss
