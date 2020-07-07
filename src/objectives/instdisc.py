import math
import torch
import numpy as np

from src.utils.utils import l2_normalize


class InstDisc(object):

    def __init__(self, indices, outputs, memory_bank, k=4096, t=0.07, m=0.5, **kwargs):
        self.k, self.t, self.m = k, t, m

        self.indices = indices.detach()
        self.outputs = l2_normalize(outputs, dim=1)

        self.memory_bank = memory_bank
        self.device = outputs.device
        self.data_len = memory_bank.size

    def updated_new_data_memory(self):
        data_memory = self.memory_bank.at_idxs(self.indices)
        new_data_memory = data_memory * self.m + (1 - self.m) * self.outputs
        return l2_normalize(new_data_memory, dim=1)

    def get_loss(self, *args, **kwargs):
        batch_size = self.indices.size(0)

        witness_score = self.memory_bank.get_dot_products(self.outputs, self.indices)

        noise_indx = torch.randint(0, self.data_len, (batch_size, self.k),
                                   device=self.device)  # U(0, data_len)
        noise_indx = noise_indx.long()
        witness_norm = self.memory_bank.get_dot_products(self.outputs, noise_indx)
        witness_norm = torch.logsumexp(witness_norm, dim=1)

        loss = -torch.mean(witness_score - witness_norm)
        return loss


class NCE(InstDisc):

    def get_loss(self):
        batch_size = self.outputs.size(0)
        witness_score = self.memory_bank.get_dot_products(self.outputs, self.indices)
        noise_indx = torch.randint(0, self.data_len, (batch_size, self.k-1), device=self.device).long()
        noise_indx = torch.cat([self.indices.unsqueeze(1), noise_indx], dim=1)
        witness_norm = self.memory_bank.get_dot_products(self.outputs, noise_indx)
        witness_norm = torch.logsumexp(witness_norm / self.t, dim=1) - math.log(self.k)
        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss


class Ball(NCE):

    def get_loss(self):
        batch_size = self.outputs.size(0)
        witness_score = self.memory_bank.get_dot_products(self.outputs, self.indices)
        all_dps = self.memory_bank.get_all_dot_products(self.outputs)
        topk_dps, _ = torch.topk(all_dps, int(self.nsp_back*all_dps.size(1)), sorted=False, dim=1)
        noise_indx = torch.randint(0, topk_dps.size(1), (batch_size, self.k-1), device=self.device).long()
        noise_indx = torch.cat([self.indices.unsqueeze(1), noise_indx], dim=1)
        back_nei_dps = torch.gather(topk_dps, 1, noise_indx)
        witness_norm = torch.logsumexp(back_nei_dps / self.t, dim=1) - math.log(self.k)
        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss


class Ring(NCE):

    def get_loss(self):
        batch_size = self.outputs.size(0)
        witness_score = self.memory_bank.get_dot_products(self.outputs, self.indices)
        all_dps = self.memory_bank.get_all_dot_products(self.outputs)
        sorted_dps, _ = torch.sort(all_dps, dim=1, descending=True)
        sorted_dps = sorted_dps[:, :int(self.nsp_back * sorted_dps.size(1))]
        sorted_dps = sorted_dps[:, int(self.nsp_close * sorted_dps.size(1)):]
        noise_indx = torch.randint(0, sorted_dps.size(1), (batch_size, self.k-1), device=self.device).long()
        noise_indx = torch.cat([self.indices.unsqueeze(1), noise_indx], dim=1)
        back_nei_dps = torch.gather(sorted_dps, 1, noise_indx)
        witness_norm = torch.logsumexp(back_nei_dps / self.t, dim=1) - math.log(self.k)
        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss


class SimCLR(object):

    def __init__(self, outputs1, outputs2, t=0.07):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.t = t

    def get_loss(self):
        batch_size = self.outputs1.size(0)  # batch_size x out_dim
        witness_score = torch.sum(self.outputs1 * self.outputs2, dim=1)
        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        # overcounts a constant
        witness_norm = self.outputs1 @ outputs12.T
        witness_norm = torch.logsumexp(witness_norm / self.t, dim=1) - math.log(batch_size)
        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss
