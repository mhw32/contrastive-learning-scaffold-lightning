import os
import numpy as np
from dotmap import DotMap
from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets import datasets
from src.models.resnet import resnet18
from src.objectives.memory import MemoryBank
from src.models.logreg import LogisticRegression
from src.objectives.instdisc import InstDisc, NCE, Ball, Ring, SimCLR
from src.objectives.moco import MoCo
from src.utils import utils

import pytorch_lightning as pl
torch.autograd.set_detect_anomaly(True)


def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset, 
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle, 
        pin_memory=True,
        drop_last=True,
        num_workers=config.data_loader_workers,
    )
    return loader


class PretrainSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset, self.val_dataset = datasets.get_datasets(
            config.data_params.dataset)
        train_labels = self.train_dataset.dataset.targets
        self.train_ordered_labels = np.array(train_labels)
        self.model = self.create_encoder()
        self.memory_bank = MemoryBank(len(self.train_dataset), 
                                      self.config.model_params.out_dim, 
                                      device=self.device)

    def create_encoder(self):
        model = resnet18(low_dim=self.config.model_params.out_dim)
        if self.config.model_params.projection_head:
            mlp_dim = model.fc.weight.size(1)
            model.fc = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                model.fc,
            )
        return model

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(),
                                lr=self.config.optim_params.learning_rate,
                                momentum=self.config.optim_params.momentum,
                                weight_decay=self.config.optim_params.weight_decay)
        return [optim], []

    def forward(self, img):
        return self.model(img)

    def get_losses_for_batch(self, batch, train=True):
        indices, img, _, = batch
        outputs = self.forward(img)
        loss_fn = NCE(indices, outputs, self.memory_bank,
                      k=self.config.loss_params.k,
                      t=self.config.loss_params.t,
                      m=self.config.loss_params.m)
        loss = loss_fn.get_loss()

        if train:
            with torch.no_grad():
                new_data_memory = loss_fn.updated_new_data_memory()
                self.memory_bank.update(indices, new_data_memory)

        return loss

    def get_nearest_neighbor_label(self, batch):
        """
        NOTE: ONLY TO BE USED FOR VALIDATION.

        For each image in validation, find the nearest image in the 
        training dataset using the memory bank. Assume its label as
        the predicted label.
        """
        _, img, label = batch
        outputs = self.forward(img)

        all_dps = self.memory_bank.get_all_dot_products(outputs)
        _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
        neighbor_idxs = neighbor_idxs.squeeze(1)
        neighbor_idxs = neighbor_idxs.cpu().numpy()

        neighbor_labels = self.train_ordered_labels[neighbor_idxs]
        neighbor_labels = torch.from_numpy(neighbor_labels).long()

        num_correct = torch.sum(neighbor_labels.cpu() == label.cpu()).item()
        return num_correct, img.size(0)

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, train=True)
        metrics = {'loss': loss}
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, train=False)
        num_correct, batch_size = self.get_nearest_neighbor_label(batch)
        output = OrderedDict({'val_loss': loss,
                              'val_num_correct': num_correct,
                              'val_num_total': batch_size})
        return output
    
    def validation_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
        num_correct = sum([out['val_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        return {'val_loss': metrics['val_loss'], 'log': metrics, 'val_acc': val_acc}

    @pl.data_loader
    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    @pl.data_loader
    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)


class PretrainTwoViewsSystem(PretrainSystem):

    def __init__(self, config):
        super().__init__(config)

        self.loss_name = self.config.loss_params.name
        if self.loss_name == 'MoCo':
            self.model_k = self.create_encoder()

            for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False     # do not update

            # create queue (k x out_dim)
            moco_queue = torch.randn(
                self.config.loss_params.k,
                self.config.model_params.out_dim, 
            )
            self.register_buffer("moco_queue", moco_queue)
            self.moco_queue = utils.l2_normalize(moco_queue, dim=1)
            self.register_buffer("moco_queue_ptr", torch.zeros(1, dtype=torch.long))

    def get_losses_for_batch(self, batch, train=True):
        indices, img1, img2, _, = batch
        outputs1 = self.forward(img1)

        if self.loss_name == 'SimCLR':
            outputs2 = self.forward(img2)
            loss_fn = SimCLR(outputs1, outputs2, 
                             t=self.config.loss_params.t)
            loss = loss_fn.get_loss()
        elif self.loss_name == 'MoCo':
            with torch.no_grad():
                self._momentum_update_key_encoder()
                if self.use_ddp or self.use_ddp2:
                    img2, idx_unshuffle = self._batch_shuffle_ddp(img2)
                outputs2 = self.model_k(img2)
                if self.use_ddp or self.use_ddp2:
                    outputs_k = self._batch_unshuffle_ddp(outputs_k, idx_unshuffle)

            loss_fn = MoCo(outputs1, outputs2, 
                           self.moco_queue.clone().detach(),
                           t=self.config.loss_params.t)
            loss = loss_fn.get_loss()

            if train:
                outputs_k = utils.l2_normalize(outputs2, dim=1)
                self._dequeue_and_enqueue(outputs_k)
        else:
            raise Exception(f'Loss {self.loss_name} not supported.')

        if train:
            with torch.no_grad():
                new_data_memory = utils.l2_normalize(outputs1)
                self.memory_bank.update(indices, new_data_memory)

        return loss

    def get_nearest_neighbor_label(self, batch):
        """
        NOTE: ONLY TO BE USED FOR VALIDATION.

        For each image in validation, find the nearest image in the 
        training dataset using the memory bank. Assume its label as
        the predicted label.
        """
        _, img, _, label = batch
        outputs = self.forward(img)

        all_dps = self.memory_bank.get_all_dot_products(outputs)
        _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
        neighbor_idxs = neighbor_idxs.squeeze(1)
        neighbor_idxs = neighbor_idxs.cpu().numpy()

        neighbor_labels = self.train_ordered_labels[neighbor_idxs]
        neighbor_labels = torch.from_numpy(neighbor_labels).long()

        num_correct = torch.sum(neighbor_labels.cpu() == label.cpu()).item()
        return num_correct, img.size(0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        m = self.config.loss_params.m
        for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def _momentum_update_viewmaker(self):
        m = self.config.loss_params.m
        for param_q, param_k in zip(self.viewmaker.parameters(), self.viewmaker_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = utils.concat_all_gather(keys)

        batch_size = keys.size(0)

        k = self.config.loss_params.k
        ptr = int(self.moco_queue_ptr)
        assert k % batch_size == 0  # why?

        # replace keys at ptr
        self.moco_queue[ptr:ptr+batch_size] = keys
        # move config by full batch size even if current batch is smaller
        ptr = (ptr + batch_size) % k

        self.moco_queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no-cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no-cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


class TransferSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = f'cuda:{config.gpu_device}' if config.cuda else 'cpu'
        self.train_dataset, self.val_dataset = datasets.get_datasets(
            config.data_params.dataset)
        self.encoder = self.load_pretrained_model()
        utils.frozen_params(self.encoder)
        self.model = self.create_model()

    def load_pretrained_model(self):
        base_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)
        # overwrite GPU to load on same as current agent
        config.gpu_device = self.config.gpu_device

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file)
        system.load_state_dict(checkpoint['state_dict'])

        encoder = system.model.eval()
        return encoder

    def create_model(self):
        dataset_name = self.config.data_params.dataset
        NUM_CLASS_DICT = {'cifar10': 10, 'imagenet': 1000}
        model = LogisticRegression(512*7*7, NUM_CLASS_DICT[dataset_name])
        return model

    def forward(self, img):
        batch_size = img.size(0)
        embs = self.encoder(img, layer=5)
        embs = embs.view(batch_size, -1)
        return self.model(embs)

    def get_losses_for_batch(self, batch):
        _, img, label = batch
        logits = self.forward(img)
        return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch):
        _, img, label = batch
        logits = self.forward(img)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        preds = preds.long().cpu()
        num_correct = torch.sum(preds == label.long().cpu()).item()
        num_total = img.size(0)
        return num_correct, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        with torch.no_grad():
            num_correct, num_total = self.get_accuracies_for_batch(batch)
            metrics = {
                'train_loss': loss,
                'train_num_correct': num_correct,
                'train_num_total': num_total,
                'train_acc': num_correct / float(num_total),
            }
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        num_correct, num_total = self.get_accuracies_for_batch(batch)
        return OrderedDict({
            'val_loss': loss,
            'val_num_correct': num_correct,
            'val_num_total': num_total,
            'val_acc': num_correct / float(num_total)
        })

    def validation_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
        num_correct = sum([out['val_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        return {'val_loss': metrics['val_loss'], 'log': metrics, 'val_acc': val_acc}

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )
        return [optim], []

    @pl.data_loader
    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    @pl.data_loader
    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)
