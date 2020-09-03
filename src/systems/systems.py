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
        return resnet18(low_dim=self.config.model_params.out_dim)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(),
                                lr=self.config.optim_params.learning_rate,
                                momentum=self.config.optim_params.momentum,
                                weight_decay=self.config.optim_params.weight_decay)
        return [optim], []

    def forward(self, img):
        outputs = self.model(img)
        return outputs

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

    def get_losses_for_batch(self, batch, train=True):
        indices, img1, img2, _, = batch
        outputs1 = self.forward(img1)
        outputs2 = self.forward(img2)
        loss_fn = SimCLR(outputs1, outputs2, 
                         t=self.config.loss_params.t)
        loss = loss_fn.get_loss()

        if train:
            with torch.no_grad():
                new_data_memory = (outputs1 + outputs2) / 2.
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
        model = LogisticRegression(512*4*4, NUM_CLASS_DICT[dataset_name])
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
