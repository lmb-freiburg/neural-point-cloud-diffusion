import torch
import torch.nn as nn

from npcd.utils import logging, subsample_gt


class NeuralPointCloudKLLoss(nn.Module):
    def __init__(self, model, weight=1, verbose=True):

        super().__init__()

        self.weight = weight
        self.verbose = verbose

        if self.verbose:
            logging.info(f"Initializing {self.name} loss.")

        if self.verbose:
            logging.info(f"\tWeight: {self.weight}")
            logging.info(f"Finished initializing {self.name} loss.")
            logging.info()

    @property
    def name(self):
        name = type(self).__name__
        return name

    def forward(self, sample, pred, aux, iteration):

        sub_losses = {}
        pointwise_losses = {}
        
        feats_mean = aux['feats_mean']
        feats_log_var = aux['feats_log_var']
        
        kld = -0.5 * torch.sum(1 + feats_log_var - feats_mean.pow(2) - feats_log_var.exp(), dim=-1)  # batch, num_points
        kld = kld * self.weight
        pointwise_losses['00_neural_point_cloud_kl'] = kld
        
        kld = kld.mean()
        sub_losses['00_neural_point_cloud_kl'] = kld
        total_loss = kld
        
        return total_loss, sub_losses, pointwise_losses
