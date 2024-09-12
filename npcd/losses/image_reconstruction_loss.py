import torch.nn as nn
import torch.nn.functional as F

from npcd.utils import logging, subsample_gt


class ImageReconstructionLoss(nn.Module):
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
        gt_img = sample['images']
        gt_img = subsample_gt(gt_img, pred.get("ray_idx", None))
        pred_img = pred.channels.contiguous()
                
        loss = ((pred_img - gt_img)**2).mean() * self.weight
        sub_losses['pointnerf_reconstruction'] = loss
        total_loss = loss
        
        return total_loss, sub_losses, pointwise_losses
