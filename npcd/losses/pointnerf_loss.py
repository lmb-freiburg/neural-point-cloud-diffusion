import torch.nn as nn

from .image_reconstruction_loss import ImageReconstructionLoss
from .neural_point_cloud_kl_loss import NeuralPointCloudKLLoss
from .neural_point_cloud_tv_loss import NeuralPointCloudTVLoss

from npcd.utils import logging


class PointNeRFLoss(nn.Module):
    def __init__(self, 
                 model, 
                 image_reconstruction_loss_weight=1, 
                 neural_point_cloud_kl_loss_weight=1, 
                 neural_point_cloud_tv_loss_weight=1,
                 verbose=True):

        super().__init__()

        self.verbose = verbose

        if self.verbose:
            logging.info(f"Initializing {self.name} loss.")
            
        self.image_reconstruction_loss = ImageReconstructionLoss(model=model, weight=image_reconstruction_loss_weight, verbose=verbose)
        self.neural_point_cloud_kl_loss = NeuralPointCloudKLLoss(model=model, weight=neural_point_cloud_kl_loss_weight, verbose=verbose)
        self.neural_point_cloud_tv_loss = NeuralPointCloudTVLoss(model=model, weight=neural_point_cloud_tv_loss_weight, verbose=verbose)
        
        if self.verbose:
            logging.info(f"Finished initializing {self.name} loss.")
            logging.info()

    @property
    def name(self):
        name = type(self).__name__
        return name

    def forward(self, sample, pred, aux, iteration):

        sub_losses = {}
        pointwise_losses = {}
        
        image_reconstruction_loss, _, _ = self.image_reconstruction_loss(sample, pred, aux, iteration)
        neural_point_cloud_kl_loss, _, _ = self.neural_point_cloud_kl_loss(sample, pred, aux, iteration)
        neural_point_cloud_tv_loss, _, _ = self.neural_point_cloud_tv_loss(sample, pred, aux, iteration)
        sub_losses['00_image_reconstruction_loss'] = image_reconstruction_loss
        sub_losses['01_neural_point_cloud_kl'] = neural_point_cloud_kl_loss
        sub_losses['02_neural_point_cloud_tv'] = neural_point_cloud_tv_loss
        total_loss = image_reconstruction_loss + neural_point_cloud_kl_loss + neural_point_cloud_tv_loss
        
        return total_loss, sub_losses, pointwise_losses
