from typing import Dict, List, Optional, Tuple
import warnings

import torch
from torch import Tensor

from npcd.utils.flex_embedding import FlexEmbedding


class VariationalEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_kp: int,
        out_dim: int,
        n_obj: int,
        gpu: bool = True
    ) -> None:
        super().__init__()
        self.n_kp = n_kp
        self.out_dim = out_dim
        self.n_obj = n_obj
        self.gpu = gpu
        self.sample_embedding = True
        self.init_emb()  # hack to keep the Embedding on cpu despite moving everything to GPUs
    
    def init_emb(self):
        emb = FlexEmbedding(self.n_obj, self.n_kp * self.out_dim * 2)
        torch.nn.init.zeros_(emb.weight)
        self.emb = emb if self.gpu else [emb]
        
    def train(self, mode=True):
        super().train(mode)
        self.sample_embedding = mode
        return self

    def forward(self, idx: Tensor) -> Tensor:
        """
        Arguments:
            idx: [B]
        Returns:
            out: [B, n_kp, out_dim]
        """
        device = idx.device
        if self.gpu:
            emb = self.emb
        else:
            emb = self.emb[0]
            idx = idx.cpu()
        emb = emb(idx).to(device=device).view(-1, self.n_kp, self.out_dim * 2)  # index the embeddings with the object index
        mean = emb[:, :, :self.out_dim]
        if self.sample_embedding:
            log_var = emb[:, :, self.out_dim:]
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)  # reparametrization trick
            out = mean + std * eps
        else:
            out = mean
        return out
    
    def get_mean_log_var_std(self, idx: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        device = idx.device
        if self.gpu:
            emb = self.emb
        else:
            emb = self.emb[0]
            idx = idx.cpu()
        emb = emb(idx).to(device=device).view(-1, self.n_kp, self.out_dim * 2)  # index the embeddings with the object index
        mean = emb[:, :, :self.out_dim]
        log_var = emb[:, :, self.out_dim:]
        std = torch.exp(0.5 * log_var)
        return mean, log_var, std
    
    def get_emb(self) -> Tensor:
        if self.gpu:
            emb = self.emb
        else:
            emb = self.emb[0]
        return emb
    
    # Ensures saving and loading embedding despite of hack for keeping it on CPU
    def get_extra_state(self):
        emb = self.emb if self.gpu else self.emb[0]
        return {"emb": emb.get_extra_state()}
    
    def set_extra_state(self, state):
        if state is not None and "emb" in state:
            emb = self.emb if self.gpu else self.emb[0]
            emb.set_extra_state(state["emb"])

    def freeze(self, emb: bool = False):
        if emb:
            e = self.emb if self.gpu else self.emb[0]
            for param in e.parameters():
                param.requires_grad = False
            e.eval()