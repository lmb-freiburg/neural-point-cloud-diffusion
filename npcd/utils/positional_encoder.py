from math import ceil

import torch
from torch import nn


class PositionalEncoder1D(nn.Module):
    """
    Sine-cosine positional encoder for input points.
    """
    def __init__(self, n_freqs: int, freq_mult: float = 1):
        super().__init__()
        self.n_freqs = n_freqs
        self.register_buffer("freq_bands", freq_mult * 2 ** torch.arange(self.n_freqs) * torch.pi, persistent=False)
  
    def forward(self, x) -> torch.Tensor:
        spectrum = x[...,None] * self.freq_bands
        sin,cos = spectrum.sin(), spectrum.cos()
        x_enc = torch.flatten(torch.cat([sin,cos],dim=-1), start_dim=-2)
        return torch.cat((x, x_enc), dim=-1)
    
    def d_out(self, d_in):
        return d_in * (1 + 2 * self.n_freqs)


class PositionalEncoder2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The channel (second) dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoder2D, self).__init__()
        assert channels % 4 == 0, "Input channels has to be divisible by 4!"
        self.channels = channels // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    @staticmethod
    def get_emb(sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, h, w, device, dtype):
        """
        :return: Positional Encoding Matrix of size (h, w, channels)
        """
        if self.cached_penc is not None and self.cached_penc.shape[:2] == (h, w):
            return self.cached_penc.to(device=device)

        self.cached_penc = None
        pos_x = torch.arange(h, device=device).type(self.inv_freq.type())
        pos_y = torch.arange(w, device=device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1)
        emb_y = self.get_emb(sin_inp_y)
        self.cached_penc = torch.zeros((h, w, self.channels * 2), device=device, dtype=dtype)
        self.cached_penc[:, :, : self.channels] = emb_x
        self.cached_penc[:, :, self.channels : 2 * self.channels] = emb_y
        return self.cached_penc
