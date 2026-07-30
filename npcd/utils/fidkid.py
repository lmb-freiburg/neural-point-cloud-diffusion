# Copied and only slightly adapted from https://github.com/Lakonik/SSDNeRF.
# Original license:
# MIT License

# Copyright (c) 2023 Hansheng Chen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle

import torch
import numpy as np
from scipy import linalg


try:
    from mmgen.core.evaluation.metrics import FID as MMGenFID
except ImportError:
    MMGenFID = None


class FID:
    """Reimplementation of the FID metric from mmgeneration (mmgen.core.evaluation.metrics.FID),
    for environments where mmgeneration is not installed.
    """
    name = 'FID'

    def __init__(self,
                 num_images,
                 inception_pkl=None,
                 bgr2rgb=True,
                 inception_args=None):
        assert inception_args is not None and inception_args.get('type') == 'StyleGAN', \
            "Only Tero Karras' StyleGAN Inception model is supported. Install mmgeneration for other options."

        self.num_images = num_images
        self.num_real_need = num_images
        self.num_fake_need = num_images
        self.num_real_feeded = 0  # record of the fed real images
        self.num_fake_feeded = 0  # record of the fed fake images
        self.inception_pkl = inception_pkl
        self.bgr2rgb = bgr2rgb
        self.real_feats = []
        self.fake_feats = []
        self.real_mean = None
        self.real_cov = None
        self._result_str = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.inception_net = torch.jit.load(inception_args['inception_path']).eval().to(self.device)
        print(f"Loaded Tero's Inception model from {inception_args['inception_path']}")

    @torch.no_grad()
    def feed(self, batch, mode):
        if mode == 'reals':
            end = min(batch.shape[0], self.num_real_need - self.num_real_feeded)
        elif mode == 'fakes':
            end = min(batch.shape[0], self.num_fake_need - self.num_fake_feeded)
        else:
            raise ValueError("The expected mode should be set to 'reals' or 'fakes', "
                             f"but got '{mode}'")

        if end <= 0:
            return 0

        batch = batch[:end, ...]
        if self.bgr2rgb:
            batch = batch[:, [2, 1, 0]]
        batch = (batch.to(self.device) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        feat = self.inception_net(batch, return_features=True).cpu()

        if mode == 'reals':
            self.real_feats.append(feat)
            self.num_real_feeded += end
        else:
            self.fake_feats.append(feat)
            self.num_fake_feeded += end

        return end

    @staticmethod
    def _calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
        # note: the `disp` argument that mmgen passes here was removed in scipy 1.18:
        cov_sqrt = linalg.sqrtm(sample_cov @ real_cov)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        return fid, mean_norm, trace


# Use mmgeneration's FID if it is installed (i.e. the original environment), to keep the code
# execution path for reproducing the results of the publication unchanged:
class FIDKID(MMGenFID if MMGenFID is not None else FID):
    name = 'FIDKID'

    def __init__(self,
                 num_images,
                 num_subsets=100,
                 max_subset_size=1000,
                 **kwargs):
        super().__init__(num_images, **kwargs)
        self.num_subsets = num_subsets
        self.max_subset_size = max_subset_size
        self.real_feats_np = None

    def prepare(self):
        if self.inception_pkl is not None:
            with open(self.inception_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.real_mean = reference['mean']
                self.real_cov = reference['cov']
                self.real_feats_np = reference['feats_np']
                print(f'Load reference inception pkl from {self.inception_pkl}')
            self.num_real_feeded = self.real_feats_np.shape[0]

    @staticmethod
    def _calc_kid(real_feat, fake_feat, num_subsets, max_subset_size):
        """Refer to the implementation from:
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/kernel_inception_distance.py#L18  # noqa
        Args:
            real_feat (np.array): Features of the real samples.
            fake_feat (np.array): Features of the fake samples.
            num_subsets (int): Number of subsets to calculate KID.
            max_subset_size (int): The max size of each subset.
        Returns:
            float: The calculated kid metric.
        """
        n = real_feat.shape[1]
        m = min(min(real_feat.shape[0], fake_feat.shape[0]), max_subset_size)
        t = 0
        for _ in range(num_subsets):
            x = fake_feat[np.random.choice(
                fake_feat.shape[0], m, replace=False)]
            y = real_feat[np.random.choice(
                real_feat.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1)**3 + (y @ y.T / n + 1)**3
            b = (x @ y.T / n + 1)**3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

        kid = t / num_subsets / m
        return float(kid)

    @torch.no_grad()
    def summary(self):
        if self.real_feats_np is None:
            feats = torch.cat(self.real_feats, dim=0)
            assert feats.shape[0] >= self.num_images
            feats = feats[:self.num_images]
            feats_np = feats.numpy()
            self.real_feats_np = feats_np
            self.real_mean = np.mean(feats_np, 0)
            self.real_cov = np.cov(feats_np, rowvar=False)

        fake_feats = torch.cat(self.fake_feats, dim=0)
        assert fake_feats.shape[0] == self.num_images
        fake_feats = fake_feats[:self.num_images]
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)
        kid = self._calc_kid(self.real_feats_np, fake_feats_np, self.num_subsets, self.max_subset_size) * 1000

        self._result_str = f'{fid:.4f} ({mean:.5f}/{cov:.5f}), {kid:.4f}'
        self._result_dict = dict(fid=fid, fid_mean=mean, fid_cov=cov, kid=kid)

        return fid, mean, cov, kid
