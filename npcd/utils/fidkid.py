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

import mmcv
from mmgen.core.evaluation.metrics import FID


class FIDKID(FID):
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
        if self.inception_pkl is not None and mmcv.is_filepath(self.inception_pkl):
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