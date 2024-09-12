import numpy as np
import torch
import torch.nn as nn

from npcd.utils import normal_kl, discretized_gaussian_log_likelihood, mean_flat

def get_beta_schedule(schedule_type, *, num_diffusion_steps, beta_start=None, beta_end=None):
    if schedule_type == 'linear':
        beta_start = 1000 / num_diffusion_steps * 0.0001 if beta_start is None else beta_start
        beta_end = 1000 / num_diffusion_steps * 0.02 if beta_end is None else beta_end
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps)
        
    else:
        raise NotImplementedError(schedule_type)
    
    return betas


class GaussianDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        betas = get_beta_schedule('linear', num_diffusion_steps=1000)
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        # nth element of the list is beta_(n+1), e.g. element at index 0 is beta_1 which is used in the distribution q(x_1 | x_0) (equation 2 in the DDPM paper)
        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        self.sqrt_one_minus_betas = torch.sqrt(1. - self.betas).float()
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance
        
    def q_sample(self, data_start, t, noise=None):
        if noise is None:
            noise = torch.randn(data_start.shape, device=data_start.device)
        assert noise.shape == data_start.shape
        
        return (
                self._extract(self.sqrt_alphas_cumprod.to(data_start.device), t, data_start.shape) * data_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(data_start.device), t, data_start.shape) * noise
        )
        
    def q_sample_next(self, data_t, t, noise=None):
        if noise is None:
            noise = torch.randn(data_t.shape, device=data_t.device)
        assert noise.shape == data_t.shape
        
        return (
                self._extract(self.sqrt_one_minus_betas.to(data_t.device), t, data_t.shape) * data_t +
                self._extract(self.betas.to(data_t.device), t, data_t.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, coords_t, feats_t, t, coords_clip_range=None, feats_clipping_range=None):
        
        model_variance = self.posterior_variance.to(coords_t.device)
        model_log_variance = self.posterior_log_variance_clipped.to(coords_t.device)
        
        model_out_coords, model_out_feats = denoise_fn(coords_t, feats_t, t)
        
        model_variance_coords = self._extract(model_variance, t, coords_t.shape) * torch.ones_like(coords_t)
        model_log_variance_coords = self._extract(model_log_variance, t, coords_t.shape) * torch.ones_like(coords_t)
        
        coords_recon = self._predict_xstart_from_eps(coords_t, t=t, eps=model_out_coords)
        if coords_clip_range is not None:
            coords_clip_min, coords_clip_max = coords_clip_range
            coords_recon = torch.clamp(coords_recon, coords_clip_min, coords_clip_max)
        model_coords_mean, _, _ = self.q_posterior_mean_variance(x_start=coords_recon, x_t=coords_t, t=t)
            
        model_variance_feats = self._extract(model_variance, t, feats_t.shape) * torch.ones_like(feats_t)
        model_log_variance_feats = self._extract(model_log_variance, t, feats_t.shape) * torch.ones_like(feats_t)
        
        feats_recon = self._predict_xstart_from_eps(feats_t, t=t, eps=model_out_feats)
        if feats_clipping_range is not None:
            feat_clip_min, feat_clip_max = feats_clipping_range
            feats_recon = torch.clamp(feats_recon, feat_clip_min, feat_clip_max)
        model_feats_mean, _, _ = self.q_posterior_mean_variance(x_start=feats_recon, x_t=feats_t, t=t)

        return model_coords_mean, model_variance_coords, model_log_variance_coords, coords_recon, model_feats_mean, model_variance_feats, model_log_variance_feats, feats_recon

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        
    def _predict_eps_from_xstart(self, x_t, t, x_start):
        assert x_start.shape == x_t.shape
        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - x_start) / self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_start.shape)

    def p_sample(self, denoise_fn, coords_t, feats_t, t, coords_clip_range=None, feats_clipping_range=None):
        model_coords_mean, model_variance_coords, model_log_variance_coords, coords_recon, model_feats_mean, model_variance_feats, model_log_variance_feats, feats_recon = self.p_mean_variance(denoise_fn, coords_t=coords_t, feats_t=feats_t, t=t, coords_clip_range=coords_clip_range, feats_clipping_range=feats_clipping_range)
        
        coords_noise = torch.randn_like(coords_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(coords_t.shape) - 1)))  # no noise when t == 0
        coords_next = model_coords_mean + nonzero_mask * torch.exp(0.5 * model_log_variance_coords) * coords_noise
            
        feats_noise = torch.randn_like(feats_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(feats_t.shape) - 1)))  # no noise when t == 0
        feats_next = model_feats_mean + nonzero_mask * torch.exp(0.5 * model_log_variance_feats) * feats_noise
            
        return coords_next, coords_recon, feats_next, feats_recon

    def p_sample_loop_trajectory(self, denoise_fn, 
                                 coords_start, feats_start, 
                                 coords_clip_range=None, feats_clip_range=None,
                                 progress=False):
        device = coords_start.device
        
        coords_t = coords_start
        feats_t = feats_start
        
        coords_ts = [coords_t]
        feats_ts = [feats_t]

        coords_recons = []
        feats_recons = []
        
        indices = list(range(self.num_timesteps))[::-1]  # go through reverse trajectory in order (e.g. from x_T to x_T-1, ..., to x_0); t=999 means going from x_1000 to x_999
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        
        for t in indices:
            t_batch = torch.tensor([t] * coords_start.shape[0], device=device)
            out = self.p_sample(denoise_fn=denoise_fn, coords_t=coords_t, feats_t=feats_t, t=t_batch, coords_clip_range=coords_clip_range, feats_clipping_range=feats_clip_range)
            coords_t, coords_recon, feats_t, feats_recon = out
            coords_ts.append(coords_t)
            coords_recons.append(coords_recon)
            feats_ts.append(feats_t)
            feats_recons.append(feats_recon)
            
        return coords_ts, coords_recons, feats_ts, feats_recons

    def _vb_terms_bpd(self, denoise_fn, coords_start, coords_t, feats_start, feats_t, t):
        
        model_coords_mean, model_variance_coords, model_log_variance_coords, coords_recon, model_feats_mean, model_variance_feats, model_log_variance_feats, feats_recon = self.p_mean_variance(denoise_fn, coords_t=coords_t, feats_t=feats_t, t=t)
        
        true_coords_mean, _, true_coords_log_variance_clipped = self.q_posterior_mean_variance(x_start=coords_start, x_t=coords_t, t=t)
        kl_coords = normal_kl(true_coords_mean, true_coords_log_variance_clipped, model_coords_mean, model_log_variance_coords)
        kl_coords = mean_flat(kl_coords) / np.log(2.)  # mean over all non-batch dimensions
        decoder_nll_coords = -discretized_gaussian_log_likelihood(coords_start, means=model_coords_mean, log_scales=0.5 * model_log_variance_coords)
        decoder_nll_coords = mean_flat(decoder_nll_coords) / np.log(2.)
        kl_coords = torch.where((t == 0), decoder_nll_coords, kl_coords)
        
        true_feats_mean, _, true_feats_log_variance_clipped = self.q_posterior_mean_variance(x_start=feats_start, x_t=feats_t, t=t)
        kl_feats = normal_kl(true_feats_mean, true_feats_log_variance_clipped, model_feats_mean, model_log_variance_feats)
        kl_feats = mean_flat(kl_feats) / np.log(2.)  # mean over all non-batch dimensions
        decoder_nll_feats = -discretized_gaussian_log_likelihood(feats_start, means=model_feats_mean, log_scales=0.5 * model_log_variance_feats)
        decoder_nll_feats = mean_flat(decoder_nll_feats) / np.log(2.)
        kl_feats = torch.where((t == 0), decoder_nll_feats, kl_feats)
        
        return kl_coords, coords_recon, kl_feats, feats_recon

    def p_losses(self, denoise_fn, coords_start, feats_start, t, coords_noise=None, feats_noise=None):
        """
        Training loss calculation
        """
        N, coords_dim, num_points = coords_start.shape
        _, feats_dim, _ = feats_start.shape
        assert t.shape == torch.Size([N])

        if coords_noise is None:
            coords_noise = torch.randn(coords_start.shape, dtype=coords_start.dtype, device=coords_start.device)
        assert coords_noise.shape == coords_start.shape and coords_noise.dtype == coords_start.dtype
        coords_t = self.q_sample(data_start=coords_start, t=t, noise=coords_noise)
        
        if feats_noise is None:
            feats_noise = torch.randn(feats_start.shape, dtype=feats_start.dtype, device=feats_start.device)
        assert feats_noise.shape == feats_start.shape and feats_noise.dtype == feats_start.dtype
        feats_t = self.q_sample(data_start=feats_start, t=t, noise=feats_noise)
        
        eps_coords, eps_feats = denoise_fn(coords_t, feats_t, t)  # input shapes: [N, 3, num_points], [N, feat_dim, num_points], [N]; output shapes: [N, 3, num_points], [N, feat_dim, num_points]
        pointwise_coords_loss = (coords_noise - eps_coords)**2  # [N, 3, num_points]
        pointwise_feats_loss = (feats_noise - eps_feats)**2  # [N, feat_dim, num_points]
        
        pointwise_coords_loss = pointwise_coords_loss / 2.  # Divide by 2 to take the average of the coords and the feats losses
        pointwise_feats_loss = pointwise_feats_loss / 2.  # Divide by 2 to take the average of the coords and the feats losses
        coords_loss = pointwise_coords_loss.mean()
        feats_loss = pointwise_feats_loss.mean()
        loss = coords_loss + feats_loss
        
        pointwise_losses = {'pointwise_coords_loss': pointwise_coords_loss, 'pointwise_feats_loss': pointwise_feats_loss}
        sub_losses = {'00_coords_loss': coords_loss, '01_feats_loss': feats_loss}
        
        return loss, sub_losses, pointwise_losses

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            N, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(N, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, coords_start, feats_start):

        with torch.no_grad():
            batch_size, T = coords_start.shape[0], self.num_timesteps
            device = coords_start.device
            
            vbs_coords = []
            xstart_mses_coords = []
            mses_coords = []
            vbs_feats = []
            xstart_mses_feats = []
            mses_feats = []
            
            for t in list(range(self.num_timesteps))[::-1]:  # go through reverse trajectory in order (e.g. from x_T to x_T-1, ..., to x_0); t=999 means going from x_1000 to x_999

                t_batch = torch.tensor([t] * batch_size, device=device)
                    
                noise_coords = torch.randn_like(coords_start)
                coords_t = self.q_sample(data_start=coords_start, t=t_batch, noise=noise_coords)
                
                noise_feats = torch.randn_like(feats_start)
                feats_t = self.q_sample(data_start=feats_start, t=t_batch, noise=noise_feats)
                    
                with torch.no_grad():
                
                    vb_terms_bpd_result = self._vb_terms_bpd(
                        denoise_fn, 
                        coords_start=coords_start, coords_t=coords_t, 
                        feats_start=feats_start, feats_t=feats_t, 
                        t=t_batch,)
                
                kl_coords, coords_recon, kl_feats, feats_recon = vb_terms_bpd_result
                
                vbs_coords.append(kl_coords)
                vbs_feats.append(kl_feats)
                
                xstart_mse_coords = mean_flat((coords_recon-coords_start)**2)
                xstart_mse_feats = mean_flat((feats_recon-feats_start)**2)
                xstart_mses_coords.append(xstart_mse_coords)
                xstart_mses_feats.append(xstart_mse_feats)
                
                eps_coords = self._predict_eps_from_xstart(coords_t, t_batch, coords_recon)
                mse_coords = mean_flat((eps_coords - noise_coords) ** 2)
                mses_coords.append(mse_coords)
                
                eps_feats = self._predict_eps_from_xstart(feats_t, t_batch, feats_recon)
                mse_feats = mean_flat((eps_feats - noise_feats) ** 2)
                mses_feats.append(mse_feats)

            vbs_coords = torch.stack(vbs_coords, dim=1)  # shape N, T
            xstart_mses_coords = torch.stack(xstart_mses_coords, dim=1)
            mses_coords = torch.stack(mses_coords, dim=1)
            
            prior_bpd_coords = self._prior_bpd(coords_start)
            total_bpd_coords = vbs_coords.sum(dim=1) + prior_bpd_coords
            
            vbs_feats = torch.stack(vbs_feats, dim=1)
            xstart_mses_feats = torch.stack(xstart_mses_feats, dim=1)
            mses_feats = torch.stack(mses_feats, dim=1)
            
            prior_bpd_feats = self._prior_bpd(feats_start)
            total_bpd_feats = vbs_feats.sum(dim=1) + prior_bpd_feats
                
            return total_bpd_coords, vbs_coords, prior_bpd_coords, xstart_mses_coords, mses_coords, \
                total_bpd_feats, vbs_feats, prior_bpd_feats, xstart_mses_feats, mses_feats
