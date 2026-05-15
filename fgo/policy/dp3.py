import torch
import torch.nn.functional as F
from typing import Optional, Dict
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from fgo.model.common.normalizer import LinearNormalizer
from fgo.policy.base_policy import BasePolicy
from fgo.model.vision.pointnet_extractor import DP3Encoder
from fgo.model.diffusion.conditional_unet1d import ConditionalUnet1D
from fgo.common.pytorch_util import dict_apply
from fgo.common.model_util import print_params


class DP3(BasePolicy):

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: DP3Encoder,
        model_cfg: Dict,
        noise_scheduler: DDIMScheduler,
        n_obs_steps: int,
        horizon: int,
        n_action_steps: int,
        num_inference_steps: Optional[int]=None,
    ):
        super().__init__()

        self.obs_encoder = obs_encoder
        self.obs_feature_dim = self.obs_encoder.output_shape()

        self.condition_type = model_cfg['condition_type']
        if 'cross_attention' in self.condition_type:
            global_cond_dim = self.obs_feature_dim
        else:
            global_cond_dim = self.obs_feature_dim * n_obs_steps
        model_cfg['global_cond_dim'] = global_cond_dim
        model_cfg['input_dim'] = self.action_dim = shape_meta['action']['shape'][0]
        self.model = ConditionalUnet1D(**model_cfg)

        self.noise_scheduler = noise_scheduler    
        self.normalizer = LinearNormalizer()
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # normalize inputs
        obs = self.normalizer.normalize(obs_dict)
        
        B = next(iter(obs.values())).shape[0]
        To = self.n_obs_steps
        T = self.horizon
        Ta = self.n_action_steps
        Da = self.action_dim
        
        # condition through global feature
        obs = dict_apply(obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(obs)
        if 'cross_attention' in self.condition_type:
            global_cond = obs_features.reshape(B, To, -1)
        else:
            global_cond = obs_features.reshape(B, -1)
        
        trajectory = torch.randn((B, T, Da), device=self.device, dtype=self.dtype)
        self.noise_scheduler.set_timesteps(self.num_inference_steps, self.device)
        for t in self.noise_scheduler.timesteps:
            pred = self.model(
                sample=trajectory,
                global_cond=global_cond,
                timestep=t,
            )
            trajectory = self.noise_scheduler.step(pred, t, trajectory).prev_sample
        
        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(trajectory)

        # get action
        start = To - 1
        end = start + Ta
        action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict) -> Dict[str, float]:
        # normalize input
        obs = self.normalizer.normalize(batch['obs'])
        obs = dict_apply(obs, lambda x: x.to(self.dtype))
        trajectory = self.normalizer['action'].normalize(batch['action']).to(self.dtype)
        
        B = next(iter(obs.values())).shape[0]
        To = self.n_obs_steps

        # condition through global feature
        obs = dict_apply(obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(obs)
        if 'cross_attention' in self.condition_type:
            global_cond = obs_features.reshape(B, To, -1)
        else:
            global_cond = obs_features.reshape(B, -1)

        # sample noise
        noise = torch.randn_like(trajectory)

        # sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps, 
            (B,),
            dtype=torch.long,
            device=trajectory.device
        )

        # add noise to the clean actions (forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # predict
        pred = self.model(
            sample=noisy_trajectory,
            global_cond=global_cond,
            timestep=timesteps, 
        )

        # compute loss
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)
        loss_dict = {
            'bc_loss': loss.item(),
        }
        
        return loss, loss_dict
