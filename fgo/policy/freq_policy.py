import torch
import torch.nn as nn
from functools import partial
from typing import Dict
from fgo.model.common.normalizer import LinearNormalizer
from fgo.policy.base_policy import BasePolicy
from fgo.model.vision.pointnet_extractor import DP3Encoder
from fgo.model.autoregressive.mae import MaskedAutoencoderViT
from fgo.common.pytorch_util import dict_apply
from fgo.common.model_util import print_params


class FreqPolicy(BasePolicy):

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: DP3Encoder,
        model_cfg: Dict,
        n_obs_steps: int,
        horizon: int,
        n_action_steps: int
    ):
        super().__init__()

        self.obs_encoder = obs_encoder
        self.obs_feature_dim = self.obs_encoder.output_shape()

        model_cfg['condition_dim'] = self.obs_feature_dim
        model_cfg['trajectory_dim'] = self.action_dim = shape_meta['action']['shape'][0]
        model_cfg['norm_layer'] = partial(nn.LayerNorm, eps=1e-6)
        self.model = MaskedAutoencoderViT(**model_cfg)

        self.normalizer = LinearNormalizer()
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps

        print_params(self)

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # normalize inputs
        obs = self.normalizer.normalize(obs_dict)
        
        B = next(iter(obs.values())).shape[0]
        To = self.n_obs_steps
        Ta = self.n_action_steps
        
        # condition through global feature
        obs = dict_apply(obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(obs)
        global_cond = obs_features.reshape(B, To, self.obs_feature_dim)
        
        with torch.no_grad():
            trajectory = self.model.sample_tokens_mask(B, global_cond)
        
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
        global_cond = obs_features.reshape(B, To, self.obs_feature_dim)

        # compute loss
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.model(trajectory, global_cond)
        loss_dict = {
            'bc_loss': loss.item(),
        }
        
        return loss, loss_dict
