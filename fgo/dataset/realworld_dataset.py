import copy
import torch
from typing import Optional, Dict
from fgo.common.pytorch_util import dict_apply
from fgo.common.replay_buffer import ReplayBuffer
from fgo.model.common.normalizer import LinearNormalizer
from fgo.dataset.base_dataset import BaseDataset
from fgo.common.sampler import (
    get_val_mask,
    downsample_mask,
    SequenceSampler
)
from fgo.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
    realworld_abs_action_normalizer_from_stat
)


class RealWorldDataset(BaseDataset):

    def __init__(
        self,
        zarr_path: str,
        shape_meta: Dict,
        n_obs_steps: int,
        horizon: int,
        pad_before: Optional[int]=0,
        pad_after: Optional[int]=0,
        seed: Optional[int]=42,
        val_ratio: Optional[float]=0.0,
        max_train_episodes: Optional[int]=None,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=list(shape_meta['obs'].keys()) + ['action']
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed
        )
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
        )
        self.train_mask = train_mask
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self) -> BaseDataset:
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # action
        stat = array_to_stats(self.replay_buffer['action'])
        normalizer['action'] = realworld_abs_action_normalizer_from_stat(stat)
        # obs
        for key in self.shape_meta['obs'].keys():
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith('pos'):
                normalizer[key] = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                normalizer[key] = get_range_normalizer_from_stat(stat)
            elif key.endswith('point_cloud'):
                normalizer[key] = get_range_normalizer_from_stat(stat)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)
        obs_dict = dict()
        for key in self.shape_meta['obs'].keys():
            obs_dict[key] = data[key][T_slice]
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'])
        }
        return torch_data
