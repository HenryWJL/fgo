import copy
import torch
import numpy as np
from typing import Optional, Dict
from fgo.common.pytorch_util import dict_apply
from fgo.common.replay_buffer import ReplayBuffer
from fgo.dataset.base_dataset import BaseDataset
from fgo.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer
)
from fgo.common.sampler import (
    get_val_mask,
    downsample_mask,
    SequenceSampler
)


class DexArtDataset(BaseDataset):

    def __init__(
        self,
        zarr_path: str, 
        horizon: Optional[int]=1,
        pad_before: Optional[int]=0,
        pad_after: Optional[int]=0,
        seed: Optional[int]=42,
        val_ratio: Optional[float]=0.0,
        max_train_episodes: Optional[int]=None,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'imagin_robot', 'img']
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

    def get_normalizer(self, mode: Optional[str]='limits', **kwargs) -> LinearNormalizer:
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['imagin_robot'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict:
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 512, 3)
        imagin_robot = sample['imagin_robot'][:,].astype(np.float32) # (T, 96, 7)
        
        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 512, 3
                'imagin_robot': imagin_robot, # T, 96, 7
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
