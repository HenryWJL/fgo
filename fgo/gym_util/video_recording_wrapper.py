try:
    import gym
except ImportError:
    import gymnasium as gym
import numpy as np
from typing import Optional, Literal, Tuple


class SimpleVideoRecordingWrapper(gym.Wrapper):

    def __init__(
        self, 
        env: gym.Env, 
        mode: Literal['human', 'rgb_array']='rgb_array',
        steps_per_render: Optional[int]=1,
    ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        self.env = env
        self.mode = mode
        self.steps_per_render = steps_per_render
        self.step_count = 0

    def reset(self, **kwargs) -> None:
        obs = super().reset(**kwargs)
        self.frames = list()

        frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        self.step_count = 1
        return obs
    
    def step(self, action: np.ndarray) -> Tuple:
        result = super().step(action)
        self.step_count += 1
        
        frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        return result
    
    def get_video(self) -> np.ndarray:
        video = np.stack(self.frames, axis=0) # (T, H, W, C)
        # to store as mp4 in wandb, we need (T, H, W, C) -> (T, C, H, W)
        video = video.transpose(0, 3, 1, 2)
        return video
