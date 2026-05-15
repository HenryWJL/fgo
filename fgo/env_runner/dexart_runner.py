import tqdm
import imageio
import torch
import numpy as np
from termcolor import cprint
from typing import Optional
from fgo.env.dexart_env import DexArtEnv
from fgo.gym_util.multistep_wrapper import MultiStepWrapper
from fgo.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from fgo.policy.base_policy import BasePolicy
from fgo.common.pytorch_util import dict_apply
from fgo.env_runner.base_runner import BaseRunner
from fgo.common.logger_util import LargestKRecorder


class DexArtRunner(BaseRunner):

    def __init__(
        self,
        output_dir: str,
        n_train: Optional[int]=50,
        max_steps: Optional[int]=200,
        n_obs_steps: Optional[int]=8,
        n_action_steps: Optional[int]=8,
        tqdm_interval_sec: Optional[float]=5.0,
        task_name: Optional[str]=None,
    ):
        super().__init__(output_dir)

        def env_fn(is_test=True):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    env=DexArtEnv(
                        task_name=task_name,
                        use_test_set=is_test,
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.env_train = env_fn(is_test=False)
        self.episode_train = n_train
        self.task_name = task_name
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_train = LargestKRecorder(K=3)
        self.logger_util_train10 = LargestKRecorder(K=5)

        
    def run(self, policy: BasePolicy):
        device = policy.device
        env_train = self.env_train

        all_returns_train = []
        all_success_rates_train = []
        videos = []

        ##############################
        # train env loop
        for _ in tqdm.tqdm(
            range(self.episode_train),
            desc=f"Eval in DexArt {self.task_name} Train Env",
            leave=False,
            mininterval=self.tqdm_interval_sec
        ):
            # start rollout
            obs = env_train.reset()
            policy.reset()

            done = False
            reward_sum = 0.
            for _ in range(self.max_steps):
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)
                # device_transfer
                np_action_dict = dict_apply(action_dict,lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, _ = env_train.step(action)
                reward_sum += reward
                done = np.all(done)
                if done:
                    break

            all_returns_train.append(reward_sum)
            all_success_rates_train.append(env_train.is_success())
            videos.append(env_train.env.get_video())
       
        SR_mean_train = np.mean(all_success_rates_train)
        returns_mean_train = np.mean(all_returns_train)

        # log
        log_data = dict()
        log_data['mean_success_rates_train'] = SR_mean_train
        log_data['mean_returns_train'] = returns_mean_train
        log_data['test_mean_score'] = SR_mean_train
        self.logger_util_train.record(SR_mean_train)
        self.logger_util_train10.record(SR_mean_train)

        log_data['SR_train_L3'] = self.logger_util_train.average_of_largest_K()
        log_data['SR_train_L5'] = self.logger_util_train10.average_of_largest_K()
        cprint( f"Mean SR train: {SR_mean_train:.3f}", 'green')

        # save videos
        videos = np.transpose(np.concatenate(videos), (0, 2, 3, 1))  # -> (T, H, W, C)
        imageio.mimwrite("video.mp4", videos, fps=30, codec='libx264')

        # clear out video buffer
        _ = env_train.reset()
        videos = None
        del env_train

        return log_data
