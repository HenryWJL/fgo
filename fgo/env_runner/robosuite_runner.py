import tqdm
import torch
import numpy as np
from termcolor import cprint
from typing import Optional, Dict
from fgo.env.robosuite_env import RobosuiteEnv
from fgo.gym_util.multistep_wrapper import MultiStepWrapper
from fgo.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from fgo.policy.base_policy import BasePolicy
from fgo.common.pytorch_util import dict_apply
from fgo.env_runner.base_runner import BaseRunner
from fgo.common.logger_util import LargestKRecorder


class RobosuiteRunner(BaseRunner):

    def __init__(
        self,
        output_dir: str,
        shape_meta: Dict,
        eval_episodes: Optional[int]=50,
        max_steps: Optional[int]=200,
        n_obs_steps: Optional[int]=8,
        n_action_steps: Optional[int]=8,
        abs_action: Optional[bool]=True,
        render_size: Optional[int]=84,
        tqdm_interval_sec: Optional[float]=5.0,
        task_name: Optional[str]=None,
        bounding_boxes: Dict=dict(),
    ):
        super().__init__(output_dir)

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    env=RobosuiteEnv(
                        env_name=task_name,
                        robots="Panda",
                        camera_names=list(bounding_boxes.keys()),
                        bounding_boxes=bounding_boxes,
                        delta_action=not abs_action,
                        render_image_size=(render_size, render_size)
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.env = env_fn()
        self.eval_episodes = eval_episodes
        self.shape_meta = shape_meta
        self.task_name = task_name
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = LargestKRecorder(K=3)
        self.logger_util_test10 = LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        env = self.env
        device = policy.device
        test_start_seed = 10000

        all_goal_achieved = []
        all_success_rates = []
        videos = []
        for episode_idx in tqdm.tqdm(
            range(self.eval_episodes),
            desc=f"Eval in Robosuite {self.task_name} Pointcloud Env",
            leave=False,
            mininterval=self.tqdm_interval_sec
        ):
            # start rollout
            env.env.env.seed(test_start_seed + episode_idx)
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            while not done:
                # create obs dict
                np_obs_dict = {key: obs[key] for key in self.shape_meta['obs'].keys() if not key.endswith('pc_mask')}
                # device transfer
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device=device))
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                # device_transfer
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, _, done, info = env.step(action)
                num_goal_achieved += np.sum(info['is_success'])
                done = np.all(done)
                actual_step_count += 1

            all_success_rates.append(np.sum(info['is_success']))
            all_goal_achieved.append(num_goal_achieved)
            videos.append(env.env.get_video())

        # log
        log_data = dict()
        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['test_mean_score'] = np.mean(all_success_rates)
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        
        # save videos
        import imageio
        videos = np.transpose(np.concatenate(videos), (0, 2, 3, 1))  # -> (T, H, W, C)
        imageio.mimwrite("video.mp4", videos, fps=30, codec='libx264')

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data
