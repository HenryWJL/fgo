"""
Reference code:
https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/gym_wrapper.py
https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_robosuite.py
https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py
"""
import fpsample
import numpy as np

try:
    import gym
    from gym import spaces
except ImportError:
    # Most APIs between gym and gymnasium are compatible
    print("WARNING! openai gym is not installed. We will try to use gymnasium instead.")
    import gymnasium as gym
    from gymnasium import spaces

import mimicgen    
import robosuite as suite
from omegaconf import ListConfig
from typing import Optional, Union, Tuple, Literal, List, Dict
from robosuite.controllers import load_controller_config
from robosuite.utils.camera_utils import (
    get_real_depth_map,
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix
)
from fgo.common.pc_util import pc_normalize, depth2pc
from fgo.common.transform_util import RotationTransformer


def get_box_space(sample: np.ndarray) -> spaces.Box:
    if np.issubdtype(sample.dtype, np.integer):
        low = np.iinfo(sample.dtype).min
        high = np.iinfo(sample.dtype).max
    elif np.issubdtype(sample.dtype, np.inexact):
        low = float("-inf")
        high = float("inf")
    else:
        raise ValueError()
    return spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)


def compute_smoothness_metrics(actions, accelerations, dt=1.0):
    """
    Computes ATV and JerkRMS.
    
    Args:
        actions: np.ndarray of shape (N, D) - e.g., (N, 7) for delta_actions
        accelerations: np.ndarray of shape (N, C) - e.g., (N, 7) joint accelerations
        dt: float, the control timestep (default 1.0 if time-scaling is ignored)
        
    Returns:
        atv (float): Action Total Variance
        jerk_rms (float): Jerk Root Mean Square
    """
    
    # ==========================================
    # 1. Action Total Variance (ATV)
    # ==========================================
    # The formula divides the total sum of absolute differences by M*(T-1).
    # Since M*(T-1) is exactly the total number of elements in the difference array,
    # this is mathematically identical to simply taking the global mean.
    
    action_diffs = np.abs(actions[1:] - actions[:-1])
    atv = np.mean(action_diffs)
    
    
    # ==========================================
    # 2. Jerk Root Mean Square (JerkRMS)
    # ==========================================
    # Calculate jerk using a single backward difference of the acceleration.
    # Resulting shape: (N-1, C)
    jerks = (accelerations[1:] - accelerations[:-1]) / dt
    
    # Compute the squared L2 norm for each timestep: || jerk_t ||_2^2
    # We sum the squares across the joint dimension (axis=1)
    jerk_l2_squared = np.sum(jerks**2, axis=1)
    
    # Take the mean across the time dimension (1 / (N-1) * sum), 
    # then apply the square root.
    jerk_rms = np.sqrt(np.mean(jerk_l2_squared))
    
    return atv, jerk_rms


class RobosuiteEnv(gym.Env):

    def __init__(
        self,
        env_name: str,
        robots: Union[str, List[str]],
        camera_names: Union[str, List[str]],
        use_object_obs: Optional[bool] = False,
        use_pc_obs: Optional[bool] = True,
        use_mask_obs: Optional[bool] = False,
        image_size: Optional[Tuple[int, int]] = (84, 84),
        num_points: Optional[int] = 512,
        bounding_boxes: Optional[Dict] = dict(),
        normalize_pc: Optional[bool] = True,
        controller: Optional[str] = "OSC_POSE",
        delta_action: Optional[bool] = False,
        control_freq: Optional[int] = 20,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        render_camera: Optional[str] = "agentview",
        render_image_size: Optional[Tuple[int, int]] = (256, 256)
    ) -> None:
        """
        Gym wrapper class for Robosuite (https://robosuite.ai/docs/overview.html).

        Args:
            env_name (str): Robosuite environment (https://robosuite.ai/docs/modules/environments.html).
            robots (str or list): Robot name(s) (https://robosuite.ai/docs/modules/robots.html).
            camera_names (str or list): Camera name(s) (https://robosuite.ai/docs/modules/sensors.html).
            use_object_obs (bool): If True, returns object states.
            use_pc_obs (bool): If True, returns point cloud observations.
            use_mask_obs (bool): If True, returns binary robot masks.
            image_size (tuple): height and width of returned images.
            num_points (int): Number of points in the point clouds.
            bounding_boxes (dict): Per-camera bounding boxes of the point clouds.
            normalize_pc (bool): If True, returns normalized point cloud observations.
            controller (str): Controller type (https://robosuite.ai/docs/modules/controllers.html).
            delta_action (bool): If True, uses delta action control.
            control_freq (int): control frequency.
            render_mode (str): "human" or "rgb_array". If "human", opens a
                real-time onscreen viewer; otherwise, returns a NumPy array.
        """
        # Robosuite strictly requires @robots and @camera_names to be either str or list type.
        # This may cause errors when instantiating with hydra (which returns ListConfig type).
        if isinstance(robots, ListConfig):
            robots = list(robots)
        if isinstance(camera_names, ListConfig):
            camera_names = list(camera_names)
        env_kwargs = dict(
            env_name=env_name,
            robots=robots,
            camera_names=camera_names,
            use_object_obs=use_object_obs,
            use_camera_obs=use_pc_obs,
            camera_depths=use_pc_obs,
            camera_heights=image_size[0],
            camera_widths=image_size[1],
            control_freq=control_freq,
            ignore_done=True
        )
        # ==================================================================== #
        # ======================= Observation Settings ======================= #
        # ==================================================================== #
        self.camera_names = [camera_names] if isinstance(camera_names, str) else camera_names
        self.image_size = image_size
        if use_mask_obs:
            env_kwargs['camera_segmentations'] = "element"
        self.use_pc_obs = use_pc_obs
        self.use_mask_obs = use_mask_obs
        self.num_points = num_points
        self.bounding_boxes = bounding_boxes
        self.normalize_pc = normalize_pc
        # =================================================================== #
        # ======================= Controller Settings ======================= #
        # =================================================================== #
        controller_config = load_controller_config(default_controller=controller)
        controller_config['control_delta'] = delta_action
        env_kwargs['controller_configs'] = controller_config
        self.rotation_transformer = RotationTransformer("rotation_6d", "axis_angle")
        # ================================================================= #
        # ======================= Renderer Settings ======================= #
        # ================================================================= #
        self.render_mode = render_mode
        self.render_camera = render_camera
        self.render_image_size = render_image_size
        env_kwargs['has_renderer'] = render_mode == "human"
        env_kwargs['has_offscreen_renderer'] = True if use_pc_obs else False
        # Create environments
        self._env = suite.make(**env_kwargs)
        # Set up observation and action spaces
        obs = self._extract_obs(self._env.reset())
        self.observation_space = spaces.Dict({
            key: get_box_space(obs[key]) for key in obs.keys()
        })
        low, high = self._env.action_spec
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )

        self._seed = None
        self.seed_state_map = {}
        self.task_completion_hold_count = -1

        #============== Evaluation ==============#
        self.action_buffer = []
        self.acceleration_buffer = []
        self.atv_buffer = []
        self.jerk_rms_buffer = []
        self.global_step = 0
        self.control_freq = control_freq
        #========================================#

    def _extract_seg_mask(self, seg_image: np.ndarray) -> np.ndarray:
        """
        Extract robot segmentation masks.

        Returns:
            seg_mask (np.ndarray): binary segmentation masks where
                1 for pixels occupied by the robot and 0 otherwise.
        """
        arm_geom_names = self._env.robots[0].robot_model.visual_geoms
        gripper_geom_names = self._env.robots[0].gripper.visual_geoms
        robot_geom_names = arm_geom_names + gripper_geom_names
        robot_geom_ids = [self._env.sim.model.geom_name2id(n) for n in robot_geom_names]
        seg_mask = np.isin(seg_image, robot_geom_ids).astype(np.uint8)
        return seg_mask

    def _extract_obs(self, raw_obs: Dict) -> Dict:
        obs = {}
        # Point clouds and segmentation masks
        if self.use_pc_obs:
            for camera_name in self.camera_names:
                # By default the camera intrinsic matrix computed from
                # MuJoCo’s camera parameters already assumes image flip.
                cam_intrin_mat = self.get_camera_intrinsic_matrix(camera_name)
                depth = get_real_depth_map(self._env.sim, raw_obs[f'{camera_name}_depth'].copy()).astype(np.float64)
                seg_mask = None
                if self.use_mask_obs:
                    seg_mask = self._extract_seg_mask(raw_obs[f'{camera_name}_segmentation_element'].copy())
                pc, seg_mask = depth2pc(
                    depth=depth,
                    camera_intrinsic_matrix=cam_intrin_mat,
                    bounding_box=self.bounding_boxes.get(camera_name),
                    seg_mask=seg_mask
                )
                fps_idx = fpsample.bucket_fps_kdline_sampling(pc, self.num_points, h=3)
                pc = pc[fps_idx]
                if self.normalize_pc:
                    pc = pc_normalize(pc)
                obs[f'{camera_name}_pc'] = pc
                if seg_mask is not None:
                    obs[f'{camera_name}_pc_mask'] = seg_mask[fps_idx]
        # Low dims
        for key in raw_obs.keys():
            if not key.endswith("image") and not key.endswith("depth") and not key.endswith("segmentation_element"):
                obs[key] = raw_obs[key].copy()
        return obs

    def seed(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)
        self._seed = seed

    def reset(self, **kwargs) -> Dict:
        #============== Evaluation ==============#
        self.action_buffer = []
        self.acceleration_buffer = []
        self.global_step = 0
        #========================================#

        self.task_completion_hold_count = -1
        if self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                raw_obs = self.reset_to({'states': self.seed_state_map[seed]})
            else:
                np.random.seed(seed=seed)
                raw_obs = self._env.reset()
                state = self._env.sim.get_state()
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            raw_obs = self._env.reset()
        obs = self._extract_obs(raw_obs)
        return obs
    
    def reset_to(self, state: Dict) -> Dict:
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment.
                - model (str): mujoco scene xml.
        
        Returns:
            raw_obs (dict): observation dictionary after setting the simulator
                state (only if "states" is in @state).
        """
        self.task_completion_hold_count = -1
        should_ret = False
        if "model" in state:
            self.reset(unset_ep_meta=False)
            xml = self._env.edit_model_xml(state["model"])
            self._env.reset_from_xml_string(xml)
            self._env.sim.reset()
        if "states" in state:
            self._env.sim.set_state(state["states"])
            self._env.sim.forward()
            should_ret = True
        if "goal" in state:
            if hasattr(self._env, "set_goal"):
                self._env.set_goal(**state["goal"])
            else:
                print("Warning: Environment does not support goal setting.")
        if should_ret:
            raw_obs = self._env._get_observations(force_update=True)
            return raw_obs
        return None

    def step(self, action: np.ndarray) -> Tuple:
        if action.shape[0] == 10:
            trans = action[:3]
            rot = action[3:9]
            gripper = action[[-1]]
            rot = self.rotation_transformer.forward(rot)
            action = np.concatenate([trans, rot, gripper])
        raw_obs, reward, _, info = self._env.step(action)
        obs = self._extract_obs(raw_obs)
        # Task is done if having a success for 10 consecutive timesteps. Code is adapted from
        # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/collect_human_demonstrations.py
        if self._env._check_success():
            if self.task_completion_hold_count > 0:
                self.task_completion_hold_count -= 1
            elif self.task_completion_hold_count < 0:
                self.task_completion_hold_count = 10
        else:
            self.task_completion_hold_count = -1
        done = not self.task_completion_hold_count
        info['is_success'] = done

        #============== Evaluation ==============#
        # action = self._env.sim.data.qpos[self._env.robots[0]._ref_joint_vel_indexes]
        # self.action_buffer.append(action)
        # qacc = self._env.sim.data.qacc[self._env.robots[0]._ref_joint_vel_indexes]
        # self.acceleration_buffer.append(qacc)
        # self.global_step += 1
        # if self.global_step == 32:
        #     actions = np.stack(self.action_buffer)
        #     accelerations = np.stack(self.acceleration_buffer)
        #     atv, jerk_rms = compute_smoothness_metrics(actions, accelerations, 1 / self.control_freq)
        #     self.atv_buffer.append(atv)
        #     self.jerk_rms_buffer.append(jerk_rms)
        #     print("Current mean ATV: ", np.mean(self.atv_buffer))
        #     print("Current mean JerkRMS: ", np.mean(self.jerk_rms_buffer))
        #     done = True
        #========================================#
        return obs, reward, done, info

    def render(self, **kwargs) -> None:
        """Render from simulation to either an on-screen window or off-screen to RGB array."""
        if self.render_mode == "human":
            cam_id = self._env.sim.model.camera_name2id(self.render_camera)
            self._env.viewer.set_camera(cam_id)
            return self._env.render()
        elif self.render_mode == "rgb_array":
            image = self._env.sim.render(
                height=self.render_image_size[0],
                width=self.render_image_size[1],
                camera_name=self.render_camera
            )
            return image[::-1]

    def get_camera_intrinsic_matrix(self, camera_name: str) -> np.ndarray:
        """Return the intrinsic matrix of @camera_name."""
        return get_camera_intrinsic_matrix(
            self._env.sim, camera_name, *self.image_size
        )

    def get_camera_extrinsic_matrix(self, camera_name: str) -> np.ndarray:
        """Return the extrinsic matrix of @camera_name in the world frame."""
        return get_camera_extrinsic_matrix(self._env.sim, camera_name)
    

def test():
    import open3d as o3d

    camera_name = "frontview"

    env = RobosuiteEnv(
        env_name="Lift",
        robots="Panda",
        camera_names=camera_name,
        use_object_obs=False,
        use_pc_obs=True,
        use_mask_obs=True,
        bounding_boxes=dict(
            frontview=dict(
                lower_bound=[-0.5, -0.4, 1.0],
                upper_bound=[0.5, 2.0, 2.35]
            ),
            robot0_eye_in_hand=dict(
                lower_bound=[-0.5, -0.5, 0.0],
                upper_bound=[0.5, 0.5, 0.5]
            )
        )
    )
    obs = env.reset(seed=42)
    env.render()
    env.step(np.random.rand(10))
    pc = obs[f'{camera_name}_pc']
    pc_mask = obs[f'{camera_name}_pc_mask']

    # Visualize point clouds
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc)
    colors = np.zeros_like(pc)
    colors[pc_mask == 1] = [1, 0, 0]   # red = robot
    colors[pc_mask == 0] = [0, 1, 0]   # green = environment
    pc_o3d.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc_o3d])