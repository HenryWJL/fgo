import time
import zarr
import click
import fpsample
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Union, List
from scipy.spatial.transform import Rotation as R


BOUNDING_BOX = {
    'lower_bound': [-0.25, -0.5, 0],
    'upper_bound': [1.0, 0.4, 1.0]
}
LOW_DIM_KEYS = ['arm_qpos', 'arm_qvel', 'ee_pos', 'ee_quat', 'gripper_qpos']


def visualize_pc_seq(pc: np.ndarray):
    """
    Args:
        pc: [B, N, 3] or [B, N, 6]
    """
    xyz_seq = pc[..., :3]
    if pc.shape[-1] == 6:
        rgb_seq = pc[..., 3:]
    else:
        rgb_seq = None
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Sequence", width=960, height=540)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz_seq[0])
    if rgb_seq is not None:
        pc.colors = o3d.utility.Vector3dVector(rgb_seq[0])
    vis.add_geometry(pc)

    for i, xyz in enumerate(xyz_seq):
        pc.points = o3d.utility.Vector3dVector(xyz)
        if rgb_seq is not None:
            pc.colors = o3d.utility.Vector3dVector(rgb_seq[i])
        vis.update_geometry(pc)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    vis.destroy_window()


def frame_to_cloud(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray]=None,
    use_color: Optional[bool]=False,
    num_points: Optional[int]=1024,
    bounding_box: Optional[dict]=None
):
    """
    Args:
        xyz: [H, W, 3] or [N, 3]
        rgb: [H, W, 3] or [N, 3]
    """
    if len(xyz.shape) == 3:
        xyz = xyz.reshape(-1, 3)

    if rgb is not None and use_color:
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        if len(rgb.shape) == 3:
            rgb = rgb.reshape(-1, 3)
    
    # Check finity
    is_finite = np.isfinite(xyz).all(axis=-1)
    xyz = xyz[is_finite]
    if rgb is not None and use_color:
        rgb = rgb[is_finite]

    # Crop
    if bounding_box is not None:
        mask = (
            (np.all(xyz > bounding_box['lower_bound'], axis=-1))
            & (np.all(xyz < bounding_box['upper_bound'], axis=-1))
        )
        xyz = xyz[mask]
        if rgb is not None and use_color:
            rgb = rgb[mask]
    
    # FPS
    if len(xyz) >= num_points:
        fps_idx = fpsample.bucket_fps_kdline_sampling(xyz, num_points, h=3)
        xyz = xyz[fps_idx]
        if rgb is not None and use_color:
            rgb = rgb[fps_idx]

    if rgb is not None and use_color:
        pc = np.concatenate([xyz, rgb], axis=-1)
    else:
        pc = xyz

    return pc


def matrix_to_rotation_6d(mat: np.ndarray) -> np.ndarray:
    col0 = mat[..., :, 0]
    col1 = mat[..., :, 1]
    return np.concatenate([col0, col1], axis=-1)


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    a1 = rot_6d[..., 0: 3]
    a2 = rot_6d[..., 3: 6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    mat = np.stack((b1, b2, b3), axis=-1)
    return mat

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/common/rotation_transformer.py
class RotationTransformer:
    valid_reps = [
        "axis_angle",
        "euler_angles",
        "quaternion",
        "rotation_6d",
        "matrix",
    ]

    def __init__(
        self,
        from_rep: Optional[str] = "axis_angle",
        to_rep: Optional[str] = "rotation_6d", 
        from_convention: Optional[str] = None,
        to_convention: Optional[str] = None
    ) -> None:
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = []
        inverse_funcs = []

        if from_rep != "matrix":
            if from_rep == "axis_angle":
                forward_funcs.append(lambda x: R.from_rotvec(x).as_matrix())
                inverse_funcs.append(lambda x: R.from_matrix(x).as_rotvec())
            elif from_rep == "euler_angles":
                forward_funcs.append(lambda x: R.from_euler(from_convention, x).as_matrix())
                inverse_funcs.append(lambda x: R.from_matrix(x).as_euler(from_convention))
            elif from_rep == "quaternion":
                forward_funcs.append(lambda x: R.from_quat(x).as_matrix())
                inverse_funcs.append(lambda x: R.from_matrix(x).as_quat())
            elif from_rep == "rotation_6d":
                forward_funcs.append(rotation_6d_to_matrix)
                inverse_funcs.append(matrix_to_rotation_6d)

        if to_rep != "matrix":
            if to_rep == "axis_angle":
                forward_funcs.append(lambda x: R.from_matrix(x).as_rotvec())
                inverse_funcs.append(lambda x: R.from_rotvec(x).as_matrix())
            elif to_rep == "euler_angles":
                forward_funcs.append(lambda x: R.from_matrix(x).as_euler(to_convention))
                inverse_funcs.append(lambda x: R.from_euler(to_convention, x).as_matrix())
            elif to_rep == "quaternion":
                forward_funcs.append(lambda x: R.from_matrix(x).as_quat())
                inverse_funcs.append(lambda x: R.from_quat(x).as_matrix())
            elif to_rep == "rotation_6d":
                forward_funcs.append(matrix_to_rotation_6d)
                inverse_funcs.append(rotation_6d_to_matrix)

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: List) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, torch.Tensor):
            x_ = x.detach().cpu().numpy()
        for func in funcs:
            x_ = func(x_)
        if isinstance(x, torch.Tensor):
            return torch.from_numpy(x_)
        return x_

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


@click.command(help="Build datatsets from raw data.")
@click.option("-r", "--root", type=str, required=True, help="Root directory of raw data.")
@click.option("-sp", "--save_path", type=str, default="data/realworld_cup.zarr", help="Dataset saving path.")
@click.option("-uc", "--use_color", is_flag=True, help="Whether to store point cloud colors.")
@click.option("-np", "--num_points", type=int, default=1024, help="Number of downsampled points.")
@click.option("-rt", "--rotation_type", type=str, default="rotation_6d", help="Type of action rotation components.")
def main(root, save_path, use_color, num_points, rotation_type):
    rot_trans = RotationTransformer(from_rep="quaternion", to_rep=rotation_type)
    pc_paths = Path(root).expanduser().absolute().glob("**/points_*hz.npz")
    save_path = Path(save_path).expanduser().absolute()

    point_cloud = []
    arm_qpos = []
    arm_qvel = []
    ee_pos = []
    ee_quat = []
    gripper_qpos = []
    action = []
    episode_ends = []
    for pc_path in pc_paths:
        # Point cloud
        d = np.load(str(pc_path), allow_pickle=True)
        pc_xyz = d["xyz"]
        pc_rgb = d["rgb"]

        pc = []
        for i, xyz in enumerate(pc_xyz):
            rgb = pc_rgb[i] if use_color else None
            pc.append(frame_to_cloud(xyz, rgb, use_color, num_points, BOUNDING_BOX))
        pc = np.stack(pc)
        point_cloud.append(pc)
        # print("Point Cloud Shape: ", pc.shape)
        # visualize_pc_seq(pc)

        # Low dimensional data (proprioception and action)
        low_dim_path = pc_path.parent.joinpath(pc_path.name.replace("points", "actions"))
        f = np.load(str(low_dim_path), allow_pickle=True)
        arm_qpos.append(f['q'])
        arm_qvel.append(f['q_dot'])
        ee_pos.append(f['ee_pos'])
        ee_quat.append(f['ee_quat'])
        gripper_qpos.append(f['gripper_pos'][:, np.newaxis])

        target_ee_pos = f['ee_pos']
        target_ee_quat = f['ee_quat']
        target_ee_rotation6d = rot_trans.forward(target_ee_quat)
        gripper_desired = f['gripper_desired'][:, np.newaxis]
        action.append(np.concatenate([target_ee_pos, target_ee_rotation6d, gripper_desired], axis=-1))

        # Episode length
        episode_ends.append(len(pc))

    point_cloud = np.concatenate(point_cloud)
    low_dims = dict(
        arm_qpos=np.concatenate(arm_qpos),
        arm_qvel=np.concatenate(arm_qvel),
        ee_pos=np.concatenate(ee_pos),
        ee_quat=np.concatenate(ee_quat),
        gripper_qpos=np.concatenate(gripper_qpos),
    )
    action = np.concatenate(action)
    episode_ends = np.cumsum(episode_ends)
    print("Number of episodes: ", len(episode_ends))
    print("Totol episode length: ", len(action))
    print("All episode ends: ", episode_ends)

    with zarr.open(str(save_path), 'w') as z:
        z['data/point_cloud'] = point_cloud
        for key in LOW_DIM_KEYS:
            z[f'data/{key}'] = low_dims[key]
        z['data/action'] = action
        z['meta/episode_ends'] = episode_ends


if __name__ == "__main__":
    main()
