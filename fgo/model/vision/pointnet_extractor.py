import copy
import torch
import torch.nn as nn
from termcolor import cprint
from typing import Optional, Dict, Tuple, Union, List, Type


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Optional[nn.Module] = nn.ReLU,
    squash_output: Optional[bool] = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    Args:
        input_dim: Dimension of the input vector
        output_dim: Dimension of the output vector
        net_arch: Architecture of the neural net. It represents the number of units
            per layer. The length of this list is the number of layers.
        activation_fn: The activation function to use after each layer.
        squash_output: Whether to squash the output using a Tanh activation function
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud XYZ and RGB"""
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int]=1024,
        use_layernorm: Optional[bool]=False,
        final_norm: Optional[str]='none',
    ):
        """
        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')

        assert in_channels == 6, cprint(f"PointNetEncoderXYZRGB only supports 6 channels, but got {in_channels}", "red")
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud XYZ only"""
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int]=1024,
        use_layernorm: Optional[bool]=False,
        final_norm: Optional[str]='none',
    ):
        """
        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class DP3Encoder(nn.Module):

    def __init__(
        self,
        shape_meta: Dict,
        pointcloud_encoder_cfg: Dict,
        state_mlp_size: List[int]=[64, 64],
        use_pc_color: Optional[bool]=False
    ):
        super().__init__()
        obs_shape_meta = shape_meta['obs']
        self.state_key = [key for key in obs_shape_meta.keys() if obs_shape_meta[key]['type'] == 'low_dim']
        self.point_cloud_key = [key for key in obs_shape_meta.keys() if obs_shape_meta[key]['type'] == 'point_cloud']
        self.point_cloud_shape = obs_shape_meta[self.point_cloud_key[0]]['shape']
        self.state_shape = [sum(obs_shape_meta[key]['shape'][0] for key in self.state_key)]
        self.use_imagined_robot = 'imagin_robot' in obs_shape_meta.keys()
        if self.use_imagined_robot:
            self.imagination_shape = obs_shape_meta['imagin_robot']['shape']
        else:
            self.imagination_shape = None       
 
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")

        # point cloud encoder
        self.use_pc_color = use_pc_color
        if self.use_pc_color:
            backbone = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
        else:
            backbone = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        self.extractor = nn.ModuleDict({key: copy.deepcopy(backbone) for key in self.point_cloud_key})
        self.n_output_channels = pointcloud_encoder_cfg['out_channels'] * len(self.point_cloud_key)

        # state encoder
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]
        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        if self.use_imagined_robot:
            points = observations['point_cloud'] if self.use_pc_color else observations['point_cloud'][..., :3]
            img_points = observations['imagin_robot'][..., :points.shape[-1]]
            points = torch.concat([points, img_points], dim=1)
            pn_feat = self.extractor['point_cloud'](points)
        else:
            pn_feat = torch.cat([
                self.extractor[key](observations[key] if self.use_pc_color else observations[key][..., :3]) for key in self.point_cloud_key
            ], dim=-1)
        state = torch.cat([observations[key] for key in self.state_key], dim=-1)
        state_feat = self.state_mlp(state)
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat

    def output_shape(self) -> int:
        return self.n_output_channels
