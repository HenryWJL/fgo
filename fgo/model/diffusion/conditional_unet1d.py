import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Optional, Union, List
from fgo.model.diffusion.positional_embedding import SinusoidalPosEmb


class Downsample1d(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: Optional[int]=8
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    

class CrossAttention(nn.Module):

    def __init__(self, in_dim: int, cond_dim: int, out_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: [batch_size, T_act, in_dim]
            cond: [batch_size, T_obs, cond_dim]
        
        Returns:
            attn_output: [batch_size, T_act, out_dim]
        '''
        # Project x and cond to query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(cond)
        value = self.value_proj(cond)
        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        # Apply attention
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output


class ConditionalResidualBlock1D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: Optional[int]=3,
        n_groups: Optional[int]=8,
        condition_type: Optional[str]='film'
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(
                in_channels,
                out_channels,
                kernel_size,
                n_groups=n_groups
            ),
            Conv1dBlock(
                out_channels,
                out_channels,
                kernel_size,
                n_groups=n_groups
            ),
        ])
        
        self.condition_type = condition_type
        cond_channels = out_channels
        if condition_type == 'film': 
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'add':
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'cross_attention_add':
            self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels)
        elif condition_type == 'cross_attention_film':
            cond_channels = out_channels * 2
            self.cond_encoder = CrossAttention(in_channels, cond_dim, cond_channels)
        elif condition_type == 'mlp_film':
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_dim),
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        else:
            raise NotImplementedError(f"condition_type {condition_type} not implemented")
        
        self.out_channels = out_channels
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]=None) -> torch.Tensor:
        '''
        Args:
            x: [batch_size, in_channels, horizon]
            cond: [batch_size, cond_dim]

        Returns:
            out: [batch_size, out_channels, horizon]
        '''
        out = self.blocks[0](x)  
        if cond is not None:      
            if self.condition_type == 'film':
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'add':
                embed = self.cond_encoder(cond)
                out = out + embed
            elif self.condition_type == 'cross_attention_add':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1)
                out = out + embed
            elif self.condition_type == 'cross_attention_film':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'mlp_film':
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            else:
                raise NotImplementedError(f"condition_type {self.condition_type} not implemented")
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: Optional[int]=256,
        freq_embed_dim: Optional[int] = 256,
        down_dims: List[int]=[256,512,1024],
        kernel_size: Optional[int]=3,
        n_groups: Optional[int]=8,
        condition_type: Optional[str]='film',
        use_down_condition: Optional[bool]=True,
        use_mid_condition: Optional[bool]=True,
        use_up_condition: Optional[bool]=True,
        use_freq_encoder: Optional[bool]=False
    ):
        super().__init__()
        
        cond_dim = global_cond_dim
        # diffusion step encoder
        cond_dim += diffusion_step_embed_dim
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        # frequency encoder
        if use_freq_encoder:
            cond_dim += freq_embed_dim
            self.freq_encoder = nn.Sequential(
                SinusoidalPosEmb(freq_embed_dim),
                nn.Linear(freq_embed_dim, freq_embed_dim * 4),
                nn.Mish(),
                nn.Linear(freq_embed_dim * 4, freq_embed_dim),
            )

        # U-Net
        all_dims = [input_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # downsampling
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_in, dim_out, cond_dim=cond_dim, 
                        kernel_size=kernel_size, n_groups=n_groups,
                        condition_type=condition_type
                    ),
                    ConditionalResidualBlock1D(
                        dim_out, dim_out, cond_dim=cond_dim, 
                        kernel_size=kernel_size, n_groups=n_groups,
                        condition_type=condition_type
                    ),
                    Downsample1d(dim_out) if not is_last else nn.Identity()
                ])
            )

        # backbone
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
        ])

        # upsampling
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_out*2, dim_in, cond_dim=cond_dim,
                        kernel_size=kernel_size, n_groups=n_groups,
                        condition_type=condition_type
                    ),
                    ConditionalResidualBlock1D(
                        dim_in, dim_in, cond_dim=cond_dim,
                        kernel_size=kernel_size, n_groups=n_groups,
                        condition_type=condition_type
                    ),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                ])
            )
        
        # final projection
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )
        
        self.condition_type = condition_type
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition
        self.use_freq_encoder = use_freq_encoder

    def forward(
        self,
        sample: torch.Tensor,
        global_cond: torch.Tensor,
        timestep: Union[torch.Tensor, float, int], 
        frequency: Union[torch.Tensor, float, int] = None
    ):
        """
        Args:
            sample: [batch_size, horizon, input_dim]
            global_cond: [batch_size, global_cond_dim]
            timestep: [B,]
            frequency: [B,]
        
        Returns:
            output: [batch_size, horizon, input_dim]
        """
        batch_size = sample.shape[0]
        sample = rearrange(sample, 'b h t -> b t h')

        # frequency encoding
        if self.use_freq_encoder:
            assert frequency is not None
            if not torch.is_tensor(frequency):
                frequency = torch.tensor([frequency], dtype=torch.long, device=sample.device)
            elif torch.is_tensor(frequency) and len(frequency.shape) == 0:
                frequency = frequency[None].to(sample.device)
            frequency = frequency.expand(batch_size)
            frequency_embed = self.freq_encoder(frequency)
            # combine with global conditionings
            if self.condition_type == 'cross_attention':
                frequency_embed = frequency_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            global_cond = torch.cat([frequency_embed, global_cond], axis=-1)

        # diffusion timestep encoding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(batch_size)
        timestep_embed = self.diffusion_step_encoder(timestep)
        # combine with global conditionings
        if self.condition_type == 'cross_attention':
            timestep_embed = timestep_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
        global_cond = torch.cat([timestep_embed, global_cond], axis=-1)
        
        x = sample
        h = []
        for (resnet, resnet2, downsample) in self.down_modules:
            if self.use_down_condition:
                x = resnet(x, global_cond)
                x = resnet2(x, global_cond)
            else:
                x = resnet(x)
                x = resnet2(x)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                x = mid_module(x, global_cond)
            else:
                x = mid_module(x)

        for (resnet, resnet2, upsample) in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                x = resnet(x, global_cond)
                x = resnet2(x, global_cond)
            else:
                x = resnet(x)
                x = resnet2(x)
            x = upsample(x)

        output = self.final_conv(x)
        output = rearrange(output, 'b t h -> b h t')

        return output
