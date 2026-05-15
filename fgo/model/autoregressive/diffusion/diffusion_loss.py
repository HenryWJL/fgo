import math
import torch
import torch.nn as nn
from typing import Optional
from . import create_diffusion


class DiffusionLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        depth: int,
        width: int,
        num_sampling_steps: str
    ):
        super(DiffusionLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels*2,
            z_channels=z_channels,
            num_res_blocks=depth,
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(
        self,
        z: torch.Tensor,
        target: torch.Tensor,
        index: Optional[torch.Tensor]=None,
        loss_weight: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z, index=index)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if loss_weight is not None:
            loss = loss * loss_weight

        return loss.mean()

    def sample(
        self,
        z: torch.Tensor,
        temperature: Optional[float]=1.0,
        cfg: Optional[float]=1.0,
        index: Optional[torch.Tensor]=None,
        num_steps: Optional[int]=None
    ) -> torch.Tensor:
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).to(z.device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg, index=index)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).to(z.device)
            model_kwargs = dict(c=z, index=index)
            sample_fn = self.net.forward
            
        diffusion = self.gen_diffusion
        if num_steps is not None:
            diffusion = create_diffusion(timestep_respacing='ddim'+str(num_steps), noise_schedule="cosine")
        sampled_token_latent = diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )
        return sampled_token_latent


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: Optional[int]=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: Optional[int]=10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels+32
        self.cond_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(self.model_channels)
        self.cond_embed = nn.Linear(z_channels, self.cond_channels)
        self.index_cond_embed = nn.Linear(1, 32)

        self.input_proj = nn.Linear(in_channels, self.model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                self.model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(self.model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        index: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        if index != None:
            index = self.index_cond_embed(index)
            c = torch.cat([c, index], dim=-1)
            y = t + c 
        else:
            y = t + c

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cfg_scale: float,
        index: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c, index)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
