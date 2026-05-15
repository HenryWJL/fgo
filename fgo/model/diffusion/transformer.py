import torch
import torch.nn as nn
from typing import Optional
from fgo.model.diffusion.positional_embedding import SinusoidalPosEmb
from fgo.model.diffusion.attention import AdaLNAttentionBlock, AdaLNFinalLayer

# Adapted from https://github.com/WEIRDLabUW/unified-world-model/blob/main/models/dp/transformer.py
class TransformerNoisePredictionNet(nn.Module):

    def __init__(
        self,
        input_len: int,
        input_dim: int,
        global_cond_dim: int,
        timestep_embed_dim: Optional[int]=256,
        embed_dim: Optional[int]=768,
        depth: Optional[int]=12,
        num_heads: Optional[int]=12,
        mlp_ratio: Optional[float]=4.0,
        qkv_bias: Optional[bool]=True,
    ):
        super().__init__()
        self.input_len = input_len

        # Input encoder and decoder
        hidden_dim = int(max(input_dim, embed_dim) * mlp_ratio)
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Timestep encoder
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

        # Model components
        self.pos_embed = nn.Parameter(
            torch.empty(1, input_len, embed_dim).normal_(std=0.02)
        )
        cond_dim = global_cond_dim + timestep_embed_dim
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)

        # AdaLN-specific weight initialization
        self.initialize_weights()

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def initialize_weights(self) -> None:
        # Base initialization
        self.apply(self.init_weights)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor
    ) -> torch.Tensor:
        # Encode input
        embed = self.input_encoder(sample)

        # Encode timestep
        if len(timestep.shape) == 0:
            timestep = timestep.expand(sample.shape[0]).to(
                dtype=torch.long, device=sample.device
            )
        temb = self.timestep_encoder(timestep)

        # Concatenate timestep and condition along the sequence dimension
        x = embed + self.pos_embed
        cond = torch.cat([global_cond, temb], dim=-1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.head(x, cond)

        # Decode output
        out = self.output_decoder(x)
        return out
