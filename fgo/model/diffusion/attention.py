# Adapted from https://github.com/WEIRDLabUW/unified-world-model/blob/main/models/common/attention.py
# and https://github.com/WEIRDLabUW/unified-world-model/blob/main/models/common/adaln_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rotary_embed(x: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """Rotates the input tensors by the positional embeddings.

    Args:
        x: a tensor of shape (..., seq_len, dim).
        thetas: a tensor of shape (..., seq_len, dim) of positional embeddings.

    Returns:
        x: a tensor of shape (..., seq_len, dim) of the rotated input tensors.
    """
    assert x.shape[-2:] == thetas.shape[-2:]
    x1, x2 = x.chunk(2, dim=-1)
    x_rotate_half = torch.cat([-x2, x1], dim=-1)
    return x * thetas.cos() + x_rotate_half * thetas.sin()


class MLP(nn.Module):
    """Multilayer perceptron with two hidden layers."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act: Optional[nn.Module]=nn.GELU,
        drop: Optional[float]=0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multiheaded self-attention."""
    def __init__(
        self,
        dim: int,
        num_heads: Optional[int]=8,
        qkv_bias: Optional[bool]=False,
        attn_drop: Optional[float]=0.0,
        proj_drop: Optional[float]=0.0,
        is_causal: Optional[bool]=False,
        causal_block: Optional[int]=1,
        use_sdpa: Optional[bool]=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.causal_block = causal_block

        if is_causal and causal_block > 1:
            print("Disabling torch spda kernel for block causal attention")
            self.use_sdpa = False
        else:
            self.use_sdpa = use_sdpa

        if not self.use_sdpa:
            self.causal_block_mat = nn.Parameter(
                torch.ones((causal_block, causal_block)).bool(),
                requires_grad=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Optional[torch.Tensor]=None,
        attn_mask: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        # Attention mask has shape (B, N, N) and dtype torch.bool where a
        # value of True indicates that the element should take part in attention.
        if attn_mask is not None:
            assert len(attn_mask.shape) == 3
            attn_mask = attn_mask.unsqueeze(1)

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        if pos_embed is not None:
            q = apply_rotary_embed(q, pos_embed)
            k = apply_rotary_embed(k, pos_embed)

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p=self.attn_drop.p, is_causal=self.is_causal
            )
        else:
            attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            if self.is_causal:
                assert attn_mask is None
                assert N % self.causal_block == 0
                num_blocks = N // self.causal_block
                block_diag_mat = torch.block_diag(
                    *[self.causal_block_mat for _ in range(num_blocks)]
                )
                triu_mat = torch.triu(
                    torch.ones(N, N, device=x.device), diagonal=1
                ).bool()
                mask = torch.logical_and(~block_diag_mat, triu_mat)
                attn = attn.masked_fill(mask.view(1, 1, N, N), float("-inf"))
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AdaLNAttentionBlock(nn.Module):
    """Multiheaded self-attention block with adaptive layer normalization modulation."""
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: Optional[int]=8,
        mlp_ratio: Optional[float]=4.0,
        qkv_bias: Optional[bool]=False,
        drop: Optional[float]=0.0,
        attn_drop: Optional[float]=0.0,
        act: Optional[nn.Module]=nn.GELU,
        norm: Optional[nn.Module]=nn.LayerNorm,
        is_causal: Optional[bool]=False,
        causal_block: Optional[int]=1,
    ):
        super().__init__()
        self.norm1 = norm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
            causal_block=causal_block,
        )
        self.norm2 = norm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            out_dim=dim,
            act=act,
            drop=drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pos_embed: Optional[torch.Tensor]=None,
        attn_mask: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), pos_embed, attn_mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class AdaLNFinalLayer(nn.Module):

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = self.linear(modulate(self.norm(x), shift, scale))
        return x
