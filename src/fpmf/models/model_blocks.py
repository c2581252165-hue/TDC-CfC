from __future__ import annotations

import math

import torch
from torch import nn


# Sentinel-2A nominal centre wavelengths and bandwidths (nm).  They are sensor
# constants, never learned from validation/test labels.  The project uses these
# metadata rather than pretending that B2...B12 form an equally spaced axis.
S2_CENTER_NM = torch.tensor(
    [492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 1613.7, 2202.4],
    dtype=torch.float32,
)
S2_FWHM_NM = torch.tensor(
    [66.0, 36.0, 31.0, 15.0, 15.0, 20.0, 106.0, 21.0, 91.0, 175.0],
    dtype=torch.float32,
)
TARGET_INDICES = torch.tensor([0, 1, 2, 6, 8, 9], dtype=torch.long)


def normalized_sensor_metadata() -> torch.Tensor:
    centre = (S2_CENTER_NM - S2_CENTER_NM.mean()) / S2_CENTER_NM.std()
    width = torch.log1p(S2_FWHM_NM)
    width = (width - width.mean()) / width.std()
    return torch.stack((centre, width), dim=-1)


def wavelength_graph(scale_nm: float = 180.0) -> torch.Tensor:
    distance = S2_CENTER_NM[:, None] - S2_CENTER_NM[None, :]
    graph = torch.exp(-0.5 * (distance / float(scale_nm)).square())
    graph = graph / graph.sum(dim=-1, keepdim=True)
    return graph


def spectral_factor_initial_logits(factor_count: int = 4) -> torch.Tensor:
    if factor_count != 4:
        raise ValueError("The paper model uses four spectral factors")
    centres = torch.tensor([500.0, 700.0, 850.0, 1800.0], dtype=torch.float32)
    scales = torch.tensor([120.0, 90.0, 100.0, 500.0], dtype=torch.float32)
    scores = -0.5 * (
        (S2_CENTER_NM.unsqueeze(0) - centres.unsqueeze(1)) / scales.unsqueeze(1)
    ).square()
    # Avoid exact softmax underflow for physically distant band/factor pairs;
    # every factor remains trainable from the first optimizer step.
    return scores.clamp_min(-20.0)


def next_month_calendar(calendar: torch.Tensor) -> torch.Tensor:
    delta = math.pi / 6.0
    last_sin, last_cos = calendar[:, -1, 0], calendar[:, -1, 1]
    return torch.stack(
        (
            last_sin * math.cos(delta) + last_cos * math.sin(delta),
            last_cos * math.cos(delta) - last_sin * math.sin(delta),
        ),
        dim=-1,
    )


def history_reliability(mask: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
    """A deterministic, window-local reliability score in [0, 1]."""

    return mask.clamp(0.0, 1.0) * torch.exp(-age.clamp_min(0.0))


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )


class SensorMetadataMixin:
    def _register_sensor_metadata(self) -> None:
        self.register_buffer(
            "sensor_metadata", normalized_sensor_metadata(), persistent=True
        )
        self.register_buffer("target_indices", TARGET_INDICES.clone(), persistent=True)


class TargetWavelengthReadout(nn.Module, SensorMetadataMixin):
    """Six physical-band queries read a shared memory without target leakage."""

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self._register_sensor_metadata()
        self.metadata_projection = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.calendar_projection = nn.Linear(2, d_model, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.scalar = nn.Linear(d_model, 1)

    def forward(
        self, memory: torch.Tensor, calendar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = memory.shape[0]
        target_metadata = self.sensor_metadata.index_select(0, self.target_indices)
        queries = self.metadata_projection(target_metadata).unsqueeze(0)
        queries = queries.expand(batch_size, -1, -1)
        queries = queries + self.calendar_projection(next_month_calendar(calendar)).unsqueeze(1)
        decoded, weights = self.cross_attention(
            queries, memory, memory, need_weights=True
        )
        decoded = self.norm(decoded + queries)
        return self.scalar(decoded).squeeze(-1), weights


class BiasedSelfAttentionBlock(nn.Module):
    """Pre-norm attention with a per-sample log key-importance bias."""

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_importance: torch.Tensor,
        *,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_importance.shape != tokens.shape[:2]:
            raise ValueError("key_importance must have shape [batch, tokens]")
        batch_size, length, width = tokens.shape
        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized).reshape(
            batch_size, length, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (part.transpose(1, 2) for part in (q, k, v))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + key_importance.clamp_min(1.0e-4).log()[:, None, None, :]
        if causal:
            scores = scores.masked_fill(
                causal_mask(length, tokens.device)[None, None],
                torch.finfo(scores.dtype).min,
            )
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        context = torch.matmul(self.dropout(attention), v)
        context = context.transpose(1, 2).reshape(batch_size, length, width)
        tokens = tokens + self.dropout(self.output(context))
        tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
        return tokens, attention


class BiasedTransformerStack(nn.Module):
    def __init__(self, d_model: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [BiasedSelfAttentionBlock(d_model, heads, dropout) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, tokens: torch.Tensor, key_importance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = tokens.new_empty(0)
        for block in self.blocks:
            tokens, attention = block(tokens, key_importance, causal=True)
        return self.final_norm(tokens), attention
