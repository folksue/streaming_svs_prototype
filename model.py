from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Phoneme + pitch + timing/progress -> condition embedding.
    """

    def __init__(
        self,
        phoneme_vocab_size: int,
        phoneme_emb_dim: int = 128,
        pitch_proj_dim: int = 64,
        time_proj_dim: int = 64,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.phoneme_emb = nn.Embedding(phoneme_vocab_size, phoneme_emb_dim)
        self.pitch_proj = nn.Sequential(
            nn.Linear(1, pitch_proj_dim),
            nn.SiLU(),
            nn.Linear(pitch_proj_dim, pitch_proj_dim),
            nn.SiLU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(6, time_proj_dim),
            nn.SiLU(),
            nn.Linear(time_proj_dim, time_proj_dim),
            nn.SiLU(),
        )
        in_dim = phoneme_emb_dim + pitch_proj_dim + time_proj_dim
        self.out = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, phoneme_id: torch.Tensor, cond_num: torch.Tensor) -> torch.Tensor:
        # cond_num = [pitch, note_on, note_off, t_since_on, t_to_end, note_prog, ph_prog]
        pitch = cond_num[..., 0:1]
        timing = cond_num[..., 1:]

        p = self.phoneme_emb(phoneme_id)
        pp = self.pitch_proj(pitch)
        tp = self.time_proj(timing)
        x = torch.cat([p, pp, tp], dim=-1)
        return self.out(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = dim * hidden_mult
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class FlowStyleGenerator(nn.Module):
    """
    Lightweight flow-matching style generator:
      z_pred = G(eps, cond, ctx)
    """

    def __init__(self, data_dim: int, cond_dim: int, ctx_dim: int, model_dim: int = 512, num_blocks: int = 6):
        super().__init__()
        self.in_proj = nn.Linear(data_dim + cond_dim + ctx_dim, model_dim)
        self.blocks = nn.ModuleList([ResidualBlock(model_dim, hidden_mult=4) for _ in range(num_blocks)])
        self.out = nn.Linear(model_dim, data_dim)

    def forward(self, eps: torch.Tensor, cond: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([eps, cond, ctx], dim=-1)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


class StreamingSVSModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        frames_per_chunk: int,
        phoneme_vocab_size: int,
        phoneme_emb_dim: int,
        pitch_proj_dim: int,
        time_proj_dim: int,
        cond_dim: int,
        ctx_hidden: int,
        ctx_layers: int,
        model_dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.frames_per_chunk = frames_per_chunk
        self.data_dim = latent_dim * frames_per_chunk

        self.cond_enc = ConditionEncoder(
            phoneme_vocab_size=phoneme_vocab_size,
            phoneme_emb_dim=phoneme_emb_dim,
            pitch_proj_dim=pitch_proj_dim,
            time_proj_dim=time_proj_dim,
            cond_dim=cond_dim,
        )
        self.ctx_gru = nn.GRU(
            input_size=self.data_dim,
            hidden_size=ctx_hidden,
            num_layers=ctx_layers,
            batch_first=True,
        )
        self.gen = FlowStyleGenerator(
            data_dim=self.data_dim,
            cond_dim=cond_dim,
            ctx_dim=ctx_hidden,
            model_dim=model_dim,
            num_blocks=num_blocks,
        )

    def _shift_right(self, z_flat: torch.Tensor) -> torch.Tensor:
        # teacher forcing prev latent: [0, z_1, z_2, ..., z_{T-1}]
        b, t, d = z_flat.shape
        out = torch.zeros_like(z_flat)
        if t > 1:
            out[:, 1:] = z_flat[:, :-1]
        return out

    def forward_train(
        self,
        z_gt: torch.Tensor,          # [B, T, F, D]
        phoneme_id: torch.Tensor,    # [B, T]
        cond_num: torch.Tensor,      # [B, T, 7]
    ) -> torch.Tensor:
        b, t, f, d = z_gt.shape
        z_flat = z_gt.view(b, t, f * d)

        cond = self.cond_enc(phoneme_id, cond_num)
        prev = self._shift_right(z_flat)
        ctx, _ = self.ctx_gru(prev)

        eps = torch.randn_like(z_flat)
        pred = self.gen(eps, cond, ctx)
        return pred.view(b, t, f, d)

    @torch.no_grad()
    def generate(
        self,
        phoneme_id: torch.Tensor,   # [B, T]
        cond_num: torch.Tensor,     # [B, T, 7]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        b, t = phoneme_id.shape
        cond = self.cond_enc(phoneme_id, cond_num)

        hidden = None
        prev = torch.zeros(b, 1, self.data_dim, device=phoneme_id.device)
        out_chunks = []

        for i in range(t):
            ctx, hidden = self.ctx_gru(prev, hidden)
            eps = torch.randn(b, 1, self.data_dim, device=phoneme_id.device) * temperature
            pred = self.gen(eps, cond[:, i : i + 1], ctx)
            out_chunks.append(pred)
            prev = pred

        z = torch.cat(out_chunks, dim=1)
        return z.view(b, t, self.frames_per_chunk, self.latent_dim)
