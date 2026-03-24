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


class StreamingSVSModel(nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        frames_per_chunk: int,
        codebook_size: int,
        phoneme_vocab_size: int,
        phoneme_emb_dim: int,
        pitch_proj_dim: int,
        time_proj_dim: int,
        cond_dim: int,
        token_emb_dim: int,
        prev_chunk_dim: int,
        model_dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.frames_per_chunk = frames_per_chunk
        self.codebook_size = codebook_size

        self.cond_enc = ConditionEncoder(
            phoneme_vocab_size=phoneme_vocab_size,
            phoneme_emb_dim=phoneme_emb_dim,
            pitch_proj_dim=pitch_proj_dim,
            time_proj_dim=time_proj_dim,
            cond_dim=cond_dim,
        )
        self.token_emb = nn.Embedding(codebook_size, token_emb_dim)
        self.prev_proj = nn.Sequential(
            nn.Linear(token_emb_dim, prev_chunk_dim),
            nn.SiLU(),
            nn.Linear(prev_chunk_dim, prev_chunk_dim),
        )
        self.bos_chunk = nn.Parameter(torch.zeros(prev_chunk_dim))
        self.fuse = nn.Sequential(
            nn.Linear(cond_dim + prev_chunk_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.blocks = nn.ModuleList([ResidualBlock(model_dim, hidden_mult=4) for _ in range(num_blocks)])
        self.slot_emb = nn.Parameter(torch.randn(frames_per_chunk, num_codebooks, model_dim) * 0.02)
        self.classifier = nn.Linear(model_dim, codebook_size)

    def _encode_chunk(self, codes: torch.Tensor) -> torch.Tensor:
        emb = self.token_emb(codes)  # [B, T, F, K, E]
        pooled = emb.mean(dim=(2, 3))
        return self.prev_proj(pooled)

    def _shift_right_chunk_repr(self, chunk_repr: torch.Tensor) -> torch.Tensor:
        b, t, d = chunk_repr.shape
        out = self.bos_chunk.view(1, 1, d).expand(b, t, d).clone()
        if t > 1:
            out[:, 1:] = chunk_repr[:, :-1]
        return out

    def _predict_logits(self, cond: torch.Tensor, prev_repr: torch.Tensor) -> torch.Tensor:
        h = self.fuse(torch.cat([cond, prev_repr], dim=-1))  # [B, T, H]
        slot = self.slot_emb.view(1, 1, self.frames_per_chunk, self.num_codebooks, h.size(-1))
        h = h.unsqueeze(2).unsqueeze(3) + slot
        for blk in self.blocks:
            h = blk(h)
        return self.classifier(h)

    def forward_train(
        self,
        codes: torch.Tensor,        # [B, T, F, K]
        phoneme_id: torch.Tensor,   # [B, T]
        cond_num: torch.Tensor,     # [B, T, 7]
    ) -> torch.Tensor:
        cond = self.cond_enc(phoneme_id, cond_num)
        prev_repr = self._shift_right_chunk_repr(self._encode_chunk(codes))
        return self._predict_logits(cond, prev_repr)

    @torch.no_grad()
    def generate(
        self,
        phoneme_id: torch.Tensor,   # [B, T]
        cond_num: torch.Tensor,     # [B, T, 7]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        b, t = phoneme_id.shape
        cond = self.cond_enc(phoneme_id, cond_num)
        prev_repr = self.bos_chunk.view(1, -1).expand(b, -1)
        out_chunks = []

        for i in range(t):
            logits = self._predict_logits(cond[:, i : i + 1], prev_repr.unsqueeze(1)).squeeze(1)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_codes = torch.distributions.Categorical(probs=probs).sample()
            else:
                next_codes = logits.argmax(dim=-1)
            out_chunks.append(next_codes)
            prev_repr = self._encode_chunk(next_codes.unsqueeze(1)).squeeze(1)

        return torch.stack(out_chunks, dim=1)
