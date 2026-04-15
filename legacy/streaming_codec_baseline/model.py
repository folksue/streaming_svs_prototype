from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Discrete note/phoneme/slur/phone-progress -> condition embedding.
    """

    def __init__(
        self,
        note_vocab_size: int,
        phoneme_vocab_size: int,
        note_emb_dim: int = 128,
        phoneme_emb_dim: int = 128,
        slur_emb_dim: int = 16,
        phone_progress_bins: int = 8,
        phone_progress_emb_dim: int = 32,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.note_emb = nn.Embedding(note_vocab_size, note_emb_dim)
        self.phoneme_emb = nn.Embedding(phoneme_vocab_size, phoneme_emb_dim)
        self.slur_emb = nn.Embedding(2, slur_emb_dim)
        self.phone_progress_emb = nn.Embedding(phone_progress_bins, phone_progress_emb_dim)
        in_dim = note_emb_dim + phoneme_emb_dim + slur_emb_dim + phone_progress_emb_dim
        self.out = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(
        self,
        note_id: torch.Tensor,
        phoneme_id: torch.Tensor,
        slur: torch.Tensor,
        phone_progress: torch.Tensor,
    ) -> torch.Tensor:
        n = self.note_emb(note_id)
        p = self.phoneme_emb(phoneme_id)
        s = self.slur_emb(slur.clamp(min=0, max=1))
        pg = self.phone_progress_emb(phone_progress)
        x = torch.cat([n, p, s, pg], dim=-1)
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


class LocalCausalSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class StreamingSVSModel(nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        frames_per_chunk: int,
        codebook_size: int,
        note_vocab_size: int,
        phoneme_vocab_size: int,
        note_emb_dim: int,
        phoneme_emb_dim: int,
        slur_emb_dim: int,
        phone_progress_bins: int,
        phone_progress_emb_dim: int,
        cond_dim: int,
        token_emb_dim: int,
        prev_chunk_dim: int,
        model_dim: int,
        attn_heads: int,
        attn_layers: int,
        history_window: int,
        num_blocks: int,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.frames_per_chunk = frames_per_chunk
        self.codebook_size = codebook_size
        self.history_window = max(1, history_window)

        self.cond_enc = ConditionEncoder(
            note_vocab_size=note_vocab_size,
            phoneme_vocab_size=phoneme_vocab_size,
            note_emb_dim=note_emb_dim,
            phoneme_emb_dim=phoneme_emb_dim,
            slur_emb_dim=slur_emb_dim,
            phone_progress_bins=phone_progress_bins,
            phone_progress_emb_dim=phone_progress_emb_dim,
            cond_dim=cond_dim,
        )
        self.token_emb = nn.Embedding(codebook_size, token_emb_dim)
        self.prev_proj = nn.Sequential(
            nn.Linear(token_emb_dim, prev_chunk_dim),
            nn.SiLU(),
            nn.Linear(prev_chunk_dim, prev_chunk_dim),
        )
        self.bos_chunk = nn.Parameter(torch.zeros(prev_chunk_dim))
        self.chunk_in = nn.Sequential(
            nn.Linear(cond_dim + prev_chunk_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.context_blocks = nn.ModuleList(
            [
                LocalCausalSelfAttentionBlock(model_dim, num_heads=attn_heads, hidden_mult=4)
                for _ in range(attn_layers)
            ]
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

    def _build_local_attn_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        dist = idx[:, None] - idx[None, :]
        return (dist < 0) | (dist >= self.history_window)

    def _contextualize_chunks(
        self,
        cond: torch.Tensor,
        prev_repr: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Chunk history remains local-causal; only the condition schema changed in v1.
        x = self.chunk_in(torch.cat([cond, prev_repr], dim=-1))  # [B, T, H]
        attn_mask = self._build_local_attn_mask(x.size(1), x.device)
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()
        for blk in self.context_blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x

    def _predict_logits(self, chunk_h: torch.Tensor) -> torch.Tensor:
        h = chunk_h
        # Chunk-level hidden state is broadcast to all frame/codebook slots, then classified in parallel.
        slot = self.slot_emb.view(1, 1, self.frames_per_chunk, self.num_codebooks, h.size(-1))
        h = h.unsqueeze(2).unsqueeze(3) + slot
        for blk in self.blocks:
            h = blk(h)
        return self.classifier(h)

    def _sample_next_codes(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.distributions.Categorical(probs=probs).sample()
        return logits.argmax(dim=-1)

    def forward_train(
        self,
        codes: torch.Tensor,        # [B, T, F, K]
        note_id: torch.Tensor,      # [B, T]
        phoneme_id: torch.Tensor,   # [B, T]
        slur: torch.Tensor,         # [B, T]
        phone_progress: torch.Tensor,  # [B, T]
        mask: torch.Tensor | None = None,  # [B, T]
    ) -> torch.Tensor:
        cond = self.cond_enc(note_id, phoneme_id, slur, phone_progress)
        prev_repr = self._shift_right_chunk_repr(self._encode_chunk(codes))
        chunk_h = self._contextualize_chunks(cond, prev_repr, valid_mask=mask)
        return self._predict_logits(chunk_h)

    @torch.no_grad()
    def generate(
        self,
        note_id: torch.Tensor,      # [B, T]
        phoneme_id: torch.Tensor,   # [B, T]
        slur: torch.Tensor,         # [B, T]
        phone_progress: torch.Tensor,  # [B, T]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        out_chunks = list(
            self.generate_stream(
                note_id=note_id,
                phoneme_id=phoneme_id,
                slur=slur,
                phone_progress=phone_progress,
                temperature=temperature,
            )
        )
        return torch.stack(out_chunks, dim=1)

    @torch.no_grad()
    def generate_stream(
        self,
        note_id: torch.Tensor,      # [B, T]
        phoneme_id: torch.Tensor,   # [B, T]
        slur: torch.Tensor,         # [B, T]
        phone_progress: torch.Tensor,  # [B, T]
        temperature: float = 1.0,
    ):
        """
        Chunk-wise streaming generation.

        Yields one predicted chunk at a time with shape [B, F, K].
        """
        b, t = phoneme_id.shape
        cond = self.cond_enc(note_id, phoneme_id, slur, phone_progress)
        prev_repr = self.bos_chunk.view(1, -1).expand(b, -1)
        hist_cond = []
        hist_prev = []

        for i in range(t):
            hist_cond.append(cond[:, i])
            hist_prev.append(prev_repr)
            if len(hist_cond) > self.history_window:
                hist_cond = hist_cond[-self.history_window :]
                hist_prev = hist_prev[-self.history_window :]

            cond_window = torch.stack(hist_cond, dim=1)
            prev_window = torch.stack(hist_prev, dim=1)
            chunk_h = self._contextualize_chunks(cond_window, prev_window)
            logits = self._predict_logits(chunk_h[:, -1:]).squeeze(1)
            next_codes = self._sample_next_codes(logits, temperature=temperature)
            yield next_codes
            prev_repr = self._encode_chunk(next_codes.unsqueeze(1)).squeeze(1)
