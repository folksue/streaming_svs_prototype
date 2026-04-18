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
        tokens_per_step: int,
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
        prev_step_dim: int,
        model_dim: int,
        attn_heads: int,
        attn_layers: int,
        history_window: int,
        num_blocks: int,
        audio_history_window: int = 64,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.tokens_per_step = tokens_per_step
        self.codebook_size = codebook_size
        self.history_window = max(1, history_window)
        self.audio_seq_len = tokens_per_step
        self.audio_history_window = max(1, audio_history_window)

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
            nn.Linear(token_emb_dim, prev_step_dim),
            nn.SiLU(),
            nn.Linear(prev_step_dim, prev_step_dim),
        )
        self.bos_step = nn.Parameter(torch.zeros(prev_step_dim))
        self.step_in = nn.Sequential(
            nn.Linear(cond_dim + prev_step_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.context_blocks = nn.ModuleList(
            [
                LocalCausalSelfAttentionBlock(model_dim, num_heads=attn_heads, hidden_mult=4)
                for _ in range(attn_layers)
            ]
        )
        self.audio_in = nn.Linear(token_emb_dim, model_dim)
        self.audio_bos = nn.Parameter(torch.zeros(model_dim))
        self.frame_pos_emb = nn.Parameter(torch.randn(self.audio_seq_len, model_dim) * 0.02)
        self.audio_blocks = nn.ModuleList(
            [
                LocalCausalSelfAttentionBlock(model_dim, num_heads=attn_heads, hidden_mult=4)
                for _ in range(num_blocks)
            ]
        )
        self.first_codebook_head = nn.Linear(model_dim, codebook_size)
        self.residual_cond = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.residual_heads = nn.ModuleList(
            [nn.Linear(model_dim, codebook_size) for _ in range(max(0, num_codebooks - 1))]
        )

    def _build_local_attn_mask(self, seq_len: int, device: torch.device, window: int) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        dist = idx[:, None] - idx[None, :]
        return (dist < 0) | (dist >= window)

    def _encode_step(self, first_codes: torch.Tensor) -> torch.Tensor:
        emb = self.token_emb(first_codes)  # [B, T, S, E]
        pooled = emb.mean(dim=2)
        return self.prev_proj(pooled)

    def _shift_right_step_repr(self, step_repr: torch.Tensor) -> torch.Tensor:
        b, t, d = step_repr.shape
        out = self.bos_step.view(1, 1, d).expand(b, t, d).clone()
        if t > 1:
            out[:, 1:] = step_repr[:, :-1]
        return out

    def _contextualize_steps(
        self,
        cond: torch.Tensor,
        prev_repr: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.step_in(torch.cat([cond, prev_repr], dim=-1))
        attn_mask = self._build_local_attn_mask(x.size(1), x.device, self.history_window)
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()
        for blk in self.context_blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x

    def _build_audio_inputs(self, first_codes: torch.Tensor, step_h: torch.Tensor) -> torch.Tensor:
        b, t, s = first_codes.shape
        tok_emb = self.audio_in(self.token_emb(first_codes))
        bos = self.audio_bos.view(1, 1, 1, -1).expand(b, t, 1, -1)
        shifted = torch.cat([bos, tok_emb[:, :, :-1, :]], dim=2)
        slot = self.frame_pos_emb.view(1, 1, s, -1)
        return shifted + slot + step_h.unsqueeze(2)

    def _decode_audio_frames(self, audio_in: torch.Tensor) -> torch.Tensor:
        b, t, s, h = audio_in.shape
        x = audio_in.view(b * t, s, h)
        attn_mask = self._build_local_attn_mask(s, x.device, min(self.audio_history_window, s))
        for blk in self.audio_blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=None)
        return x.view(b, t, s, h)

    def _predict_logits(self, step_h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        first_codes = codes[..., 0]
        audio_in = self._build_audio_inputs(first_codes, step_h)
        audio_h = self._decode_audio_frames(audio_in)

        first_logits = self.first_codebook_head(audio_h).unsqueeze(3)
        if self.num_codebooks == 1:
            return first_logits

        teacher_first = self.audio_in(self.token_emb(first_codes))
        residual_h = self.residual_cond(torch.cat([audio_h, teacher_first], dim=-1))
        residual_logits = [head(residual_h).unsqueeze(3) for head in self.residual_heads]
        return torch.cat([first_logits] + residual_logits, dim=3)

    def _sample_next_codes(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.distributions.Categorical(probs=probs).sample()
        return logits.argmax(dim=-1)

    def forward_train(
        self,
        codes: torch.Tensor,  # [B, T, S, K]
        note_id: torch.Tensor,  # [B, T]
        phoneme_id: torch.Tensor,  # [B, T]
        slur: torch.Tensor,  # [B, T]
        phone_progress: torch.Tensor,  # [B, T]
        mask: torch.Tensor | None = None,  # [B, T]
    ) -> torch.Tensor:
        cond = self.cond_enc(note_id, phoneme_id, slur, phone_progress)
        prev_repr = self._shift_right_step_repr(self._encode_step(codes[..., 0]))
        step_h = self._contextualize_steps(cond, prev_repr, valid_mask=mask)
        return self._predict_logits(step_h, codes)

    @torch.no_grad()
    def generate(
        self,
        note_id: torch.Tensor,  # [B, T]
        phoneme_id: torch.Tensor,  # [B, T]
        slur: torch.Tensor,  # [B, T]
        phone_progress: torch.Tensor,  # [B, T]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        out_steps = list(
            self.generate_stream(
                note_id=note_id,
                phoneme_id=phoneme_id,
                slur=slur,
                phone_progress=phone_progress,
                temperature=temperature,
            )
        )
        return torch.stack(out_steps, dim=1)

    @torch.no_grad()
    def generate_stream(
        self,
        note_id: torch.Tensor,  # [B, T]
        phoneme_id: torch.Tensor,  # [B, T]
        slur: torch.Tensor,  # [B, T]
        phone_progress: torch.Tensor,  # [B, T]
        temperature: float = 1.0,
    ):
        """
        Step-wise streaming generation.

        Yields one predicted step at a time with shape [B, S, K].
        """
        b, t = phoneme_id.shape
        cond = self.cond_enc(note_id, phoneme_id, slur, phone_progress)
        prev_repr = self.bos_step.view(1, -1).expand(b, -1)
        hist_cond: list[torch.Tensor] = []
        hist_prev: list[torch.Tensor] = []

        for i in range(t):
            hist_cond.append(cond[:, i])
            hist_prev.append(prev_repr)
            if len(hist_cond) > self.history_window:
                hist_cond = hist_cond[-self.history_window :]
                hist_prev = hist_prev[-self.history_window :]

            cond_window = torch.stack(hist_cond, dim=1)
            prev_window = torch.stack(hist_prev, dim=1)
            step_h = self._contextualize_steps(cond_window, prev_window)
            next_codes = self._generate_step_tokens(step_h[:, -1], temperature=temperature)
            yield next_codes
            prev_repr = self._encode_step(next_codes[..., 0].unsqueeze(1)).squeeze(1)

    def _generate_step_tokens(self, step_h: torch.Tensor, temperature: float) -> torch.Tensor:
        b = step_h.size(0)
        first_generated = []
        step_outputs = []
        for pos in range(self.audio_seq_len):
            if first_generated:
                prev_ids = torch.stack(first_generated, dim=1)
                tok_emb = self.audio_in(self.token_emb(prev_ids))
                bos = self.audio_bos.view(1, 1, -1).expand(b, 1, -1)
                x = torch.cat([bos, tok_emb], dim=1)
            else:
                x = self.audio_bos.view(1, 1, -1).expand(b, 1, -1)
            x = x + self.frame_pos_emb[: pos + 1].unsqueeze(0) + step_h.unsqueeze(1)
            attn_mask = self._build_local_attn_mask(pos + 1, x.device, min(self.audio_history_window, pos + 1))
            for blk in self.audio_blocks:
                x = blk(x, attn_mask=attn_mask, key_padding_mask=None)
            frame_h = x[:, -1]
            first_logits = self.first_codebook_head(frame_h)
            next_first = self._sample_next_codes(first_logits, temperature=temperature)
            first_generated.append(next_first)

            if self.num_codebooks == 1:
                frame_codes = next_first.unsqueeze(1)
            else:
                first_cond = self.audio_in(self.token_emb(next_first))
                residual_h = self.residual_cond(torch.cat([frame_h, first_cond], dim=-1))
                residual_codes = [
                    self._sample_next_codes(head(residual_h), temperature=temperature)
                    for head in self.residual_heads
                ]
                frame_codes = torch.cat(
                    [next_first.unsqueeze(1)] + [code.unsqueeze(1) for code in residual_codes],
                    dim=1,
                )
            step_outputs.append(frame_codes)
        return torch.stack(step_outputs, dim=1)
