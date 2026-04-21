from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(rms + self.eps)
        return (x_norm.type_as(x)) * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class GRN(nn.Module):
    """
    Global Response Normalization, same core form used in ConvNeXtV2-style blocks.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block1D(nn.Module):
    """
    Lightweight 1D ConvNeXtV2-style block for condition sequence encoding.
    Input/Output: [B, T, H]
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


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
    class RotaryEmbedding(nn.Module):
        def __init__(self, head_dim: int, base: float = 10000.0):
            super().__init__()
            if head_dim % 2 != 0:
                raise ValueError(f"RoPE head_dim must be even, got {head_dim}.")
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.head_dim = int(head_dim)

        def cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
            # position_ids: [L]
            freqs = torch.outer(position_ids.float(), self.inv_freq)  # [L, D/2]
            emb = torch.cat([freqs, freqs], dim=-1)  # [L, D]
            cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1,1,L,D]
            sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1,1,L,D]
            return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.size(-1) // 2]
        x2 = x[..., x.size(-1) // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.rope.cos_sin(position_ids=position_ids, dtype=q.dtype)
        q_out = (q * cos) + (self._rotate_half(q) * sin)
        k_out = (k * cos) + (self._rotate_half(k) * sin)
        return q_out, k_out

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope_base: float = 10000.0,
        ffn_mult: float = 8.0 / 3.0,
        attn_bias: bool = False,
        ffn_bias: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.q_proj = nn.Linear(dim, dim, bias=attn_bias)
        self.k_proj = nn.Linear(dim, dim, bias=attn_bias)
        self.v_proj = nn.Linear(dim, dim, bias=attn_bias)
        self.o_proj = nn.Linear(dim, dim, bias=attn_bias)
        self.rope = self.RotaryEmbedding(head_dim=self.head_dim, base=rope_base)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        hidden = int(dim * ffn_mult)
        self.ffn = SwiGLUFeedForward(dim, hidden_dim=hidden, bias=ffn_bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        b, l, _ = h.shape
        q = self.q_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,D]
        k = self.k_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(l, device=h.device, dtype=torch.long)
        q, k = self._apply_rope(q=q, k=k, position_ids=position_ids)

        attn_bias = torch.zeros((l, l), device=h.device, dtype=h.dtype)
        attn_bias = attn_bias.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :]  # [B,1,1,L], True means disallow
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0) + torch.zeros(
                (b, 1, 1, l), device=h.device, dtype=h.dtype
            ).masked_fill(kpm, float("-inf"))
        else:
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, self.dim)
        attn_out = self.o_proj(attn_out)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_incremental(
        self,
        x_new: torch.Tensor,  # [B, 1, D]
        cache_k: torch.Tensor | None,  # [B, H, Lc, Dh]
        cache_v: torch.Tensor | None,  # [B, H, Lc, Dh]
        position_id: torch.Tensor,  # [1]
        window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.norm1(x_new)
        b, l, _ = h.shape
        if l != 1:
            raise ValueError(f"forward_incremental expects one token at a time, got shape {tuple(h.shape)}")

        q = self.q_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self._apply_rope(q=q, k=k, position_ids=position_id)

        if cache_k is None:
            k_cat = k
            v_cat = v
        else:
            k_cat = torch.cat([cache_k, k], dim=2)
            v_cat = torch.cat([cache_v, v], dim=2)

        if k_cat.size(2) > window_size:
            k_cat = k_cat[:, :, -window_size:, :]
            v_cat = v_cat[:, :, -window_size:, :]

        attn_out = F.scaled_dot_product_attention(
            q,
            k_cat,
            v_cat,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, self.dim)
        attn_out = self.o_proj(attn_out)

        x = x_new + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, k_cat, v_cat


class StreamingSVSModel(nn.Module):
    """
    Single-stream decoder-only model.
    Sequence layout per step:
      [CONTROL] [FRAME_0] [FRAME_1] ... [FRAME_(S-1)]
    where each frame slot predicts K codebooks in parallel with multi-head outputs.
    """

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
        prev_step_dim: int,  # kept for config compatibility, unused by single-stream architecture
        model_dim: int,
        attn_heads: int,
        attn_layers: int,
        history_window: int,
        num_blocks: int,
        audio_history_window: int = 64,  # kept for config compatibility
        control_encoder_type: str = "mlp",
        control_conv_layers: int = 0,
        control_conv_hidden_mult: float = 2.0,
        control_conv_kernel_size: int = 7,
        control_conv_dilation: int = 1,
        prefix_pad_steps: int = 0,
        rope_base: float = 10000.0,
        ffn_mult: float = 8.0 / 3.0,
        attn_bias: bool = False,
        ffn_bias: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        del prev_step_dim, audio_history_window

        self.num_codebooks = int(num_codebooks)
        self.tokens_per_step = int(tokens_per_step)
        self.codebook_size = int(codebook_size)
        # One frame-slot per step position; each frame predicts K codebooks in parallel.
        self.audio_slots_per_step = self.tokens_per_step
        self.seq_len_per_step = 1 + self.audio_slots_per_step
        self.token_history_window = max(1, int(history_window)) * self.seq_len_per_step
        self.prefix_pad_steps = max(0, int(prefix_pad_steps))

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
        self.codebook_token_emb = nn.ModuleList(
            [nn.Embedding(self.codebook_size, token_emb_dim) for _ in range(self.num_codebooks)]
        )
        self.audio_in = nn.Linear(token_emb_dim, model_dim)
        self.control_encoder_type = str(control_encoder_type).lower().strip()
        self.control_conv_layers = int(control_conv_layers)
        if self.control_encoder_type not in {"mlp", "convnext"}:
            raise ValueError(
                f"Unsupported control_encoder_type={control_encoder_type!r}, expected 'mlp' or 'convnext'."
            )
        if self.control_encoder_type == "mlp":
            self.control_in = nn.Sequential(
                nn.Linear(cond_dim, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )
            self.control_preflow = None
        else:
            # Keep projection simple; use optional lightweight temporal conv blocks for control encoding.
            self.control_in = nn.Linear(cond_dim, model_dim)
            hidden = max(model_dim, int(model_dim * float(control_conv_hidden_mult)))
            self.control_preflow = nn.ModuleList(
                [
                    ConvNeXtV2Block1D(
                        dim=model_dim,
                        intermediate_dim=hidden,
                        kernel_size=int(control_conv_kernel_size),
                        dilation=int(control_conv_dilation),
                    )
                    for _ in range(self.control_conv_layers)
                ]
            )

        self.audio_bos_id = 0
        self.token_type_emb = nn.Embedding(2, model_dim)  # 0=control, 1=audio
        self.within_step_pos_emb = nn.Parameter(torch.randn(self.seq_len_per_step, model_dim) * 0.02)
        # Learned virtual-step paddings used by ablation `prefix_pad_steps`.
        self.prefix_control_pad = nn.Parameter(torch.zeros(model_dim))
        self.prefix_frame_pad = nn.Parameter(torch.zeros(model_dim))

        total_layers = int(attn_layers) + int(num_blocks)
        self.blocks = nn.ModuleList(
            [
                LocalCausalSelfAttentionBlock(
                    model_dim,
                    num_heads=attn_heads,
                    rope_base=rope_base,
                    ffn_mult=ffn_mult,
                    attn_bias=attn_bias,
                    ffn_bias=ffn_bias,
                    norm_eps=norm_eps,
                )
                for _ in range(total_layers)
            ]
        )
        self.out_norm = RMSNorm(model_dim, eps=norm_eps)
        self.audio_heads = nn.ModuleList(
            [nn.Linear(model_dim, self.codebook_size) for _ in range(self.num_codebooks)]
        )

    def _encode_control(self, cond: torch.Tensor) -> torch.Tensor:
        x = self.control_in(cond)
        if self.control_encoder_type == "mlp":
            return x
        for blk in self.control_preflow:
            x = blk(x)
        return x

    def _build_local_attn_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        dist = idx[:, None] - idx[None, :]
        return (dist < 0) | (dist >= self.token_history_window)

    def _embed_frame_tokens(self, frame_codes: torch.Tensor) -> torch.Tensor:
        # frame_codes: [B, T, S, K]
        emb_sum = 0.0
        for k in range(self.num_codebooks):
            emb_sum = emb_sum + self.codebook_token_emb[k](frame_codes[..., k])
        return emb_sum

    def _prepare_sequence_inputs(
        self,
        cond: torch.Tensor,  # [B, T, C]
        codes: torch.Tensor,  # [B, T, S, K]
        step_mask: torch.Tensor | None,  # [B, T]
        frame_valid_flat: torch.Tensor | None = None,  # [B, T*S]
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, s, k = codes.shape
        if s != self.tokens_per_step or k != self.num_codebooks:
            raise ValueError(
                f"codes shape mismatch, expected [B,T,{self.tokens_per_step},{self.num_codebooks}], got {tuple(codes.shape)}"
            )

        # Shift-right on frame axis (global over T*S); each frame input is previous frame codes.
        frame_targets_flat = codes.view(b, t * s, k)
        bos_frame = torch.full((b, 1, k), fill_value=self.audio_bos_id, dtype=torch.long, device=codes.device)
        frame_inputs_flat = torch.cat([bos_frame, frame_targets_flat[:, :-1, :]], dim=1)
        frame_inputs = frame_inputs_flat.view(b, t, s, k)

        t_eff = t + self.prefix_pad_steps
        seq_len = t_eff * self.seq_len_per_step
        seq_in = torch.zeros(b, seq_len, self.audio_heads[0].in_features, device=codes.device, dtype=cond.dtype)
        valid_pos = torch.zeros(b, seq_len, device=codes.device, dtype=torch.bool)

        control_tok = self._encode_control(cond)
        frame_tok = self.audio_in(self._embed_frame_tokens(frame_inputs))

        step_ids = torch.arange(t, device=codes.device) + self.prefix_pad_steps
        step_start = step_ids * self.seq_len_per_step

        control_idx = step_start
        seq_in[:, control_idx, :] = control_tok
        valid_control = torch.ones((b, t), device=codes.device, dtype=torch.bool)

        step_offsets = torch.arange(self.audio_slots_per_step, device=codes.device)
        frame_idx = step_start.unsqueeze(1) + 1 + step_offsets.unsqueeze(0)  # [T, S]
        frame_idx_flat = frame_idx.reshape(-1)

        seq_in[:, frame_idx_flat, :] = frame_tok.reshape(b, t * self.audio_slots_per_step, -1)

        if step_mask is None:
            valid_step = torch.ones((b, t), device=codes.device, dtype=torch.bool)
        else:
            valid_step = step_mask.bool()
        if frame_valid_flat is None:
            valid_audio = valid_step.unsqueeze(-1).expand(b, t, self.audio_slots_per_step).reshape(
                b, t * self.audio_slots_per_step
            )
        else:
            valid_audio = frame_valid_flat.bool()
        valid_control = valid_control & valid_step

        valid_pos[:, control_idx] = valid_control
        valid_pos[:, frame_idx_flat] = valid_audio

        # Optional virtual left-padding steps for ablation:
        # prepend learned dummy steps so the first real step has synthetic history.
        if self.prefix_pad_steps > 0:
            pad_step_ids = torch.arange(self.prefix_pad_steps, device=codes.device)
            pad_step_start = pad_step_ids * self.seq_len_per_step
            pad_control_idx = pad_step_start
            pad_offsets = torch.arange(self.audio_slots_per_step, device=codes.device)
            pad_frame_idx = pad_step_start.unsqueeze(1) + 1 + pad_offsets.unsqueeze(0)
            pad_frame_idx_flat = pad_frame_idx.reshape(-1)

            seq_in[:, pad_control_idx, :] = self.prefix_control_pad.to(seq_in.dtype).view(1, 1, -1)
            seq_in[:, pad_frame_idx_flat, :] = self.prefix_frame_pad.to(seq_in.dtype).view(1, 1, -1)
            valid_pos[:, pad_control_idx] = True
            valid_pos[:, pad_frame_idx_flat] = True

        type_ids = torch.zeros(seq_len, device=codes.device, dtype=torch.long)
        all_step_start = torch.arange(t_eff, device=codes.device) * self.seq_len_per_step
        all_offsets = torch.arange(self.audio_slots_per_step, device=codes.device)
        all_frame_idx = all_step_start.unsqueeze(1) + 1 + all_offsets.unsqueeze(0)
        type_ids[all_frame_idx.reshape(-1)] = 1
        seq_in = seq_in + self.token_type_emb(type_ids).unsqueeze(0)

        within_step_idx = torch.arange(self.seq_len_per_step, device=codes.device).repeat(t_eff)
        seq_in = seq_in + self.within_step_pos_emb[within_step_idx].unsqueeze(0)

        position_ids = torch.arange(seq_len, device=codes.device, dtype=torch.long) + int(position_offset)

        return seq_in, valid_pos, position_ids

    def _forward_sequence(
        self,
        seq_in: torch.Tensor,
        valid_pos: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = self._build_local_attn_mask(seq_in.size(1), seq_in.device)
        key_padding_mask = ~valid_pos
        x = seq_in
        for blk in self.blocks:
            x = blk(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                position_ids=position_ids,
            )
        return self.out_norm(x)

    def _forward_one_token_incremental(
        self,
        x_new: torch.Tensor,  # [B,1,H]
        position_id: int,
        kv_cache: list[tuple[torch.Tensor | None, torch.Tensor | None]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor | None, torch.Tensor | None]]]:
        x = x_new
        pos = torch.tensor([int(position_id)], device=x.device, dtype=torch.long)
        new_cache: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
        for layer_idx, blk in enumerate(self.blocks):
            cache_k, cache_v = kv_cache[layer_idx]
            x, next_k, next_v = blk.forward_incremental(
                x_new=x,
                cache_k=cache_k,
                cache_v=cache_v,
                position_id=pos,
                window_size=self.token_history_window,
            )
            new_cache.append((next_k, next_v))
        return self.out_norm(x), new_cache

    def _extract_audio_logits(self, hidden: torch.Tensor, t: int) -> torch.Tensor:
        step_ids = torch.arange(t, device=hidden.device) + self.prefix_pad_steps
        step_start = step_ids * self.seq_len_per_step
        step_offsets = torch.arange(self.audio_slots_per_step, device=hidden.device)
        audio_idx = step_start.unsqueeze(1) + 1 + step_offsets.unsqueeze(0)
        audio_idx_flat = audio_idx.reshape(-1)
        frame_hidden_flat = hidden[:, audio_idx_flat, :]
        frame_hidden = frame_hidden_flat.view(hidden.size(0), t, self.tokens_per_step, hidden.size(-1))
        logits_per_codebook = [head(frame_hidden) for head in self.audio_heads]  # K x [B,T,S,V]
        return torch.stack(logits_per_codebook, dim=3)  # [B,T,S,K,V]

    def _sample_next(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
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
        seq_in, valid_pos, position_ids = self._prepare_sequence_inputs(
            cond=cond,
            codes=codes,
            step_mask=mask,
            position_offset=0,
        )
        hidden = self._forward_sequence(seq_in=seq_in, valid_pos=valid_pos, position_ids=position_ids)
        return self._extract_audio_logits(hidden, t=codes.size(1))

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
        b, t = note_id.shape
        cond = self.cond_enc(note_id, phoneme_id, slur, phone_progress)
        kv_cache: list[tuple[torch.Tensor | None, torch.Tensor | None]] = [
            (None, None) for _ in range(len(self.blocks))
        ]
        pos = 0
        prev_frame_codes = torch.full(
            (b, self.num_codebooks),
            fill_value=self.audio_bos_id,
            dtype=torch.long,
            device=note_id.device,
        )

        # Optional virtual left-padding steps (ablation) are injected into cache first.
        if self.prefix_pad_steps > 0:
            pad_control = self.prefix_control_pad.to(cond.dtype).view(1, 1, -1).expand(b, 1, -1)
            pad_frame = self.prefix_frame_pad.to(cond.dtype).view(1, 1, -1).expand(b, 1, -1)
            for _ in range(self.prefix_pad_steps):
                control_in = pad_control + self.token_type_emb.weight[0].view(1, 1, -1) + self.within_step_pos_emb[
                    0
                ].to(cond.dtype).view(1, 1, -1)
                _, kv_cache = self._forward_one_token_incremental(control_in, pos, kv_cache)
                pos += 1
                for frame_slot in range(self.audio_slots_per_step):
                    frame_in = pad_frame + self.token_type_emb.weight[1].view(1, 1, -1) + self.within_step_pos_emb[
                        1 + frame_slot
                    ].to(cond.dtype).view(1, 1, -1)
                    _, kv_cache = self._forward_one_token_incremental(frame_in, pos, kv_cache)
                    pos += 1

        for step_idx in range(t):
            step_tokens = []
            control_tok = self._encode_control(cond[:, step_idx : step_idx + 1, :])
            control_in = control_tok + self.token_type_emb.weight[0].view(1, 1, -1) + self.within_step_pos_emb[
                0
            ].to(cond.dtype).view(1, 1, -1)
            _, kv_cache = self._forward_one_token_incremental(control_in, pos, kv_cache)
            pos += 1

            for frame_idx in range(self.audio_slots_per_step):
                frame_codes_in = prev_frame_codes.view(b, 1, 1, self.num_codebooks)
                frame_tok = self.audio_in(self._embed_frame_tokens(frame_codes_in)).view(b, 1, -1)
                frame_in = frame_tok + self.token_type_emb.weight[1].view(1, 1, -1) + self.within_step_pos_emb[
                    1 + frame_idx
                ].to(cond.dtype).view(1, 1, -1)
                frame_hidden, kv_cache = self._forward_one_token_incremental(frame_in, pos, kv_cache)
                pos += 1

                frame_hidden_flat = frame_hidden[:, 0, :]  # [B,H]
                logits_per_codebook = [head(frame_hidden_flat) for head in self.audio_heads]
                next_logits = torch.stack(logits_per_codebook, dim=1)  # [B,K,V]
                next_tokens = self._sample_next(next_logits, temperature=temperature).long()  # [B,K]
                prev_frame_codes = next_tokens
                step_tokens.append(next_tokens)

            yield torch.stack(step_tokens, dim=1)  # [B,S,K]
