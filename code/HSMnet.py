"""
LWCSS: Light-Weight Continuous Speech Separation
=================================================
Hierarchical Streaming Memory (HSM) + IASFC Encoder + FFIBs + EDA

Two-Stage Plan A pipeline:
  Stage 1: IASFC Encoder → FFIBs → HSM compress & fuse → EDA attractors
  Stage 2: Attractors → FiLM condition → FFIBs → IASFC Decoder → waveforms

Architecture:
  ┌──────────── Stage 1: Local Extraction ────────────┐
  │ IASFC Encoder → FFIBs → HSM compress & fuse      │
  │   → EDA attractor generation                     │
  └────────────────────────────────────────────────────┘
                         ↓
  ┌────────── Stage 2: Speaker-Conditioned ───────────┐
  │ FiLM condition FFIB features → IASFC Decoder      │
  └────────────────────────────────────────────────────┘

Key references:
  - IASFC: Saijo & Bando (2026 TASLP) SFC-Mamba
  - FFIBs: Li et al. (2025 ICLR) TIGER
  - HSM: Emformer (Shi 2021) + Papez AWM (Oh 2023) + Mamba3 compression
  - EDA: SepTDA (Lee 2024) TDA + A-DCSS (Wang 2025) null attractor
  - Mamba3: Lahoti et al. (2026 arXiv:2603.15569)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .tiger import (
    ConvNorm, ConvNormAct, GlobLN, DilatedConvNorm,
    UConvBlock, MultiHeadSelfAttention2D, Mlp, InjectionMultiSum,
)
from ..layers.normalizations import LayerNormalization4D

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("[LWCSS] WARNING: mamba_ssm not installed. HSM compression will be disabled.")


# ═══════════════════════════════════════════════════════════════
# Utility Layers (from SFC-Mamba: enc_dec_base.py)
# ═══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMS Normalization (Saijo & Bando 2026, SFC-Mamba)."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = (x * x).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP for decoder adaptive query generation (Saijo & Bando 2026)."""
    def __init__(self, d_model, d_inner, d_input=None):
        super().__init__()
        self.linear1 = nn.Linear(d_model if d_input is None else d_input, d_inner * 2)
        self.linear2 = nn.Linear(d_inner, d_model)

    def forward(self, x):
        a, b = self.linear1(x).chunk(2, dim=-1)
        b = F.silu(b)
        return self.linear2(a * b)


# ═══════════════════════════════════════════════════════════════
# Mamba3 Building Block
# ═══════════════════════════════════════════════════════════════

class Mamba3Block(nn.Module):
    """Mamba3 block (Lahoti et al., 2026 arXiv:2603.15569).

    Improvements over Mamba-2:
    - Exponential-trapezoidal discretization (3-term recurrence)
    - Data-dependent RoPE from complex-valued SSM
    - BCNorm (RMSNorm on B,C + learnable head bias)

    SISO mode (is_mimo=False) used throughout: short sequences in LWCSS
    (F+K≈321 for SFC; B+M≈80 for HSM) do not benefit from MIMO's
    increased arithmetic intensity.
    """
    def __init__(self, d_model, use_norm=True, use_res=True,
                 d_state=128, expand=2, headdim=64, ngroups=1,
                 rope_fraction=0.5, mimo_rank=4, is_mimo=False,
                 chunk_size=64):
        super().__init__()
        self.use_norm = use_norm
        self.use_res = use_res
        if use_norm:
            self.norm = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        if HAS_MAMBA:
            try:
                from mamba_ssm.modules.mamba3 import Mamba3
                self.mixer = Mamba3(
                    d_model=d_model, d_state=d_state, expand=expand,
                    headdim=headdim, ngroups=ngroups,
                    rope_fraction=rope_fraction, mimo_rank=mimo_rank,
                    is_mimo=is_mimo, chunk_size=chunk_size,
                )
                self._is_mamba3 = True
            except ImportError:
                from mamba_ssm import Mamba
                self.mixer = Mamba(
                    d_model=d_model, d_state=d_state, d_conv=4, expand=expand
                )
                self._is_mamba3 = False
        else:
            self.mixer = nn.GRU(
                d_model, d_model, num_layers=1,
                batch_first=True, bidirectional=True
            )
            self.mixer_proj = nn.Linear(2 * d_model, d_model)
            self._is_mamba3 = False

    def forward(self, hidden_states):
        residual = hidden_states
        if self.use_norm:
            hidden_states = self.norm(hidden_states)
        if HAS_MAMBA:
            hidden_states = self.mixer(hidden_states)
        else:
            hidden_states, _ = self.mixer(hidden_states)
            hidden_states = self.mixer_proj(hidden_states)
        if self.use_res:
            hidden_states = residual + hidden_states
        return hidden_states


# ═══════════════════════════════════════════════════════════════
# SFC-Mamba Encoder/Decoder (faithful to Saijo & Bando 2026)
# Modifications: (1) speech band split, (2) Mamba3 upgrade
# ═══════════════════════════════════════════════════════════════

class SFCEncoder(nn.Module):
    """SFC-Mamba Encoder: spectral feature compression.

    Faithfully follows Saijo & Bando (2026 TASLP) with two modifications:
    1. Speech-adaptive non-uniform band split
    2. Mamba3 replaces Mamba1

    Design: Conv2d(3×3)+RMSNorm → adaptive query → bidir Mamba3(use_res=False)
            → Conv2d(3×3)+RMSNorm → returns (query_out, emb)
    """
    def __init__(self, n_freq, n_bands, d_inner, d_model, band_indices,
                 d_state=8, expand=1):
        super().__init__()
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.d_inner = d_inner
        self.d_model = d_model
        self.band_indices = band_indices

        # Input Conv2d(3×3) + RMSNorm
        self.input_conv = nn.Sequential(
            nn.Conv2d(2, d_inner, kernel_size=(3, 3), padding=(1, 1)),
            _RMSNorm4D(d_inner),
        )
        # Output Conv2d(3×3) + RMSNorm (bidirectional → d_model)
        self.output_conv = nn.Sequential(
            nn.Conv2d(d_inner * 2, d_model, kernel_size=(3, 3), padding=(1, 1)),
            _RMSNorm4D(d_model),
        )

        # Adaptive query: vectorized weighted mean
        widths = torch.tensor([e - s for s, e in band_indices], dtype=torch.long)
        self.register_buffer("widths", widths, persistent=False)
        flat_idx_list, band_ids_list = [], []
        for b, (s, e) in enumerate(band_indices):
            if e > s:
                flat_idx_list.append(torch.arange(s, e, dtype=torch.long))
                band_ids_list.append(torch.full((e - s,), b, dtype=torch.long))
        self.register_buffer("flat_idx", torch.cat(flat_idx_list), persistent=False)
        self.register_buffer("band_ids", torch.cat(band_ids_list), persistent=False)
        self.freq_weights = nn.Parameter(
            torch.cat([torch.ones(w, dtype=torch.float32) / max(int(w), 1)
                       for w in widths.tolist()])
        )

        # Band-middle interleaving indices
        self.emb_indices, self.query_indices = _prepare_query_indices(band_indices)

        # Bidirectional Mamba3 (use_res=False, original SFC)
        self.forward_block = Mamba3Block(d_inner, d_state=d_state * 8, expand=expand,
                                         is_mimo=False, use_res=False)
        self.backward_block = Mamba3Block(d_inner, d_state=d_state * 8, expand=expand,
                                          is_mimo=False, use_res=False)

    def forward(self, x):
        """Args: x: (B, 2, T, F). Returns: (B, d_model, T, K), (B*T, F, d_inner*2)"""
        B, _, T, F = x.shape
        emb_4d = self.input_conv(x)
        emb = emb_4d.permute(0, 2, 3, 1).reshape(B * T, F, self.d_inner)
        query = self._compute_adaptive_query(emb)
        query_fwd, emb_fwd = self._process_mamba(self.forward_block, emb, query, True)
        query_bwd, emb_bwd = self._process_mamba(self.backward_block, emb, query, False)
        query_cat = torch.cat([query_fwd, query_bwd], dim=-1)
        emb_cat = torch.cat([emb_fwd, emb_bwd], dim=-1)
        query_4d = query_cat.reshape(B, T, self.n_bands, -1).permute(0, 3, 1, 2)
        query_out = self.output_conv(query_4d)
        return query_out, emb_cat

    def _compute_adaptive_query(self, emb):
        BxT, S, H = emb.shape
        K = int(self.widths.numel())
        flat = emb.index_select(1, self.flat_idx)
        w = self.freq_weights.to(dtype=emb.dtype)
        weighted = flat * w[None, :, None]
        out = emb.new_zeros(BxT, K, H)
        out.index_add_(1, self.band_ids, weighted)
        denom = emb.new_zeros(K)
        denom.index_add_(0, self.band_ids, w)
        return out / denom[None, :, None].clamp_min(1e-8)

    def _process_mamba(self, block, emb, query, forward=True):
        BxT = emb.shape[0]
        n_total = self.n_freq + self.n_bands
        combined = torch.empty(BxT, n_total, self.d_inner, dtype=emb.dtype, device=emb.device)
        combined[:, self.emb_indices] = emb
        combined[:, self.query_indices] = query
        if not forward:
            combined = combined.flip([1])
        combined = block(combined)
        if not forward:
            combined = combined.flip([1])
        return combined[:, self.query_indices], combined[:, self.emb_indices].contiguous()


class SFCDecoder(nn.Module):
    """SFC-Mamba Decoder with adaptive queries from encoder embeddings."""
    def __init__(self, n_freq, n_bands, d_inner, d_model, band_indices,
                 d_state=8, expand=1):
        super().__init__()
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.d_inner = d_inner
        self.d_model = d_model
        self.input_conv = nn.Sequential(
            nn.Conv2d(d_model, d_inner, kernel_size=(3, 3), padding=(1, 1)),
            _RMSNorm4D(d_inner),
        )
        self.query_mlp = nn.Sequential(
            RMSNorm(d_inner * 2),
            SwiGLUMLP(d_inner, d_inner * 2, d_input=d_inner * 2),
        )
        enc_emb_idx, enc_query_idx = _prepare_query_indices(band_indices)
        self.query_indices = enc_emb_idx
        self.emb_indices = enc_query_idx
        self.forward_block = Mamba3Block(d_inner, d_state=d_state * 8, expand=expand,
                                         is_mimo=False, use_res=False)
        self.backward_block = Mamba3Block(d_inner, d_state=d_state * 8, expand=expand,
                                          is_mimo=False, use_res=False)
        self.output_conv = nn.Conv2d(d_inner * 2, 2, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, z, encoder_emb):
        """Args: z: (B, d_model, T, K), encoder_emb: (B*T, F, d_inner*2). Returns: (B, 2, T, F)"""
        B, _, T, K = z.shape
        z_proj = self.input_conv(z)
        emb = z_proj.permute(0, 2, 3, 1).reshape(B * T, K, self.d_inner)
        query = self.query_mlp(encoder_emb)
        query_fwd, _ = self._process_mamba(self.forward_block, emb, query, True)
        query_bwd, _ = self._process_mamba(self.backward_block, emb, query, False)
        query_cat = torch.cat([query_fwd, query_bwd], dim=-1)
        query_4d = query_cat.reshape(B, T, self.n_freq, -1).permute(0, 3, 1, 2)
        return self.output_conv(query_4d)

    def _process_mamba(self, block, emb, query, forward=True):
        BxT = emb.shape[0]
        n_total = self.n_freq + self.n_bands
        combined = torch.empty(BxT, n_total, self.d_inner, dtype=emb.dtype, device=emb.device)
        combined[:, self.emb_indices] = emb
        combined[:, self.query_indices] = query
        if not forward:
            combined = combined.flip([1])
        combined = block(combined)
        if not forward:
            combined = combined.flip([1])
        return combined[:, self.query_indices], combined[:, self.emb_indices]


class _RMSNorm4D(nn.Module):
    """RMSNorm applied to 4D tensor (B, C, T, F) along channel dim."""
    def __init__(self, n_channels):
        super().__init__()
        self.norm = RMSNorm(n_channels)

    def forward(self, x):
        B, C, T, F = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def _prepare_query_indices(band_indices):
    """Band middle strategy: insert query at band center (original SFC)."""
    seq_len = band_indices[-1][1]
    centers = [(b[0] + b[1]) // 2 for b in band_indices]
    centers = torch.tensor(centers, dtype=torch.long)
    query_indices = centers + torch.arange(len(band_indices))
    is_query = torch.zeros(seq_len + len(band_indices), dtype=torch.bool)
    is_query[query_indices] = True
    emb_indices = torch.where(~is_query)[0]
    return emb_indices, query_indices


# ═══════════════════════════════════════════════════════════════
# FFIBs — Frequency-Frame Interleaved Blocks (TIGER, Li 2025)
# No extra post-iteration F3A (回复2: 删除)
# ═══════════════════════════════════════════════════════════════

class FFIB(nn.Module):
    """Single Frequency-Frame Interleaved Block (TIGER faithful)."""
    def __init__(self, out_channels=128, in_channels=512,
                 upsampling_depth=4, n_head=4, att_hid_chan=4):
        super().__init__()
        self.freq_path = nn.ModuleList([
            UConvBlock(out_channels, in_channels, upsampling_depth),
            MultiHeadSelfAttention2D(
                out_channels, 1, n_head=n_head, hid_chan=att_hid_chan,
                act_type="prelu", norm_type="LayerNormalization4D", dim=4
            ),
            LayerNormalization4D((out_channels, 1)),
        ])
        self.frame_path = nn.ModuleList([
            UConvBlock(out_channels, in_channels, upsampling_depth),
            MultiHeadSelfAttention2D(
                out_channels, 1, n_head=n_head, hid_chan=att_hid_chan,
                act_type="prelu", norm_type="LayerNormalization4D", dim=4
            ),
            LayerNormalization4D((out_channels, 1)),
        ])
        self.residual_1 = nn.Conv2d(out_channels, out_channels, 1)
        self.residual_2 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        """x: (B, N, nband, T) → (B, N, nband, T)"""
        B, N, nband, T = x.shape
        # Frequency Path
        residual_1 = x
        x_f = x.permute(0, 3, 1, 2).reshape(B * T, N, nband)
        freq_fea = self.freq_path[0](x_f)
        freq_fea = freq_fea.view(B, T, N, nband).permute(0, 2, 1, 3)
        freq_fea = self.freq_path[1](freq_fea)
        freq_fea = self.freq_path[2](freq_fea)
        freq_fea = freq_fea.permute(0, 1, 3, 2)
        x = self.residual_1(freq_fea) + residual_1
        # Frame Path
        residual_2 = x
        x_t = x.permute(0, 2, 1, 3).reshape(B * nband, N, T)
        frame_fea = self.frame_path[0](x_t)
        frame_fea = frame_fea.view(B, nband, N, T).permute(0, 2, 3, 1)
        frame_fea = self.frame_path[1](frame_fea)
        frame_fea = self.frame_path[2](frame_fea)
        frame_fea = frame_fea.permute(0, 1, 3, 2)
        x = self.residual_2(frame_fea) + residual_2
        return x


class FFIBStack(nn.Module):
    """Parameter-shared FFIB stack (no extra F3A — removed per user instruction)."""
    def __init__(self, out_channels=128, in_channels=512,
                 upsampling_depth=4, n_head=4, att_hid_chan=4, n_iter=4):
        super().__init__()
        self.n_iter = n_iter
        self.block = FFIB(out_channels, in_channels, upsampling_depth, n_head, att_hid_chan)
        self.concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, groups=out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        """x: (B, N, nband, T) → (B, N, nband, T)"""
        mixture = x.clone()
        for i in range(self.n_iter):
            if i == 0:
                x = self.block(x)
            else:
                x = self.block(self.concat_block(mixture + x))
        return x


# ═══════════════════════════════════════════════════════════════
# HSM — Hierarchical Streaming Memory (Core Innovation)
# 方案1B: Attention-based fusion + M=16 + [memory; chunk] order
# ═══════════════════════════════════════════════════════════════

class HierarchicalStreamingMemory(nn.Module):
    """Three-tier HSM with attention-based memory-chunk fusion.

    Bottom: FFIBStack local extraction → H_c
    Middle: Unidirectional Mamba3([G_{c-1}; S_{c-1}]) → G̃_c
            EMA: G_c = α·G̃_c + (1-α)·G_{c-1}
    Top: Per-band attention fusion (方案1B from HSM-Emformer report):
         attn = softmax(H_c @ G_c^T)  → (B, nband, M)
         context = attn @ G_c         → (B, nband, D)
         S_c = (1+δ_γ(context)) * H_pooled + β(context)

    Concat order: [memory; chunk] following Papez AWM design.
    M=16 following Papez ablation (M=16 >> M=4 by 1.3dB).

    Refs: Papez AWM (Oh et al., ICASSP 2023), Emformer (Shi et al., 2021)
    """
    def __init__(self, n_bands, d_model, n_memory_slots=16,
                 d_state=8, expand=1, ema_alpha=0.6):
        super().__init__()
        self.n_bands = n_bands
        self.d_model = d_model
        self.n_memory_slots = n_memory_slots

        # Unidirectional Mamba3 compression
        self.compress_mamba = Mamba3Block(
            d_model, d_state=d_state * 8, expand=expand,
            is_mimo=False, use_res=True
        )

        # Learnable EMA alpha (sigmoid-gated)
        self.ema_alpha = nn.Parameter(torch.tensor(ema_alpha))

        # 方案1B: Per-band attention fusion (replaces FiLM mean-pool)
        # Each band attends to M memory slots independently
        self.attn_proj_q = nn.Linear(d_model, d_model)  # Project H_pooled for query
        self.attn_proj_k = nn.Linear(d_model, d_model)  # Project G_c for key
        self.fusion_gamma = nn.Linear(d_model, d_model)  # context → δ_gamma
        self.fusion_beta = nn.Linear(d_model, d_model)   # context → beta
        # Initialize near identity
        nn.init.zeros_(self.fusion_gamma.bias)
        nn.init.zeros_(self.fusion_gamma.weight)
        nn.init.zeros_(self.fusion_beta.bias)
        nn.init.zeros_(self.fusion_beta.weight)

        # Learnable initial memory G_0
        self.g0 = nn.Parameter(torch.randn(1, n_memory_slots, d_model) * 0.02)

    def forward(self, H_c, G_prev=None, S_prev=None):
        """Process one chunk through HSM.

        Args:
            H_c: (B, n_bands, d_model, T_c) — local FFIB output
            G_prev: (B, M, d_model) — previous global state
            S_prev: (B, n_bands, d_model) — previous fused features

        Returns:
            S_c: (B, n_bands, d_model)
            G_c: (B, M, d_model)
        """
        B, n_bands, d_model, T_c = H_c.shape
        H_pooled = H_c.mean(dim=-1)  # (B, n_bands, d_model)

        if S_prev is None:
            S_prev = torch.zeros(B, n_bands, d_model, device=H_c.device, dtype=H_c.dtype)
        if G_prev is None:
            G_prev = self.g0.expand(B, -1, -1)

        # ── Middle: Mamba3 compression ──
        # Concat order: [memory; chunk] (Papez AWM: memory tokens first)
        seq = torch.cat([G_prev, S_prev], dim=1)  # (B, M+B_bands, d_model)
        compressed = self.compress_mamba(seq)
        G_tilde = compressed[:, :self.n_memory_slots]  # First M slots (memory positions)

        # EMA smoothing
        alpha = torch.sigmoid(self.ema_alpha)
        G_c = alpha * G_tilde + (1 - alpha) * G_prev

        # ── Top: Per-band attention fusion (方案1B) ──
        # Each of B bands attends to M memory slots
        Q = self.attn_proj_q(H_pooled)  # (B, n_bands, D)
        K = self.attn_proj_k(G_c)       # (B, M, D)

        # Attention: (B, n_bands, D) × (B, D, M) → (B, n_bands, M)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_model)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_bands, M)

        # Context: weighted combination of memory slots per band
        context = torch.bmm(attn_weights, G_c)  # (B, n_bands, D)

        # Modulation: gamma=1+δ, beta
        delta_gamma = self.fusion_gamma(context)  # (B, n_bands, D)
        beta = self.fusion_beta(context)          # (B, n_bands, D)

        S_c = (1.0 + delta_gamma) * H_pooled + beta

        return S_c, G_c


# ═══════════════════════════════════════════════════════════════
# EDA — Encoder-Decoder Attractor
# 方案A2 + B + 2层: Time-sampled KV + Masked Self-Attention + 2-Layer Decoder
# ═══════════════════════════════════════════════════════════════

class EncoderDecoderAttractor(nn.Module):
    """2-Layer Transformer Decoder Attractor with time-sampled KV.

    Design (from EDA-Dual-Source report 方案A2+B+2层):
    - KV: S_c bands (B tokens) + time-sampled H_c (T//stride tokens)
      → Total ~74 KV tokens (64 + 10) instead of 2 pooled tokens
    - K+1 learnable speaker queries (last = null attractor)
    - Layer 1: Cross-attention only (DETR practice: skip self-attn in first layer)
    - Layer 2: Masked self-attention + cross-attention
    - Existence head for speaker counting

    Refs: SepTDA (Lee 2024), A-DCSS (Wang 2025), DETR (Carion 2020)
    """
    def __init__(self, d_model, max_speakers=4, n_head=4, d_ff=256,
                 dropout=0.1, time_stride=5):
        super().__init__()
        self.max_speakers = max_speakers
        self.d_model = d_model
        self.time_stride = time_stride
        self.n_queries = max_speakers + 1  # K+1 (including null)

        # K+1 learnable speaker queries
        self.speaker_queries = nn.Parameter(
            torch.randn(self.n_queries, d_model) * 0.02
        )

        # ── Layer 1: Cross-Attention only ──
        self.cross_attn_1 = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm_ca_1 = nn.LayerNorm(d_model)
        self.ffn_1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm_ffn_1 = nn.LayerNorm(d_model)

        # ── Layer 2: Masked Self-Attention + Cross-Attention ──
        self.self_attn_2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm_sa_2 = nn.LayerNorm(d_model)
        self.cross_attn_2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm_ca_2 = nn.LayerNorm(d_model)
        self.ffn_2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm_ffn_2 = nn.LayerNorm(d_model)

        # Causal mask for self-attention (speaker competition)
        self.register_buffer('causal_mask',
            torch.triu(torch.ones(self.n_queries, self.n_queries) * float('-inf'), diagonal=1))

        # Existence head
        self.existence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1), nn.Sigmoid(),
        )

        # FiLM projection (gamma=1+δ for Stage 2 conditioning)
        self.film_proj = nn.Linear(d_model, d_model * 2)
        nn.init.zeros_(self.film_proj.bias)
        nn.init.normal_(self.film_proj.weight, std=0.01)

    def forward(self, S_c, H_c_local):
        """Generate speaker attractors via 2-layer Transformer Decoder.

        Args:
            S_c: (B, n_bands, d_model) — HSM-fused features
            H_c_local: (B, n_bands, d_model, T_c) — local chunk features

        Returns:
            attractors: (B, K+1, d_model)
            p_exist: (B, K+1, 1)
        """
        B, n_bands, d_model = S_c.shape
        T_c = H_c_local.shape[-1]

        # ── Build KV: 方案A2 (time-sampled + band-level) ──
        # S_c as band-level tokens (B, n_bands, D) → n_bands tokens
        S_tokens = S_c  # (B, n_bands, D)

        # Time-sampled H_c: pool over bands, subsample time
        H_band_pooled = H_c_local.mean(dim=1)  # (B, D, T_c)
        H_time_sampled = H_band_pooled[:, :, ::self.time_stride]  # (B, D, T_c//stride)
        H_time_sampled = H_time_sampled.permute(0, 2, 1)  # (B, T_c//stride, D)

        # Concatenate: [S_c bands; H time-sampled]
        kv = torch.cat([S_tokens, H_time_sampled], dim=1)  # (B, n_bands + T//stride, D)

        # Speaker queries
        queries = self.speaker_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K+1, D)

        # ── Layer 1: Cross-Attention only (DETR practice) ──
        ca_out, _ = self.cross_attn_1(queries, kv, kv)
        x = self.norm_ca_1(queries + ca_out)
        x = self.norm_ffn_1(x + self.ffn_1(x))

        # ── Layer 2: Masked Self-Attention + Cross-Attention ──
        sa_out, _ = self.self_attn_2(x, x, x, attn_mask=self.causal_mask)
        x = self.norm_sa_2(x + sa_out)
        ca_out, _ = self.cross_attn_2(x, kv, kv)
        x = self.norm_ca_2(x + ca_out)
        attractors = self.norm_ffn_2(x + self.ffn_2(x))

        # Existence probabilities
        p_exist = self.existence_head(attractors)

        return attractors, p_exist

    def get_film_params(self, attractors):
        """Generate FiLM parameters (gamma=1+δ, beta)."""
        film_out = self.film_proj(attractors)
        delta_gamma = film_out[..., :self.d_model]
        beta = film_out[..., self.d_model:]
        return 1.0 + delta_gamma, beta


# ═══════════════════════════════════════════════════════════════
# LWCSS — Main Model
# ═══════════════════════════════════════════════════════════════

class LWCSS(BaseModel):
    """Light-Weight Continuous Speech Separation.

    Full streaming CSS with HSM (M=16, attention fusion) + EDA (2-layer decoder).
    """
    def __init__(
        self,
        sample_rate=8000,
        win=512,
        stride=256,
        out_channels=128,
        in_channels=512,
        num_blocks=4,
        upsampling_depth=4,
        att_n_head=4,
        att_hid_chan=4,
        n_memory_slots=16,      # V5: M=16 (Papez optimal)
        max_speakers=4,
        num_sources=None,
        d_state=8,
    ):
        super().__init__(sample_rate=sample_rate)

        self.win = win
        self.stride = stride
        self.n_fft = win
        self.enc_dim = win // 2 + 1
        self.out_channels = out_channels
        self.max_speakers = max_speakers
        self.n_memory_slots = n_memory_slots
        self.num_sources = num_sources if num_sources is not None else max_speakers + 1

        self._build_band_config()

        # IASFC Encoder
        self.sfc_encoder = SFCEncoder(
            n_freq=self.enc_dim, n_bands=self.n_bands,
            d_inner=out_channels, d_model=out_channels,
            band_indices=self.band_indices, d_state=d_state,
        )

        # Stage 1 FFIBs
        self.ffib_stage1 = FFIBStack(
            out_channels=out_channels, in_channels=in_channels,
            upsampling_depth=upsampling_depth, n_head=att_n_head,
            att_hid_chan=att_hid_chan, n_iter=num_blocks,
        )

        # HSM (M=16, attention fusion)
        self.hsm = HierarchicalStreamingMemory(
            n_bands=self.n_bands, d_model=out_channels,
            n_memory_slots=n_memory_slots, d_state=d_state,
        )

        # EDA (2-layer decoder, time-sampled KV)
        self.eda = EncoderDecoderAttractor(
            d_model=out_channels, max_speakers=max_speakers,
            n_head=att_n_head, d_ff=in_channels,
        )

        # Stage 2 FFIBs
        self.ffib_stage2 = FFIBStack(
            out_channels=out_channels, in_channels=in_channels,
            upsampling_depth=upsampling_depth, n_head=att_n_head,
            att_hid_chan=att_hid_chan, n_iter=max(num_blocks // 2, 2),
        )

        # IASFC Decoder
        self.sfc_decoder = SFCDecoder(
            n_freq=self.enc_dim, n_bands=self.n_bands,
            d_inner=out_channels, d_model=out_channels,
            band_indices=self.band_indices, d_state=d_state,
        )

        self.eps = torch.finfo(torch.float32).eps

    def _build_band_config(self):
        """64 bands for 8kHz/512FFT (257 bins)."""
        freq_res = self.sample_rate / self.n_fft
        bin_1k = round(1000 / freq_res)
        bin_2k = round(2000 / freq_res)
        n_low, n_mid, n_high = 32, 16, 16
        bw_low = max(1, bin_1k // n_low)
        bw_mid = max(1, (bin_2k - bin_1k) // n_mid)
        remaining = self.enc_dim - (n_low * bw_low + n_mid * bw_mid)
        bw_high = max(1, remaining // n_high)
        self.band_width = [bw_low] * n_low + [bw_mid] * n_mid + [bw_high] * n_high
        total = sum(self.band_width)
        if total != self.enc_dim:
            self.band_width[-1] += (self.enc_dim - total)
        self.n_bands = len(self.band_width)
        self.band_indices = []
        offset = 0
        for w in self.band_width:
            self.band_indices.append((offset, offset + w))
            offset += w
        print(f"[LWCSS] {self.n_bands} bands, widths: {self.band_width[:3]}...{self.band_width[-3:]}")

    def forward(self, input, G_prev=None, S_prev=None):
        """Forward pass.

        Args:
            input: (B, C, T) waveform | (B, T) | (T,)
            G_prev: (B, M, D) | None
            S_prev: (B, n_bands, D) | None

        Returns:
            output: (B, num_sources, T)
            G_cur: (B, M, D)
            S_cur: (B, n_bands, D)
            p_exist: (B, K+1)
        """
        squeeze_output = False
        if input.ndim == 1:
            squeeze_output = True
            input = input.unsqueeze(0).unsqueeze(1)
        elif input.ndim == 2:
            squeeze_output = True
            input = input.unsqueeze(1)

        B, nch, nsample = input.shape
        input_flat = input.view(B * nch, -1)

        # STFT
        spec = torch.stft(
            input_flat, n_fft=self.n_fft, hop_length=self.stride,
            window=torch.hann_window(self.n_fft, device=input.device, dtype=input.dtype),
            return_complex=True
        )
        F_bins, T_frames = spec.shape[1], spec.shape[2]

        # IASFC Encoder
        spec_RI = torch.stack([spec.real, spec.imag], dim=1).permute(0, 1, 3, 2)  # (B, 2, T, F)
        enc_out, encoder_emb = self.sfc_encoder(spec_RI)  # (B, D, T, K), (B*T, F, D*2)
        subband_feats = enc_out.permute(0, 1, 3, 2)  # (B, N, nband, T)

        # Stage 1 FFIBs
        H_c = self.ffib_stage1(subband_feats)  # (B, N, nband, T)

        # HSM
        H_c_hsm = H_c.permute(0, 2, 1, 3)  # (B, nband, N, T)
        S_c, G_cur = self.hsm(H_c_hsm, G_prev, S_prev)

        # EDA
        attractors, p_exist_raw = self.eda(S_c, H_c_hsm)

        # Stage 2: FiLM condition + FFIBs
        gamma, beta = self.eda.get_film_params(attractors)
        H_expanded = H_c.unsqueeze(1).expand(-1, self.num_sources, -1, -1, -1)
        gamma_exp = gamma.unsqueeze(-1).unsqueeze(-1)
        beta_exp = beta.unsqueeze(-1).unsqueeze(-1)
        H_conditioned = gamma_exp * H_expanded + beta_exp

        B_k = B * self.num_sources
        H_in = H_conditioned.view(B_k, self.out_channels, self.n_bands, T_frames)
        H_out = self.ffib_stage2(H_in)
        H_out = H_out.view(B, self.num_sources, self.out_channels, self.n_bands, T_frames)

        # IASFC Decoder
        sep_specs = []
        for k in range(self.num_sources):
            feat_k = H_out[:, k].permute(0, 1, 3, 2)  # (B, D, T, K)
            mask_k = self.sfc_decoder(feat_k, encoder_emb)  # (B, 2, T, F)
            mask_k = mask_k.permute(0, 3, 1, 2)  # (B, F, 2, T)
            mask_complex = torch.complex(mask_k[:, :, 0], mask_k[:, :, 1])
            sep_specs.append(spec * mask_complex)

        sep_spec = torch.stack(sep_specs, dim=1)
        sep_spec_flat = sep_spec.view(B * self.num_sources, F_bins, T_frames)
        output = torch.istft(
            sep_spec_flat, n_fft=self.n_fft, hop_length=self.stride,
            window=torch.hann_window(self.n_fft, device=input.device, dtype=input.dtype),
            length=nsample
        )
        output = output.view(B, self.num_sources, -1)
        p_exist = p_exist_raw.squeeze(-1)

        if squeeze_output:
            output = output.squeeze(0)

        return output, G_cur, S_c, p_exist

    def get_model_args(self):
        return {"sample_rate": self.sample_rate}
