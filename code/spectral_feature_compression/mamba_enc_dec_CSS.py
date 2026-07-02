# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

# 假设原有的导入路径保持不变
from spectral_feature_compression.core.model.enc_dec_base import DecoderBase, EncoderBase, RMSNorm


# ---------------------------------------------------------
# [新增模块]：用于替代普通 Conv2d 的流式因果卷积包装器
# ---------------------------------------------------------
class CausalConv2dWrapper(nn.Module):
    def __init__(self, conv2d_module: nn.Conv2d):
        super().__init__()
        self.conv = conv2d_module
        # 获取原始卷积的 kernel_size, 假设为 (K_t, K_f)
        self.k_t = self.conv.kernel_size[0]
        self.k_f = self.conv.kernel_size[1]
        self.stride_t = self.conv.stride[0]
        self.stride_f = self.conv.stride[1]

        # 强制修改原始卷积的 Padding
        # 在频率轴 (F) 保持对称 Padding，在时间轴 (T) 设为 0 (我们手动 Padding)
        pad_f = self.conv.padding[1]
        self.conv.padding = (0, pad_f)

    def forward(self, x: torch.Tensor, conv_state: torch.Tensor | None = None):
        """
        x: (B, C, T_chunk, F)
        conv_state: (B, C, K_t - 1, F) 上一次卷积遗留的历史数据
        """
        pad_t = self.k_t - 1

        if conv_state is not None:
            # 流式推理：将历史状态拼接到当前输入的前面
            x = torch.cat([conv_state, x], dim=2)
        else:
            # 初始状态或离线训练：在时间轴左侧补零 (Causal Padding)
            x = F.pad(x, (0, 0, pad_t, 0))

        # 提取用于下一个 Chunk 的状态 (取最后 K_t - 1 帧)
        next_conv_state = x[:, :, -pad_t:, :].detach() if pad_t > 0 else None

        # 执行卷积
        out = self.conv(x)
        return out, next_conv_state


# ---------------------------------------------------------
# [修改模块]：重构基类，加入流式状态流转
# ---------------------------------------------------------
class MambaEncDecForward:
    forward_block: nn.Module
    backward_block: nn.Module
    input_conv: nn.Module  # 在初始化时，需要被包裹为 CausalConv2dWrapper
    emb_indices: list
    query_indices: list
    bidirectional: bool

    def forward(self,
                input: torch.Tensor,
                query: torch.Tensor | None = None,
                conv_state: torch.Tensor | None = None,
                query_state: torch.Tensor | None = None):
        """
        input: torch.Tensor (n_batch, n_channels, n_frames_chunk, n_freq)
            注意：流式模式下，这里传入的 n_frames_chunk 仅为 C (当前) + F (前瞻)
        query: torch.Tensor (n_batch * n_frames, n_bands, n_hidden)
        conv_state: 卷积的缓存状态
        query_state: 上一个 Chunk 的 Query 状态，用于时间平滑
        """
        if input.is_complex():
            input = torch.cat((input.real, input.imag), dim=1)

        n_batch, _, n_frames, n_freq = input.shape

        # 1. 严格因果的 2D 卷积处理
        emb_orig, next_conv_state = self.input_conv(input, conv_state)

        # 展平时间维度 (B*T)，使得 Mamba 逐帧沿着频率轴扫描
        emb_orig = emb_orig.permute(0, 2, 3, 1).reshape(n_batch * n_frames, -1, self.d_inner)
        emb = emb_orig.clone()

        # 2. 生成 Query
        query = self._prepare_query(emb, query_orig=query)

        # 3. 流式自适应 Query 平滑 (Inductive Bias for Streaming)
        # 如果提供了历史 query_state，我们在时间维度上给 query 增加一个动量平滑
        if query_state is not None:
            # 将 query_state 扩展到当前 batch*frames 形状
            # query_state 形状一般设计为 (n_batch, 1, n_bands, d_inner)
            qs_expanded = query_state.expand(n_batch, n_frames, -1, -1).reshape_as(query)
            alpha = 0.8  # 平滑系数，可设为可学习参数 nn.Parameter
            query = alpha * query + (1 - alpha) * qs_expanded

        # 提取当前 chunk 的最后几帧作为下一个 chunk 的 query_state
        next_query_state = query.view(n_batch, n_frames, query.shape[-2], -1)[:, -1:, :, :].detach()

        f_new = query.shape[-2]

        # 4. 频率轴双向 Mamba 扫描
        query_forward, emb_forward = self._process_input(
            self.forward_block, emb, query, self.emb_indices, self.query_indices, forward=True
        )
        if self.bidirectional:
            query_backward, emb_backward = self._process_input(
                self.backward_block, emb, query, self.emb_indices, self.query_indices, forward=False
            )
            query = torch.cat((query_forward, query_backward), dim=-1)
            emb = torch.cat((emb_forward, emb_backward), dim=-1)
        else:
            query = query_forward
            emb = emb_forward

        query = query.reshape(n_batch, n_frames, f_new, -1)
        query = self._output_proj(query)

        # 返回时，连同状态一并返回，供外部流式引擎 (Streaming Wrapper) 保存
        return query, emb, next_conv_state, next_query_state

    # _process_input 和 _prepare_combined_tokens 保持原样不变
    # ...