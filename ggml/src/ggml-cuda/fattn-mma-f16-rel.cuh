#pragma once

#include "common.cuh"

// Dedicated SM75+ Tensor-Core FlashAttention (DKQ=128, DV=128) with:
//   1. On-the-fly Inkling-2 relative-position bias (src[6]=rel_table, src[7]=k_pos, src[8]=q_pos)
//      eliminating the materialized per-head kq_b buffer while preserving GQA KV sharing.
//   2. Direct shared-memory F8_E4M3 (double-buffered cp.async + branchless e4m3x8->f16) and
//      Q8_0 KV tile loaders, avoiding per-call F16 KV expansion.
//   3. FP32 VKQ accumulation for low-drift long-context prefill and decode.

bool ggml_cuda_flash_attn_ext_mma_f16_rel_supported(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_mma_f16_rel(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
