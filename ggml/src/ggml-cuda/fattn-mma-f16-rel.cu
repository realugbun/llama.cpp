#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16-rel.cuh"

#include <climits>
#include <cstdlib>

using namespace ggml_cuda_mma;

// ------------------------------------------------------------------------------------------------------------------
// Branchless E4M3 -> FP16 converter (MIT, adapted from project-maya ik2_kernels.cu):
//   - Bit-exact for all 254 finite E4M3 values (normals, subnormals, +/-0).
//   - Maps E4M3 NaN codes 0x7F/0xFF to finite +/-480.0 so unwritten/masked KV cells can never
//     poison attention logits with NaN + (-inf) = NaN.
// ------------------------------------------------------------------------------------------------------------------
static __device__ __forceinline__ void fattn_rel_e4m3x4_h4(const uint32_t w, uint32_t & lo, uint32_t & hi) {
    const uint32_t t = w & 0x78787878u;
    uint32_t z = (t - 0x01010101u) & ~t & 0x80808080u;
    z -= z >> 7;
    const uint32_t v  = w & 0x07070707u;
    const uint32_t u  = v | (v >> 4);
    const uint32_t dd = __byte_perm(0x070A1140u, 0x01020304u, __byte_perm(u, 0u, 0x4420u)) & z;
    const uint32_t x0 = __byte_perm(w, 0u, 0x1404u);
    const uint32_t x1 = __byte_perm(w, 0u, 0x3424u);
    const uint32_t y0 = __byte_perm(dd, 0u, 0x1404u);
    const uint32_t y1 = __byte_perm(dd, 0u, 0x3424u);
    lo = x0 - (((x0 & 0x7F007F00u) + y0) >> 1) + 0x20002000u;
    hi = x1 - (((x1 & 0x7F007F00u) + y1) >> 1) + 0x20002000u;
}

static __device__ __forceinline__ uint4 fattn_rel_e4m3x8_h8(const uint2 p) {
    uint4 h;
    fattn_rel_e4m3x4_h4(p.x, h.x, h.y);
    fattn_rel_e4m3x4_h4(p.y, h.z, h.w);
    return h;
}

// ------------------------------------------------------------------------------------------------------------------
// Tile configs for DKQ=128, DV=128 on SM75+ (Turing / Ampere / Ada / Hopper / Blackwell)
// Fixed nbatch_fa = 64, nthreads = 128 (4 warps), occupancy = 2, Q_in_reg = true, swizzled = true.
// Shared memory:
//   - F16 2-stage:   tile_K (16 KB) + tile_V (16 KB) + tile_mask (<=2.3 KB) + tile_k_pos (256 B) <= 34.6 KB -> occ 2
//   - E4M3 2-stage:  tile_KV (16 KB) + raw_K (8 KB) + raw_V (8 KB) + tile_mask + tile_k_pos      <= 34.6 KB -> occ 2
//   - Q8_0 1-stage:  tile_KV (16 KB) + tile_mask + tile_k_pos                                    <= 18.6 KB -> occ 2
// ------------------------------------------------------------------------------------------------------------------
static constexpr int FATTN_REL_DKQ            = 128;
static constexpr int FATTN_REL_DV             = 128;
static constexpr int FATTN_REL_NTHREADS       = 128;
static constexpr int FATTN_REL_NWARPS         = 4;
static constexpr int FATTN_REL_NBATCH_FA      = 64;
static constexpr int FATTN_REL_STRIDE_TILE_KV = 64; // half2s (128 halfs = 256 bytes, swizzled)
static constexpr int FATTN_REL_STRIDE_TILE_Q  = 68; // DKQ/2 + 4
static constexpr int FATTN_REL_NBATCH_COMBINE = 64;

template <int ncols, bool fp32_vkq = false> struct fattn_rel_mma_tiles {
    using T_A_KQ  = tile<16,  8, half2>;
    using T_B_KQ  = tile<16,  8, half2>;
    using T_C_KQ  = tile<16, 16, float>;
    using T_A_VKQ = tile<16,  8, half2>;
    using T_B_VKQ = tile<16,  8, half2>;
    using T_C_VKQ = typename std::conditional<fp32_vkq, tile<16, 16, float>, tile<16, 8, half2>>::type;
};

template <bool fp32_vkq> struct fattn_rel_mma_tiles<8, fp32_vkq> {
    using T_A_KQ  = tile<16,  8, half2>;
    using T_B_KQ  = tile< 8,  8, half2>;
    using T_C_KQ  = tile<16,  8, float>;
    using T_A_VKQ = tile<16,  8, half2>;
    using T_B_VKQ = tile< 8,  8, half2>;
    using T_C_VKQ = typename std::conditional<fp32_vkq, tile<16, 8, float>, tile<16, 4, half2>>::type;
};

// ------------------------------------------------------------------------------------------------------------------
// KV Tile Loaders (F16, F8_E4M3, Q8_0) -> swizzled half2[FATTN_REL_NBATCH_FA][FATTN_REL_STRIDE_TILE_KV]
// ------------------------------------------------------------------------------------------------------------------

// F16 loader (cp.async or synchronous)
template <bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void fattn_rel_load_kv_f16(
        const char * const __restrict__ KV_bytes,
        half2 * const __restrict__ tile_KV,
        const int stride_KV_bytes,
        const int k_VKQ_0,
        const int i_sup,
        const int32_t * const __restrict__ indices) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    constexpr int chunks_per_row = 16; // 128 halfs / 8 halfs per 16B chunk
    constexpr int rows_per_iter  = FATTN_REL_NTHREADS / chunks_per_row; // 8

    if constexpr (use_cp_async) {
        const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);
#pragma unroll
        for (int i0 = 0; i0 < FATTN_REL_NBATCH_FA; i0 += rows_per_iter) {
            const int row   = i0 + (tid >> 4);
            const int chunk = tid & 15;
            const int64_t i_KV = indices ? max(indices[k_VKQ_0 + row], 0) : (int64_t)(k_VKQ_0 + row);
            const char * src = KV_bytes + i_KV * stride_KV_bytes + chunk * 16;
            cp_async_cg_16<64>(
                tile_KV_32 + swizzle<FATTN_REL_STRIDE_TILE_KV * sizeof(half2), char>(row * FATTN_REL_STRIDE_TILE_KV * sizeof(half2) + chunk * 16, row),
                src);
        }
    } else {
        const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int i0 = 0; i0 < FATTN_REL_NBATCH_FA; i0 += rows_per_iter) {
            const int row   = i0 + (tid >> 4);
            const int chunk = tid & 15;
            uint4 val = zero;
            if (!oob_check || row < i_sup) {
                const int64_t i_KV = indices ? max(indices[k_VKQ_0 + row], 0) : (int64_t)(k_VKQ_0 + row);
                val = *(const uint4 *)(KV_bytes + i_KV * stride_KV_bytes + chunk * 16);
            }
            *(uint4 *) swizzle<FATTN_REL_STRIDE_TILE_KV>(tile_KV, row * FATTN_REL_STRIDE_TILE_KV + chunk * 4, row) = val;
        }
    }
}

// F8_E4M3 raw cp.async loader -> uint8_t[FATTN_REL_NBATCH_FA][128]
static __device__ __forceinline__ void fattn_rel_load_kv_f8_raw_async(
        const char * const __restrict__ KV_bytes,
        uint8_t * const __restrict__ tile_raw,
        const int stride_KV_bytes,
        const int k_VKQ_0,
        const int32_t * const __restrict__ indices) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    constexpr int chunks_per_row = 8; // 128 bytes / 16B per cp.async
    constexpr int rows_per_iter  = FATTN_REL_NTHREADS / chunks_per_row; // 16
    const unsigned int raw_32 = ggml_cuda_cvta_generic_to_shared(tile_raw);

#pragma unroll
    for (int i0 = 0; i0 < FATTN_REL_NBATCH_FA; i0 += rows_per_iter) {
        const int row   = i0 + (tid >> 3);
        const int chunk = tid & 7;
        const int64_t i_KV = indices ? max(indices[k_VKQ_0 + row], 0) : (int64_t)(k_VKQ_0 + row);
        cp_async_cg_16<64>(raw_32 + row * 128 + chunk * 16, KV_bytes + i_KV * stride_KV_bytes + chunk * 16);
    }
}

// Convert staged F8_E4M3 raw smem -> swizzled half2 tile_KV
static __device__ __forceinline__ void fattn_rel_convert_f8_raw_to_tile(
        const uint8_t * const __restrict__ tile_raw,
        half2 * const __restrict__ tile_KV) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    constexpr int chunks_per_row = 16; // 128 elems / 8 elems per thread
    constexpr int rows_per_iter  = FATTN_REL_NTHREADS / chunks_per_row; // 8

#pragma unroll
    for (int i0 = 0; i0 < FATTN_REL_NBATCH_FA; i0 += rows_per_iter) {
        const int row   = i0 + (tid >> 4);
        const int chunk = tid & 15;
        const uint2 raw8 = *(const uint2 *)(tile_raw + row * 128 + chunk * 8);
        *(uint4 *) swizzle<FATTN_REL_STRIDE_TILE_KV>(tile_KV, row * FATTN_REL_STRIDE_TILE_KV + chunk * 4, row) = fattn_rel_e4m3x8_h8(raw8);
    }
}

// Synchronous F8_E4M3 loader (used when oob_check=true or cp.async unavailable)
template <bool oob_check>
static __device__ __forceinline__ void fattn_rel_load_kv_f8_sync(
        const char * const __restrict__ KV_bytes,
        half2 * const __restrict__ tile_KV,
        const int stride_KV_bytes,
        const int k_VKQ_0,
        const int i_sup,
        const int32_t * const __restrict__ indices) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    constexpr int chunks_per_row = 16;
    constexpr int rows_per_iter  = FATTN_REL_NTHREADS / chunks_per_row; // 8

#pragma unroll
    for (int i0 = 0; i0 < FATTN_REL_NBATCH_FA; i0 += rows_per_iter) {
        const int row   = i0 + (tid >> 4);
        const int chunk = tid & 15;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (!oob_check || row < i_sup) {
            const int64_t i_KV = indices ? max(indices[k_VKQ_0 + row], 0) : (int64_t)(k_VKQ_0 + row);
            const uint2 raw8 = *(const uint2 *)(KV_bytes + i_KV * stride_KV_bytes + chunk * 8);
            val = fattn_rel_e4m3x8_h8(raw8);
        }
        *(uint4 *) swizzle<FATTN_REL_STRIDE_TILE_KV>(tile_KV, row * FATTN_REL_STRIDE_TILE_KV + chunk * 4, row) = val;
    }
}

// Synchronous Q8_0 loader: 4 block_q8_0 per 128-dim row (136 bytes/row, 2-byte aligned).
// Matches dequantize_block_q8_0: F32 multiply (__half2float(d) * float(int8)) -> half2.
template <bool oob_check>
static __device__ __forceinline__ void fattn_rel_load_kv_q8_0(
        const char * const __restrict__ KV_bytes,
        half2 * const __restrict__ tile_KV,
        const int stride_KV_bytes,
        const int k_VKQ_0,
        const int i_sup,
        const int32_t * const __restrict__ indices) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    constexpr int chunks_per_row = 16; // 4 blocks * 4 chunks of 8 int8s
    constexpr int rows_per_iter  = FATTN_REL_NTHREADS / chunks_per_row; // 8

#pragma unroll
    for (int i0 = 0; i0 < FATTN_REL_NBATCH_FA; i0 += rows_per_iter) {
        const int row   = i0 + (tid >> 4);
        const int chunk = tid & 15;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (!oob_check || row < i_sup) {
            const int64_t i_KV = indices ? max(indices[k_VKQ_0 + row], 0) : (int64_t)(k_VKQ_0 + row);
            const int ib   = chunk >> 2;
            const int qoff = (chunk & 3) * 8;
            const block_q8_0 * blk = (const block_q8_0 *)(KV_bytes + i_KV * stride_KV_bytes) + ib;
            const float d = __half2float(blk->d);
            int8_t q[8];
            ggml_cuda_memcpy_1<8, 2>(q, blk->qs + qoff);
            half2 h2[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                h2[m] = make_half2(d * (float) q[2 * m + 0], d * (float) q[2 * m + 1]);
            }
            val = *(const uint4 *) h2;
        }
        *(uint4 *) swizzle<FATTN_REL_STRIDE_TILE_KV>(tile_KV, row * FATTN_REL_STRIDE_TILE_KV + chunk * 4, row) = val;
    }
}

// Mask loader (head-free F16 mask [n_kv, n_tps, 1, n_stream])
template <int ncols1, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void fattn_rel_load_mask(
        const half * const __restrict__ mask_h,
        half * const __restrict__ tile_mask,
        const int stride_mask,
        const int k_VKQ_0,
        const int i_sup,
        const int j0,
        const uint3 ne01) {
    if (!mask_h) {
        return;
    }
    if constexpr (use_cp_async) {
        constexpr int cols_per_warp = 8 * 32 / FATTN_REL_NBATCH_FA; // 4
        constexpr int stride_j      = FATTN_REL_NWARPS * cols_per_warp; // 16
        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y * cols_per_warp + threadIdx.x / (32 / cols_per_warp);
            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }
            const int j_vram = fastmodulo(j0 + j_sram, ne01);
            const int i      = 8 * (threadIdx.x % (FATTN_REL_NBATCH_FA / 8));
            cp_async_cg_16<64>(
                tile_mask_32 + j_sram * (FATTN_REL_NBATCH_FA * sizeof(half) + 16) + i * sizeof(half),
                mask_h + int64_t(j_vram) * stride_mask + k_VKQ_0 + i);
        }
    } else if constexpr (oob_check) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += FATTN_REL_NWARPS) {
            const int j_sram = j1 + threadIdx.y;
            if (j1 + FATTN_REL_NWARPS > ncols1 && j_sram >= ncols1) {
                break;
            }
            const int j_vram = fastmodulo(j0 + j_sram, ne01);
#pragma unroll
            for (int i0 = 0; i0 < FATTN_REL_NBATCH_FA; i0 += 32) {
                const int i = i0 + threadIdx.x;
                tile_mask[j_sram * (FATTN_REL_NBATCH_FA + 8) + i] =
                    i < i_sup ? mask_h[int64_t(j_vram) * stride_mask + k_VKQ_0 + i] : half(0.0f);
            }
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += FATTN_REL_NWARPS) {
            const int j_sram = j1 + threadIdx.y;
            if (j1 + FATTN_REL_NWARPS > ncols1 && j_sram >= ncols1) {
                break;
            }
            const int j_vram = fastmodulo(j0 + j_sram, ne01);
            const int i = 2 * threadIdx.x;
            ggml_cuda_memcpy_1<sizeof(half2)>(
                tile_mask + j_sram * (FATTN_REL_NBATCH_FA + 8) + i,
                mask_h + int64_t(j_vram) * stride_mask + k_VKQ_0 + i);
        }
    }
}

// Load tile_k_pos[0..63] (INT32_MIN for OOB)
template <bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void fattn_rel_load_k_pos(
        const int32_t * const __restrict__ k_pos_seq,
        int32_t * const __restrict__ tile_k_pos,
        const int k_VKQ_0,
        const int i_sup) {
    if (!k_pos_seq) {
        return;
    }
    const int tid = threadIdx.y * 32 + threadIdx.x;
    if constexpr (use_cp_async) {
        if (tid < FATTN_REL_NBATCH_FA / 4) { // 16 threads x 16B = 256B = 64 int32s
            const unsigned int dst_32 = ggml_cuda_cvta_generic_to_shared(tile_k_pos);
            cp_async_cg_16<64>(dst_32 + tid * 16, k_pos_seq + k_VKQ_0 + tid * 4);
        }
    } else {
        if (tid < FATTN_REL_NBATCH_FA) {
            tile_k_pos[tid] = (!oob_check || tid < i_sup) ? k_pos_seq[k_VKQ_0 + tid] : INT32_MIN;
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Core iteration over one 64-key KV tile
// ------------------------------------------------------------------------------------------------------------------
template <int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool use_cp_async_2stage, bool last_iter, bool oob_check, bool fp32_vkq>
static __device__ __forceinline__ void fattn_rel_iter(
        const char    * const __restrict__ K_bytes,
        const char    * const __restrict__ V_bytes,
        const half    * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        const int32_t * const __restrict__ k_pos_seq,
        const int32_t q_pos_col[2],
        const float   * const rel_row[2],
        const float   far_bias[2],
        const int32_t rel_extent,
        const uint3   ne01,
        const int     stride_K_bytes,
        const int     stride_V_bytes,
        const int     stride_mask,
        half2   * const __restrict__ tile_K,
        half2   * const __restrict__ tile_V,
        uint8_t * const __restrict__ tile_raw_K,
        uint8_t * const __restrict__ tile_raw_V,
        half    * const __restrict__ tile_mask,
        int32_t * const __restrict__ tile_k_pos,
        typename fattn_rel_mma_tiles<ncols1 * ncols2, fp32_vkq>::T_B_KQ  * const __restrict__ Q_B,
        typename fattn_rel_mma_tiles<ncols1 * ncols2, fp32_vkq>::T_C_VKQ * const __restrict__ VKQ_C,
        float * const __restrict__ KQ_max,
        float * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int kb0_next,
        const int k_VKQ_sup) {
#if defined(TURING_MMA_AVAILABLE)
    constexpr int ncols         = ncols1 * ncols2;
    using T_A_KQ  = typename fattn_rel_mma_tiles<ncols, fp32_vkq>::T_A_KQ;
    using T_B_KQ  = typename fattn_rel_mma_tiles<ncols, fp32_vkq>::T_B_KQ;
    using T_C_KQ  = typename fattn_rel_mma_tiles<ncols, fp32_vkq>::T_C_KQ;
    using T_A_VKQ = typename fattn_rel_mma_tiles<ncols, fp32_vkq>::T_A_VKQ;
    using T_B_VKQ = typename fattn_rel_mma_tiles<ncols, fp32_vkq>::T_B_VKQ;
    using T_C_VKQ = typename fattn_rel_mma_tiles<ncols, fp32_vkq>::T_C_VKQ;

    constexpr int cols_per_warp = T_B_KQ::I;
    constexpr int np            = FATTN_REL_NWARPS * cols_per_warp / ncols;
    const int k_VKQ_0           = kb0 * FATTN_REL_NBATCH_FA;

    T_C_KQ KQ_C[FATTN_REL_NBATCH_FA / (np * (cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];

    // --- Stage A: ensure K tile, mask, and k_pos are ready; overlap V load if 2-stage ---
    if constexpr (use_cp_async_2stage) {
        cp_async_wait_all();
        __syncthreads();

        // Issue V preload while we compute K convert + Q*K^T + softmax!
        if constexpr (type_V == GGML_TYPE_F16) {
            fattn_rel_load_kv_f16<true, false>(V_bytes, tile_V, stride_V_bytes, k_VKQ_0, k_VKQ_sup, indices);
        } else if constexpr (type_V == GGML_TYPE_F8_E4M3) {
            fattn_rel_load_kv_f8_raw_async(V_bytes, tile_raw_V, stride_V_bytes, k_VKQ_0, indices);
        }

        if constexpr (type_K == GGML_TYPE_F8_E4M3) {
            fattn_rel_convert_f8_raw_to_tile(tile_raw_K, tile_K);
            __syncthreads();
        }
    } else {
        fattn_rel_load_mask<ncols1, false, oob_check>(mask_h, tile_mask, stride_mask, k_VKQ_0, k_VKQ_sup, jt * ncols1, ne01);
        fattn_rel_load_k_pos<false, oob_check>(k_pos_seq, tile_k_pos, k_VKQ_0, k_VKQ_sup);

        if constexpr (type_K == GGML_TYPE_F16) {
            fattn_rel_load_kv_f16<false, oob_check>(K_bytes, tile_K, stride_K_bytes, k_VKQ_0, k_VKQ_sup, indices);
        } else if constexpr (type_K == GGML_TYPE_F8_E4M3) {
            fattn_rel_load_kv_f8_sync<oob_check>(K_bytes, tile_K, stride_K_bytes, k_VKQ_0, k_VKQ_sup, indices);
        } else {
            fattn_rel_load_kv_q8_0<oob_check>(K_bytes, tile_K, stride_K_bytes, k_VKQ_0, k_VKQ_sup, indices);
        }
        __syncthreads();
    }

    // --- Stage B: Q * K^T MMA ---
#pragma unroll
    for (int i_KQ_00 = 0; i_KQ_00 < FATTN_REL_NBATCH_FA; i_KQ_00 += np * T_A_KQ::I) {
        const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np) * T_A_KQ::I;
#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < FATTN_REL_DKQ / 2; k_KQ_0 += T_A_KQ::J) {
            T_A_KQ K_A;
            load_ldmatrix_swizzled<FATTN_REL_STRIDE_TILE_KV>(K_A, tile_K, i_KQ_0 * FATTN_REL_STRIDE_TILE_KV + k_KQ_0);
            if constexpr (cols_per_warp == 8) {
                mma(KQ_C[i_KQ_00 / (np * T_A_KQ::I)], K_A, Q_B[k_KQ_0 / T_A_KQ::J]);
            } else {
                mma(KQ_C[i_KQ_00 / (np * T_A_KQ::I)], Q_B[k_KQ_0 / T_A_KQ::J], K_A);
            }
        }
    }

    // --- Stage C: Mask + On-the-fly Relative-Position Bias + Online Softmax ---
    float KQ_max_new[2]    = { KQ_max[0], KQ_max[1] };
    float KQ_rowsum_add[2] = { 0.0f, 0.0f };

    // Tile-level far_bias fast-path: on 99%+ of long-context tiles, all 64 keys have dist >= rel_extent
    // (or are INT32_MIN empty cells), so every key gets the constant far_bias[col] = -sink(q, h).
    // IMPORTANT: branch condition (k_pos_seq != nullptr) is 100% block-uniform so all 32 lanes
    // participate in warp shuffles and __all_sync with zero warp divergence on SM70+.
    bool warp_all_far = false;
    if (k_pos_seq) {
        int32_t tk_max = max(tile_k_pos[threadIdx.x], tile_k_pos[threadIdx.x + 32]);
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            tk_max = max(tk_max, ggml_cuda_shfl_xor(tk_max, off, 32));
        }
        const bool f0 = !rel_row[0] || (tk_max == INT32_MIN) ||
                        ((int64_t) q_pos_col[0] - (int64_t) tk_max >= (int64_t) rel_extent);
        const bool f1 = !rel_row[1] || (tk_max == INT32_MIN) ||
                        ((int64_t) q_pos_col[1] - (int64_t) tk_max >= (int64_t) rel_extent);
        warp_all_far = __all_sync(0xFFFFFFFFu, f0 && f1);
    }

    if constexpr (cols_per_warp == 8) {
        if (warp_all_far) {
#pragma unroll
            for (int i00 = 0; i00 < FATTN_REL_NBATCH_FA; i00 += np * T_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np) * T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i   = i0 + T_C_KQ::get_i(l);
                    const int col = l % 2;
                    float bias = far_bias[col];
                    if (mask_h) {
                        const int j = ((threadIdx.y / np) * T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;
                        bias += __half2float(tile_mask[j * (FATTN_REL_NBATCH_FA + 8) + i]);
                    }
                    KQ_C[i00 / (np * T_C_KQ::I)].x[l] += bias;
                }
            }
        } else {
#pragma unroll
            for (int i00 = 0; i00 < FATTN_REL_NBATCH_FA; i00 += np * T_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np) * T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i   = i0 + T_C_KQ::get_i(l);
                    const int col = l % 2;
                    float bias = 0.0f;
                    if (mask_h) {
                        const int j = ((threadIdx.y / np) * T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;
                        bias = __half2float(tile_mask[j * (FATTN_REL_NBATCH_FA + 8) + i]);
                    }
                    if (rel_row[col]) {
                        const int32_t kp   = tile_k_pos[i];
                        const int64_t dist = (kp == INT32_MIN) ? -1 : (int64_t) q_pos_col[col] - (int64_t) kp;
                        const float   rb   = (dist >= 0 && dist < rel_extent) ? rel_row[col][(int32_t) dist] : far_bias[col];
                        bias += rb;
                    }
                    KQ_C[i00 / (np * T_C_KQ::I)].x[l] += bias;
                }
            }
        }

#pragma unroll
        for (int k0 = 0; k0 < FATTN_REL_NBATCH_FA; k0 += np * T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np) * T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
                    const int KQ_idx = l % 2;
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0 / (np * T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < 2; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], ggml_cuda_shfl_xor(KQ_max_new[col], offset, 32));
            }
        }

#pragma unroll
        for (int k0 = 0; k0 < FATTN_REL_NBATCH_FA; k0 += np * T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np) * T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
                    const int KQ_idx = l % 2;
                    KQ_C[k0 / (np * T_C_KQ::I)].x[l] = expf(KQ_C[k0 / (np * T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0 / (np * T_C_KQ::I)].x[l];
                } else {
                    KQ_C[k0 / (np * T_C_KQ::I)].x[l] = 0.0f;
                }
            }
        }
    } else {
        // cols_per_warp == 16
        if (warp_all_far) {
#pragma unroll
            for (int i00 = 0; i00 < FATTN_REL_NBATCH_FA; i00 += np * T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np) * T_C_KQ::J;
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i2  = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int col = (l0 / 2) % 2;

                    float m0 = far_bias[col];
                    float m1 = far_bias[col];
                    if (mask_h) {
                        const int j = ((threadIdx.y / np) * cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;
                        const float2 tmp = __half22float2(((const half2 *) tile_mask)[j * (FATTN_REL_NBATCH_FA / 2 + 4) + i2]);
                        m0 += tmp.x;
                        m1 += tmp.y;
                    }
                    KQ_C[i00 / (np * T_C_KQ::J)].x[l0 + 0] += m0;
                    KQ_C[i00 / (np * T_C_KQ::J)].x[l0 + 1] += m1;
                }
            }
        } else {
#pragma unroll
            for (int i00 = 0; i00 < FATTN_REL_NBATCH_FA; i00 += np * T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np) * T_C_KQ::J;
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i2  = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j   = ((threadIdx.y / np) * cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;
                    const int col = (l0 / 2) % 2;

                    float m0 = 0.0f;
                    float m1 = 0.0f;
                    if (mask_h) {
                        const float2 tmp = __half22float2(((const half2 *) tile_mask)[j * (FATTN_REL_NBATCH_FA / 2 + 4) + i2]);
                        m0 = tmp.x;
                        m1 = tmp.y;
                    }
                    if (rel_row[col]) {
                        const int32_t kp0   = tile_k_pos[2 * i2 + 0];
                        const int32_t kp1   = tile_k_pos[2 * i2 + 1];
                        const int64_t dist0 = (kp0 == INT32_MIN) ? -1 : (int64_t) q_pos_col[col] - (int64_t) kp0;
                        const int64_t dist1 = (kp1 == INT32_MIN) ? -1 : (int64_t) q_pos_col[col] - (int64_t) kp1;
                        const float   rb0   = (dist0 >= 0 && dist0 < rel_extent) ? rel_row[col][(int32_t) dist0] : far_bias[col];
                        const float   rb1   = (dist1 >= 0 && dist1 < rel_extent) ? rel_row[col][(int32_t) dist1] : far_bias[col];
                        m0 += rb0;
                        m1 += rb1;
                    }
                    KQ_C[i00 / (np * T_C_KQ::J)].x[l0 + 0] += m0;
                    KQ_C[i00 / (np * T_C_KQ::J)].x[l0 + 1] += m1;
                }
            }
        }

#pragma unroll
        for (int k0 = 0; k0 < FATTN_REL_NBATCH_FA; k0 += np * T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np) * T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
                    const int KQ_idx = (l / 2) % 2;
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0 / (np * T_C_KQ::J)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < 2; ++col) {
            KQ_max_new[col] = fmaxf(KQ_max_new[col], ggml_cuda_shfl_xor(KQ_max_new[col], 2, 32));
            KQ_max_new[col] = fmaxf(KQ_max_new[col], ggml_cuda_shfl_xor(KQ_max_new[col], 1, 32));
        }

#pragma unroll
        for (int k0 = 0; k0 < FATTN_REL_NBATCH_FA; k0 += np * T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np) * T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
                    const int KQ_idx = (l / 2) % 2;
                    KQ_C[k0 / (np * T_C_KQ::J)].x[l] = expf(KQ_C[k0 / (np * T_C_KQ::J)].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0 / (np * T_C_KQ::J)].x[l];
                } else {
                    KQ_C[k0 / (np * T_C_KQ::J)].x[l] = 0.0f;
                }
            }
        }
    }

    // Rescale previous VKQ accumulators
    {
        float KQ_max_scale[2];
#pragma unroll
        for (int col = 0; col < 2; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];
            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;
            KQ_rowsum[col] = KQ_max_scale[col] * KQ_rowsum[col] + KQ_rowsum_add[col];
        }

        if constexpr (fp32_vkq) {
            if constexpr (cols_per_warp == 8) {
#pragma unroll
                for (int i = 0; i < FATTN_REL_DV / T_C_VKQ::I; ++i) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        VKQ_C[i].x[l] *= KQ_max_scale[l % 2];
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < FATTN_REL_DV / T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        VKQ_C[i].x[l] *= KQ_max_scale[(l / 2) % 2];
                    }
                }
            }
        } else if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < FATTN_REL_DV / T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < 2; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (FATTN_REL_DV / 2) / T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
    }

    // Pack KQ softmax weights into B registers for P * V
    T_B_VKQ B[FATTN_REL_NBATCH_FA / (np * 2 * T_B_VKQ::J)];
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < FATTN_REL_NBATCH_FA / (np * 2 * T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
#pragma unroll
        for (int k = 0; k < FATTN_REL_NBATCH_FA / (np * 2 * T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

    // --- Stage D: Prepare V tile; overlap next iter's K + mask + k_pos if 2-stage ---
    if constexpr (use_cp_async_2stage) {
        cp_async_wait_all();
        __syncthreads(); // All warps done reading tile_mask, tile_k_pos, tile_K; V is ready

        if constexpr (!last_iter) {
            fattn_rel_load_mask<ncols1, true, false>(mask_h, tile_mask, stride_mask, kb0_next * FATTN_REL_NBATCH_FA, k_VKQ_sup, jt * ncols1, ne01);
            fattn_rel_load_k_pos<true, false>(k_pos_seq, tile_k_pos, kb0_next * FATTN_REL_NBATCH_FA, k_VKQ_sup);
            if constexpr (type_K == GGML_TYPE_F16) {
                fattn_rel_load_kv_f16<true, false>(K_bytes, tile_K, stride_K_bytes, kb0_next * FATTN_REL_NBATCH_FA, k_VKQ_sup, indices);
            } else if constexpr (type_K == GGML_TYPE_F8_E4M3) {
                fattn_rel_load_kv_f8_raw_async(K_bytes, tile_raw_K, stride_K_bytes, kb0_next * FATTN_REL_NBATCH_FA, indices);
            }
        }

        if constexpr (type_V == GGML_TYPE_F8_E4M3) {
            fattn_rel_convert_f8_raw_to_tile(tile_raw_V, tile_V);
            __syncthreads();
        }
    } else {
        __syncthreads(); // Done reading tile_K (which aliases tile_V in 1-stage)
        if constexpr (type_V == GGML_TYPE_F16) {
            fattn_rel_load_kv_f16<false, oob_check>(V_bytes, tile_V, stride_V_bytes, k_VKQ_0, k_VKQ_sup, indices);
        } else if constexpr (type_V == GGML_TYPE_F8_E4M3) {
            fattn_rel_load_kv_f8_sync<oob_check>(V_bytes, tile_V, stride_V_bytes, k_VKQ_0, k_VKQ_sup, indices);
        } else {
            fattn_rel_load_kv_q8_0<oob_check>(V_bytes, tile_V, stride_V_bytes, k_VKQ_0, k_VKQ_sup, indices);
        }
        __syncthreads();
    }

    // --- Stage E: P * V MMA ---
#pragma unroll
    for (int i_VKQ_0 = 0; i_VKQ_0 < FATTN_REL_DV; i_VKQ_0 += T_A_VKQ::I) {
#pragma unroll
        for (int k00 = 0; k00 < FATTN_REL_NBATCH_FA / 2; k00 += np * T_A_VKQ::J) {
            const int k0 = k00 + (threadIdx.y % np) * T_A_VKQ::J;
            T_A_VKQ A;
            load_ldmatrix_trans_swizzled<FATTN_REL_STRIDE_TILE_KV>(A, tile_V, 2 * k0 * FATTN_REL_STRIDE_TILE_KV + i_VKQ_0 / 2);
            if constexpr (cols_per_warp == 8) {
                mma(VKQ_C[i_VKQ_0 / T_A_VKQ::I], A, B[k00 / (np * T_A_VKQ::J)]);
            } else {
                mma(VKQ_C[i_VKQ_0 / T_A_VKQ::I], B[k00 / (np * T_A_VKQ::J)], A);
            }
        }
    }

    if constexpr (!use_cp_async_2stage) {
        __syncthreads(); // Done reading tile_V before next iter overwrites tile_K
    }
#else
    GGML_UNUSED_VARS(K_bytes, V_bytes, mask_h, indices, k_pos_seq, q_pos_col, rel_row, far_bias, rel_extent,
        ne01, stride_K_bytes, stride_V_bytes, stride_mask, tile_K, tile_V, tile_raw_K, tile_raw_V, tile_mask, tile_k_pos,
        Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, kb0_next, k_VKQ_sup);
    NO_DEVICE_CODE;
#endif
}

// ------------------------------------------------------------------------------------------------------------------
// Process one output tile (jt, zt_gqa, z_KV, sequence) across [kb0_start, kb0_stop)
// ------------------------------------------------------------------------------------------------------------------
template <int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool kv_aligned, bool needs_fixup, bool is_fixup, bool fp32_vkq>
static __device__ __forceinline__ void fattn_rel_process_tile(
        const float2  * const __restrict__ Q_f2,
        const char    * const __restrict__ K_bytes,
        const char    * const __restrict__ V_bytes,
        const half    * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        const float   * const __restrict__ sinks_f,
        const float   * const __restrict__ rel_table_seq,
        const int32_t * const __restrict__ k_pos_seq,
        const int32_t * const __restrict__ q_pos_seq,
        float2        * const __restrict__ dstk,
        float2        * const __restrict__ dstk_fixup,
        const float   scale,
        const uint3   ne01,
        const int     ne02,
        const int     gqa_ratio,
        const int     ne11,
        const int     stride_Q1,
        const int     stride_Q2,
        const int     stride_K_bytes,
        const int     stride_V_bytes,
        const int     stride_mask,
        const int     rel_extent,
        const int64_t rel_s1,
        const int64_t rel_s2,
        const int     jt,
        const int     zt_gqa,
        const int     zt_Q,
        const int     kb0_start,
        const int     kb0_stop,
        const int     kb0_max) {
#if defined(TURING_MMA_AVAILABLE)
    constexpr int ncols = ncols1 * ncols2;
    using T_A_KQ  = typename fattn_rel_mma_tiles<ncols, fp32_vkq>::T_A_KQ;
    using T_B_KQ  = typename fattn_rel_mma_tiles<ncols, fp32_vkq>::T_B_KQ;
    using T_C_KQ  = typename fattn_rel_mma_tiles<ncols, fp32_vkq>::T_C_KQ;
    using T_C_VKQ = typename fattn_rel_mma_tiles<ncols, fp32_vkq>::T_C_VKQ;

    constexpr int cols_per_warp = T_B_KQ::I;
    constexpr int np            = FATTN_REL_NWARPS * cols_per_warp / ncols;
    constexpr bool kv_supports_2stage = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_F8_E4M3) &&
                                        (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_F8_E4M3);
#ifdef CP_ASYNC_AVAILABLE
    constexpr bool use_cp_async_2stage = kv_supports_2stage && kv_aligned;
#else
    constexpr bool use_cp_async_2stage = false;
#endif
    constexpr bool alias_kv_converted = (!use_cp_async_2stage) ||
                                        (type_K == GGML_TYPE_F8_E4M3 && type_V == GGML_TYPE_F8_E4M3);

    // Shared memory layout (all 16-byte aligned)
    extern __shared__ half2 tile_Q[];
    half2 * tile_K = tile_Q; // Q_in_reg = true, so tile_K aliases tile_Q after Q is loaded to registers
    half2 * tile_V = alias_kv_converted ? tile_K : (tile_K + FATTN_REL_NBATCH_FA * FATTN_REL_STRIDE_TILE_KV);
    char  * smem_after_kv = (char *)(tile_V + FATTN_REL_NBATCH_FA * FATTN_REL_STRIDE_TILE_KV);

    uint8_t * tile_raw_K = nullptr;
    uint8_t * tile_raw_V = nullptr;
    if constexpr (use_cp_async_2stage && type_K == GGML_TYPE_F8_E4M3) {
        tile_raw_K = (uint8_t *) smem_after_kv;
        smem_after_kv += FATTN_REL_NBATCH_FA * FATTN_REL_DKQ;
    }
    if constexpr (use_cp_async_2stage && type_V == GGML_TYPE_F8_E4M3) {
        tile_raw_V = (uint8_t *) smem_after_kv;
        smem_after_kv += FATTN_REL_NBATCH_FA * FATTN_REL_DV;
    }
    half    * tile_mask  = (half *) smem_after_kv;
    int32_t * tile_k_pos = (int32_t *)(tile_mask + ncols1 * (FATTN_REL_NBATCH_FA + 8));

    T_B_KQ  Q_B[FATTN_REL_DKQ / (2 * T_B_KQ::J)];
    T_C_VKQ VKQ_C[FATTN_REL_DV / 16];

    float KQ_rowsum[2] = { 0.0f, 0.0f };
    float KQ_max[2]    = { -FLT_MAX / 2.0f, -FLT_MAX / 2.0f };

    // Load Q into tile_Q (scaled by scale_h2)
    const half2 scale_h2 = make_half2(scale, scale);
    {
        // DKQ/2 == 64 half2s: 32 threads x 2 iters per row
#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += FATTN_REL_NWARPS) {
            const int jc = jc0 + threadIdx.y;
            if (jc0 + FATTN_REL_NWARPS > ncols && jc >= ncols) {
                break;
            }
            const int j = jc / ncols2;
            const int c = jc % ncols2;
            const bool valid = (ncols1 == 1 || jt * ncols1 + j < int(ne01.z)) &&
                               (ncols2 == 1 || zt_gqa * ncols2 + c < gqa_ratio);
#pragma unroll
            for (int k0 = 0; k0 < FATTN_REL_DKQ / 2; k0 += 32) {
                const int k = k0 + threadIdx.x;
                if (valid) {
                    const float2 tmp = Q_f2[(jt * ncols1 + j) * stride_Q1 + c * stride_Q2 + k];
                    tile_Q[jc * FATTN_REL_STRIDE_TILE_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                } else {
                    tile_Q[jc * FATTN_REL_STRIDE_TILE_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }
    __syncthreads();

    // Load Q into MMA registers
    {
        const int j0 = (threadIdx.y / np) * cols_per_warp;
#pragma unroll
        for (int k0 = 0; k0 < FATTN_REL_DKQ / 2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0 / T_B_KQ::J], tile_Q + j0 * FATTN_REL_STRIDE_TILE_Q + k0, FATTN_REL_STRIDE_TILE_Q);
        }
    }
    __syncthreads();

    // Preload per-thread query positions, rel_table row pointers, and far_bias (-sink(q, h)) into registers ONCE!
    int32_t       q_pos_col[2] = { 0, 0 };
    const float * rel_row[2]   = { nullptr, nullptr };
    float         far_bias[2]  = { 0.0f, 0.0f };
    if (rel_table_seq) {
#pragma unroll
        for (int col = 0; col < 2; ++col) {
            const int jc = (threadIdx.y / np) * cols_per_warp +
                (cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2 * col));
            const int j_local = jc / ncols2;
            const int c_local = jc % ncols2;
            const int q_tok   = jt * ncols1 + j_local;
            const int q_head  = zt_Q + c_local;
            if (q_tok < int(ne01.z) && zt_gqa * ncols2 + c_local < gqa_ratio) {
                q_pos_col[col] = q_pos_seq[q_tok];
                rel_row[col]   = rel_table_seq + (int64_t) q_tok * rel_s2 + (int64_t) q_head * rel_s1;
                far_bias[col]  = rel_row[col][rel_extent];
            }
        }
    }

    // KV block loop
    const int iter_k   = (ne11 + FATTN_REL_NBATCH_FA - 1) / FATTN_REL_NBATCH_FA;
    const int nkb_pre  = max(0, min(kb0_stop, kb0_max) - kb0_start);
    const int nkb_post = max(0, min(kb0_stop - iter_k, kb0_max));
    const int nkb      = max(1, nkb_pre + nkb_post);
    int       kb0      = (use_cp_async_2stage && nkb_pre == 0 && nkb_post > 0) ? 0 : kb0_start;

    if constexpr (use_cp_async_2stage) {
        // Preload iter 0: mask + k_pos + K
        fattn_rel_load_mask<ncols1, true, false>(mask_h, tile_mask, stride_mask, kb0 * FATTN_REL_NBATCH_FA, FATTN_REL_NBATCH_FA, jt * ncols1, ne01);
        fattn_rel_load_k_pos<true, false>(k_pos_seq, tile_k_pos, kb0 * FATTN_REL_NBATCH_FA, FATTN_REL_NBATCH_FA);
        if constexpr (type_K == GGML_TYPE_F16) {
            fattn_rel_load_kv_f16<true, false>(K_bytes, tile_K, stride_K_bytes, kb0 * FATTN_REL_NBATCH_FA, FATTN_REL_NBATCH_FA, indices);
        } else if constexpr (type_K == GGML_TYPE_F8_E4M3) {
            fattn_rel_load_kv_f8_raw_async(K_bytes, tile_raw_K, stride_K_bytes, kb0 * FATTN_REL_NBATCH_FA, indices);
        }

        for (int ikb = 1; ikb < nkb; ++ikb) {
            const int kb0_next = (kb0 + 1 == kb0_max) ? 0 : (kb0 + 1);
            fattn_rel_iter<ncols1, ncols2, type_K, type_V, true, false, false, fp32_vkq>(
                K_bytes, V_bytes, mask_h, indices, k_pos_seq,
                q_pos_col, rel_row, far_bias, rel_extent,
                ne01, stride_K_bytes, stride_V_bytes, stride_mask,
                tile_K, tile_V, tile_raw_K, tile_raw_V, tile_mask, tile_k_pos,
                Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, kb0_next, FATTN_REL_NBATCH_FA);
            kb0 = kb0_next;
        }
        fattn_rel_iter<ncols1, ncols2, type_K, type_V, true, true, false, fp32_vkq>(
            K_bytes, V_bytes, mask_h, indices, k_pos_seq,
            q_pos_col, rel_row, far_bias, rel_extent,
            ne01, stride_K_bytes, stride_V_bytes, stride_mask,
            tile_K, tile_V, tile_raw_K, tile_raw_V, tile_mask, tile_k_pos,
            Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, kb0, FATTN_REL_NBATCH_FA);
    } else if constexpr (!kv_aligned) {
        for (; kb0 < kb0_stop - 1; ++kb0) {
            fattn_rel_iter<ncols1, ncols2, type_K, type_V, false, false, true, fp32_vkq>(
                K_bytes, V_bytes, mask_h, indices, k_pos_seq,
                q_pos_col, rel_row, far_bias, rel_extent,
                ne01, stride_K_bytes, stride_V_bytes, stride_mask,
                tile_K, tile_V, tile_raw_K, tile_raw_V, tile_mask, tile_k_pos,
                Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, kb0 + 1, FATTN_REL_NBATCH_FA);
        }
        const int k_VKQ_sup = ne11 - kb0 * FATTN_REL_NBATCH_FA;
        fattn_rel_iter<ncols1, ncols2, type_K, type_V, false, true, true, fp32_vkq>(
            K_bytes, V_bytes, mask_h, indices, k_pos_seq,
            q_pos_col, rel_row, far_bias, rel_extent,
            ne01, stride_K_bytes, stride_V_bytes, stride_mask,
            tile_K, tile_V, tile_raw_K, tile_raw_V, tile_mask, tile_k_pos,
            Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, kb0, k_VKQ_sup);
    } else {
        for (; kb0 < kb0_stop - 1; ++kb0) {
            fattn_rel_iter<ncols1, ncols2, type_K, type_V, false, false, false, fp32_vkq>(
                K_bytes, V_bytes, mask_h, indices, k_pos_seq,
                q_pos_col, rel_row, far_bias, rel_extent,
                ne01, stride_K_bytes, stride_V_bytes, stride_mask,
                tile_K, tile_V, tile_raw_K, tile_raw_V, tile_mask, tile_k_pos,
                Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, kb0 + 1, FATTN_REL_NBATCH_FA);
        }
        fattn_rel_iter<ncols1, ncols2, type_K, type_V, false, true, false, fp32_vkq>(
            K_bytes, V_bytes, mask_h, indices, k_pos_seq,
            q_pos_col, rel_row, far_bias, rel_extent,
            ne01, stride_K_bytes, stride_V_bytes, stride_mask,
            tile_K, tile_V, tile_raw_K, tile_raw_V, tile_mask, tile_k_pos,
            Q_B, VKQ_C, KQ_max, KQ_rowsum, jt, kb0, kb0, FATTN_REL_NBATCH_FA);
    }

    if constexpr (use_cp_async_2stage) {
        __syncthreads();
    }

    // Reduce KQ_rowsum across warp
    {
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#pragma unroll
        for (int col = 0; col < 2; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += ggml_cuda_shfl_xor(KQ_rowsum[col], offset, 32);
            }
        }
    }

    // Apply attention sinks (for Inkling-2, sinks_f is 0.0f and -sink(q, h) is baked into rel_table)
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[2];
#pragma unroll
        for (int col = 0; col < 2; ++col) {
            const int jc      = (threadIdx.y / np) * cols_per_warp +
                                (cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2 * col));
            const int c_local = jc % ncols2;
            const bool valid_head = (ncols2 == 1) || (zt_gqa * ncols2 + c_local < gqa_ratio);
            if (!valid_head) {
                KQ_max_scale[col] = 1.0f;
                continue;
            }
            const float sink = sinks_f[c_local];

            const float KQ_max_new  = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col]       = KQ_max_new;
            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col] * KQ_rowsum[col] + KQ_max_add;
        }

        if constexpr (fp32_vkq) {
            if constexpr (cols_per_warp == 8) {
#pragma unroll
                for (int i = 0; i < FATTN_REL_DV / T_C_VKQ::I; ++i) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        VKQ_C[i].x[l] *= KQ_max_scale[l % 2];
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < FATTN_REL_DV / T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        VKQ_C[i].x[l] *= KQ_max_scale[(l / 2) % 2];
                    }
                }
            }
        } else if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < FATTN_REL_DV / T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < 2; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (FATTN_REL_DV / 2) / T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
    }

    // Combine VKQ across parallel warps (np) in FP32 via shared memory & write to dstk / dstk_fixup.
    // Keeping combine partials in FP32 avoids FP16 overflow on unnormalized sums before rowsum division.
    constexpr int tile_stride_f2 = FATTN_REL_NBATCH_COMBINE + 2; // 64 float2s for DV=128 + metadata slot at index 64
    float2 * const tile_comb     = (float2 *) tile_Q;
    using T_C_VKQ_H2 = tile<16, cols_per_warp == 8 ? 4 : 8, half2>;
    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2 * T_C_VKQ_H2::J)) / T_C_VKQ_H2::J;
        const int jc_cwm  = threadIdx.y * (2 * T_C_VKQ_H2::J) + 2 * T_C_VKQ_H2::get_j(-1) + jc_cwmo;
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]);
        __syncthreads();
        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2 * T_C_VKQ_H2::J) {
            tile_comb[jc_cwm * tile_stride_f2 + FATTN_REL_NBATCH_COMBINE] = KQ_cmr;
        }
        __syncthreads();
        if (np == 1) {
            if (needs_fixup && threadIdx.x < T_B_KQ::I) {
                dstk_fixup[blockIdx.x * ncols + jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < T_B_KQ::I) {
                dstk_fixup[(gridDim.x + blockIdx.x) * ncols + jc_cwm] = KQ_cmr;
            }
        }
    } else {
        const int jc_cwm = threadIdx.y * cols_per_warp + T_C_VKQ_H2::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % 2], KQ_rowsum[threadIdx.x % 2]);
        const bool thread_should_write = (threadIdx.x % 4) < 2;
        __syncthreads();
        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            tile_comb[jc_cwm * tile_stride_f2 + FATTN_REL_NBATCH_COMBINE] = KQ_cmr;
        }
        __syncthreads();
        if (np == 1) {
            if (needs_fixup && thread_should_write) {
                dstk_fixup[blockIdx.x * ncols + jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                dstk_fixup[(gridDim.x + blockIdx.x) * ncols + jc_cwm] = KQ_cmr;
            }
        }
    }

    if (np > 1) {
        constexpr int nmeta = (np * cols_per_warp >= 32) ? (np * cols_per_warp / 32) : 1;
        float KQ_cmn;
        float KQ_cms[nmeta];
        float KQ_crs;
        const int jc_meta = threadIdx.y * cols_per_warp + (np * cols_per_warp < 32 ? threadIdx.x % (np * cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = tile_comb + jc_meta * tile_stride_f2 + FATTN_REL_NBATCH_COMBINE;

        if (threadIdx.y % np == 0) {
            float2 meta[nmeta];
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                meta[imeta] = meta_ptr[imeta * 32 * tile_stride_f2];
            }
            KQ_cmn = meta[0].x;
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
            }
#pragma unroll
            for (int offset = np * cols_per_warp / 2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < 32) {
                    KQ_cmn = fmaxf(KQ_cmn, ggml_cuda_shfl_xor(KQ_cmn, offset, 32));
                }
            }
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
            }
            KQ_crs = KQ_cms[0] * meta[0].y;
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_crs += KQ_cms[imeta] * meta[imeta].y;
            }
#pragma unroll
            for (int offset = np * cols_per_warp / 2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < 32) {
                    KQ_crs += ggml_cuda_shfl_xor(KQ_crs, offset, 32);
                }
            }
        }
        __syncthreads();
        if (threadIdx.y % np == 0) {
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                if (np * cols_per_warp >= 32 || threadIdx.x < np * cols_per_warp) {
                    meta_ptr[imeta * 32 * tile_stride_f2] = make_float2(KQ_cms[imeta], KQ_crs);
                }
            }
            if (needs_fixup && threadIdx.x < cols_per_warp) {
                dstk_fixup[blockIdx.x * ncols + (threadIdx.y / np) * cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
            if (is_fixup && threadIdx.x < cols_per_warp) {
                dstk_fixup[(gridDim.x + blockIdx.x) * ncols + (threadIdx.y / np) * cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
        }
    }

    // Write FP32 VKQ_C directly to tile_comb (DV/2 == 64 == FATTN_REL_NBATCH_COMBINE -> single pass, zero half narrowing!)
    if constexpr (cols_per_warp == 8) {
        float * const tile_comb_f = (float *) tile_comb;
        const int j0 = threadIdx.y * cols_per_warp;
#pragma unroll
        for (int k1 = 0; k1 < FATTN_REL_NBATCH_COMBINE; k1 += T_B_KQ::J) {
            const auto & V_tile = VKQ_C[k1 / T_B_KQ::J];
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                const int j = j0 + T_C_VKQ::get_j(l);
                const int d = 2 * k1 + T_C_VKQ::get_i(l);
                tile_comb_f[j * (2 * tile_stride_f2) + d] = V_tile.x[l];
            }
        }
    } else {
        const int j0 = threadIdx.y * cols_per_warp;
#pragma unroll
        for (int k1 = 0; k1 < FATTN_REL_NBATCH_COMBINE; k1 += T_C_VKQ_H2::J) {
            const auto & V_tile = VKQ_C[k1 / T_C_VKQ_H2::J];
#pragma unroll
            for (int l = 0; l < T_C_VKQ_H2::ne; ++l) {
                const int j = j0 + T_C_VKQ_H2::get_i(l);
                const int k = k1 + T_C_VKQ_H2::get_j(l);
                tile_comb[j * tile_stride_f2 + k] = make_float2(V_tile.x[2 * l + 0], V_tile.x[2 * l + 1]);
            }
        }
    }
    __syncthreads();

    if (np == 1 || threadIdx.y % np == 0) {
        float2 * dstk_fixup_data = dstk_fixup + gridDim.x * (2 * ncols) + blockIdx.x * (ncols * (FATTN_REL_DV / 2));
#pragma unroll
        for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (FATTN_REL_NWARPS / np)) {
            const int jc_dst = jc0_dst + (threadIdx.y / np);
            if (jc0_dst + (FATTN_REL_NWARPS / np) > ncols && jc_dst >= ncols) {
                break;
            }
            const int jc_tile_K = (jc_dst / cols_per_warp) * (np * cols_per_warp) + jc_dst % cols_per_warp;
            const int j_dst     = jc_dst / ncols2;
            const int c_dst     = jc_dst % ncols2;
            if (!is_fixup && ((ncols1 > 1 && jt * ncols1 + j_dst >= int(ne01.z)) ||
                              (ncols2 > 1 && zt_gqa * ncols2 + c_dst >= gqa_ratio))) {
                continue;
            }
            const float2 * meta_j = tile_comb + jc_tile_K * tile_stride_f2 + FATTN_REL_NBATCH_COMBINE;
#pragma unroll
            for (int k0 = 0; k0 < FATTN_REL_NBATCH_COMBINE; k0 += 32) {
                const int k = k0 + threadIdx.x;
                float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                for (int ip = 0; ip < np; ++ip) {
                    const float  KQ_crs       = (np == 1) ? 1.0f : meta_j[ip * cols_per_warp * tile_stride_f2].x;
                    const float2 dstk_val_add = tile_comb[(jc_tile_K + ip * cols_per_warp) * tile_stride_f2 + k];
                    dstk_val.x += dstk_val_add.x * KQ_crs;
                    dstk_val.y += dstk_val_add.y * KQ_crs;
                }
                if (!needs_fixup && !is_fixup) {
                    const float sum     = meta_j[0].y;
                    const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                    dstk_val.x *= inv_sum;
                    dstk_val.y *= inv_sum;
                }
                if (is_fixup) {
                    dstk_fixup_data[jc_dst * (FATTN_REL_DV / 2) + k] = dstk_val;
                } else {
                    dstk[((jt * ncols1 + j_dst) * ne02 + c_dst) * (FATTN_REL_DV / 2) + k] = dstk_val;
                }
            }
        }
    }
    // End-of-tile barrier: prevent non-reducer warps (when np > 1) from overwriting tile_Q on the next tile
    // before reducer warps finish reading tile_comb.
    __syncthreads();
#else
    GGML_UNUSED_VARS(Q_f2, K_bytes, V_bytes, mask_h, indices, sinks_f, rel_table_seq, k_pos_seq, q_pos_seq,
        dstk, dstk_fixup, scale, ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K_bytes, stride_V_bytes,
        stride_mask, rel_extent, rel_s1, rel_s2, jt, zt_gqa, zt_Q, kb0_start, kb0_stop, kb0_max);
    NO_DEVICE_CODE;
#endif
}

// ------------------------------------------------------------------------------------------------------------------
// Stream-K FlashAttention Kernel Entry Point
// ------------------------------------------------------------------------------------------------------------------
template <int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool kv_aligned, bool fp32_vkq>
__launch_bounds__(FATTN_REL_NTHREADS, 2)
static __global__ void flash_attn_ext_mma_f16_rel_kernel(
        const char    * Q_ptr,
        const char    * K_ptr,
        const char    * V_ptr,
        const char    * mask_ptr,
        const char    * sinks_ptr,
        const int     * KV_max_ptr,
        const int     * KV_rows_ptr,
        const float   * rel_table_ptr,
        const int32_t * k_pos_ptr,
        const int32_t * q_pos_ptr,
        float         * dst_ptr,
        float2        * dst_meta_ptr,
        const float     scale,
        const int32_t   rel_extent,
        const int64_t   rel_s1,
        const int64_t   rel_s2,
        const int64_t   rel_s3,
        const uint3     ne01, const int32_t ne02, const int32_t ne03,
        const int32_t   nb01, const int32_t nb02, const int32_t nb03,
        const int32_t   ne11, const int32_t ne12,
        const int32_t   nb11, const int32_t nb12, const int64_t nb13,
        const int32_t   nb21, const int32_t nb22, const int64_t nb23,
        const int32_t   ne32, const int32_t ne33,
        const int32_t   nb31, const int64_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_sync();
#if defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE)
    const int gqa_ratio   = ne02 / ne12;
    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_mask = nb31 / sizeof(half);

    const int iter_k     = (ne11      + (FATTN_REL_NBATCH_FA - 1)) / FATTN_REL_NBATCH_FA;
    const int iter_j     = (ne01.z    + (ncols1          - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2          - 1)) / ncols2;

    int       kbc      = int64_t(blockIdx.x + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / gridDim.x;

    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    constexpr bool kv_supports_2stage = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_F8_E4M3) &&
                                        (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_F8_E4M3);
#ifdef CP_ASYNC_AVAILABLE
    constexpr bool rotate = kv_supports_2stage && kv_aligned;
#else
    constexpr bool rotate = false;
#endif

    const int kbc0     = kbc;
    int       kbc_tile = kbc;
    if (rotate && kb0_start != 0) {
        const int64_t nkbc    = int64_t(iter_k) * iter_j * iter_z_gqa * ne12 * ne03;
        const int64_t b_first = ((kbc - kb0_start + 1) * int64_t(gridDim.x) - 1) / nkbc;
        kbc_tile = b_first * nkbc / gridDim.x;
    }

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int sequence =  kbc / (iter_k * iter_j * iter_z_gqa * ne12);
        const int z_KV     = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence) / (iter_k * iter_j * iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence - iter_k * iter_j * iter_z_gqa * z_KV) / (iter_k * iter_j);
        const int jt       = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence - iter_k * iter_j * iter_z_gqa * z_KV - iter_k * iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2;

        const float2  * Q_f2    = (const float2 *)(Q_ptr + nb03 * sequence + nb02 * zt_Q);
        const char    * K_bytes = K_ptr + nb13 * sequence + nb12 * z_KV;
        const char    * V_bytes = V_ptr + nb23 * sequence + nb22 * z_KV;
        const half    * mask_h  = mask_ptr ? (const half *)(mask_ptr + nb33 * (sequence % ne33) + nb32 * (zt_Q % ne32)) : nullptr;
        const float   * sinks_f = sinks_ptr ? (const float *) sinks_ptr + zt_Q : nullptr;
        const int32_t * indices = KV_rows_ptr ? KV_rows_ptr + int64_t(sequence) * ne11 : nullptr;
        const float   * rel_seq = rel_table_ptr ? rel_table_ptr + int64_t(sequence) * rel_s3 : nullptr;
        const int32_t * kp_seq  = k_pos_ptr ? k_pos_ptr + int64_t(sequence) * ne11 : nullptr;
        const int32_t * qp_seq  = q_pos_ptr ? q_pos_ptr + int64_t(sequence) * ne01.z : nullptr;
        float2        * dstk    = ((float2 *) dst_ptr) + (sequence * ne01.z * ne02 + zt_Q) * (FATTN_REL_DV / 2);

        if (KV_max_ptr && !rotate) {
            kb0_stop = min(kb0_stop, KV_max_ptr[sequence * iter_j + jt] / FATTN_REL_NBATCH_FA);
        }
        const int kb0_max = (rotate && KV_max_ptr) ? (KV_max_ptr[sequence * iter_j + jt] / FATTN_REL_NBATCH_FA) : iter_k;
        const int kb0_rot = rotate ? ((kbc - kbc_tile) % iter_k) : kb0_start;

        if (kb0_start == 0) {
            fattn_rel_process_tile<ncols1, ncols2, type_K, type_V, kv_aligned, false, false, fp32_vkq>(
                Q_f2, K_bytes, V_bytes, mask_h, indices, sinks_f, rel_seq, kp_seq, qp_seq,
                dstk, dst_meta_ptr, scale, ne01, ne02, gqa_ratio, ne11,
                stride_Q1, stride_Q2, nb11, nb21, stride_mask,
                rel_extent, rel_s1, rel_s2, jt, zt_gqa, zt_Q,
                kb0_rot, kb0_rot + kb0_stop - kb0_start, kb0_max);
        } else {
            fattn_rel_process_tile<ncols1, ncols2, type_K, type_V, kv_aligned, true, false, fp32_vkq>(
                Q_f2, K_bytes, V_bytes, mask_h, indices, sinks_f, rel_seq, kp_seq, qp_seq,
                dstk, dst_meta_ptr, scale, ne01, ne02, gqa_ratio, ne11,
                stride_Q1, stride_Q2, nb11, nb21, stride_mask,
                rel_extent, rel_s1, rel_s2, jt, zt_gqa, zt_Q,
                kb0_rot, kb0_rot + kb0_stop - kb0_start, kb0_max);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;
        kbc_tile  = kbc0;
        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int sequence =  kbc / (iter_k * iter_j * iter_z_gqa * ne12);
    const int z_KV     = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence) / (iter_k * iter_j * iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence - iter_k * iter_j * iter_z_gqa * z_KV) / (iter_k * iter_j);
    const int jt       = (kbc - iter_k * iter_j * iter_z_gqa * ne12 * sequence - iter_k * iter_j * iter_z_gqa * z_KV - iter_k * iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2;

    const float2  * Q_f2    = (const float2 *)(Q_ptr + nb03 * sequence + nb02 * zt_Q);
    const char    * K_bytes = K_ptr + nb13 * sequence + nb12 * z_KV;
    const char    * V_bytes = V_ptr + nb23 * sequence + nb22 * z_KV;
    const half    * mask_h  = mask_ptr ? (const half *)(mask_ptr + nb33 * (sequence % ne33) + nb32 * (zt_Q % ne32)) : nullptr;
    const float   * sinks_f = sinks_ptr ? (const float *) sinks_ptr + zt_Q : nullptr;
    const int32_t * indices = KV_rows_ptr ? KV_rows_ptr + int64_t(sequence) * ne11 : nullptr;
    const float   * rel_seq = rel_table_ptr ? rel_table_ptr + int64_t(sequence) * rel_s3 : nullptr;
    const int32_t * kp_seq  = k_pos_ptr ? k_pos_ptr + int64_t(sequence) * ne11 : nullptr;
    const int32_t * qp_seq  = q_pos_ptr ? q_pos_ptr + int64_t(sequence) * ne01.z : nullptr;
    float2        * dstk    = ((float2 *) dst_ptr) + (sequence * ne01.z * ne02 + zt_Q) * (FATTN_REL_DV / 2);

    if (KV_max_ptr && !rotate) {
        kb0_stop = min(kb0_stop, KV_max_ptr[sequence * iter_j + jt] / FATTN_REL_NBATCH_FA);
    }
    const int kb0_max = (rotate && KV_max_ptr) ? (KV_max_ptr[sequence * iter_j + jt] / FATTN_REL_NBATCH_FA) : iter_k;
    const int kb0_rot = rotate ? ((kbc - kbc_tile) % iter_k) : kb0_start;

    fattn_rel_process_tile<ncols1, ncols2, type_K, type_V, kv_aligned, false, true, fp32_vkq>(
        Q_f2, K_bytes, V_bytes, mask_h, indices, sinks_f, rel_seq, kp_seq, qp_seq,
        dstk, dst_meta_ptr, scale, ne01, ne02, gqa_ratio, ne11,
        stride_Q1, stride_Q2, nb11, nb21, stride_mask,
        rel_extent, rel_s1, rel_s2, jt, zt_gqa, zt_Q,
        kb0_rot, kb0_rot + kb0_stop - kb0_start, kb0_max);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr,
        rel_table_ptr, k_pos_ptr, q_pos_ptr, dst_ptr, dst_meta_ptr,
        scale, rel_extent, rel_s1, rel_s2, rel_s3,
        ne01, ne02, ne03, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb13, nb21, nb22, nb23,
        ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif
}

// ------------------------------------------------------------------------------------------------------------------
// Host Launch Wrapper
// ------------------------------------------------------------------------------------------------------------------

template <int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool kv_aligned, bool fp32_vkq>
static void launch_fattn_mma_f16_rel(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q         = dst->src[0];
    const ggml_tensor * K         = dst->src[1];
    const ggml_tensor * V         = dst->src[2];
    const ggml_tensor * mask      = dst->src[3];
    const ggml_tensor * sinks     = dst->src[4];
    const ggml_tensor * kv_rows   = dst->src[5];
    const ggml_tensor * rel_table = dst->src[6];
    const ggml_tensor * k_pos     = dst->src[7];
    const ggml_tensor * q_pos     = dst->src[8];

    const int64_t n_kv_cols = kv_rows ? kv_rows->ne[0] : K->ne[1];
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool & pool    = ctx.pool();
    cudaStream_t main_stream = ctx.stream();

    // Direct KV types never need F16 scratch; any other or mismatched KV type converts to F16 via extra_data.
    const bool need_f16_K = (K->type != type_K);
    const bool need_f16_V = (V->type != type_V);

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1], nb12 = K->nb[2], nb13 = K->nb[3];
    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1], nb22 = V->nb[2], nb23 = V->nb[3];

    if (need_f16_K) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);
        half * K_f16 = (half *) f16_extra.K;
        if (ggml_is_contiguously_allocated(K)) {
            ggml_get_to_fp16_cuda(K->type)(K_data, K_f16, ggml_nelements(K), main_stream);
            nb11 = nb11 * bs * sizeof(half) / ts;
            nb12 = nb12 * bs * sizeof(half) / ts;
            nb13 = nb13 * bs * sizeof(half) / ts;
        } else {
            ggml_get_to_fp16_nc_cuda(K->type)(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                nb11 / ts, nb12 / ts, nb13 / ts, main_stream);
            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (const char *) K_f16;
    }
    if (need_f16_V) {
        const size_t bs = ggml_blck_size(V->type);
        const size_t ts = ggml_type_size(V->type);
        half * V_f16 = (half *) f16_extra.V;
        if (ggml_is_contiguously_allocated(V)) {
            ggml_get_to_fp16_cuda(V->type)(V_data, V_f16, ggml_nelements(V), main_stream);
            nb21 = nb21 * bs * sizeof(half) / ts;
            nb22 = nb22 * bs * sizeof(half) / ts;
            nb23 = nb23 * bs * sizeof(half) / ts;
        } else {
            ggml_get_to_fp16_nc_cuda(V->type)(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                nb21 / ts, nb22 / ts, nb23 / ts, main_stream);
            nb21 = V->ne[0] * sizeof(half);
            nb22 = V->ne[1] * nb21;
            nb23 = V->ne[2] * nb22;
        }
        V_data = (const char *) V_f16;
    }

    if (kv_rows) {
        nb13 = 0;
        nb23 = 0;
    }

    // Shared memory calculation matching fattn_rel_process_tile
    const bool kv_supports_2stage  = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_F8_E4M3) &&
                                     (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_F8_E4M3);
    const bool use_cp_async_2stage = cp_async_available(cc) && kv_supports_2stage && kv_aligned;
    const bool alias_kv_converted  = (!use_cp_async_2stage) ||
                                     (type_K == GGML_TYPE_F8_E4M3 && type_V == GGML_TYPE_F8_E4M3);

    const size_t nbytes_kv_conv = (alias_kv_converted ? 1 : 2) * FATTN_REL_NBATCH_FA * FATTN_REL_STRIDE_TILE_KV * sizeof(half2);
    const size_t nbytes_raw_k   = (use_cp_async_2stage && type_K == GGML_TYPE_F8_E4M3) ? (FATTN_REL_NBATCH_FA * FATTN_REL_DKQ) : 0;
    const size_t nbytes_raw_v   = (use_cp_async_2stage && type_V == GGML_TYPE_F8_E4M3) ? (FATTN_REL_NBATCH_FA * FATTN_REL_DV)  : 0;
    const size_t nbytes_mask    = ncols1 * (FATTN_REL_NBATCH_FA + 8) * sizeof(half);
    const size_t nbytes_kpos    = FATTN_REL_NBATCH_FA * sizeof(int32_t);
    const size_t nbytes_q       = ncols * FATTN_REL_STRIDE_TILE_Q * sizeof(half2);
    const size_t nbytes_combine = FATTN_REL_NWARPS * std::min(ncols, 16) * (FATTN_REL_NBATCH_COMBINE + 2) * sizeof(float2);
    const size_t nbytes_shared  = std::max({ nbytes_combine, nbytes_q, nbytes_kv_conv + nbytes_raw_k + nbytes_raw_v + nbytes_mask + nbytes_kpos });

    auto kernel = flash_attn_ext_mma_f16_rel_kernel<ncols1, ncols2, type_K, type_V, kv_aligned, fp32_vkq>;
    static bool shared_raised[GGML_CUDA_MAX_DEVICES] = { false };
    if (!shared_raised[id] && nbytes_shared > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared));
        shared_raised[id] = true;
    }

    const int ntiles_x     = (Q->ne[1] + ncols1 - 1) / ncols1;
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];
    const int ntiles_KV    = (n_kv_cols + FATTN_REL_NBATCH_FA - 1) / FATTN_REL_NBATCH_FA;

    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const bool scan_mask = mask && mask->ne[2] == 1 && n_kv_cols % FATTN_KQ_STRIDE == 0 &&
                           mask->ne[1] >= (int64_t) ntiles_x * ncols1 && mask->ne[3] == Q->ne[3] &&
                           (Q->ne[1] >= 64 || Q->ne[3] > 1 || kv_rows);
    if (scan_mask) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);
        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);
        const int iter_k_scan = n_kv_cols / FATTN_KQ_STRIDE;

        KV_max.alloc(blocks_num_KV_max.x * blocks_num_KV_max.y);
        ggml_cuda_kernel_launch_params lp(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, lp,
            (const half2 *) mask->data, KV_max.ptr, iter_k_scan, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(32, FATTN_REL_NWARPS, 1);
    int max_blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, FATTN_REL_NTHREADS, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);

    const int max_blocks = max_blocks_per_sm * nsm;
    const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
    const int tiles_eff    = 100 * ntiles_dst / (max_blocks * tiles_nwaves);
    const bool use_stream_k = (cc >= GGML_CUDA_CC_ADA_LOVELACE) || (tiles_eff < 75);

    dim3 blocks_num(ntiles_dst, 1, 1);
    if (use_stream_k) {
        const int raw     = std::min(max_blocks, ntiles_KV * ntiles_dst);
        const int rounded = (raw / ntiles_dst) * ntiles_dst;
        const int loss    = rounded > 0 ? (100 * (raw - rounded) / raw) : 100;
        blocks_num.x      = (loss <= 5) ? rounded : raw;
    }
    if (ntiles_dst % blocks_num.x != 0) {
        dst_tmp_meta.alloc((size_t) blocks_num.x * ncols * (2 + FATTN_REL_DV / 2));
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const uint3   ne01       = init_fastdiv_values(Q->ne[1]);
    const int32_t rel_extent = rel_table ? (int32_t)(rel_table->ne[0] - 1) : 0;
    const int64_t rel_s1     = rel_table ? (int64_t)(rel_table->nb[1] / sizeof(float)) : 0;
    const int64_t rel_s2     = rel_table ? (int64_t)(rel_table->nb[2] / sizeof(float)) : 0;
    const int64_t rel_s3     = rel_table ? (int64_t)(rel_table->nb[3] / sizeof(float)) : 0;

    ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? (const char *) mask->data : nullptr,
        sinks ? (const char *) sinks->data : nullptr,
        KV_max.ptr,
        kv_rows ? (const int *) kv_rows->data : nullptr,
        rel_table ? (const float *) rel_table->data : nullptr,
        k_pos ? (const int32_t *) k_pos->data : nullptr,
        q_pos ? (const int32_t *) q_pos->data : nullptr,
        (float *) dst->data,
        dst_tmp_meta.ptr,
        scale,
        rel_extent, rel_s1, rel_s2, rel_s3,
        ne01, (int32_t) Q->ne[2], (int32_t) Q->ne[3],
        (int32_t) Q->nb[1], (int32_t) Q->nb[2], (int32_t) Q->nb[3],
        (int32_t) n_kv_cols, (int32_t) K->ne[2],
        (int32_t) nb11, (int32_t) nb12, (int64_t) nb13,
        (int32_t) nb21, (int32_t) nb22, (int64_t) nb23,
        mask ? (int32_t) mask->ne[2] : 0, mask ? (int32_t) mask->ne[3] : 0,
        mask ? (int32_t) mask->nb[1] : 0, mask ? (int64_t) mask->nb[2] : 0, mask ? (int64_t) mask->nb[3] : 0);
    CUDA_CHECK(cudaGetLastError());

    // Stream-K fixup
    if ((int) blocks_num.x % ntiles_dst == 0 && (int) blocks_num.x > ntiles_dst) {
        const int nblocks_sk = (int) blocks_num.x;
        const int bpt        = nblocks_sk / ntiles_dst;
        const uint3 fd0      = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
        const uint3 fd1      = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
        const uint3 fd2      = init_fastdiv_values(ntiles_x);

        const dim3 block_dim_combine(FATTN_REL_DV, 1, 1);
        const dim3 blocks_num_combine((unsigned) ntiles_dst, ncols1, ncols2);
        ggml_cuda_kernel_launch_params lp(blocks_num_combine, block_dim_combine, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<FATTN_REL_DV, ncols1, ncols2>, lp,
            (float *) dst->data, dst_tmp_meta.ptr,
            (int32_t) Q->ne[1], (int32_t) Q->ne[2], (int32_t) K->ne[2], nblocks_sk,
            gqa_ratio, bpt, fd0, fd1, fd2);
    } else if (ntiles_dst % blocks_num.x != 0) {
        const int total_work       = ntiles_KV * ntiles_dst;
        const uint3 fd_k_j_z_ne12  = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
        const uint3 fd_k_j_z       = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
        const uint3 fd_k_j         = init_fastdiv_values(ntiles_KV * ntiles_x);
        const uint3 fd_k           = init_fastdiv_values(ntiles_KV);

        const dim3 block_dim_combine(FATTN_REL_DV, 1, 1);
        const dim3 blocks_num_combine(blocks_num.x, ncols1, ncols2);
        ggml_cuda_kernel_launch_params lp(blocks_num_combine, block_dim_combine, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<FATTN_REL_DV, ncols1, ncols2>, lp,
            (float *) dst->data, dst_tmp_meta.ptr,
            (int32_t) Q->ne[1], (int32_t) Q->ne[2], gqa_ratio, total_work,
            fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool kv_aligned>
static void dispatch_fattn_mma_f16_rel_acc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    launch_fattn_mma_f16_rel<ncols1, ncols2, type_K, type_V, kv_aligned, true>(ctx, dst);
}

template <int ncols1, int ncols2, bool kv_aligned>
static void dispatch_fattn_mma_f16_rel_kv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_type tk = dst->src[1]->type;
    const ggml_type tv = dst->src[2]->type;

    if (tk == GGML_TYPE_F8_E4M3 && tv == GGML_TYPE_F8_E4M3) {
        dispatch_fattn_mma_f16_rel_acc<ncols1, ncols2, GGML_TYPE_F8_E4M3, GGML_TYPE_F8_E4M3, kv_aligned>(ctx, dst);
    } else if (tk == GGML_TYPE_Q8_0 && tv == GGML_TYPE_Q8_0) {
        dispatch_fattn_mma_f16_rel_acc<ncols1, ncols2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, kv_aligned>(ctx, dst);
    } else {
        dispatch_fattn_mma_f16_rel_acc<ncols1, ncols2, GGML_TYPE_F16, GGML_TYPE_F16, kv_aligned>(ctx, dst);
    }
}

template <int ncols2, bool kv_aligned>
static void dispatch_fattn_mma_f16_rel_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t n_q = dst->src[0]->ne[1];
    const int     cc  = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if constexpr (ncols2 <= 8) {
        if (n_q <= 8 / ncols2) {
            dispatch_fattn_mma_f16_rel_kv<8 / ncols2, ncols2, kv_aligned>(ctx, dst);
            return;
        }
    }
    if (n_q <= 16 / ncols2) {
        dispatch_fattn_mma_f16_rel_kv<16 / ncols2, ncols2, kv_aligned>(ctx, dst);
        return;
    }
    if (n_q <= 32 / ncols2 || ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) {
        dispatch_fattn_mma_f16_rel_kv<32 / ncols2, ncols2, kv_aligned>(ctx, dst);
        return;
    }
    dispatch_fattn_mma_f16_rel_kv<64 / ncols2, ncols2, kv_aligned>(ctx, dst);
}

bool ggml_cuda_flash_attn_ext_mma_f16_rel_supported(int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED_VARS(device, dst);
    return false;
#else
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc)) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (Q->ne[0] != FATTN_REL_DKQ || K->ne[0] != FATTN_REL_DKQ || V->ne[0] != FATTN_REL_DV) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    // Require causal mask, sinks (Stream-K finalization assumes non-zero rowsum), and attached rel_bias;
    // reject compacted kv_rows (dst->src[5]) so unsupported shapes fall back cleanly.
    if (dst->src[5] != nullptr || dst->src[3] == nullptr || dst->src[4] == nullptr ||
        dst->src[6] == nullptr || dst->src[7] == nullptr || dst->src[8] == nullptr) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->src[3]->type != GGML_TYPE_F16 || dst->src[4]->type != GGML_TYPE_F32) {
        return false;
    }
    const bool kv_ok =
        (K->type == GGML_TYPE_F16     && V->type == GGML_TYPE_F16) ||
        (K->type == GGML_TYPE_Q8_0    && V->type == GGML_TYPE_Q8_0) ||
        (K->type == GGML_TYPE_F8_E4M3 && V->type == GGML_TYPE_F8_E4M3);
    if (!kv_ok) {
        return false;
    }
    // Check element contiguity and vector-load alignment (Q uses float2 loads; F16 KV uses uint4; F8 KV uses uint2),
    // including view_offs on unallocated views where data == nullptr.
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % sizeof(float2) != 0 || Q->nb[2] % sizeof(float2) != 0 || Q->nb[3] % sizeof(float2) != 0 ||
        Q->view_offs % sizeof(float2) != 0) {
        return false;
    }
    if (Q->data != nullptr && ((uintptr_t) Q->data) % sizeof(float2) != 0) {
        return false;
    }
    const size_t kv_elem_nb0 = ggml_type_size(K->type);
    const size_t kv_vec_align = (K->type == GGML_TYPE_F16) ? 16 : (K->type == GGML_TYPE_F8_E4M3 ? 8 : 1);
    for (const ggml_tensor * kv_t : { K, V }) {
        if (kv_t->nb[0] != kv_elem_nb0) {
            return false;
        }
        if (kv_vec_align > 1) {
            if (kv_t->nb[1] % kv_vec_align != 0 || kv_t->nb[2] % kv_vec_align != 0 || kv_t->nb[3] % kv_vec_align != 0 ||
                kv_t->view_offs % kv_vec_align != 0) {
                return false;
            }
            if (kv_t->data != nullptr && ((uintptr_t) kv_t->data) % kv_vec_align != 0) {
                return false;
            }
        }
    }
    return true;
#endif
}

void ggml_cuda_flash_attn_ext_mma_f16_rel(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[5] == nullptr && "rel_bias FA does not support compacted kv_rows");
    GGML_ASSERT(dst->src[3] != nullptr && "rel_bias FA requires causal attention mask");
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int64_t n_kv       = dst->src[5] ? dst->src[5]->ne[0] : K->ne[1];
    const int gqa_ratio      = (int)(Q->ne[2] / K->ne[2]);

    bool use_gqa_opt = mask && mask->ne[2] == 1 && (n_kv % FATTN_KQ_STRIDE == 0);
    for (const ggml_tensor * t : { Q, K, V, mask }) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    if (use_gqa_opt) {
        // Odd GQA ratio (e.g. Inkling-2 gqa_ratio=3) during long prefill (n_q >= 64):
        // ncols1=64, ncols2=1 with 2-stage cp.async eliminates the 25% wasted GQA padding column in ncols2=4
        // AND cuts total KV HBM reads by 25% (24 vs 32 passes per KV head at ub=512).
        if ((gqa_ratio % 2 == 1) && (Q->ne[1] >= 64)) {
            dispatch_fattn_mma_f16_rel_kv<64, 1, true>(ctx, dst);
        } else if (gqa_ratio > 4) {
            dispatch_fattn_mma_f16_rel_ncols1<8, true>(ctx, dst);
        } else if (gqa_ratio > 2) {
            dispatch_fattn_mma_f16_rel_ncols1<4, true>(ctx, dst);
        } else if (gqa_ratio > 1) {
            dispatch_fattn_mma_f16_rel_ncols1<2, true>(ctx, dst);
        } else {
            dispatch_fattn_mma_f16_rel_ncols1<1, true>(ctx, dst);
        }
    } else {
        dispatch_fattn_mma_f16_rel_ncols1<1, false>(ctx, dst);
    }
}
