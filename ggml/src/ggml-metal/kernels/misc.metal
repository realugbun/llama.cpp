#include "common.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_argmax_part_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * tmp,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  ntg[[threads_per_threadgroup]]) {
    const int row  = tgpig.y;
    const int part = tgpig.x;
    const int ne00 = (int) args.ne00;
    const int npart = (int) args.npart;
    const int chunk = (ne00 + npart - 1) / npart;
    const int start = part * chunk;
    const int end   = min(start + chunk, ne00);

    device const float * x_row = (device const float *) ((device const char *) src0 + row * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = start + tpitg.x; i00 < end; i00 += ntg.x) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        max_val = max_val_reduced;
        arg_val = arg_val_reduced;
    }

    if (tpitg.x == 0) {
        device float * tmp_max = (device float *) tmp;
        device int32_t * tmp_arg = (device int32_t *) tmp + npart * (int) args.ne01;

        tmp_max[row*npart + part] = max_val;
        tmp_arg[row*npart + part] = arg_val;
    }
}

kernel void kernel_argmax_final_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * tmp,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  ntg[[threads_per_threadgroup]]) {
    const int row = tgpig.x;
    const int npart = (int) args.npart;

    device const float * tmp_max = (device const float *) tmp;
    device const int32_t * tmp_arg = (device const int32_t *) tmp + npart * (int) args.ne01;

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i = tpitg.x; i < npart; i += ntg.x) {
        if (tmp_max[row*npart + i] > lmax) {
            lmax = tmp_max[row*npart + i];
            larg = tmp_arg[row*npart + i];
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        max_val = max_val_reduced;
        arg_val = arg_val_reduced;
    }

    if (tpitg.x == 0) {
        ((device int32_t *) dst)[row] = arg_val;
    }
}

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

constant bool FC_pad_circular [[function_constant(FC_PAD + 0)]];

// circular means on a torus, so the coordinates wrap around
static inline int32_t wrap_around(int32_t coord, int32_t size) {
    return (coord + size) % size;
}

kernel void kernel_pad_f32(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t ne00 = args.ne00;
    const int32_t ne01 = args.ne01;
    const int32_t ne02 = args.ne02;
    const int32_t ne03 = args.ne03;

    int32_t i01 = i1 - args.lp1;
    int32_t i02 = i2 - args.lp2;
    int32_t i03 = i3 - args.lp3;

    if (FC_pad_circular) {
        i01 = wrap_around(i01, ne01);
        i02 = wrap_around(i02, ne02);
        i03 = wrap_around(i03, ne03);
    }

    device float * dst_ptr = (device float *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1);

    // the row lies in the padded region, so no source row backs it
    if (i01 < 0 || i01 >= ne01 ||
        i02 < 0 || i02 >= ne02 ||
        i03 < 0 || i03 >= ne03) {
        for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
            const int32_t i0 = k0*1024 + tpitg.x + l0;
            if (i0 >= args.ne0) {
                break;
            }

            dst_ptr[i0] = 0.0f;
        }

        return;
    }

    device const char * src0_row = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        int32_t i00 = i0 - args.lp0;

        if (FC_pad_circular) {
            i00 = wrap_around(i00, ne00);
        }

        dst_ptr[i0] = i00 >= 0 && i00 < ne00 ? *((device const float *) (src0_row + i00*args.nb00)) : 0.0f;
    }
}

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T * args.C) {
        return;
    }

    const int   c  = idx / args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

kernel void kernel_inkling2_l2_clip_f32(
        constant ggml_metal_kargs_l2_clip & args,
        device const char  * x,
        device const float * s,
        device       float * dst,
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    // one simdgroup per row, ne00 % 4 == 0 and ne00 <= 128: lane i owns elements [4i, 4i + 4)
    // the reduction mirrors kernel_sum_rows_f32_f32_4 (one float4 per lane, simd_sum, sum) so the result is bit-identical
    const int r = tgpig*nsg + sgitg;
    if (r >= args.nrows) {
        return;
    }
    const int i1 = r % args.ne01;
    const int i2 = (r / args.ne01) % args.ne02;
    const int i3 = r / (args.ne01*args.ne02);
    device const float4 * xr = (device const float4 *) (x + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device const float4 * sr = args.s_ne1 == 0 ? nullptr : (device const float4 *) (s + (args.s_ne1 == 1 ? 0 : i1)*args.ne00);
    device       float4 * dr = (device float4 *) (dst + (int64_t) r*args.ne00);

    const int n4 = args.ne00/4;
    float4 y = 0.0f;
    if (tiisg < n4) {
        y = sr ? xr[tiisg]*sr[tiisg] : xr[tiisg];
    }
    const float4 s4 = simd_sum(y*y);
    const float sumf   = sum(s4);
    const float norm   = sqrt(sumf)*1.0f + args.eps;
    const float factor = clamp(args.thr/norm, args.lo, args.hi);
    if (tiisg < n4) {
        dr[tiisg] = y*factor;
    }
}

// Inkling 2 short convolution with its state store, one thread per output element (c, t, s)
kernel void kernel_inkling2_conv_state_f32(
        constant ggml_metal_kargs_conv_state & args,
        device const char  * past,
        device const char  * cur,
        device const char  * w,
        device const float * r,
        device const float * res,
        device       char  * cell,
        device       float * dst,
        uint gid [[thread_position_in_grid]]) {
    if ((int) gid >= args.n) {
        return;
    }
    const int c = gid % args.n_c;
    const int t = (gid / args.n_c) % args.n_t;
    const int s = gid / (args.n_c*args.n_t);
    const int d = args.d;

    device const float * pr = (device const float *) (past + c*args.past_nb1 + s*args.past_nb2);
    device const char  * cr = cur + c*args.cur_nb1 + s*args.cur_nb2;
    device const float * wr = (device const float *) (w + c*args.w_nb1);

    float x[8];
    for (int k = 0; k < d; ++k) {
        const int j = t + k;
        x[k] = j < d - 1 ? pr[j] : *((device const float *) (cr + (j - (d - 1))*args.cur_nb0));
    }
    // the order of kernel_ssm_conv_f32_f32(_4)
    float sumf = 0.0f;
    if (d % 4 == 0) {
        for (int k = 0; k < d; k += 4) {
            sumf += dot(float4(x[k], x[k+1], x[k+2], x[k+3]), float4(wr[k], wr[k+1], wr[k+2], wr[k+3]));
        }
    } else {
        for (int k = 0; k < d; ++k) {
            sumf += x[k]*wr[k];
        }
    }
    if (t == 0) {
        device float * cl = (device float *) (cell + s*args.cell_nb1) + c*(d - 1);
        for (int k = 0; k < d - 1; ++k) {
            const int j = args.n_t + k;
            cl[k] = j < d - 1 ? pr[j] : *((device const float *) (cr + (j - (d - 1))*args.cur_nb0));
        }
    }
    const float y = args.s0 * (r[gid] + sumf) + args.s1;
    dst[gid] = args.has_res ? res[gid] + y : y;
}

// Inkling 2 dynamic short convolution (d = 4, one sequence) with its state store
kernel void kernel_inkling2_conv_dyn_f32(
        constant ggml_metal_kargs_conv_dyn & args,
        device const char  * past,
        device const char  * cur,
        device const char  * w,
        device const char  * tap,
        device const float * r,
        device       float * cell,
        device       float * dst,
        uint gid [[thread_position_in_grid]]) {
    if ((int) gid >= args.n_c*args.n_t) {
        return;
    }
    const int c = gid % args.n_c;
    const int t = gid / args.n_c;
    const int h = c / (args.n_c/args.n_h);
    device const float * pr = (device const float *) (past + c*args.past_nb1);
    device const char  * cr = cur + c*args.cur_nb1;
    device const float * wr = (device const float *) (w + c*args.w_nb1);
    float x[4];
    for (int k = 0; k < 4; ++k) {
        const int j = t + k;
        x[k] = j < 3 ? pr[j] : *((device const float *) (cr + (j - 3)*args.cur_nb0));
    }
    float out = 0.0f;
    out += dot(float4(x[0], x[1], x[2], x[3]), float4(wr[0], wr[1], wr[2], wr[3]));
    device const float * tp = (device const float *) (tap + h*args.tap_nb1 + t*args.tap_nb2);
    for (int l = 0; l < 4; ++l) {
        out = out + metal::precise::fma(x[3 - l], tp[l], 0.0f);
    }
    if (t == 0) {
        for (int k = 0; k < 3; ++k) {
            const int j = args.n_t + k;
            cell[c*3 + k] = j < 3 ? pr[j] : *((device const float *) (cr + (j - 3)*args.cur_nb0));
        }
    }
    dst[gid] = args.s0 * (r[gid] + out) + args.s1;
}

// Inkling 2 attention over depth, one threadgroup of 576 threads per token, one float4 of the 2304 per thread.
// The norm of x replicates kernel_rms_norm_f32_4 (576 threads); the dot products and the mix sum in another order.
kernel void kernel_inkling2_mix_f32(
        constant ggml_metal_kargs_mix & args,
        device const char   * x,
        device const char   * p,
        device const float4 * q,
        device       float4 * dst,
        threadgroup  float  * shmem [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int t  = tgpig;
    const int i  = tpitg;          // float4 index, 0..575
    const int L  = args.L;
    const int nq = args.nq;

    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }
    const float4 xv = ((device const float4 *) (x + t*args.x_nb2))[i];
    float sumf = dot(xv, xv);
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = shmem[tiisg];
    sumf = simd_sum(sumf);
    const float mean = sumf/2304;
    const float scl  = 1.0f/sqrt(mean + args.eps);
    const float4 xn  = xv*scl;

    float4 st[16];
    for (int l = 0; l < L - 1; ++l) {
        st[l] = ((device const float4 *) (p + l*args.p_nb1 + t*args.p_nb2))[i];
    }
    st[L - 1] = xn;

    // partial dot products, reduced over the 18 simdgroups
    threadgroup float * red = shmem + 32;   // [18][64]
    for (int j = 0; j < L*nq; ++j) {
        const int l = j % L, k = j / L;
        float d = simd_sum(dot(st[l], q[k*576 + i]));
        if (tiisg == 0) {
            red[sgitg*64 + j] = d;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * sc = shmem + 32 + 18*64;   // [64]
    if (i < L*nq) {
        float s = 0.0f;
        for (int g = 0; g < 18; ++g) {
            s += red[g*64 + i];
        }
        sc[i] = s*args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < nq; ++k) {
        float mx = -INFINITY;
        for (int l = 0; l < L; ++l) mx = max(mx, sc[k*L + l]);
        float w[16]; float sum = 0.0f;
        for (int l = 0; l < L; ++l) { w[l] = exp(sc[k*L + l] - mx); sum += w[l]; }
        float4 acc = 0.0f;
        for (int l = 0; l < L; ++l) acc += st[l]*(w[l]/sum);
        dst[(t*nq + k)*576 + i] = acc;
    }
}

// Inkling 2 loop controller output: (c - tanh(f)*c) + tanh(g)*m
kernel void kernel_inkling2_ctrl_tail_f32(
        constant ggml_metal_kargs_ctrl_tail & args,
        device const char * c,
        device const char * f,
        device const char * g,
        device const char * m,
        device      float * dst,
        uint gid [[thread_position_in_grid]]) {
    if ((int) gid >= args.ne0*args.ne1) {
        return;
    }
    const int i0 = gid % args.ne0;
    const int i1 = gid / args.ne0;
    const float cv = ((device const float *) (c + i1*args.nb_c))[i0];
    const float tf = precise::tanh(((device const float *) (f + i1*args.nb_f))[i0]);
    const float tg = precise::tanh(((device const float *) (g + i1*args.nb_g))[i0]);
    const float mv = ((device const float *) (m + i1*args.nb_m))[i0];
    // keep the roundings of the unfused MUL, SUB, MUL, ADD kernels: no contraction into fma
    const float a  = metal::precise::fma(tf, cv, 0.0f);
    const float b  = cv - a;
    const float e  = metal::precise::fma(tg, mv, 0.0f);
    dst[gid] = b + e;
}

// Inkling 2 SiTU expert activation; the scales keep the a*x + b form of kernel_unary (SCALE)
kernel void kernel_inkling2_situ_f32(
        constant ggml_metal_kargs_situ & args,
        device const char * g,
        device const char * u,
        device      float * dst,
        uint gid [[thread_position_in_grid]]) {
    if ((int) gid >= args.ne0*args.ne1*args.ne2) {
        return;
    }
    const int i0 = gid % args.ne0;
    const int i1 = (gid / args.ne0) % args.ne1;
    const int i2 = gid / (args.ne0*args.ne1);
    const float x = ((device const float *) (g + i1*args.g_nb1 + i2*args.g_nb2))[i0];
    const float y = ((device const float *) (u + i1*args.u_nb1 + i2*args.u_nb2))[i0];
    const float a0 = args.s0 * x + args.b0;
    const float t0 = precise::tanh(a0);
    const float a1 = args.s1 * t0 + args.b1;
    const float sg = 1 / (1 + exp(-x));
    const float ac = metal::precise::fma(a1, sg, 0.0f);
    const float c0 = args.s2 * y + args.b2;
    const float t1 = precise::tanh(c0);
    const float c1 = args.s3 * t1 + args.b3;
    dst[gid] = ac*c1;
}

// Inkling 2 relative-position bias added to the attention mask
kernel void kernel_inkling2_relbias(
        constant ggml_metal_kargs_relbias & args,
        device const char * tbl,
        device const char * snk,
        device const char * rel,
        device const char * mask,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int n = args.n_kv*args.n_tt*args.n_h*args.n_s;
    if ((int) gid >= n) {
        return;
    }
    const int kv = gid % args.n_kv;
    const int tt = (gid / args.n_kv) % args.n_tt;
    const int h  = (gid / (args.n_kv*args.n_tt)) % args.n_h;
    const int s  = gid / (args.n_kv*args.n_tt*args.n_h);
    const int t  = tt + s*args.n_tt;
    const int r  = ((device const int32_t *) (rel + t*args.rel_nb1))[kv];
    const float b  = ((device const float *) (tbl + h*args.tbl_nb1 + t*args.tbl_nb2))[r];
    const float sk = *((device const float *) (snk + h*args.snk_nb1 + t*args.snk_nb2));
    device const char * mr = mask + tt*args.m_nb1 + (args.m_ne2 == 1 ? 0 : h)*args.m_nb2 + s*args.m_nb3;
    const float m  = args.m_f16 ? (float) ((device const half *) mr)[kv] : ((device const float *) mr)[kv];
    const float v  = (b - sk) + m;
    device char * dr = dst + tt*args.d_nb1 + h*args.d_nb2 + s*args.d_nb3;
    if (args.d_f16) {
        ((device half *) dr)[kv] = (half) v;
    } else {
        ((device float *) dr)[kv] = v;
    }
}

template<int N, typename src_t>
kernel void kernel_fwht(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = float(src[i*NW + lane])*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = val2 - val + 2*((lane & i) == 0)*val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

// Wide blocks: one row per threadgroup instead of per simdgroup, so each thread keeps
// N/NT values rather than N/32. Butterflies below the simdgroup width still shuffle;
// those up to NT go through threadgroup memory; the rest stay in registers.
// TODO: try avoiding branch https://github.com/ggml-org/llama.cpp/pull/29094#discussion_r4049563223
// TODO: try to unroll loops
template<int N, int NT, typename src_t>
kernel void kernel_fwht_tg(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NT;

    threadgroup float shmem[N];

    const float scale = 1.0f / sqrt((float) N);

    const int64_t r = tgpig.x;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid = sgitg * NW + tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = float(src[i*NT + tid])*scale;
    }

    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < NT; i *= 2) {
        for (int j = 0; j < NE; j++) {
            shmem[j*NT + tid] = reg[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = shmem[j*NT + (tid ^ i)];
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = NT; i < N; i *= 2) {
        const int step = i / NT;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NT + tid] = reg[i];
    }
}

typedef decltype(kernel_fwht<64, float>) kernel_fwht_f32_t;
typedef decltype(kernel_fwht<64, half>)  kernel_fwht_f16_t;

template [[host_name("kernel_fwht_f32_64")]]  kernel kernel_fwht_f32_t kernel_fwht<64,  float>;
template [[host_name("kernel_fwht_f32_128")]] kernel kernel_fwht_f32_t kernel_fwht<128, float>;
template [[host_name("kernel_fwht_f32_256")]] kernel kernel_fwht_f32_t kernel_fwht<256, float>;

template [[host_name("kernel_fwht_f16_64")]]  kernel kernel_fwht_f16_t kernel_fwht<64,  half>;
template [[host_name("kernel_fwht_f16_128")]] kernel kernel_fwht_f16_t kernel_fwht<128, half>;
template [[host_name("kernel_fwht_f16_256")]] kernel kernel_fwht_f16_t kernel_fwht<256, half>;

template [[host_name("kernel_fwht_f32_512")]]  kernel kernel_fwht_f32_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_1024")]] kernel kernel_fwht_f32_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_2048")]] kernel kernel_fwht_f32_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_4096")]] kernel kernel_fwht_f32_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_8192")]] kernel kernel_fwht_f32_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, float>;

template [[host_name("kernel_fwht_f16_512")]]  kernel kernel_fwht_f16_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_1024")]] kernel kernel_fwht_f16_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_2048")]] kernel kernel_fwht_f16_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_4096")]] kernel kernel_fwht_f16_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_8192")]] kernel kernel_fwht_f16_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, half>;

constant int FC_dsv4_hc_n_hc [[function_constant(FC_DSV4_HC + 0)]];

kernel void kernel_dsv4_hc_comb_f32(
        constant ggml_metal_kargs_dsv4_hc_comb & args,
        device const char * mixes,
        device const char * scale,
        device const char * base,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;
    constexpr ushort comb_offset = 2*hc;

    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) {
        return;
    }

    float scale_lane = 0.0f;
    if (tiisg == 0) {
        scale_lane = *(device const float *) (scale + 2*args.nb_s0);
    }
    const float scale_comb = simd_shuffle(scale_lane, 0);

    float v = 0.0f;
    if (tiisg < hc*hc) {
        v = *(device const float *) (mixes + (comb_offset + tiisg)*args.nb_m0 + it*args.nb_m1)*scale_comb
          + *(device const float *) (base   + (comb_offset + tiisg)*args.nb_b0);
    }

    // Softmax across destinations (the four contiguous lanes for each source).
    float vmax = max(v, simd_shuffle_xor(v, 1));
    vmax = max(vmax, simd_shuffle_xor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + simd_shuffle_xor(v, 1);
    sum += simd_shuffle_xor(sum, 2);
    v = v/sum + args.eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + simd_shuffle_xor(v, 4);
    sum += simd_shuffle_xor(sum, 8);
    v /= sum + args.eps;

    for (int i = 1; i < args.n_iter; ++i) {
        sum = v + simd_shuffle_xor(v, 1);
        sum += simd_shuffle_xor(sum, 2);
        v /= sum + args.eps;

        sum = v + simd_shuffle_xor(v, 4);
        sum += simd_shuffle_xor(sum, 8);
        v /= sum + args.eps;
    }

    if (tiisg < hc*hc) {
        const ushort idst = tiisg & 3;
        const ushort isrc = tiisg >> 2;
        *(device float *) (dst + idst*args.nb_d0 + isrc*args.nb_d1 + it*args.nb_d2) = v;
    }
}

kernel void kernel_dsv4_hc_pre_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * weights,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x + i0*args.nb_x0 + it*args.nb_x2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        const float wv = *(device const float *) (weights + ih*args.nb_w0 + it*args.nb_w1);
        result = fma(xv, wv, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_pre_gated_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * gate,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x    + i0*args.nb_x0 + it*args.nb_x2;
    device const char * gb = gate + i0*args.nb_w0 + it*args.nb_w2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float g  = 1.0f/(1.0f + exp(-*(device const float *) (gb + ih*args.nb_w1)));
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        result = fma(xv, g, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_post_nocomb_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float post_lane = 0.0f;
    if (tiisg < hc) {
        post_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    }

    float post_reg[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(post_lane, idst);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        const float rv = *(device const float *) (rb + idst*args.nb_r1);
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = xv*post_reg[idst] + rv;
    }
}

kernel void kernel_dsv4_hc_post_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float coeff_lane = 0.0f;
    if (tiisg < hc) {
        coeff_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    } else if (tiisg < hc + hc*hc) {
        const ushort idx  = tiisg - hc;
        const ushort idst = idx & 3;
        const ushort isrc = idx >> 2;
        coeff_lane = *(device const float *) (comb + idst*args.nb_c0 + isrc*args.nb_c1 + it*args.nb_c2);
    }

    float post_reg[hc];
    float comb_reg[hc][hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(coeff_lane, idst);
    }
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            comb_reg[isrc][idst] = simd_shuffle(coeff_lane, hc + idst + hc*isrc);
        }
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    float result[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        result[idst] = xv*post_reg[idst];
    }

    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        const float rv = *(device const float *) (rb + isrc*args.nb_r1);
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            result[idst] = fma(rv, comb_reg[isrc][idst], result[idst]);
        }
    }

    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = result[idst];
    }
}

constant bool FC_top_k_mask_has_score  [[function_constant(FC_TOP_K_MASK + 0)]];
constant bool FC_top_k_mask_has_blocks [[function_constant(FC_TOP_K_MASK + 1)]];
constant bool FC_top_k_mask_has_extra  [[function_constant(FC_TOP_K_MASK + 2)]];
constant bool FC_top_k_mask_vec        [[function_constant(FC_TOP_K_MASK + 3)]];

// the fill runs before the scatter, both are fully parallel over the whole dst and over the picks of one row
template <typename T, typename T4>
kernel void kernel_top_k_mask_fill(
        constant ggml_metal_kargs_top_k_mask & args,
        device char * dst,
        uint  tid[[thread_position_in_grid]],
        uint ntot[[threads_per_grid]]) {
    const T ninf = T(-INFINITY);

    if (FC_top_k_mask_vec) {
        device T4 * dst_4 = (device T4 *) dst;
        for (int64_t i = tid; i < args.n_elems/4; i += ntot) {
            dst_4[i] = T4(ninf);
        }
    } else {
        device T * dst_t = (device T *) dst;
        for (int64_t i = tid; i < args.n_elems; i += ntot) {
            dst_t[i] = ninf;
        }
    }
}

typedef decltype(kernel_top_k_mask_fill<half,  half4>)  kernel_top_k_mask_fill_h_t;
typedef decltype(kernel_top_k_mask_fill<float, float4>) kernel_top_k_mask_fill_f_t;

template [[host_name("kernel_top_k_mask_fill_f16")]] kernel kernel_top_k_mask_fill_h_t kernel_top_k_mask_fill<half,  half4>;
template [[host_name("kernel_top_k_mask_fill_f32")]] kernel kernel_top_k_mask_fill_f_t kernel_top_k_mask_fill<float, float4>;

// one threadgroup per row: copy the mask cells of the live picks and of the extra cells over the fill
template <typename T, typename T4>
kernel void kernel_top_k_mask(
        constant ggml_metal_kargs_top_k_mask & args,
        device const  char * mask,
        device const  char * top_k,
        device const  char * score,
        device const  char * blocks,
        device const  char * extra,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3 ntg_max[[threads_per_threadgroup]]) {
    const int i1 = tgpig.x;
    const int i2 = tgpig.y;
    const int i3 = tgpig.z;

    const int   tiitg = tpitg.x;
    const int     ntg = ntg_max.x;

    device const char * mask_row  = mask  + (i1 % args.m1)*args.nbm1 + (i2 % args.m2)*args.nbm2 + (i3 % args.m3)*args.nbm3;
    device       char * dst_row   = dst   + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;
    device const char * top_k_row = top_k + i1*args.nbk1 + i2*args.nbk2 + i3*args.nbk3;

    device const T * mask_row_t = (device const T *) mask_row;
    device       T * dst_row_t  = (device       T *) dst_row;

    for (int i = tiitg; i < args.n_top; i += ntg) {
        const int32_t p = *(device const int32_t *) (top_k_row + i*args.nbk0);
        if (p < 0 || p >= args.n_pick) {
            continue;
        }
        if (FC_top_k_mask_has_score && !(*(device const float *) (score + p*args.nbs0 + i1*args.nbs1 + i2*args.nbs2 + i3*args.nbs3) > -INFINITY)) {
            continue;
        }
        if (FC_top_k_mask_has_blocks) {
            FOR_UNROLL (int k = 0; k < args.n_blk; ++k) {
                const int32_t c = *(device const int32_t *) (blocks + k*args.nbb0 + p*args.nbb1);
                if (c >= 0 && c < args.n_kv) {
                    dst_row_t[c] = mask_row_t[c];
                }
            }
        } else if (p < args.n_kv) {
            dst_row_t[p] = mask_row_t[p];
        }
    }

    if (FC_top_k_mask_has_extra) {
        for (int i = tiitg; i < args.n_extra; i += ntg) {
            const int32_t c = *(device const int32_t *) (extra + i*args.nbe0 + i1*args.nbe1 + i2*args.nbe2 + i3*args.nbe3);
            if (c >= 0 && c < args.n_kv) {
                dst_row_t[c] = mask_row_t[c];
            }
        }
    }
}

typedef decltype(kernel_top_k_mask<half,  half4>)  kernel_top_k_mask_h_t;
typedef decltype(kernel_top_k_mask<float, float4>) kernel_top_k_mask_f_t;

template [[host_name("kernel_top_k_mask_f16")]] kernel kernel_top_k_mask_h_t kernel_top_k_mask<half,  half4>;
template [[host_name("kernel_top_k_mask_f32")]] kernel kernel_top_k_mask_f_t kernel_top_k_mask<float, float4>;
