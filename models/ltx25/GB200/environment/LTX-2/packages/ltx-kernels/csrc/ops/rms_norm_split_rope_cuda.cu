#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// CUDA kernel template for RMS norm + split RoPE
// out_t can be at::Float8_e4m3fn or at::BFloat16

using bf16 = __nv_bfloat16;
using fp8 = __nv_fp8_e4m3;
__device__ __forceinline__ void _load_x(bf16* x, float x_vals[8], int h){
    bf16 x_tmp[8];
    *reinterpret_cast<int4*>(x_tmp) = reinterpret_cast<int4*>(x + blockIdx.x * h)[threadIdx.x];
    #pragma unroll
    for(int i = 0; i < 8; i++){
        x_vals[i] = float(x_tmp[i]);
    }
}
// Load 8 freq values for this thread from a table laid out logically as
// [b, n, s, d/2] (what apply_split_rotary_emb produces -- a swapaxes view whose
// physical layout is [b, s, n, d/2]). The strides (sb, sn, ss; inner d/2 stride
// is 1) are forwarded from the host so the read is correct for both that
// non-contiguous view and a genuinely contiguous [b, n, s, d/2] tensor. Both
// head-halves map to the same freq element (mirrors the eager cos.unsqueeze(-2)).
__device__ __forceinline__ void _load_freqs(
    const bf16* freqs, float x_vals[8], int s, int d, long sb, long sn, long ss
){
    bf16 x_tmp[8];
    int threads_per_head = d / 8;
    int head_idx = threadIdx.x / threads_per_head;
    int lane = threadIdx.x % (threads_per_head / 2);
    int b_idx = blockIdx.x / s;
    int t_idx = blockIdx.x % s;
    long off = b_idx * sb + head_idx * sn + t_idx * ss + (long)lane * 8;
    *reinterpret_cast<int4*>(x_tmp) = *reinterpret_cast<const int4*>(freqs + off);
    #pragma unroll
    for(int i = 0; i < 8; i++){
        x_vals[i] = float(x_tmp[i]);
    }
}

template<typename out_t>
__global__ void _rms_norm_split_rope_kernel(bf16* x, bf16* sin_freqs, bf16* cos_freqs, void* out, bf16* weights, int b, int s, int n, int h,
    long cos_sb, long cos_sn, long cos_ss, long sin_sb, long sin_sn, long sin_ss){
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    // freqs have shape [b, s, h/2]
    // each thread block calculate one row
    // there are h/8 threads in thread block, each thread processes 8 values
    // num_of_rows = b * s
    // freqs have h/2 dim
    // gridDim is (num_of_rows, 1, 1)
    // calculate rms norm x_normed = x/x_norm * weights. x_norm is calculated across row, it means thread block wide sum reduction

    extern __shared__ float smem[];

    // Step 1: Load input values (8 per thread)
    float x_vals[8];
    _load_x(x, x_vals, h);

    float sum_sq = 0.0f;
    #pragma unroll
    for(int i = 0; i < 8; i++){
        sum_sq += x_vals[i] * x_vals[i];
    }

    // Warp-level reduction
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    if(tid % 32 == 0){
        smem[tid / 32] = sum_sq;
    }
    __syncthreads();

    // Final reduction across warps
    if(tid == 0){
        float total_sum = 0.0f;
        int num_warps = blockDim.x / 32;
        for(int i = 0; i < num_warps; i++){
            total_sum += smem[i];
        }
        // RMS: sqrt(mean(x^2))
        float rms = rsqrtf(total_sum / h + 1e-6f);  // Add epsilon for numerical stability
        smem[0] = rms;
    }
    __syncthreads();

    float inv_rms = smem[0];

    // Step 3: Apply RMS normalization (and weights if provided)
    #pragma unroll
    for(int i = 0; i < 8; i++){
        x_vals[i] *= inv_rms;
        // TODO: Apply weights if provided
        if(weights != nullptr) x_vals[i] *= float(weights[tid * 8 + i]);
    }

    // Step 4: Calculate dimensions for split RoPE
    // Conceptually: [b, s, h] -> [b, s, n, 2*d] -> [b, s, n, 2, d]
    // where h = n * 2 * d
    int d = h / n;
    float x_other_vals[8];

    int threads_per_head = d / 8;
    int head_idx = tid / threads_per_head;
    int idx_in_head = tid % threads_per_head;
    bool is_first_half = idx_in_head < (threads_per_head / 2);

    // LT-PATCH: full-warp mask. The original (1u << threads_per_head) - 1 only marks
    // the first head's lanes active, so lanes belonging to heads beyond the first are
    // not in the mask -> __shfl_xor_sync result is undefined and can corrupt RoPE. The
    // XOR pattern keeps data within each power-of-two head group, so a full-warp mask
    // is correct for every lane.
    const unsigned mask = 0xffffffffu;
    const int laneMask = threads_per_head / 2;           // 4, 8, or 16
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x_other_vals[i] = __shfl_xor_sync(mask, x_vals[i], laneMask);
    }

    float cos_vals[8], sin_vals[8];
    _load_freqs(cos_freqs, cos_vals, s, d, cos_sb, cos_sn, cos_ss);
    _load_freqs(sin_freqs, sin_vals, s, d, sin_sb, sin_sn, sin_ss);
    #pragma unroll
    for(int i = 0; i < 8; i++){
        x_vals[i] = cos_vals[i]*x_vals[i];
    }


    float sign = is_first_half ? -1.0f : 1.0f;
    for(int i = 0; i < 8; i++){
        x_vals[i] += sign*sin_vals[i]*x_other_vals[i];
    }

    // Step 6: Convert and store output
    if constexpr (std::is_same_v<out_t, at::Float8_e4m3fn>){
        fp8 out_tmp[8];
        #pragma unroll
        for(int i = 0; i < 8; i++){
            out_tmp[i] = fp8(x_vals[i]);
        }
        *reinterpret_cast<int64_t*>((fp8*)out + token_idx * h + tid * 8) = *reinterpret_cast<int64_t*>(out_tmp);
    } else {
        bf16 out_tmp[8];
        #pragma unroll
        for(int i = 0; i < 8; i++){
            out_tmp[i] = __float2bfloat16(x_vals[i]);
        }
        *reinterpret_cast<int4*>((bf16*)out + token_idx * h + tid * 8) = *reinterpret_cast<int4*>(out_tmp);
    }
}

template<typename out_t>
void rms_norm_split_rope_cuda(
    void* x,           // Input: [b, s, h]
    void* sin_freqs,   // Sin frequencies: [b, n, s, d]
    void* cos_freqs,   // Cos frequencies: [b, n, s, d]
    void* weights,
    int b,                      // Batch size
    int s,                      // Sequence length
    int n,                      // Number of heads (32)
    int h,                      // Hidden dimension (2048, 4096, or 8192)
    long cos_sb, long cos_sn, long cos_ss,  // cos_freqs strides (b, n, s)
    long sin_sb, long sin_sn, long sin_ss,  // sin_freqs strides (b, n, s)
    void* out,                // Output: [b, s, h]
    cudaStream_t stream
) {
    int num_tokens = b * s;
    int num_threads = h / 8;  // Each thread processes 8 elements
    int smem_size = (num_threads / 32 + 1) * sizeof(float);  // Shared memory for reductions

    dim3 grid(num_tokens);
    dim3 block(num_threads);

    _rms_norm_split_rope_kernel<out_t><<<grid, block, smem_size, stream>>>(
        reinterpret_cast<bf16*>(x),
        reinterpret_cast<bf16*>(sin_freqs),
        reinterpret_cast<bf16*>(cos_freqs),
        out,
        reinterpret_cast<bf16*>(weights),
        b, s, n, h,
        cos_sb, cos_sn, cos_ss,
        sin_sb, sin_sn, sin_ss
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Explicit template instantiations
template void rms_norm_split_rope_cuda<at::BFloat16>(
    void*, void*, void*, void*, int, int, int, int, long, long, long, long, long, long, void*, cudaStream_t
);

template void rms_norm_split_rope_cuda<at::Float8_e4m3fn>(
    void*, void*, void*, void*, int, int, int, int, long, long, long, long, long, long, void*, cudaStream_t
);
