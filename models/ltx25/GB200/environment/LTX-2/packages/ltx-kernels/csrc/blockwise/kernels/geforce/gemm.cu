#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cute/tensor.hpp>
#include <c10/cuda/CUDAException.h> 
#include <torch/extension.h>
#include <torch/python.h>
#include <cuda_runtime.h>
#include <iostream>

#include "kernel_traits.cuh"
#include "static_switch.h"

namespace sm89{
using namespace cute;

__device__ static void copy_1d(float* gmem_src, float* smem_dst)
{

    uint32_t smem_int_ptr = cast_smem_ptr_to_uint((void*)smem_dst);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(smem_int_ptr),
           "l"(gmem_src),
           "n"(sizeof(float)));
}

template <typename KernelTraits=gemm_traits<128, 256, 2, 4096, 2, 4, true, half_t, bfloat16_t>>
__global__ void gemm_fp8_kernel(float_e4m3_t* Aptr, float* sfa, float_e4m3_t* Bptr, float* sfb, float* bias_ptr, void* out, int M, int N, int K, int TMA_ALIGNED_M){
    using output_t = typename KernelTraits::out_t;
    using SmemLayoutA = typename KernelTraits::SmemLayoutA;
    using SmemLayoutB = typename KernelTraits::SmemLayoutB;
    using SmemLayoutC = typename KernelTraits::SmemLayoutC;
    
    constexpr int BM = KernelTraits::BM;
    constexpr int BN = KernelTraits::BN;
    constexpr int BK = KernelTraits::BK;
    constexpr int Ksfa = KernelTraits::KSF;
    constexpr bool has_bias = KernelTraits::HasBias;
    extern __shared__ float smem_[];
    float *bias_shm = smem_;
    float *sfa_shm = reinterpret_cast<float*>(bias_shm + cosize(typename KernelTraits::SmemLayoutBias{}));
    
    output_t* C_shm = reinterpret_cast<output_t*>(sfa_shm + cosize(typename KernelTraits::SmemLayoutSFA{}));
    float_e4m3_t* A_shm = reinterpret_cast<float_e4m3_t*>(sfa_shm + cosize(typename KernelTraits::SmemLayoutSFA{}));
    float_e4m3_t* B_shm = reinterpret_cast<float_e4m3_t*>(A_shm + cosize(SmemLayoutA{}));
  
    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    // sfa += BM * iy;
    sfb += KernelTraits::NUM_SFB_PER_STEP * ix * Ksfa;
    

    output_t* Cptr = reinterpret_cast<output_t*>(out);

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, Int<1>{}));
    Tensor SFA = make_tensor(make_gmem_ptr(sfa), make_shape(M, Ksfa), make_stride(Int<1>{}, TMA_ALIGNED_M));
    
    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _)); 
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _)); 
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix)); 
    Tensor gSFA = local_tile(SFA, make_tile(Int<BM>{}, Int<1>{}), make_coord(iy, _));

    auto sBias = make_tensor(make_smem_ptr(bias_shm), typename KernelTraits::SmemLayoutBias{});
    if constexpr (has_bias){
        Tensor Bias = make_tensor(make_gmem_ptr(bias_ptr), make_shape(_1{}, N), make_stride(N, Int<1>{}));
        Tensor gBias = local_tile(Bias, make_tile(Int<1>{}, Int<BN>{}), make_coord(_, ix));
        typename KernelTraits::G2SBiasCopy g2s_bias_copy;
        auto g2s_bias_thr_copy = g2s_bias_copy.get_slice(idx);
        auto tCBiasgBias = g2s_bias_thr_copy.partition_S(gBias);
        auto tCBiassBias = g2s_bias_thr_copy.partition_D(sBias);
        if(idx < BN){
            copy_1d((float*)&gBias(0) + idx, (float*)&sBias(0) + idx);
        }
    }

    auto sSFA = make_tensor(make_smem_ptr(sfa_shm), typename KernelTraits::SmemLayoutSFA{});
    auto sA = make_tensor(make_smem_ptr(A_shm), SmemLayoutA{}); 
    auto sB = make_tensor(make_smem_ptr(B_shm), SmemLayoutB{});

    typename KernelTraits::MMATile tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); 
    auto tCrD = thr_mma.partition_fragment_C(gD);           
    clear(tCrD);
    auto tCrD_fp32 = make_tensor_like<float>(tCrD);
    clear(tCrD_fp32);

    typename KernelTraits::G2STiledCopy g2s_tiled_copy;
    auto g2s_thr_copy = g2s_tiled_copy.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy.partition_S(gA); 
    auto tAsA_copy = g2s_thr_copy.partition_D(sA); 
    auto tBgB_copy = g2s_thr_copy.partition_S(gB); 
    auto tBsB_copy = g2s_thr_copy.partition_D(sB); 

    auto s2r_tiled_copy_a = make_tiled_copy_A(typename KernelTraits::S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);     
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); 


    auto s2r_tiled_copy_b = make_tiled_copy_B(typename KernelTraits::S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);    
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    auto cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
    auto tAcA = g2s_thr_copy.partition_S(cA);
    int residual = M - iy*BM;

    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;
    int ismem_read_sfa = 0;
    constexpr int kStages = KernelTraits::KStages;

    #pragma unroll
    for(int istage=0; istage<kStages - 1; ++istage){
        for (size_t m = 0; m < size<1>(tAsA_copy); m++)
        {
            for (size_t k = 0; k < size<2>(tAsA_copy); k++)
            {
                if(get<0>(tAcA(0, m, k)) < residual){
                    cute::copy(g2s_tiled_copy, tAgA_copy(_, m, k, istage), tAsA_copy(_, m, k, istage)); 
                }
            }  
        }
        if(idx < KernelTraits::THREADS_SFA_COPY && (BM * iy + idx * KernelTraits::SFA_ELEMS_PER_COPY < M)) {
            copy_1d((float*)&gSFA(0, 0, istage) + idx*KernelTraits::SFA_ELEMS_PER_COPY,  (float*)&sSFA(0, istage) + idx*KernelTraits::SFA_ELEMS_PER_COPY);
        }
        cute::copy(g2s_tiled_copy, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        cp_async_fence();
        ++itile_to_read;
        ++ismem_write;
    }

    cp_async_wait<kStages - 2>();
    __syncthreads();

    cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0)); 
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

    static constexpr int nk = size<2>(tCrA);
    auto sfa_tv = typename KernelTraits::SFAThreadLayout{};
    static constexpr int NTILES = KernelTraits::NTiles;
    #pragma unroll
    for(int itile = 0; itile < NTILES; itile++){
        clear(tCrD);
        #pragma unroll
        for(int ik = 0; ik < nk; ik++){
            int ik_next = (ik + 1) % nk;
            if(ik == nk - 1) {
                cp_async_wait<kStages - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStages;
            }
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));
            if(ik == 0){
                if(itile_to_read < NTILES){
                    for (size_t m = 0; m < size<1>(tAsA_copy); m++)
                    {
                        for (size_t k = 0; k < size<2>(tAsA_copy); k++)
                        {
                            if(get<0>(tAcA(0, m, k)) < residual){
                                cute::copy(g2s_tiled_copy, tAgA_copy(_, m, k, itile_to_read), tAsA_copy(_, m, k, ismem_write));
                            }
                        }
                    }
                    cute::copy(g2s_tiled_copy, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));
                    if(idx < KernelTraits::THREADS_SFA_COPY && (BM * iy + idx * KernelTraits::SFA_ELEMS_PER_COPY < M)) {
                        copy_1d((float*)&gSFA(0, 0, itile_to_read) + idx * KernelTraits::SFA_ELEMS_PER_COPY,  (float*)&sSFA(0, ismem_write) + idx*KernelTraits::SFA_ELEMS_PER_COPY);
                    }
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStages;
                }
                cp_async_fence();
            }
            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
       
        int sf_ind = itile / KernelTraits::TILES_PER_BLOCK;
        float sfb_val = sfb[sf_ind];
        #pragma unroll
        for(int i = 0; i < size<1>(tCrD); i++){ // (MMA, MMA_M, MMA_N) = (4, 4, 4)
            float sfa_val_1 = sSFA(sfa_tv(idx) + i * KernelTraits::MMA_WARP_M, ismem_read_sfa);
            float sfa_val_2  = sSFA(sfa_tv(idx) + 8 + i * KernelTraits::MMA_WARP_M, ismem_read_sfa);
            #pragma unroll
            for(int j = 0; j < size<2>(tCrD); j++){
                tCrD_fp32(0, i, j) += sfa_val_1 * sfb_val * float(tCrD(0, i, j));
                tCrD_fp32(1, i, j) += sfa_val_1 * sfb_val * float(tCrD(1, i, j));
                tCrD_fp32(2, i, j) += sfa_val_2 * sfb_val * float(tCrD(2, i, j));
                tCrD_fp32(3, i, j) += sfa_val_2 * sfb_val * float(tCrD(3, i, j));
            }
        } 
        ismem_read_sfa = (ismem_read_sfa + 1) % kStages;
        __syncthreads();
    }

    auto tCrBias = make_tensor<float>(Layout<Shape<_2, Int<size<2>(tCrD_fp32)>>>{});
    auto bias_threads = typename KernelTraits::BiasThreadLayout{};
    if constexpr (has_bias){
        #pragma unroll
        for(int i = 0; i<size<2>(tCrD_fp32); i++){
            tCrBias(0, i) = sBias(bias_threads(idx) + i * KernelTraits::MMA_WARP_N);
            tCrBias(1, i) = sBias(1 + bias_threads(idx) + i * KernelTraits::MMA_WARP_N);
        }
        #pragma unroll
        for(int i = 0; i<size<1>(tCrD_fp32); i++){
            #pragma unroll
            for (int j = 0; j < size<2>(tCrD_fp32) ; j++)
            {
                tCrD_fp32(0, i, j) += tCrBias(0, j);
                tCrD_fp32(1, i, j) += tCrBias(1, j);
                tCrD_fp32(2, i, j) += tCrBias(0, j);
                tCrD_fp32(3, i, j) += tCrBias(1, j);
            }
        }
    }

    auto sC = make_tensor(make_smem_ptr(C_shm), SmemLayoutC{});
    auto r2s_tiled_copy_c = make_tiled_copy_C(typename KernelTraits::R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD_fp32);  
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  

    typename KernelTraits::S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  

    int pipe = size<2>(tCsC_r2s);

    auto cC = make_identity_tensor(make_shape(size<0>(gD), size<1>(gD)));
    auto tCcC = s2g_thr_copy_c.partition_D(cC);

    for(int i = 0; i< size<1>(tCrC_r2s); i++){
        for(int j = 0; j < size<2>(tCrC_r2s); j+=pipe){
            for(int step = 0; step < pipe; ++step){
                auto fragment = make_tensor_like<output_t>(tCrC_r2s(_, i, j + step));
                cute::copy(tCrC_r2s(_, i, j + step), fragment);
                cute::copy(r2s_tiled_copy_c, fragment, tCsC_r2s(_, 0, step)); 
            }
            __syncthreads();
            if (get<0>(tCcC(0, i, j / pipe)) < residual){
                cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0), tCgC_s2g(_, i, j / pipe));                
            }
            __syncthreads();
        }
    }
}

template <bool has_bias, typename accum_type>
void fp8_kernel_launch(void* Aptr, void* sfa, void* Bptr, void* sfb, void* bias_ptr, void* out, int M, int N, int K, cudaStream_t stream) {
    int TMA_ALIGNED_M = ((M + sizeof(float) - 1) / sizeof(float)) * sizeof(float); // SIZEOF(float) = 4 
    BLOCK_K_SWITCH(K_, M_SWITCH( 
        using KernelTraits = gemm_traits<BM, BN, 3, K_, WARP_ROW, WARP_COL, has_bias, accum_type, bfloat16_t>;
        auto kernel = &gemm_fp8_kernel<KernelTraits>;
        int BX = (N + KernelTraits::BN - 1) / KernelTraits::BN;
        int BY = (M + KernelTraits::BM - 1) / KernelTraits::BM;
        dim3 block(KernelTraits::NUM_THREADS);
        dim3 gridDim(BX, BY);
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, KernelTraits::SmemSize);
        kernel<<<gridDim, KernelTraits::NUM_THREADS, KernelTraits::SmemSize, stream>>>((float_e4m3_t*)Aptr, (float*)sfa, (float_e4m3_t*)Bptr, (float*)sfb, (float*)bias_ptr, out, M, N, K, TMA_ALIGNED_M);
        C10_CUDA_KERNEL_LAUNCH_CHECK();))
}

template<bool use_fast_accum>
void fp8_bias_gemm_cuda(void* Aptr, void* SFA, void* Bptr, void* SFB, void* bias_ptr, void* out, int M, int N, int K, cudaStream_t stream){
    using accum_type = std::conditional_t<use_fast_accum, half_t, float>;
    fp8_kernel_launch<true, accum_type>(Aptr, SFA, Bptr, SFB, bias_ptr, out, M, N, K, stream);
}
// template<bool use_fast_accum>
// void fp8_gemm_cuda(void* Aptr, void* SFA, void* Bptr, void* SFB, void* out, int M, int N, int K, cudaStream_t stream){
//     using accum_type = std::conditional_t<use_fast_accum, half_t, float>;
//     BLOCK_K_SWITCH(num_acc_upcast_steps, fp8_kernel_launch<false, num_acc_upcast_steps, accum_type>(Aptr, SFA, Bptr, SFB, nullptr, out, M, N, K, stream);) 
// }

// template void fp8_gemm_cuda<true>(void* Aptr, void* SFA, void* Bptr, void* SFB, void* out, int M, int N, int K, cudaStream_t stream);
// template void fp8_gemm_cuda<false>(void* Aptr, void* SFA, void* Bptr, void* SFB, void* out, int M, int N, int K, cudaStream_t stream);

template void fp8_bias_gemm_cuda<true>(void* Aptr, void* SFA, void* Bptr, void* SFB, void* bias_ptr, void* out, int M, int N, int K, cudaStream_t stream);
template void fp8_bias_gemm_cuda<false>(void* Aptr, void* SFA, void* Bptr, void* SFB, void* bias_ptr, void* out, int M, int N, int K, cudaStream_t stream);
}; // namespace sm89