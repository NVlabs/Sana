#pragma once

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm::sm90 {

template <int N_, typename MMA>
struct FP8MMA {

    template <size_t ...Idx>
    __forceinline__ __device__ static void call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    __forceinline__ __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_/2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N>
struct FP8MMASelector {

    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 8) return MMA_64x8x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 16) return MMA_64x16x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 24) return MMA_64x24x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 32) return MMA_64x32x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 40) return MMA_64x40x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 48) return MMA_64x48x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 56) return MMA_64x56x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 64) return MMA_64x64x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 72) return MMA_64x72x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 80) return MMA_64x80x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 88) return MMA_64x88x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 96) return MMA_64x96x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 104) return MMA_64x104x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 112) return MMA_64x112x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 120) return MMA_64x120x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 136) return MMA_64x136x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 144) return MMA_64x144x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 152) return MMA_64x152x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 160) return MMA_64x160x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 168) return MMA_64x168x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 176) return MMA_64x176x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 184) return MMA_64x184x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 192) return MMA_64x192x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 200) return MMA_64x200x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 208) return MMA_64x208x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 216) return MMA_64x216x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 224) return MMA_64x224x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 232) return MMA_64x232x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 240) return MMA_64x240x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 248) return MMA_64x248x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 256) return MMA_64x256x32_F32E4M3E4M3_SS_TN();
    }

    static constexpr auto select_type() {
        return FP8MMA<N, decltype(select_mma())>();
    }

    using type = decltype(select_type());
};

template <int N_, typename MMA>
struct BF16MMA {

    template <size_t ...Idx>
    __forceinline__ __device__ static void call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    __forceinline__ __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_/2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 16;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N>
struct BF16MMASelector {

    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 8) return MMA_64x8x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 16) return MMA_64x16x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 24) return MMA_64x24x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 32) return MMA_64x32x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 40) return MMA_64x40x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 48) return MMA_64x48x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 56) return MMA_64x56x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 64) return MMA_64x64x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 72) return MMA_64x72x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 80) return MMA_64x80x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 88) return MMA_64x88x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 96) return MMA_64x96x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 104) return MMA_64x104x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 112) return MMA_64x112x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 120) return MMA_64x120x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 128) return MMA_64x128x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 136) return MMA_64x136x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 144) return MMA_64x144x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 152) return MMA_64x152x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 160) return MMA_64x160x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 168) return MMA_64x168x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 176) return MMA_64x176x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 184) return MMA_64x184x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 192) return MMA_64x192x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 200) return MMA_64x200x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 208) return MMA_64x208x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 216) return MMA_64x216x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 224) return MMA_64x224x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 232) return MMA_64x232x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 240) return MMA_64x240x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 248) return MMA_64x248x16_F32BF16BF16_SS<Major::K, Major::K>();
        if constexpr (N == 256) return MMA_64x256x16_F32BF16BF16_SS<Major::K, Major::K>();
    }

    static constexpr auto select_type() {
        return BF16MMA<N, decltype(select_mma())>();
    }

    using type = decltype(select_type());
};


template <typename dtype_t>
struct SM90_U32x2_STSM_N {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                     :: "l"(smem_dst), "r"(src[0]), "r"(src[1]));
    }
};

struct SM90_U32x2_LDSM_N {
    __device__ __forceinline__ static void
    copy(uint32_t& dst_0, uint32_t& dst_1, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(dst_0), "=r"(dst_1)
                     : "l"(smem_src));
    }
};

struct SM90_U32x4_LDSM_N {
    __device__ __forceinline__ static void
    copy(uint32_t& dst_0, uint32_t& dst_1, uint32_t& dst_2, uint32_t& dst_3, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(dst_0), "=r"(dst_1), "=r"(dst_2), "=r"(dst_3)
                     : "l"(smem_src));
    }
};

__forceinline__ __device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__forceinline__ __device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__forceinline__ __device__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

template <int N>
__forceinline__ __device__ void warpgroup_wait() {
    DG_STATIC_ASSERT(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// TODO: replace with CUTLASS solution
union GmmaDescriptor {
    __host__ __device__ constexpr GmmaDescriptor() noexcept: desc_(0) {}

    __host__ __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept: desc_(desc) {}

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept: desc_(t.desc_) {}

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept: desc_(t.desc_) {}

    __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor const &t) noexcept {
        desc_ = t.desc_;
        return *this;
    }

    __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept {
        desc_ = t.desc_;
        return *this;
    }

    uint64_t desc_;
    uint32_t reg32_[2];
    uint16_t reg16_[4];

    struct {
        uint16_t start_address_: 14, : 2;
        uint16_t leading_byte_offset_: 14, : 2;
        uint16_t stride_byte_offset_: 14, : 2;
        uint8_t : 1, base_offset_: 3, : 4;
        uint8_t : 6, layout_type_: 2;
    } bitfield;

    // Decay to an `uint64_t`
    __host__ __device__ constexpr operator uint64_t() const noexcept { return desc_; }
};

template <class PointerType>
__device__ GmmaDescriptor make_smem_desc(PointerType smem_ptr, const int& layout_type,
                                         const int& leading_byte_offset = 0,
                                         const int& stride_byte_offset = 1024) {
    GmmaDescriptor desc;
    const auto& uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

__device__ __forceinline__ void
tma_copy(void const* desc_ptr, uint64_t* barrier_ptr, void* smem_ptr,
         const uint32_t& crd_0, const uint32_t& crd_1, const uint32_t& num_tma_multicast = 1) {
    constexpr auto cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    if (num_tma_multicast == 1) {
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd_0, crd_1);
    } else if (cute::block_rank_in_cluster() == 0) {
        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, barrier_ptr, (1 << num_tma_multicast) - 1, cache_hint, smem_ptr, crd_0, crd_1);
    }
}

__device__ __forceinline__ void
tma_3d_copy(void const* desc_ptr, uint64_t* barrier_ptr, void* smem_ptr,
            const uint32_t& crd_0, const uint32_t& crd_1, const uint32_t& crd_2) {
    constexpr auto cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    cute::SM90_TMA_LOAD_3D::copy(desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd_0, crd_1, crd_2);
}

// Tensormap related
__device__ __forceinline__ void tensor_map_release_cta() {
    asm volatile ("fence.proxy.tensormap::generic.release.cta;");
}

__device__ __forceinline__ void tensor_map_acquire_cta(const cute::TmaDescriptor* gmem_desc_ptr) {
    auto gmem_int_desc = reinterpret_cast<uint64_t>(gmem_desc_ptr);
    asm volatile ("fence.proxy.tensormap::generic.acquire.cta [%0], 128;" :: "l"(gmem_int_desc) : "memory");
}

__device__ __forceinline__ void tensor_map_replace_global_addr_in_smem(cute::TmaDescriptor* smem_desc, const void* new_addr) {
    auto smem_int_desc = static_cast<uint32_t>(__cvta_generic_to_shared(smem_desc));
    const auto new_int64_addr = reinterpret_cast<uint64_t>(new_addr);
    asm volatile ("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" :: "r"(smem_int_desc), "l"(new_int64_addr));
}

__device__ __forceinline__ void tensor_map_replace_global_inner_dim_stride_in_smem(cute::TmaDescriptor* smem_desc, const uint32_t& new_dim, const uint64_t& new_stride) {
    auto smem_int_desc = __cvta_generic_to_shared(smem_desc);
    asm volatile ("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" :: "l"(smem_int_desc), "r"(new_dim));
#if ((__CUDACC_VER_MAJOR__ > 12) or ((__CUDACC_VER_MAJOR__ == 12) and (__CUDACC_VER_MINOR__ >= 3)))
    asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;" :: "l"(smem_int_desc), "l"(new_stride));
#else
    DG_STATIC_ASSERT(false, "Invalid CUDA version");
#endif
}

} // namespace `deep_gemm::sm90`
