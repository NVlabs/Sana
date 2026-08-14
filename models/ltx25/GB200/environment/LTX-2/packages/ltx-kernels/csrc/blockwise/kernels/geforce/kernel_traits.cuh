#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>

#include "mma_sm89_fp16.hpp"
#include "mma_traits_sm89_fp16.hpp"

using namespace cute;

template<int BYTES> struct BytesToType {};
template<> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};
template<> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<int BM_, int BN_, int KStages_, int K_, int WARP_ROW_=2, int WARP_COL_=2, bool HasBias_=false, typename accum_t_=cutlass::half_t, typename out_t_=cutlass::bfloat16_t>
struct gemm_traits {
    static constexpr int BLOCK_SIZE = 128;
    static constexpr int K = K_;
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = 128;
    static constexpr int TILES_PER_BLOCK = BLOCK_SIZE / BK;
    static constexpr int NUM_SFB_PER_STEP = BN / BLOCK_SIZE;
    static constexpr int NTiles = K / BK;
    static constexpr int KSF = K / BLOCK_SIZE; 
    static constexpr int KStages = KStages_;
    static constexpr int WARP_ROW = WARP_ROW_;
    static constexpr int WARP_COL = WARP_COL_;
    static constexpr int NUM_WARPS = WARP_ROW * WARP_COL;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int MMA_WARP_M = WARP_ROW * 16;
    static constexpr int MMA_WARP_N = WARP_COL * 8;
    static constexpr int MMA_WARP_K = 32;
    using accum_t = accum_t_;
    using out_t = out_t_;
    using SwizzleLayoutO = std::conditional_t<
        std::is_same_v<out_t_, cutlass::bfloat16_t>,
        Swizzle<3, 3, 3>,
        Swizzle<2, 4, 3>
    >;
    using SwizzleLayoutAB = Swizzle<2, 4, 3>;
    using MMA_Atom_SM89 = std::conditional_t<
        std::is_same_v<accum_t, cutlass::half_t>,
        MMA_Atom<SM89_16x8x32_F16E4M3E4M3F16_TN>,
        MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>
    >;
    static constexpr int INPUT_ELEMS_PER_COPY = sizeof(uint128_t) / sizeof(float_e4m3_t);
    static constexpr int OUTPUT_ELEMS_PER_COPY = sizeof(uint128_t) / sizeof(out_t_); 
    static constexpr int THREADS_PER_ROW = BK / INPUT_ELEMS_PER_COPY;
    using GMEMLayout = Layout< Shape <Int<NUM_THREADS / THREADS_PER_ROW>, Int<THREADS_PER_ROW>>, Stride<Int<THREADS_PER_ROW>, _1>>;
    using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float_e4m3_t>;
    using G2STiledCopy = decltype(
        make_tiled_copy(
            G2SCopyAtom{},
            GMEMLayout{},
            Layout<Shape<_1, Int<INPUT_ELEMS_PER_COPY>>>{}
        )
    );
    using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, float_e4m3_t>;
    using S2RCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, float_e4m3_t>;
    using SmemLayoutAtom = decltype(composition(
        Swizzle<2, 4, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStages>{}))
    );
    using SmemLayoutB = decltype(
        tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<KStages>{}))
    );
    using MMATile = decltype(
        make_tiled_mma(
            MMA_Atom_SM89{},
            Layout<Shape<Int<WARP_ROW>, Int<WARP_COL>, _1>>{},
            Tile<Int<MMA_WARP_M>, Int<MMA_WARP_N>, Int<MMA_WARP_K>>{}
        )
    );

    static constexpr int ELEMS_PER_TILE = MMA_WARP_M * MMA_WARP_N;
    static constexpr int NUM_ELEMS_PER_WRITE = NUM_THREADS * sizeof(cute::uint128_t) / sizeof(out_t_);
    static constexpr int OUT_PIPE = NUM_ELEMS_PER_WRITE / ELEMS_PER_TILE;
    // using SmemLayoutC = Layout<Shape<Int<BM>, Int<BN>>, Stride<Int<BN>, Int<1>>>;

    using SmemLayoutC = decltype(
            make_layout( 
                make_shape(Int<MMA_WARP_M>{}, Int<MMA_WARP_N*OUT_PIPE>{}), 
                make_stride(Int<MMA_WARP_N*OUT_PIPE>{}, Int<1>{})
            )
    );  
    static constexpr int THREADS_PER_ROW_WRITE = MMA_WARP_N * OUT_PIPE / OUTPUT_ELEMS_PER_COPY;
    using R2SCopyAtomC = Copy_Atom<UniversalCopy<typename BytesToType<2*sizeof(out_t)>::Type>, out_t>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, out_t>;
    using S2GCopyC = decltype(make_tiled_copy(S2GCopyAtomC{},
                            make_layout(make_shape(Int<NUM_THREADS / THREADS_PER_ROW_WRITE>{}, Int<THREADS_PER_ROW_WRITE>{}),
                                        make_stride(Int<THREADS_PER_ROW_WRITE>{}, Int<1>{})),
                            make_layout(make_shape(Int<1>{}, Int<OUTPUT_ELEMS_PER_COPY>{}))));
    
   using G2SBiasCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<float>, float>;
   using G2SBiasCopy = decltype(make_tiled_copy(G2SBiasCopyAtom{}, make_layout(
                    make_shape(Int<1>{},Int<BN>{}), make_stride(Int<BN>{}, Int<1>{})),
                    make_layout(make_shape(Int<1>{},Int<1>{}), make_stride(Int<1>{}, Int<1>{}))));
    using sfa_copy_vtype = float;
    static constexpr int SFA_ELEMS_PER_COPY = sizeof(sfa_copy_vtype)/sizeof(float);
    static constexpr int THREADS_SFA_COPY = BM * sizeof(float) / sizeof(sfa_copy_vtype); 
    // using G2SSFACopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, float>;
    static constexpr bool HasBias = HasBias_;
    using SmemLayoutBias = Layout<Shape<Int<1>, Int<BN>>, Stride<Int<BN>, Int<1>>>;
    using SmemLayoutSFA = Layout<Shape<Int<BM>, Int<KStages>>, Stride<Int<1>, Int<BM>>>;
    using BiasThreadLayout = Layout<Shape<Shape<_4, _8>, Shape<Int<WARP_ROW>, Int<WARP_COL>>>, Stride<Stride<_2, _0>, Stride<_0, _8>>>;
    using SFAThreadLayout = Layout<Shape<Shape<_4, _8>, Shape<Int<WARP_ROW>, Int<WARP_COL>>>, Stride<Stride<_0, _1>, Stride<_16, _0>>>;
    static constexpr int SmemSize = cute::max(cute::cosize(SmemLayoutA{})+cute::cosize(SmemLayoutB{}), cute::cosize(SmemLayoutC{})*sizeof(out_t)) + cute::cosize(SmemLayoutBias{}) * sizeof(float) + cute::cosize(SmemLayoutSFA{})*sizeof(float);
};
