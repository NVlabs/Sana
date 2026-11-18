#pragma once
#include <torch/python.h>
#include <cute/arch/mma_sm100_umma.hpp>
#include "utils.hpp"
#include "exceptions.hpp"

namespace blockwise{
struct MulticastConfig {
    int num_multicast;
    bool is_multicast_on_a;

    MulticastConfig(const int& num_multicast, const bool& is_multicast_on_a):
        num_multicast(num_multicast), is_multicast_on_a(is_multicast_on_a) {
        DG_HOST_ASSERT(1 <= num_multicast and num_multicast <= 2);
    }
};

struct SharedMemoryConfig {
    int smem_size;
    int swizzle_a_mode;
    int swizzle_b_mode;
    int swizzle_cd_mode;
};

struct ThreadConfig {
    int num_threads;

    // SM90
    int num_tma_threads;
    int num_math_threads;

    // SM100
    int num_non_epilogue_threads;
    int num_epilogue_threads;

    static ThreadConfig sm90(const int& num_tma_threads,
                             const int& num_math_threads) {
        auto config = ThreadConfig();
        config.num_threads = num_tma_threads + num_math_threads;
        config.num_tma_threads = num_tma_threads;
        config.num_math_threads = num_math_threads;
        return config;
    }

    static ThreadConfig sm100(const int& num_non_epilogue_threads,
                              const int& num_epilogue_threads) {
        auto config = ThreadConfig();
        config.num_threads = num_non_epilogue_threads + num_epilogue_threads;
        config.num_non_epilogue_threads = num_non_epilogue_threads;
        config.num_epilogue_threads = num_epilogue_threads;
        return config;
    }
};

template<int SM>
struct GemmConfig{};
// {
//     // Templated configs
    
//     at::ScalarType ab_dtype, cd_dtype;
//     bool with_accumulation;
//     int block_m, block_n, block_k;
//     int num_stages, num_last_stages;

//     // Templated device configs
//     int num_sms;

//     // Structured configs
//     MulticastConfig multicast_config;
//     SharedMemoryConfig smem_config;
//     ThreadConfig thread_config;
// };


template <> 
struct GemmConfig<90>
{
    at::ScalarType ab_dtype = torch::kFloat8_e4m3fn;
    at::ScalarType cd_dtype = torch::kBFloat16;
    bool with_accumulation = false; 
    int block_m = 256;
    int block_n = 128;
    int block_k = 128;
    int num_stages = 3;
    int num_last_stages = 2;
    int num_sms = 132;
    MulticastConfig multicast_config{2, true};
    SharedMemoryConfig smem_config{216240, 128, 128, 128};
    ThreadConfig thread_config = ThreadConfig::sm90(128, 256);
};

};