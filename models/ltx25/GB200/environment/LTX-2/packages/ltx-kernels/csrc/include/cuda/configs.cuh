/**
 * @file configs.cuh
 * @brief Configuration constants and compile-time settings for ltx-kernels.
 *
 * This header defines the tunable parameters and constants used throughout
 * the ltx-kernels communication library. These values are chosen to balance
 * performance across different GPU architectures.
 */

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ltx_kernels {
// =============================================================================
// Synchronization Configuration
// =============================================================================

/**
 * @brief Default barrier timeout in seconds.
 *
 * If a barrier wait exceeds this timeout, the kernel traps to indicate a deadlock or
 * communication failure. All2All converts it to clock cycles using the device's peak SM clock
 * (cudaDeviceGetAttribute(cudaDevAttrClockRate)), so the wall-clock guard holds regardless of GPU.
 */
constexpr double DEFAULT_BARRIER_TIMEOUT_SECONDS = 10.0;

// =============================================================================
// Hardware Limits
// =============================================================================

/**
 * @brief Maximum number of peer GPUs supported for IPC communication.
 *
 * This limits the size of static arrays for buffer pointers and barrier signals.
 * Set to 8 to support up to 8-way tensor parallelism (common for DGX systems).
 */
constexpr int MAX_NUM_PEERS = 8;

// =============================================================================
// Kernel Configuration
// =============================================================================

/**
 * @brief Default number of threads per block for All2All kernels.
 *
 * Used by send_recv_all2all and gather_heads kernels. The value 512 provides
 * good occupancy while leaving registers for complex pointer arithmetic.
 */
constexpr int DEFAULT_KERNEL_THREADS = 512;

/**
 * @brief Number of threads per block for the AllGather kernel.
 *
 * AllGather uses more threads (1024) because its memory access pattern
 * is simpler (no head selection), allowing higher thread-level parallelism.
 */
constexpr int ALLGATHER_KERNEL_THREADS = 1024;

} // namespace ltx_kernels

// =============================================================================
// Torch/CUDA Compatibility Fixes
// =============================================================================

/*
 * PyTorch sometimes disables CUDA half/bfloat16 operators and conversions
 * to avoid ambiguity in template resolution. We re-enable them here since
 * our kernels explicitly handle these types.
 */

#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif
