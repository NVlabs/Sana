/**
 * @file utils.cuh
 * @brief Low-level CUDA utility functions for memory operations and synchronization.
 *
 * This header provides optimized PTX assembly wrappers for memory operations
 * that bypass cache hierarchy or use specific memory ordering semantics.
 * These are critical for achieving peak bandwidth in multi-GPU communication.
 *
 * ## Memory Operation Types
 *
 * - **Non-allocating stores (st_na)**: Bypass L1 cache to avoid polluting it
 *   with data that won't be reused locally
 * - **Non-caching loads (ld_nc)**: Bypass L1 cache for streaming reads
 * - **Acquire/Release**: Memory ordering for synchronization
 * - **System scope (sys)**: Visibility across all GPUs, not just this one
 *
 * ## Cache Hints
 *
 * - L1::no_allocate: Don't allocate in L1 on miss (streaming pattern)
 * - L2::256B: Use 256-byte L2 cache lines
 * - volatile: Bypass all caches, always go to memory
 */

#pragma once
#include <stdint.h>

// =============================================================================
// PTX Instruction Selection
// =============================================================================

/**
 * Store instruction macro. When DISABLE_AGGRESSIVE_PTX_INSTRS is not defined,
 * uses non-allocating stores to avoid polluting L1 cache with write-only data.
 */
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

/**
 * Load instruction macro. When DISABLE_AGGRESSIVE_PTX_INSTRS is not defined,
 * uses non-caching loads optimized for streaming access patterns.
 */
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global.L2::256B"
#endif

namespace ltx_kernels {

// =============================================================================
// Round-Robin SM Distribution Helpers
// =============================================================================

/**
 * @brief Compute target rank for a given SM using round-robin distribution.
 *
 * Round-robin assignment ensures all SMs are utilized even when num_sms
 * is not evenly divisible by world_size.
 *
 * @param sm_id The SM/block ID (blockIdx.x)
 * @param world_size Total number of ranks
 * @return Target rank for this SM
 */
__device__ __forceinline__ int get_target_rank(int sm_id, int world_size) { return sm_id % world_size; }

/**
 * @brief Compute local SM index within a rank's SM group.
 *
 * With round-robin, SM i is the (i / world_size)-th SM assigned to its rank.
 *
 * @param sm_id The SM/block ID (blockIdx.x)
 * @param world_size Total number of ranks
 * @return Local index of this SM within its assigned rank's group
 */
__device__ __forceinline__ int get_rank_local_sm_id(int sm_id, int world_size) { return sm_id / world_size; }

/**
 * @brief Compute number of SMs assigned to a specific rank.
 *
 * With round-robin distribution:
 *   - Ranks [0, extra) get (base + 1) SMs each
 *   - Ranks [extra, world_size) get base SMs each
 * where base = num_sms / world_size, extra = num_sms % world_size
 *
 * @param target_rank The rank to query
 * @param num_sms Total number of SMs launched
 * @param world_size Total number of ranks
 * @return Number of SMs assigned to target_rank
 */
__device__ __forceinline__ int get_num_sms_for_rank(int target_rank, int num_sms, int world_size) {
  int base_sms = num_sms / world_size;
  int extra_sms = num_sms % world_size;
  return base_sms + (target_rank < extra_sms ? 1 : 0);
}

// =============================================================================
// Control Flow
// =============================================================================

/**
 * @brief Triggers a GPU trap (fatal error).
 *
 * Used for unrecoverable errors like synchronization timeout.
 * Causes the kernel to abort and report an error to the host.
 */
__device__ __forceinline__ void trap() { asm("trap;"); }

// =============================================================================
// Memory Ordering Operations (for synchronization)
// =============================================================================

/**
 * @brief System-scope store with release ordering.
 *
 * Ensures all prior memory operations are visible before this store.
 * System scope means visibility across all GPUs (for IPC communication).
 *
 * @param ptr Pointer to global memory
 * @param val Value to store
 */
__device__ __forceinline__ void st_release_sys_global(const int *ptr, int val) {
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

/**
 * @brief System-scope store with relaxed ordering.
 *
 * No ordering guarantees - fastest store but requires external synchronization.
 *
 * @param ptr Pointer to global memory
 * @param val Value to store
 */
__device__ __forceinline__ void st_relaxed_sys_global(const int *ptr, int val) {
  asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

/**
 * @brief CTA-scope store with release ordering.
 *
 * Ensures visibility within the thread block (CTA = Cooperative Thread Array).
 *
 * @param ptr Pointer to global memory
 * @param val Value to store
 */
__device__ __forceinline__ void st_release_cta(const int *ptr, int val) {
  asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

/**
 * @brief System-scope load with acquire ordering (32-bit).
 *
 * Ensures subsequent memory operations are ordered after this load.
 * System scope for IPC visibility across GPUs.
 *
 * @param ptr Pointer to global memory
 * @return Loaded value
 */
__device__ __forceinline__ int ld_acquire_sys_global(const int *ptr) {
  int ret;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

/**
 * @brief System-scope load with acquire ordering (64-bit).
 *
 * @param ptr Pointer to global memory
 * @return Loaded value
 */
__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t *ptr) {
  uint64_t ret;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

/**
 * @brief GPU-scope load with acquire ordering.
 *
 * Visibility limited to this GPU (not for IPC).
 *
 * @param ptr Pointer to global memory
 * @return Loaded value
 */
__device__ __forceinline__ int ld_acquire_global(const int *ptr) {
  int ret;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

/**
 * @brief Volatile load bypassing all caches.
 *
 * Always reads from memory, never from cache. Used for polling
 * synchronization variables that may be updated by other GPUs.
 *
 * @param ptr Pointer to global memory
 * @return Loaded value
 */
__device__ __forceinline__ int ld_volatile_global(const int *ptr) {
  int ret;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

// =============================================================================
// Optimized Bulk Memory Operations
// =============================================================================

/**
 * @brief Non-allocating 128-bit store.
 *
 * Stores an int4 (128 bits / 16 bytes) without allocating in L1 cache.
 * Optimal for write-streaming patterns where data won't be read locally.
 *
 * @param ptr Destination pointer (must be 16-byte aligned)
 * @param value Data to store
 */
__device__ __forceinline__ void st_na_global(const int4 *ptr, const int4 &value) {
  asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z),
               "r"(value.w));
}

/**
 * @brief Non-caching 128-bit load.
 *
 * Loads an int4 bypassing L1 cache with optimized L2 caching (256B lines).
 * Optimal for read-streaming patterns.
 *
 * @param ptr Source pointer (must be 16-byte aligned)
 * @return Loaded int4 value
 */
__device__ __forceinline__ int4 ld_nc_global(const int4 *ptr) {
  int4 ret;
  asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
}

/**
 * @brief Barrier synchronization pattern for multi-GPU communication.
 *
 * This function implements a barrier synchronization protocol used in All2All
 * and AllGather operations. It signals completion to target ranks and waits
 * for all expected signals to arrive before resetting the barrier.
 *
 * Protocol:
 * 1. Thread 0 of each block signals completion to the target rank
 * 2. Block 0 waits for all ranks to signal (with timeout protection)
 * 3. Once all signals received, reset the barrier counters
 *
 * @param barrier_signal_ptrs Array of pointers to barrier signal buffers for each rank
 * @param target_rank The rank this block is sending data to
 * @param rank This GPU's rank
 * @param world_size Total number of GPUs/ranks
 * @param expected_count Number of signals expected (typically num_sms_per_rank)
 * @param sm_id The SM/block ID (blockIdx.x)
 * @param thread_id The thread ID within the block (threadIdx.x)
 * @param timeout_cycles Number of cycles to wait before timeout
 */
__device__ __forceinline__ void barrier_wait_and_reset(int **barrier_signal_ptrs, int target_rank, int rank,
                                                       int world_size, int expected_count, int sm_id, int thread_id,
                                                       uint64_t timeout_cycles) {
  // Release: fence so peers see our data writes, then sync before signaling.
  __threadfence_system();
  __syncthreads();

  // Thread 0 signals completion to target rank
  if (thread_id == 0) {
    atomicAdd_system(barrier_signal_ptrs[target_rank] + rank, 1);
  }

  // Synchronize before checking signals
  __syncthreads();

  // Only block 0 waits for all signals and resets the barrier
  if (sm_id == 0 && thread_id < world_size) {
    auto start_time = clock64();
    while (true) {
      // Acquire: seeing the signal guarantees the peer's data is visible.
      int recv_count = ld_acquire_sys_global(barrier_signal_ptrs[rank] + thread_id);
      if (recv_count == expected_count) {
        break;
      }
      if (clock64() - start_time >= timeout_cycles) {
        printf("All2All barrier timeout: rank=%d, waiting_for_source=%d, expected=%d, got=%d\n", rank, thread_id,
               expected_count, recv_count);
        trap();
      }
    }
    // Reset barrier for next use
    atomicSub_system(barrier_signal_ptrs[rank] + thread_id, expected_count);
  }
}

/**
 * @brief Barrier synchronization for round-robin SM distribution.
 *
 * Similar to barrier_wait_and_reset, but handles the case where SMs are
 * distributed round-robin across ranks, resulting in different target ranks
 * receiving different numbers of signals.
 *
 * With round-robin: target ranks [0, extra) receive (base + 1) signals from
 * each source, and target ranks [extra, world_size) receive base signals
 * from each source. Note that ALL sources send the same count to a given
 * receiver - the count depends on the receiver's rank position.
 *
 * @param barrier_signal_ptrs Array of pointers to barrier signal buffers for each rank
 * @param target_rank The rank this block is sending data to
 * @param rank This GPU's rank
 * @param world_size Total number of GPUs/ranks
 * @param num_sms Total number of SMs launched (used to compute expected counts)
 * @param sm_id The SM/block ID (blockIdx.x)
 * @param thread_id The thread ID within the block (threadIdx.x)
 * @param timeout_cycles Number of cycles to wait before timeout
 */
__device__ __forceinline__ void barrier_wait_and_reset_roundrobin(int **barrier_signal_ptrs, int target_rank, int rank,
                                                                  int world_size, int num_sms, int sm_id, int thread_id,
                                                                  uint64_t timeout_cycles) {
  // Release: fence so peers see our data writes, then sync before signaling.
  __threadfence_system();
  __syncthreads();

  // Thread 0 signals completion to target rank
  if (thread_id == 0) {
    atomicAdd_system(barrier_signal_ptrs[target_rank] + rank, 1);
  }

  // Synchronize before checking signals
  __syncthreads();

  // Only block 0 waits for all signals and resets the barrier
  // Each thread handles one source rank
  if (sm_id == 0 && thread_id < world_size) {
    // All sources send the same number of signals to THIS receiver.
    // The count depends on how many SMs target this rank (the receiver).
    int expected_from_each_source = get_num_sms_for_rank(rank, num_sms, world_size);

    auto start_time = clock64();
    while (true) {
      // Acquire: seeing the signal guarantees the peer's data is visible.
      int recv_count = ld_acquire_sys_global(barrier_signal_ptrs[rank] + thread_id);
      if (recv_count == expected_from_each_source) {
        break;
      }
      if (clock64() - start_time >= timeout_cycles) {
        printf("All2All barrier timeout (roundrobin): rank=%d, waiting_for_source=%d, expected=%d, got=%d\n", rank,
               thread_id, expected_from_each_source, recv_count);
        trap();
      }
    }
    // Reset barrier for next use
    atomicSub_system(barrier_signal_ptrs[rank] + thread_id, expected_from_each_source);
  }
}

} // namespace ltx_kernels
