/**
 * @file all2all_heads.cu
 * @brief CUDA kernels for All2All attention head redistribution.
 *
 * This file implements the GPU kernels for redistributing attention heads across
 * multiple GPUs using IPC-based direct memory access. The kernels are designed
 * for tensor-parallel transformer models where attention heads need to be
 * exchanged between GPUs.
 *
 * ## Algorithm Overview
 *
 * The kernels use a direct-write approach where each GPU writes its data directly
 * to the target GPU's memory buffer via IPC. This avoids intermediate copies and
 * achieves near-peak memory bandwidth utilization.
 *
 * ## SM Work Distribution (Round-Robin)
 *
 * SMs are distributed round-robin among target ranks to handle non-divisible SM counts:
 *   - SM i writes to rank (i % world_size)
 *   - With 132 SMs and 8 GPUs: ranks 0-3 get 17 SMs, ranks 4-7 get 16 SMs
 *   - Each SM group processes all tokens for its assigned target rank
 *   - Within each group, SMs cooperate to cover all tokens in strided fashion
 *
 * ## Synchronization Protocol
 *
 * After data transfer, a barrier synchronization ensures all ranks have completed:
 *   1. Each SM atomically increments the target rank's barrier counter for this rank
 *   2. SM 0 waits until it has received signals from all ranks
 *   3. Barrier counters are reset for the next operation
 */

#include "cuda/configs.cuh"
#include "cuda/exceptions.cuh"
#include "cuda/utils.cuh"
#include <ATen/cuda/CUDADataType.h>

namespace ltx_kernels {
namespace all2all {
namespace all2all_cuda {

/**
 * @brief All2All kernel for redistributing attention heads across GPUs.
 *
 * This kernel performs the "send" phase of All2All: each GPU writes its assigned
 * subset of attention heads to all other GPUs. The data layout transformation is:
 *
 *   Source: [batch, num_tokens, num_heads, head_size]
 *   Dest:   [batch, total_tokens, heads_per_rank, head_size]
 *
 * Each GPU writes heads [target_rank * heads_per_rank : (target_rank+1) * heads_per_rank]
 * to target_rank's buffer at token offset prefix_rank_tokens[rank].
 *
 * ## Memory Layout
 *
 * Input tensor x (row-major, contiguous):
 *   - Batch dimension: outermost
 *   - Token dimension: batch_stride = num_tokens * num_heads * head_size
 *   - Head dimension: token_stride = num_heads * head_size
 *   - Head element: head_stride = head_size
 *
 * Output buffer (per target rank):
 *   - Similar layout but with heads_per_rank instead of num_heads
 *   - Tokens from this rank placed at offset prefix_rank_tokens[rank]
 *
 * ## Thread Block Organization
 *
 * Each thread block handles multiple tokens cooperatively:
 *   - Threads are organized in a 2D logical grid (rows=tokens, cols=elements)
 *   - Each thread copies 16 bytes (int4) per iteration
 *   - num_threads_per_token = (heads_per_rank * head_size) / elements_per_thread
 *   - num_tokens_per_copy = num_threads / num_threads_per_token
 *
 * @tparam ELEM_T Element type (at::BFloat16 or at::Float8_e4m3fn)
 * @param buffer_ptrs Device array of pointers to each rank's data buffer
 * @param barrier_signal_ptrs Device array of pointers to each rank's barrier signals
 * @param x Source tensor data pointer
 * @param rank This GPU's rank
 * @param world_size Total number of GPUs
 * @param batch_size Number of batches
 * @param num_tokens Number of tokens on this rank
 * @param num_heads Total number of attention heads
 * @param head_size Size of each attention head
 * @param total_tokens Sum of tokens across all ranks
 * @param prefix_rank_tokens Cumulative token counts for offset calculation
 */
template <typename ELEM_T>
__global__ void send_recv_all2all(void **buffer_ptrs, int **barrier_signal_ptrs, void *x, int rank, int world_size,
                                  int batch_size, int num_tokens, int num_heads, int head_size, int total_tokens,
                                  int *prefix_rank_tokens, uint64_t timeout_cycles) {
  // Grid dimensions
  int num_sms = gridDim.x;
  int sm_id = blockIdx.x;
  int num_threads = blockDim.x;

  // === SM Work Distribution (Round-Robin) ===
  // Use modular assignment to handle num_sms not divisible by world_size.
  // This ensures all SMs are utilized: some ranks get ceil(num_sms/world_size)
  // SMs, others get floor(num_sms/world_size) SMs.
  int64_t target_rank = get_target_rank(sm_id, world_size);
  int64_t rank_local_sm_id = get_rank_local_sm_id(sm_id, world_size);
  int64_t num_sms_for_this_rank = get_num_sms_for_rank(target_rank, num_sms, world_size);

  // === Head Assignment ===
  // Heads are partitioned evenly: rank i gets heads [i*hpr : (i+1)*hpr]
  int64_t heads_per_rank = num_heads / world_size;
  int64_t head_id = target_rank * heads_per_rank; // Starting head for target rank

  // === Thread Mapping ===
  // Each thread copies an int4 (16 bytes) per memory operation
  // Threads form a 2D grid: (tokens_per_copy, threads_per_token)
  int64_t num_elems_per_thread = sizeof(int4) / sizeof(ELEM_T);
  int64_t num_threads_per_token = heads_per_rank * head_size / num_elems_per_thread;
  int64_t num_tokens_per_copy = num_threads / num_threads_per_token;

  // 2D thread coordinates within the logical grid
  int64_t copy_thr_col_idx = threadIdx.x % num_threads_per_token; // Element offset
  int64_t copy_thr_row_idx = threadIdx.x / num_threads_per_token; // Token offset

  // Get target rank's buffer pointer
  auto ptr = reinterpret_cast<void *>(static_cast<int8_t *>(buffer_ptrs[target_rank]));

  // Use 64-bit arithmetic to avoid overflow for large tensors
  int64_t num_tokens_64b = int64_t(num_tokens);
  int64_t num_heads_64b = int64_t(num_heads);
  int64_t head_size_64b = int64_t(head_size);

  // === Main Copy Loop ===
  // Iterate over batches and tokens, with SMs in the same group
  // working on different token ranges in strided fashion
  for (int64_t batch_ind = 0; batch_ind < batch_size; batch_ind++) {
    // Strided token iteration: each SM in the group handles different token ranges
    for (int64_t token_idx = rank_local_sm_id * num_tokens_per_copy; token_idx < num_tokens;
         token_idx += num_tokens_per_copy * num_sms_for_this_rank) {
      int64_t copy_token_idx = token_idx + copy_thr_row_idx;
      // Destination token index accounts for this rank's offset in the global sequence
      int64_t dst_token_idx = prefix_rank_tokens[rank] + copy_token_idx;

      if (copy_token_idx >= num_tokens)
        break;

      // === Pointer Arithmetic ===
      // Source: Read from this rank's input tensor at [batch, token, head_id:head_id+hpr, :]
      // Note: We read a contiguous chunk of heads starting at head_id
      int4 *shuffled_x_ptr =
          reinterpret_cast<int4 *>(reinterpret_cast<uint8_t *>(x) +
                                   batch_ind * num_tokens_64b * num_heads_64b * head_size_64b * sizeof(ELEM_T) +
                                   copy_token_idx * num_heads_64b * head_size_64b * sizeof(ELEM_T) +
                                   head_id * head_size_64b * sizeof(ELEM_T)) +
          copy_thr_col_idx;

      // Destination: Write to target rank's buffer at [batch, dst_token, :, :]
      // The buffer has layout [batch, total_tokens, heads_per_rank, head_size]
      int4 *shuffled_buffer_ptr =
          reinterpret_cast<int4 *>(reinterpret_cast<uint8_t *>(ptr) +
                                   batch_ind * total_tokens * heads_per_rank * head_size_64b * sizeof(ELEM_T) +
                                   dst_token_idx * heads_per_rank * head_size_64b * sizeof(ELEM_T)) +
          copy_thr_col_idx;

      // Non-allocating store to avoid polluting L1 cache
      st_na_global(shuffled_buffer_ptr, __ldg(shuffled_x_ptr));
    }
  }

  // === Barrier Synchronization ===
  // Signal completion to target rank and wait for all ranks to finish
  barrier_wait_and_reset_roundrobin(barrier_signal_ptrs, target_rank, rank, world_size, num_sms, sm_id, threadIdx.x,
                                    timeout_cycles);
}

/**
 * @brief All2All kernel for gathering attention heads back to original distribution.
 *
 * This kernel performs the inverse of send_recv_all2all: it gathers heads from
 * all ranks back to reconstruct the original tensor layout. Each GPU reads from
 * its local buffer and writes its portion of heads to all target ranks.
 *
 * Data layout transformation:
 *   Source: [batch, total_tokens, heads_per_rank, head_size]  (per GPU)
 *   Dest:   [batch, rank_tokens[target], num_heads, head_size]  (per target GPU)
 *
 * ## Memory Layout
 *
 * Input tensor x (this rank's portion after send_recv_all2all):
 *   - Contains all tokens but only heads_per_rank heads
 *   - Layout: [batch, total_tokens, heads_per_rank, head_size]
 *
 * Output buffer (per target rank):
 *   - Contains only that rank's tokens but all heads
 *   - Layout: [batch, rank_tokens[target], num_heads, head_size]
 *   - This rank writes heads [rank * heads_per_rank : (rank+1) * heads_per_rank]
 *
 * @tparam ELEM_T Element type (at::BFloat16 or at::Float8_e4m3fn)
 * @param buffer_ptrs Device array of pointers to each rank's data buffer
 * @param barrier_signal_ptrs Device array of pointers to barrier signals
 * @param x Source tensor data (this rank's buffer after send_recv)
 * @param rank This GPU's rank
 * @param world_size Total number of GPUs
 * @param batch_size Number of batches
 * @param num_heads Total number of heads (reconstructed)
 * @param head_size Size of each attention head
 * @param rank_tokens Number of tokens for each rank
 * @param total_tokens Sum of tokens across all ranks
 * @param prefix_rank_tokens Cumulative token counts for offset calculation
 */
template <typename ELEM_T>
__global__ void gather_heads(void **buffer_ptrs, int **barrier_signal_ptrs, void *x, int rank, int world_size,
                             int batch_size, int num_heads, int head_size, const int *__restrict__ rank_tokens,
                             int total_tokens, int *prefix_rank_tokens, uint64_t timeout_cycles) {
  // Grid dimensions
  int num_sms = gridDim.x;
  int sm_id = blockIdx.x;
  int num_threads = blockDim.x;

  // === SM Work Distribution (Round-Robin) ===
  // Same partitioning as send_recv_all2all
  int64_t target_rank = get_target_rank(sm_id, world_size);
  int64_t rank_local_sm_id = get_rank_local_sm_id(sm_id, world_size);
  int64_t num_sms_for_this_rank = get_num_sms_for_rank(target_rank, num_sms, world_size);
  int64_t heads_per_rank = num_heads / world_size;

  // === Thread Mapping ===
  int64_t num_elems_per_thread = sizeof(int4) / sizeof(ELEM_T);
  int64_t num_threads_per_token = heads_per_rank * head_size / num_elems_per_thread;
  int64_t num_tokens_per_copy = num_threads / num_threads_per_token;

  int64_t copy_thr_col_idx = threadIdx.x % num_threads_per_token;
  int64_t copy_thr_row_idx = threadIdx.x / num_threads_per_token;

  // Number of tokens owned by target rank
  const int64_t tgt_tokens = int64_t(rank_tokens[target_rank]);

  // This rank writes its heads at offset [rank * heads_per_rank] in the output
  int64_t head_idx = rank * heads_per_rank;
  int64_t num_heads_64b = int64_t(num_heads);
  int64_t head_size_64b = int64_t(head_size);
  int64_t total_tokens_64b = int64_t(total_tokens);

  // Get target rank's buffer pointer
  auto ptr = reinterpret_cast<void *>(static_cast<int8_t *>(buffer_ptrs[target_rank]));

  // === Main Copy Loop ===
  // Process target rank's tokens: read from global position, write to local position
  for (int64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int64_t token_idx = rank_local_sm_id * num_tokens_per_copy; token_idx < tgt_tokens;
         token_idx += num_tokens_per_copy * num_sms_for_this_rank) {
      int64_t copy_token = token_idx + copy_thr_row_idx;
      if (copy_token >= tgt_tokens)
        break;

      // Source: Read from global token position (target rank's tokens in our buffer)
      int64_t src_token_idx = prefix_rank_tokens[target_rank] + copy_token;
      // Destination: Write to local token position in target's buffer
      int64_t dst_token_idx = copy_token;

      // Source pointer: our input tensor at [batch, src_token, :, :]
      int4 *shuffled_x_ptr =
          reinterpret_cast<int4 *>(reinterpret_cast<uint8_t *>(x) +
                                   batch_idx * total_tokens_64b * heads_per_rank * head_size_64b * sizeof(ELEM_T) +
                                   src_token_idx * heads_per_rank * head_size_64b * sizeof(ELEM_T)) +
          copy_thr_col_idx;

      // Destination pointer: target's buffer at [batch, dst_token, head_idx:head_idx+hpr, :]
      int4 *shuffled_buffer_ptr =
          reinterpret_cast<int4 *>(reinterpret_cast<uint8_t *>(ptr) +
                                   batch_idx * tgt_tokens * num_heads_64b * head_size_64b * sizeof(ELEM_T) +
                                   dst_token_idx * num_heads_64b * head_size_64b * sizeof(ELEM_T) +
                                   head_idx * head_size_64b * sizeof(ELEM_T)) +
          copy_thr_col_idx;

      st_na_global(shuffled_buffer_ptr, __ldg(shuffled_x_ptr));
    }
  }

  // === Barrier Synchronization ===
  barrier_wait_and_reset_roundrobin(barrier_signal_ptrs, target_rank, rank, world_size, num_sms, sm_id, threadIdx.x,
                                    timeout_cycles);
}

/**
 * @brief Host function to launch the gather_heads kernel.
 *
 * Selects the appropriate template instantiation based on tensor data type
 * and launches the kernel with the specified number of SMs.
 *
 * @param buffer_ptrs Device array of buffer pointers
 * @param barrier_signal_ptrs Device array of barrier signal pointers
 * @param x Input tensor data pointer
 * @param rank_tokens Token count per rank (device memory)
 * @param prefix_rank_tokens Cumulative token counts (device memory)
 * @param rank This GPU's rank
 * @param world_size Total number of GPUs
 * @param batch_size Number of batches
 * @param total_tokens Sum of tokens across all ranks
 * @param num_heads Total number of attention heads
 * @param head_size Size of each attention head
 * @param stream CUDA stream for async execution
 * @param num_sms Number of SMs to launch
 * @param tensor_dtype Data type (BFloat16 or Float8_e4m3fn)
 */
void all2all_head_gather_launch(void **buffer_ptrs, int **barrier_signal_ptrs, void *x, const int *rank_tokens,
                                int *prefix_rank_tokens, int rank, int world_size, int batch_size, int total_tokens,
                                int num_heads, int head_size, cudaStream_t stream, int num_sms,
                                at::ScalarType tensor_dtype, uint64_t timeout_cycles) {
  do {
    if (tensor_dtype == at::ScalarType::BFloat16) {
      gather_heads<at::BFloat16><<<num_sms, DEFAULT_KERNEL_THREADS, 0, stream>>>(
          buffer_ptrs, barrier_signal_ptrs, x, rank, world_size, batch_size, num_heads, head_size, rank_tokens,
          total_tokens, prefix_rank_tokens, timeout_cycles);
    } else if (tensor_dtype == at::ScalarType::Float8_e4m3fn) {
      gather_heads<at::Float8_e4m3fn><<<num_sms, DEFAULT_KERNEL_THREADS, 0, stream>>>(
          buffer_ptrs, barrier_signal_ptrs, x, rank, world_size, batch_size, num_heads, head_size, rank_tokens,
          total_tokens, prefix_rank_tokens, timeout_cycles);
    }

    // Check for kernel launch errors
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      EPException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e));
      fprintf(stderr, "%s\n", cuda_exception.what());
      throw cuda_exception;
    }
  } while (0);
}

/**
 * @brief Host function to launch the send_recv_all2all kernel.
 *
 * Selects the appropriate template instantiation based on tensor data type
 * and launches the kernel with the specified number of SMs.
 *
 * @param buffer_ptrs Device array of buffer pointers
 * @param barrier_signal_ptrs Device array of barrier signal pointers
 * @param x Input tensor data pointer
 * @param prefix_rank_tokens Cumulative token counts (device memory)
 * @param rank This GPU's rank
 * @param world_size Total number of GPUs
 * @param batch_size Number of batches
 * @param total_tokens Sum of tokens across all ranks
 * @param num_tokens Number of tokens on this rank
 * @param num_heads Total number of attention heads
 * @param head_size Size of each attention head
 * @param stream CUDA stream for async execution
 * @param num_sms Number of SMs to launch
 * @param tensor_dtype Data type (BFloat16 or Float8_e4m3fn)
 */
void all2all_head_launch(void **buffer_ptrs, int **barrier_signal_ptrs, void *x, int *prefix_rank_tokens, int rank,
                         int world_size, int batch_size, int total_tokens, int num_tokens, int num_heads, int head_size,
                         cudaStream_t stream, int num_sms, at::ScalarType tensor_dtype, uint64_t timeout_cycles) {
  do {
    if (tensor_dtype == at::ScalarType::BFloat16) {
      send_recv_all2all<at::BFloat16><<<num_sms, DEFAULT_KERNEL_THREADS, 0, stream>>>(
          buffer_ptrs, barrier_signal_ptrs, x, rank, world_size, batch_size, num_tokens, num_heads, head_size,
          total_tokens, prefix_rank_tokens, timeout_cycles);
    } else if (tensor_dtype == at::ScalarType::Float8_e4m3fn) {
      send_recv_all2all<at::Float8_e4m3fn><<<num_sms, DEFAULT_KERNEL_THREADS, 0, stream>>>(
          buffer_ptrs, barrier_signal_ptrs, x, rank, world_size, batch_size, num_tokens, num_heads, head_size,
          total_tokens, prefix_rank_tokens, timeout_cycles);
    }

    // Check for kernel launch errors
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      EPException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e));
      fprintf(stderr, "%s\n", cuda_exception.what());
      throw cuda_exception;
    }
  } while (0);
}

} // namespace all2all_cuda
} // namespace all2all
} // namespace ltx_kernels
