/**
 * @file allgather.cu
 * @brief CUDA kernel for AllGather operation using IPC-based direct memory access.
 *
 * This file implements the GPU kernel for gathering sequence tokens from all GPUs
 * into a complete sequence on each GPU. Unlike the head redistribution kernels,
 * this kernel preserves the head dimension and only gathers across the token
 * (sequence) dimension.
 *
 * ## Algorithm Overview
 *
 * Each GPU broadcasts its local tokens to all other GPUs' buffers:
 *   - GPU i writes its tokens to position [prefix_rank_tokens[i]] in each buffer
 *   - After completion, all buffers contain the full sequence [0:total_tokens]
 *
 * ## Use Case
 *
 * This is typically used after tensor-parallel computation to reconstruct the
 * full sequence for operations that require global context (e.g., output projection).
 */

#include "cuda/configs.cuh"
#include "cuda/exceptions.cuh"
#include "cuda/utils.cuh"
#include <ATen/cuda/CUDADataType.h>

namespace ltx_kernels {
namespace all2all {
namespace all2all_cuda {

/**
 * @brief AllGather kernel to collect sequence tokens from all ranks.
 *
 * Each GPU writes its local sequence tokens to all other GPUs' buffers at the
 * appropriate offset. After synchronization, all GPUs have the complete sequence.
 *
 * Data layout transformation:
 *   Input per GPU:  [batch, seqlen, hidden_dim]
 *   Output per GPU: [batch, total_tokens, hidden_dim]  (identical on all GPUs)
 *
 * ## Memory Layout
 *
 * Input tensor x (contiguous):
 *   - Shape: [batch, seqlen, hidden_dim]
 *   - hidden_dim = num_heads * head_size (flattened)
 *
 * Output buffer (per target rank, after gather):
 *   - Shape: [batch, total_tokens, hidden_dim]
 *   - This rank's tokens placed at offset rank_tokens_prefix[rank]
 *
 * ## Thread Mapping
 *
 * Similar to all2all_heads, threads cooperate to copy tokens:
 *   - Each thread copies 16 bytes (int4)
 *   - Threads per token = hidden_dim * sizeof(ELEM_T) / sizeof(int4)
 *   - Multiple tokens processed per thread block
 *
 * @tparam ELEM_T Element type (__nv_bfloat16 or at::Float8_e4m3fn)
 * @param x Source tensor data pointer (this rank's tokens)
 * @param buffer_ptrs Device array of pointers to each rank's data buffer
 * @param barrier_signal_ptrs Device array of pointers to barrier signals
 * @param batch_size Number of batches
 * @param seqlen Number of tokens on this rank
 * @param hidden_dim Hidden dimension size (num_heads * head_size)
 * @param world_size Total number of GPUs
 * @param rank This GPU's rank
 * @param total_tokens Sum of tokens across all ranks
 * @param rank_tokens_prefix Cumulative token counts (device memory)
 */
template <typename ELEM_T>
__global__ void allgather(void *x, void **buffer_ptrs, int **barrier_signal_ptrs, int batch_size, int seqlen,
                          int hidden_dim, int world_size, int rank, int total_tokens, int *rank_tokens_prefix,
                          uint64_t timeout_cycles) {

  // Grid dimensions
  int num_sms = gridDim.x;
  int sm_id = blockIdx.x;
  int num_threads = blockDim.x;

  // === SM Work Distribution (Round-Robin) ===
  // Use modular assignment to handle num_sms not divisible by world_size.
  // This ensures all SMs are utilized: some ranks get ceil(num_sms/world_size)
  // SMs, others get floor(num_sms/world_size) SMs.
  int tgt_rank = get_target_rank(sm_id, world_size);
  int rank_local_sm_id = get_rank_local_sm_id(sm_id, world_size);
  int num_sms_for_this_rank = get_num_sms_for_rank(tgt_rank, num_sms, world_size);

  // Get target rank's buffer pointer
  auto ptr = reinterpret_cast<void *>(static_cast<int8_t *>(buffer_ptrs[tgt_rank]));

  // === Thread Mapping ===
  // Each thread copies one int4 (16 bytes)
  int64_t num_elems_per_thread = sizeof(int4) / sizeof(ELEM_T);
  int64_t num_threads_per_token = hidden_dim / num_elems_per_thread;
  int64_t num_tokens_per_copy = num_threads / num_threads_per_token;

  // 2D thread coordinates
  int64_t copy_thr_col_idx = threadIdx.x % num_threads_per_token; // Element offset
  int64_t copy_thr_row_idx = threadIdx.x / num_threads_per_token; // Token offset

  // Use 64-bit arithmetic to avoid overflow
  int64_t hidden_dim_64b = int64_t(hidden_dim);
  int64_t total_tokens_64b = int64_t(total_tokens);
  int64_t seqlen_64b = int64_t(seqlen);

  // === Main Copy Loop ===
  // Broadcast this rank's tokens to all target ranks' buffers
  for (int64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    // Strided token iteration within SM group for this target rank
    for (int64_t token_idx = rank_local_sm_id * num_tokens_per_copy; token_idx < seqlen;
         token_idx += num_tokens_per_copy * num_sms_for_this_rank) {
      int64_t copy_token = token_idx + copy_thr_row_idx;
      if (copy_token >= seqlen)
        break;

      // Source: local token index in input tensor
      int64_t src_token_idx = copy_token;
      // Destination: global token index in output buffer
      // This rank's tokens start at prefix_rank_tokens[rank]
      int64_t dst_token_idx = copy_token + rank_tokens_prefix[rank];

      // Source pointer: input tensor at [batch, src_token, :]
      int4 *shuffled_x_ptr = reinterpret_cast<int4 *>(reinterpret_cast<uint8_t *>(x) +
                                                      batch_idx * seqlen_64b * hidden_dim_64b * sizeof(ELEM_T) +
                                                      src_token_idx * hidden_dim_64b * sizeof(ELEM_T)) +
                             copy_thr_col_idx;

      // Destination pointer: target buffer at [batch, dst_token, :]
      int4 *shuffled_buffer_ptr =
          reinterpret_cast<int4 *>(reinterpret_cast<uint8_t *>(ptr) +
                                   batch_idx * total_tokens_64b * hidden_dim_64b * sizeof(ELEM_T) +
                                   dst_token_idx * hidden_dim_64b * sizeof(ELEM_T)) +
          copy_thr_col_idx;

      // Non-allocating store for better cache behavior
      st_na_global(shuffled_buffer_ptr, __ldg(shuffled_x_ptr));
    }
  }

  // === Barrier Synchronization ===
  // Signal completion to target rank and wait for all ranks
  // Use round-robin variant since SM counts per rank may differ
  barrier_wait_and_reset_roundrobin(barrier_signal_ptrs, tgt_rank, rank, world_size, num_sms, sm_id, threadIdx.x,
                                    timeout_cycles);
}

/**
 * @brief Host function to launch the allgather kernel.
 *
 * Launches the AllGather kernel with the specified configuration.
 * Uses ALLGATHER_KERNEL_THREADS (1024) threads per block for higher
 * occupancy than the All2All kernels.
 *
 * @param buffer_ptrs Device array of buffer pointers
 * @param barrier_signal_ptrs Device array of barrier signal pointers
 * @param x Input tensor data pointer
 * @param prefix_rank_tokens Cumulative token counts (device memory)
 * @param rank This GPU's rank
 * @param world_size Total number of GPUs
 * @param batch_size Number of batches
 * @param seqlen Number of tokens on this rank
 * @param hidden_dim Hidden dimension size
 * @param total_tokens Sum of tokens across all ranks
 * @param stream CUDA stream for async execution
 * @param num_sms Number of SMs to launch
 * @param tensor_dtype Data type (BFloat16 or Float8_e4m3fn)
 */
void allgather_launch(void **buffer_ptrs, int **barrier_signal_ptrs, void *x, int *prefix_rank_tokens, int rank,
                      int world_size, int batch_size, int seqlen, int hidden_dim, int total_tokens, cudaStream_t stream,
                      int num_sms, at::ScalarType tensor_dtype, uint64_t timeout_cycles) {
  do {
    if (tensor_dtype == at::ScalarType::BFloat16) {
      allgather<at::BFloat16><<<num_sms, ALLGATHER_KERNEL_THREADS, 0, stream>>>(
          x, buffer_ptrs, barrier_signal_ptrs, batch_size, seqlen, hidden_dim, world_size, rank, total_tokens,
          prefix_rank_tokens, timeout_cycles);
    } else if (tensor_dtype == at::ScalarType::Float8_e4m3fn) {
      allgather<at::Float8_e4m3fn><<<num_sms, ALLGATHER_KERNEL_THREADS, 0, stream>>>(
          x, buffer_ptrs, barrier_signal_ptrs, batch_size, seqlen, hidden_dim, world_size, rank, total_tokens,
          prefix_rank_tokens, timeout_cycles);
    } else {
      EPException dtype_exception("allgather_launch", __FILE__, __LINE__, "Unsupported dtype");
      fprintf(stderr, "%s\n", dtype_exception.what());
      throw dtype_exception;
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
