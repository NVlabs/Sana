/**
 * @file api.cuh
 * @brief CUDA kernel launch function declarations for All2All operations.
 *
 * This header provides the host-callable interface for launching the All2All
 * CUDA kernels. These functions handle template instantiation and kernel
 * configuration based on the tensor data type.
 */

#pragma once

#include <ATen/cuda/CUDADataType.h>
#include <vector>

namespace ltx_kernels {
namespace all2all {
namespace all2all_cuda {

/**
 * @brief Launches the All2All head redistribution kernel.
 *
 * Redistributes attention heads across GPUs:
 *   Input:  [batch, num_tokens, num_heads, head_size] per GPU
 *   Output: [batch, total_tokens, num_heads/world_size, head_size] per GPU
 *
 * @param buffer_ptrs Device array of pointers to each rank's data buffer
 * @param barrier_signal_ptrs Device array of pointers to barrier signals
 * @param x Source tensor data pointer
 * @param prefix_rank_tokens Cumulative token counts per rank (device memory)
 * @param rank This GPU's rank (0 to world_size-1)
 * @param world_size Total number of GPUs
 * @param batch_size Batch dimension size
 * @param total_tokens Sum of tokens across all ranks
 * @param num_tokens Number of tokens on this rank
 * @param num_heads Total number of attention heads
 * @param head_size Size of each attention head
 * @param stream CUDA stream for async execution
 * @param num_sms Number of SMs to use for the kernel
 * @param tensor_dtype Data type (BFloat16 or Float8_e4m3fn)
 */
void all2all_head_launch(void **buffer_ptrs, int **barrier_signal_ptrs, void *x, int *prefix_rank_tokens, int rank,
                         int world_size, int batch_size, int total_tokens, int num_tokens, int num_heads, int head_size,
                         cudaStream_t stream, int num_sms, at::ScalarType tensor_dtype, uint64_t timeout_cycles);

/**
 * @brief Launches the gather heads kernel (inverse of all2all_head_launch).
 *
 * Redistributes tokens back to original head distribution:
 *   Input:  [batch, total_tokens, heads_per_rank, head_size] per GPU
 *   Output: [batch, rank_tokens[rank], num_heads, head_size] per GPU
 *
 * @param buffer_ptrs Device array of pointers to each rank's data buffer
 * @param barrier_signal_ptrs Device array of pointers to barrier signals
 * @param x Source tensor data pointer
 * @param rank_tokens Token count for each rank (device memory)
 * @param prefix_rank_tokens Cumulative token counts (device memory)
 * @param rank This GPU's rank
 * @param world_size Total number of GPUs
 * @param batch_size Batch dimension size
 * @param total_tokens Sum of tokens across all ranks
 * @param num_heads Total number of attention heads (reconstructed)
 * @param head_size Size of each attention head
 * @param stream CUDA stream for async execution
 * @param num_sms Number of SMs to use for the kernel
 * @param tensor_dtype Data type (BFloat16 or Float8_e4m3fn)
 */
void all2all_head_gather_launch(void **buffer_ptrs, int **barrier_signal_ptrs, void *x, const int *rank_tokens,
                                int *prefix_rank_tokens, int rank, int world_size, int batch_size, int total_tokens,
                                int num_heads, int head_size, cudaStream_t stream, int num_sms,
                                at::ScalarType tensor_dtype, uint64_t timeout_cycles);

/**
 * @brief Launches the AllGather kernel for sequence tokens.
 *
 * Gathers sequence tokens from all ranks:
 *   Input:  [batch, seqlen, hidden_dim] per GPU
 *   Output: [batch, total_tokens, hidden_dim] per GPU (identical on all)
 *
 * @param buffer_ptrs Device array of pointers to each rank's data buffer
 * @param barrier_signal_ptrs Device array of pointers to barrier signals
 * @param x Source tensor data pointer
 * @param prefix_rank_tokens Cumulative token counts (device memory)
 * @param rank This GPU's rank
 * @param world_size Total number of GPUs
 * @param batch_size Batch dimension size
 * @param seqlen Number of tokens on this rank
 * @param hidden_dim Hidden dimension size (num_heads * head_size)
 * @param total_tokens Sum of tokens across all ranks
 * @param stream CUDA stream for async execution
 * @param num_sms Number of SMs to use for the kernel
 * @param tensor_dtype Data type (BFloat16 or Float8_e4m3fn)
 */
void allgather_launch(void **buffer_ptrs, int **barrier_signal_ptrs, void *x, int *prefix_rank_tokens, int rank,
                      int world_size, int batch_size, int seqlen, int hidden_dim, int total_tokens, cudaStream_t stream,
                      int num_sms, at::ScalarType tensor_dtype, uint64_t timeout_cycles);

} // namespace all2all_cuda
} // namespace all2all
} // namespace ltx_kernels
