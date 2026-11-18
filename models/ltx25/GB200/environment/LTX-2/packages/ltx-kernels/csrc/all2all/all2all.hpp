/**
 * @file all2all.hpp
 * @brief High-performance All2All communication primitives for multi-GPU tensor parallelism.
 *
 * This library provides efficient All2All communication operations optimized for transformer
 * models using tensor parallelism. It uses CUDA IPC (Inter-Process Communication) for
 * zero-copy data transfer between GPUs in the same node.
 *
 * ## Architecture Overview
 *
 * The All2All class manages shared memory buffers accessible by all GPUs via IPC handles.
 * Each GPU allocates a contiguous memory region containing:
 *   - Data buffer: Stores tensor data for exchange
 *   - Barrier signals: Synchronization counters for coordination
 *   - GPU pointer arrays: Device-accessible pointers to all peer buffers
 *
 * Memory Layout (per GPU):
 * ```
 * |<---- tensor_bytes ---->|<-- barrier signals -->|<-- buffer_ptrs_gpu -->|<-- barrier_signal_ptrs_gpu -->|
 * |      Data Buffer       |    MAX_PEERS * int    |   MAX_PEERS * void*   |      MAX_PEERS * int*         |
 * ```
 *
 * ## Supported Operations
 *
 * 1. **send_recv_heads**: Redistributes attention heads across GPUs (All2All)
 *    - Input: [batch, tokens, heads, head_size] on each GPU
 *    - Output: [batch, total_tokens, heads/world_size, head_size] on each GPU
 *
 * 2. **gather_heads**: Inverse of send_recv_heads
 *    - Gathers distributed heads back to original distribution
 *
 * 3. **allgather**: Gathers sequence data from all ranks
 *    - Each GPU contributes its local tokens to form the complete sequence
 *
 * ## Thread Safety
 *
 * - The class is NOT thread-safe. Each thread/process should have its own instance.
 * - Multiple CUDA streams may use the same instance sequentially.
 * - The `destroy()` method MUST be called before destruction to properly release IPC handles.
 *
 * ## Usage Example
 *
 * ```cpp
 * // Initialize on each GPU
 * auto comm = All2All(rank, world_size, max_tokens, hidden_dim, num_sms, dtype);
 *
 * // Exchange IPC handles (via NCCL or other collective)
 * auto my_handle = comm.get_local_ipc_handle();
 * // ... gather all handles ...
 * comm.sync(all_handles);
 *
 * // Set token distribution for current batch
 * comm.set_rank_tokens({128, 128, 128, 128});  // tokens per rank
 *
 * // Perform All2All on attention heads
 * auto result = comm.send_recv_heads(input_tensor, copy_output=false);
 *
 * // Clean up
 * comm.destroy();
 * ```
 */

#pragma once

#include "cuda/configs.cuh"
#include "event.hpp"
#include <cmath>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <torch/types.h>
#include <tuple>
#include <vector>

namespace ltx_kernels {
namespace all2all {

/**
 * @class All2All
 * @brief Manages All2All communication state and operations for multi-GPU tensor parallelism.
 *
 * This class encapsulates the IPC-based communication infrastructure needed for
 * efficient All2All operations. It maintains shared memory buffers, barrier signals,
 * and provides methods for head-parallel tensor redistribution.
 */
struct All2All {
private:
  int rank;             ///< This GPU's rank (0 to world_size-1)
  int world_size;       ///< Total number of GPUs in the communication group
  int num_sms;          ///< Number of SMs to use for kernel launches
  int max_tokens;       ///< Maximum number of tokens the buffer was allocated for
  int64_t num_elems;    ///< Number of elements in the data buffer (tokens * hidden_dim)
  int64_t tensor_bytes; ///< Size of the data buffer in bytes

  /// Host array of pointers to each rank's data buffer (GPU memory)
  void *buffer_ptrs[MAX_NUM_PEERS] = {nullptr};
  /// Device-accessible array of buffer pointers (copied to GPU)
  void **buffer_ptrs_gpu = nullptr;

  /// Host array of pointers to each rank's barrier signal buffer
  int *barrier_signal_ptrs[MAX_NUM_PEERS] = {nullptr};
  /// Device-accessible array of barrier signal pointers
  int **barrier_signal_ptrs_gpu = nullptr;

  /// IPC handles for sharing memory between processes
  cudaIpcMemHandle_t ipc_handlers[MAX_NUM_PEERS];

  at::ScalarType tensor_dtype; ///< Data type of tensors (BFloat16 or Float8_e4m3fn)
  bool destroyed = false;      ///< Flag to track if resources have been released

  int total_tokens;                      ///< Sum of tokens across all ranks for current batch
  int rank_tokens[MAX_NUM_PEERS];        ///< Number of tokens on each rank
  int prefix_rank_tokens[MAX_NUM_PEERS]; ///< Cumulative sum of tokens (for offset calculation)
  int *rank_tokens_gpu = nullptr;        ///< Device copy of rank_tokens
  int *prefix_rank_tokens_gpu = nullptr; ///< Device copy of prefix_rank_tokens

  /// Device peak SM clock in Hz (from cudaDeviceGetAttribute(cudaDevAttrClockRate)), queried
  /// once at construction. Used to convert a wall-clock timeout in seconds to barrier cycles.
  double sm_clock_hz_ = 0.0;

  /// All2All barrier timeout in GPU clock cycles. The constructor sets it from
  /// DEFAULT_BARRIER_TIMEOUT_SECONDS and the queried SM clock; raise it (set_timeout_seconds)
  /// to tolerate large cross-rank kernel-launch skew during the first torch.compile forward,
  /// where one rank's recompile can delay its launch past the steady-state timeout.
  uint64_t timeout_cycles_ = 0;

public:
  /**
   * @brief Constructs an All2All communication manager.
   *
   * Allocates GPU memory for the local data buffer, barrier signals, and pointer arrays.
   * The IPC handle for the local buffer is created and can be retrieved via get_local_ipc_handle().
   *
   * @param rank This GPU's rank in the communication group (0-indexed)
   * @param world_size Total number of GPUs/ranks
   * @param num_tokens Maximum number of tokens this rank will handle
   * @param hidden_dim Hidden dimension size (heads * head_size)
   * @param num_sms Number of CUDA SMs to use for kernel execution
   * @param tensor_dtype Data type for tensors (BFloat16 or Float8_e4m3fn)
   * @param timeout_seconds Initial barrier timeout in seconds (see set_timeout_seconds); may be
   *        raised/reset at runtime for the first torch.compile forward
   */
  All2All(int rank, int world_size, int num_tokens, int hidden_dim, int num_sms, at::ScalarType tensor_dtype,
          double timeout_seconds = DEFAULT_BARRIER_TIMEOUT_SECONDS);

  /**
   * @brief Destructor - warns if destroy() was not called.
   *
   * @warning Always call destroy() explicitly before the destructor to properly
   *          release IPC handles. Failing to do so may leak resources.
   */
  ~All2All() noexcept(false);

  /**
   * @brief Synchronizes IPC handles from all ranks and opens remote memory mappings.
   *
   * This method must be called after all ranks have created their All2All instances
   * and exchanged IPC handles via an external collective (e.g., NCCL allgather).
   *
   * @param all_gathered_handles Vector of IPC handles from all ranks (indexed by rank)
   */
  void sync(const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles);

  /**
   * @brief Returns the IPC handle for this rank's shared buffer.
   *
   * The returned handle should be gathered across all ranks and passed to sync().
   *
   * @return pybind11::bytearray containing the CUDA IPC handle (CUDA_IPC_HANDLE_SIZE bytes)
   */
  pybind11::bytearray get_local_ipc_handle() const;

  /**
   * @brief Creates a tensor view or copy of the local output buffer.
   *
   * @param x Reference tensor for options (dtype, device)
   * @param batch_size Batch dimension size
   * @param out_tokens Output token dimension size
   * @param out_heads Output heads dimension size
   * @param head_size Head dimension size
   * @param should_copy If true, copies data to a new tensor; if false, returns a view
   * @param stream CUDA stream for async copy
   * @return Tensor with shape [batch_size, out_tokens, out_heads, head_size]
   */
  at::Tensor get_local_buffer_tensor(at::Tensor &x, int batch_size, int out_tokens, int out_heads, int head_size,
                                     bool should_copy, cudaStream_t stream);

  /**
   * @brief Releases all GPU resources and closes IPC handles.
   *
   * This method MUST be called before the object is destroyed. It synchronizes
   * the device, closes remote IPC mappings, and frees local GPU memory.
   */
  void destroy();

  /**
   * @brief Performs All2All communication to redistribute attention heads.
   *
   * Redistributes tensor from [batch, local_tokens, all_heads, head_size] to
   * [batch, all_tokens, local_heads, head_size]. Each rank sends its portion
   * of heads to the corresponding target rank.
   *
   * @param x Input tensor with shape [batch, num_tokens, num_heads, head_size]
   * @param copy_output If true, returns a copy; if false, returns a view of the IPC buffer
   * @return Tensor with shape [batch, total_tokens, num_heads/world_size, head_size]
   */
  at::Tensor send_recv_heads(at::Tensor &x, bool copy_output);

  /**
   * @brief Performs inverse All2All to gather heads back to original distribution.
   *
   * Inverse of send_recv_heads(). Redistributes from [batch, all_tokens, local_heads, head_size]
   * back to [batch, local_tokens, all_heads, head_size].
   *
   * @param x Input tensor with shape [batch, total_tokens, heads_per_rank, head_size]
   * @param copy_output If true, returns a copy; if false, returns a view of the IPC buffer
   * @return Tensor with shape [batch, rank_tokens[rank], num_heads, head_size]
   */
  at::Tensor gather_heads(at::Tensor &x, bool copy_output);

  /**
   * @brief Gathers sequence tokens from all ranks.
   *
   * Each rank contributes its local sequence tokens, which are gathered into
   * a complete sequence on all ranks.
   *
   * @param x Input tensor with shape [batch, seqlen, num_heads, head_size]
   * @param copy_output If true, returns a copy; if false, returns a view of the IPC buffer
   * @return Tensor with shape [batch, total_tokens, num_heads, head_size]
   */
  at::Tensor allgather(at::Tensor &x, bool copy_output);

  /**
   * @brief Sets the token count for each rank in the current batch.
   *
   * Must be called before send_recv_heads(), gather_heads(), or allgather()
   * to configure the token distribution. This allows variable-length sequences
   * across ranks.
   *
   * @param rank_num_tokens Vector of token counts, one per rank (must have world_size elements)
   */
  void set_rank_tokens(const std::vector<int> &rank_num_tokens);

  /**
   * @brief Sets the all2all barrier timeout in seconds.
   *
   * Converted to GPU clock cycles using the device's peak SM clock (queried at construction).
   * Relaxes deadlock detection during the first torch.compile forward, where asymmetric
   * per-rank recompilation can delay a rank's kernel launch beyond the steady-state timeout.
   * Reset to the default for steady-state replay.
   */
  void set_timeout_seconds(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
      throw std::invalid_argument("All2All timeout (seconds) must be finite and non-negative");
    }
    // Saturate rather than overflow the float->uint64 cast (out-of-range conversion is UB).
    const double cycles = seconds * sm_clock_hz_;
    const double max_cycles = static_cast<double>(std::numeric_limits<uint64_t>::max());
    timeout_cycles_ = cycles >= max_cycles ? std::numeric_limits<uint64_t>::max() : static_cast<uint64_t>(cycles);
  }
};

} // namespace all2all
} // namespace ltx_kernels
