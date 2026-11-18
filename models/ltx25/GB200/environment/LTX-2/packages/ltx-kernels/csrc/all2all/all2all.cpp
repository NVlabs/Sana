/**
 * @file all2all.cpp
 * @brief Implementation of All2All communication primitives for multi-GPU tensor parallelism.
 *
 * This file implements the All2All class which provides efficient inter-GPU communication
 * using CUDA IPC (Inter-Process Communication). The implementation supports:
 * - Head redistribution for tensor-parallel attention (send_recv_heads, gather_heads)
 * - Sequence gathering for cross-rank aggregation (allgather)
 *
 * All operations use a barrier-based synchronization protocol where each GPU writes
 * directly to remote GPU memory via IPC, then signals completion through atomic
 * operations on barrier counters.
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <c10/cuda/CUDAGuard.h>

#include <chrono>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/functional.h>
#include <torch/python.h>

#include "all2all.hpp"
#include "cuda/api.cuh"
#include "cuda/configs.cuh"

namespace ltx_kernels {
namespace all2all {

/**
 * Constructs the All2All communication manager.
 *
 * Memory Allocation Strategy:
 * The constructor allocates a single contiguous GPU memory block that contains:
 *   1. Data buffer (tensor_bytes): Space for tensor data exchange
 *   2. Barrier signals (MAX_NUM_PEERS * sizeof(int)): Per-rank completion counters
 *   3. Buffer pointers (MAX_NUM_PEERS * sizeof(void*)): GPU-accessible pointer array
 *   4. Barrier pointer array (MAX_NUM_PEERS * sizeof(int*)): GPU-accessible signal pointers
 *
 * This layout minimizes memory allocations and allows the entire region to be
 * shared via a single IPC handle.
 */
All2All::All2All(int rank, int world_size, int num_tokens, int hidden_dim, int num_sms, at::ScalarType tensor_dtype,
                 double timeout_seconds)
    : rank(rank), world_size(world_size), num_sms(num_sms), max_tokens(num_tokens), num_elems(0), tensor_bytes(0),
      tensor_dtype(tensor_dtype) {
  num_elems = int64_t(num_tokens) * int64_t(hidden_dim);
  tensor_bytes = num_elems * elementSize(tensor_dtype);

  // Derive the barrier timeout from the device's peak SM clock so the wall-clock guard is
  // correct on any GPU (the kernel counts SM cycles via clock64). Use cudaDeviceGetAttribute,
  // not cudaDeviceProp::clockRate, which was removed in CUDA 13. The attribute is in kHz.
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int sm_clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, device));
  sm_clock_hz_ = static_cast<double>(sm_clock_khz) * 1e3;
  set_timeout_seconds(timeout_seconds);

  // Calculate sizes for each region of the shared memory block
  int64_t ptrs_bytes = MAX_NUM_PEERS * sizeof(void *);
  int64_t barrier_signal_bytes = MAX_NUM_PEERS * sizeof(int);
  int64_t barrier_signal_ptrs_bytes = MAX_NUM_PEERS * sizeof(int *);

  // Allocate GPU memory for token count arrays (used by kernels)
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&rank_tokens_gpu), sizeof(int) * MAX_NUM_PEERS));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&prefix_rank_tokens_gpu), sizeof(int) * MAX_NUM_PEERS));

  // Allocate the main shared memory block and create IPC handle
  // Layout: [data_buffer | barrier_signals | buffer_ptrs | barrier_signal_ptrs]
  CUDA_CHECK(
      cudaMalloc(&buffer_ptrs[rank], tensor_bytes + barrier_signal_bytes + ptrs_bytes + barrier_signal_ptrs_bytes));
  CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handlers[rank], buffer_ptrs[rank]));

  // Set up pointers to each region within the allocated block
  buffer_ptrs_gpu =
      reinterpret_cast<void **>(static_cast<uint8_t *>(buffer_ptrs[rank]) + tensor_bytes + barrier_signal_bytes);
  barrier_signal_ptrs[rank] = reinterpret_cast<int *>(static_cast<uint8_t *>(buffer_ptrs[rank]) + tensor_bytes);
  barrier_signal_ptrs_gpu = reinterpret_cast<int **>(static_cast<uint8_t *>(buffer_ptrs[rank]) + tensor_bytes +
                                                     barrier_signal_bytes + ptrs_bytes);

  // Initialize barrier signals to zero
  CUDA_CHECK(cudaMemset(barrier_signal_ptrs[rank], 0, barrier_signal_bytes));
}

All2All::~All2All() noexcept(false) {
  if (!destroyed) {
    printf("WARNING: destroy() was not called, which can leak resources.\n");
    fflush(stdout);
    destroy();
  }
}

/**
 * Releases all allocated resources.
 *
 * This must be called explicitly before destruction to ensure proper cleanup of:
 * - IPC memory mappings to remote GPUs
 * - Local GPU memory allocations
 *
 * The method synchronizes the device to ensure all pending operations complete
 * before releasing resources.
 */
void All2All::destroy() {
  if (destroyed) {
    return;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Close IPC mappings to remote GPU memory (skip our own rank)
  // Only close handles that were actually opened via sync()
  for (int i = 0; i < world_size; i++) {
    if (i != rank && buffer_ptrs[i] != nullptr) {
      CUDA_CHECK(cudaIpcCloseMemHandle(buffer_ptrs[i]));
    }
  }

  // Free local GPU memory allocations
  CUDA_CHECK(cudaFree(buffer_ptrs[rank]));
  CUDA_CHECK(cudaFree(rank_tokens_gpu));
  CUDA_CHECK(cudaFree(prefix_rank_tokens_gpu));
  destroyed = true;
}

/**
 * Opens IPC memory mappings to all peer GPUs.
 *
 * This method processes IPC handles gathered from all ranks and opens memory
 * mappings to enable direct GPU-to-GPU memory access. After calling this method,
 * each GPU can read/write directly to any other GPU's buffer via buffer_ptrs.
 *
 * The barrier_signal_ptrs are also set up to point to the correct offset within
 * each peer's shared memory block.
 */
void All2All::sync(const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles) {
  for (int i = 0; i < world_size; i++) {
    auto handle_str = std::string(all_gathered_handles[i].value());
    EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);

    if (i != rank) {
      // Open IPC mapping to remote GPU's memory
      std::memcpy(ipc_handlers[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);
      CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handlers[i], cudaIpcMemLazyEnablePeerAccess));
      // Calculate offset to barrier signals in remote buffer
      barrier_signal_ptrs[i] = reinterpret_cast<int *>(static_cast<uint8_t *>(buffer_ptrs[i]) + tensor_bytes);
    } else {
      // Verify our own handle matches what we sent
      EP_HOST_ASSERT(std::memcmp(ipc_handlers[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE) == 0);
    }
  }

  // Copy pointer arrays to GPU for kernel access
  CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void *) * world_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int *) * world_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
}

pybind11::bytearray All2All::get_local_ipc_handle() const {
  return {ipc_handlers[rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

/**
 * Configures token distribution across ranks for the current batch.
 *
 * This method computes prefix sums needed by the kernels to calculate source
 * and destination offsets. It must be called before any communication operation
 * when the token distribution changes between batches.
 *
 * Example: For rank_num_tokens = {128, 96, 128, 64}
 *   - rank_tokens = {128, 96, 128, 64}
 *   - prefix_rank_tokens = {0, 128, 224, 352}
 *   - total_tokens = 416
 */
void All2All::set_rank_tokens(const std::vector<int> &rank_num_tokens) {
  EP_HOST_ASSERT(static_cast<int>(rank_num_tokens.size()) == world_size);

  // Initialize prefix sums to zero
  for (int i = 0; i < world_size; i++) {
    prefix_rank_tokens[i] = 0;
  }

  // Compute prefix sums (exclusive scan)
  for (int i = 0; i < world_size; i++) {
    rank_tokens[i] = rank_num_tokens[i];
    if (i > 0) {
      prefix_rank_tokens[i] = prefix_rank_tokens[i - 1] + rank_tokens[i - 1];
    }
  }

  // Total tokens is the sum of all rank tokens
  total_tokens = prefix_rank_tokens[world_size - 1] + rank_tokens[world_size - 1];

  // Copy to GPU for kernel access
  CUDA_CHECK(cudaMemcpy(rank_tokens_gpu, rank_tokens, sizeof(int) * MAX_NUM_PEERS, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(prefix_rank_tokens_gpu, prefix_rank_tokens, sizeof(int) * MAX_NUM_PEERS, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Creates a tensor from the local IPC buffer.
 *
 * This helper method returns either a zero-copy view of the IPC buffer or
 * a newly allocated tensor with the data copied. The zero-copy mode is more
 * efficient but the tensor lifetime is tied to the All2All instance.
 *
 * @note The buffer pointer is cast to the template type T for proper interpretation.
 */
at::Tensor All2All::get_local_buffer_tensor(at::Tensor &x, int batch_size, int out_tokens, int out_heads, int head_size,
                                            bool should_copy, cudaStream_t stream) {
  auto ptr = buffer_ptrs[rank];
  if (should_copy) {
    // Allocate new tensor and copy data from IPC buffer
    auto out_tensor = torch::empty({batch_size, out_tokens, out_heads, head_size}, x.options());
    CUDA_CHECK(cudaMemcpyAsync(out_tensor.data_ptr(), ptr,
                               int64_t(batch_size) * int64_t(out_tokens) * int64_t(out_heads) * int64_t(head_size) *
                                   int64_t(elementSize(x.scalar_type())),
                               cudaMemcpyDeviceToDevice, stream));
    return out_tensor;
  } else {
    // Return a view directly into the IPC buffer (zero-copy)
    auto out_tensor = torch::from_blob(ptr, {batch_size, out_tokens, out_heads, head_size}, x.options());
    return out_tensor;
  }
}

/**
 * All2All communication to redistribute attention heads across GPUs.
 *
 * This operation is used in tensor-parallel transformers to exchange attention heads:
 *   - Before: Each GPU has all tokens but only a subset of heads
 *   - After: Each GPU has all tokens with heads redistributed
 *
 * Tensor Layout Transformation:
 *   Input:  [batch, local_tokens, all_heads, head_size]  per GPU
 *   Output: [batch, all_tokens, heads_per_rank, head_size]  per GPU
 *
 * The operation partitions heads evenly: heads_per_rank = all_heads / world_size
 * GPU i receives heads [i*heads_per_rank : (i+1)*heads_per_rank] from all GPUs.
 */
at::Tensor All2All::send_recv_heads(at::Tensor &x, bool copy_output) {
  // Validate input tensor properties
  EP_HOST_ASSERT(x.dim() == 4 and x.is_contiguous());
  EP_HOST_ASSERT(x.dtype() == tensor_dtype);
  EP_HOST_ASSERT(x.device().is_cuda());
  EP_HOST_ASSERT(x.device().index() == rank);

  int batch_size = x.size(0);
  int num_tokens = x.size(1);
  int num_heads = x.size(2);
  int head_size = x.size(3);

  // Output dimensions after redistribution
  int out_tokens = total_tokens;          // All tokens from all ranks
  int out_heads = num_heads / world_size; // Each rank gets 1/world_size of heads

  EP_HOST_ASSERT(int64_t(batch_size) * int64_t(out_tokens) * int64_t(out_heads) * int64_t(head_size) *
                     int64_t(elementSize(x.scalar_type())) <=
                 tensor_bytes);

  at::cuda::CUDAGuard device_guard{x.device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // Launch the All2All kernel
  all2all_cuda::all2all_head_launch(buffer_ptrs_gpu, barrier_signal_ptrs_gpu, x.data_ptr(), prefix_rank_tokens_gpu,
                                    rank, world_size, batch_size, total_tokens, num_tokens, num_heads, head_size,
                                    stream, num_sms, tensor_dtype, timeout_cycles_);

  return get_local_buffer_tensor(x, batch_size, out_tokens, out_heads, head_size, copy_output, stream);
}

/**
 * Inverse All2All to gather heads back to original distribution.
 *
 * This is the inverse operation of send_recv_heads(). It redistributes data
 * so each GPU gets back its original tokens with all attention heads.
 *
 * Tensor Layout Transformation:
 *   Input:  [batch, all_tokens, heads_per_rank, head_size]  per GPU
 *   Output: [batch, local_tokens, all_heads, head_size]  per GPU
 *
 * Each GPU sends its portion of tokens to the originating rank, reconstructing
 * the original head distribution.
 */
at::Tensor All2All::gather_heads(at::Tensor &x, bool copy_output) {
  // Validate input tensor properties
  EP_HOST_ASSERT(x.dim() == 4 and x.is_contiguous());
  EP_HOST_ASSERT(x.dtype() == tensor_dtype);
  EP_HOST_ASSERT(x.device().is_cuda());
  EP_HOST_ASSERT(x.device().index() == rank);

  at::cuda::CUDAGuard device_guard{x.device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int batch_size = x.size(0);
  int num_heads = x.size(2) * world_size; // Reconstruct total head count
  int head_size = x.size(3);

  // Output dimensions: this rank's tokens with all heads
  int out_tokens = rank_tokens[rank];
  int out_heads = num_heads;

  EP_HOST_ASSERT(int64_t(batch_size) * int64_t(out_tokens) * int64_t(out_heads) * int64_t(head_size) *
                     int64_t(elementSize(x.scalar_type())) <=
                 tensor_bytes);

  // Launch the gather kernel
  all2all_cuda::all2all_head_gather_launch(buffer_ptrs_gpu, barrier_signal_ptrs_gpu, x.data_ptr(), rank_tokens_gpu,
                                           prefix_rank_tokens_gpu, rank, world_size, batch_size, total_tokens,
                                           num_heads, head_size, stream, num_sms, tensor_dtype, timeout_cycles_);

  return get_local_buffer_tensor(x, batch_size, out_tokens, out_heads, head_size, copy_output, stream);
}

/**
 * AllGather operation to collect sequence tokens from all ranks.
 *
 * Each GPU contributes its local sequence tokens, which are gathered into
 * a complete sequence replicated on all GPUs. This is typically used after
 * tensor-parallel operations to reconstruct the full sequence.
 *
 * Tensor Layout Transformation:
 *   Input:  [batch, local_seqlen, heads, head_size]  per GPU
 *   Output: [batch, total_seqlen, heads, head_size]  per GPU (identical on all GPUs)
 *
 * Each GPU's tokens are placed at offset prefix_rank_tokens[rank] in the output.
 */
at::Tensor All2All::allgather(at::Tensor &x, bool copy_output) {
  // Validate input tensor properties
  EP_HOST_ASSERT(x.dim() == 4 and x.is_contiguous());
  EP_HOST_ASSERT(x.dtype() == tensor_dtype);
  EP_HOST_ASSERT(x.device().is_cuda());
  EP_HOST_ASSERT(x.device().index() == rank);

  at::cuda::CUDAGuard device_guard{x.device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int batch_size = x.size(0);
  int seqlen = x.size(1);
  int num_heads = x.size(2);
  int head_size = x.size(3);

  // Output contains all tokens from all ranks
  int out_tokens = total_tokens;
  int out_heads = num_heads;
  int hidden_dim = num_heads * head_size;

  EP_HOST_ASSERT(int64_t(batch_size) * int64_t(out_tokens) * int64_t(out_heads) * int64_t(head_size) *
                     int64_t(elementSize(x.scalar_type())) <=
                 tensor_bytes);

  // Launch the allgather kernel
  all2all_cuda::allgather_launch(buffer_ptrs_gpu, barrier_signal_ptrs_gpu, x.data_ptr(), prefix_rank_tokens_gpu, rank,
                                 world_size, batch_size, seqlen, hidden_dim, total_tokens, stream, num_sms,
                                 tensor_dtype, timeout_cycles_);

  return get_local_buffer_tensor(x, batch_size, out_tokens, out_heads, head_size, copy_output, stream);
}

} // namespace all2all
} // namespace ltx_kernels

/**
 * Python bindings for the All2All communication library.
 *
 * Usage from Python:
 *   import all2all_cpp
 *
 *   # Create instance (one per GPU)
 *   comm = all2all_cpp.All2All(rank, world_size, max_tokens, hidden_dim, num_sms, dtype)
 *
 *   # Exchange IPC handles and synchronize
 *   handle = comm.get_local_ipc_handle()
 *   # ... gather handles via NCCL ...
 *   comm.sync(all_handles)
 *
 *   # Set token distribution
 *   comm.set_rank_tokens([128, 128, 128, 128])
 *
 *   # Perform operations
 *   output = comm.send_recv_heads(input_tensor, copy_output=False)
 *
 *   # Cleanup
 *   comm.destroy()
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "High-performance All2All communication library for multi-GPU tensor parallelism.\n\n"
            "This library provides IPC-based All2All operations optimized for transformer models.\n"
            "Supported operations:\n"
            "  - send_recv_heads: Redistribute attention heads across GPUs\n"
            "  - gather_heads: Inverse of send_recv_heads\n"
            "  - allgather: Gather sequence tokens from all ranks\n";

  pybind11::class_<ltx_kernels::all2all::All2All>(
      m, "All2All",
      "Manages All2All communication state for multi-GPU operations.\n\n"
      "Args:\n"
      "    rank: This GPU's rank (0 to world_size-1)\n"
      "    world_size: Total number of GPUs\n"
      "    num_tokens: Maximum tokens per rank\n"
      "    hidden_dim: Hidden dimension (heads * head_size)\n"
      "    num_sms: Number of SMs for kernel launches\n"
      "    tensor_dtype: Tensor data type (torch.bfloat16 or torch.float8_e4m3fn)\n"
      "    timeout_seconds: Optional initial barrier timeout in seconds (defaults to the kernel default)")
      .def(pybind11::init<int, int, int, int, int, at::ScalarType>())
      .def(pybind11::init<int, int, int, int, int, at::ScalarType, double>())
      .def("get_local_ipc_handle", &ltx_kernels::all2all::All2All::get_local_ipc_handle,
           "Returns the IPC handle for this rank's buffer.")
      .def("sync", &ltx_kernels::all2all::All2All::sync, "Opens IPC mappings to all peer GPUs using gathered handles.")
      .def("destroy", &ltx_kernels::all2all::All2All::destroy,
           "Releases all GPU resources. Must be called before destruction.")
      .def("send_recv_heads", &ltx_kernels::all2all::All2All::send_recv_heads,
           "All2All operation to redistribute attention heads.")
      .def("gather_heads", &ltx_kernels::all2all::All2All::gather_heads,
           "Inverse All2All to gather heads back to original distribution.")
      .def("allgather", &ltx_kernels::all2all::All2All::allgather, "Gathers sequence tokens from all ranks.")
      .def("set_rank_tokens", &ltx_kernels::all2all::All2All::set_rank_tokens,
           "Sets token counts per rank for the current batch.")
      .def("set_timeout_seconds", &ltx_kernels::all2all::All2All::set_timeout_seconds,
           "Sets the barrier timeout in seconds (converted to cycles via the device peak SM clock).");
}
