/**
 * @file event.hpp
 * @brief CUDA stream and event synchronization utilities.
 *
 * This header provides wrapper types and helper functions for managing
 * CUDA events and stream synchronization in PyTorch/ATen environment.
 * These utilities are used to coordinate asynchronous operations across
 * multiple CUDA streams.
 */

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <memory>

#include "cuda/exceptions.cuh"

namespace ltx_kernels {

/**
 * @struct EventHandle
 * @brief RAII wrapper for a CUDA event with automatic recording.
 *
 * EventHandle encapsulates a torch::Event and automatically records it
 * on the specified (or current) CUDA stream upon construction. This
 * provides a convenient way to capture the completion point of stream
 * operations for synchronization purposes.
 *
 * ## Usage Example
 *
 * ```cpp
 * // Record event on current stream
 * EventHandle ev1;
 *
 * // Record event on specific stream
 * EventHandle ev2(my_stream);
 *
 * // Make current stream wait for the event
 * ev1.current_stream_wait();
 * ```
 */
struct EventHandle {
  /// Shared pointer to the underlying torch::Event
  std::shared_ptr<torch::Event> event;

  /**
   * @brief Constructs an EventHandle and records on the current CUDA stream.
   *
   * The event captures the completion point of all operations submitted
   * to the current stream before this constructor is called.
   */
  EventHandle() {
    event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(at::cuda::getCurrentCUDAStream());
  }

  /**
   * @brief Constructs an EventHandle and records on the specified stream.
   *
   * @param stream The CUDA stream to record the event on
   */
  explicit EventHandle(const at::cuda::CUDAStream &stream) {
    event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(stream);
  }

  /// Copy constructor (shares the underlying event)
  EventHandle(const EventHandle &other) = default;

  /**
   * @brief Makes the current CUDA stream wait for this event.
   *
   * After this call returns, operations submitted to the current stream
   * will not execute until the event has been reached on its recording stream.
   */
  void current_stream_wait() const { at::cuda::getCurrentCUDAStream().unwrap().wait(*event); }
};

/**
 * @brief Creates and records a CUDA event on the specified stream.
 *
 * @param s The CUDA stream to record on
 * @return A torch::Event that has been recorded on stream s
 */
inline torch::Event create_event(const at::cuda::CUDAStream &s) {
  auto event = torch::Event(torch::kCUDA);
  event.record(s);
  return event;
}

/**
 * @brief Makes stream s_0 wait for stream s_1's current position.
 *
 * After this call, operations on s_0 will not execute until all operations
 * currently queued on s_1 have completed.
 *
 * @param s_0 The stream that will wait
 * @param s_1 The stream to wait for
 * @pre s_0 and s_1 must be different streams
 */
inline void stream_wait(const at::cuda::CUDAStream &s_0, const at::cuda::CUDAStream &s_1) {
  EP_HOST_ASSERT(s_0.id() != s_1.id());
  s_0.unwrap().wait(create_event(s_1));
}

/**
 * @brief Makes a stream wait for a previously recorded event.
 *
 * @param s The stream that will wait
 * @param event The event to wait for
 */
inline void stream_wait(const at::cuda::CUDAStream &s, const EventHandle &event) { s.unwrap().wait(*event.event); }

} // namespace ltx_kernels
