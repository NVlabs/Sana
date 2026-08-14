/**
 * @file exceptions.cuh
 * @brief Exception handling and assertion macros for CUDA/C++ code.
 *
 * This header provides a unified exception type and assertion macros for
 * both host and device code. The macros capture file and line information
 * for easier debugging of errors.
 *
 * ## Usage Examples
 *
 * ```cpp
 * // Check CUDA API call
 * CUDA_CHECK(cudaMalloc(&ptr, size));
 *
 * // Host-side assertion
 * EP_HOST_ASSERT(tensor.is_contiguous());
 *
 * // Device-side assertion (inside kernel)
 * EP_DEVICE_ASSERT(threadIdx.x < MAX_THREADS);
 *
 * // Compile-time assertion
 * EP_STATIC_ASSERT(sizeof(int4) == 16, "int4 must be 16 bytes");
 * ```
 */

#pragma once

#include <exception>
#include <string>

#include "configs.cuh"

// =============================================================================
// Static Assertions
// =============================================================================

/**
 * @brief Compile-time assertion macro.
 *
 * @param cond Condition that must be true at compile time
 * @param reason Human-readable error message if condition fails
 */
#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

// =============================================================================
// Exception Type
// =============================================================================

/**
 * @class EPException
 * @brief Custom exception type with file/line information.
 *
 * EPException captures the location (file, line) and context (name, error)
 * of the error for debugging. It inherits from std::exception for
 * compatibility with standard C++ exception handling.
 *
 * ## Message Format
 *
 * The what() message has the format:
 * "Failed: <name> error <file>:<line> '<error message>'"
 */
class EPException : public std::exception {
private:
  std::string message = {}; ///< Formatted error message

public:
  /**
   * @brief Constructs an EPException with location and error information.
   *
   * @param name Category of error (e.g., "CUDA", "Assertion")
   * @param file Source file where error occurred (__FILE__)
   * @param line Line number where error occurred (__LINE__)
   * @param error Description of the error
   */
  explicit EPException(const char *name, const char *file, const int line, const std::string &error) {
    message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
  }

  /**
   * @brief Returns the formatted error message.
   * @return C-string containing the error message
   */
  const char *what() const noexcept override { return message.c_str(); }
};

// =============================================================================
// Runtime Assertion Macros
// =============================================================================

/**
 * @brief Checks CUDA API return value and throws on error.
 *
 * Use this macro to wrap all CUDA runtime API calls. If the call fails,
 * an EPException is thrown with the CUDA error string.
 *
 * @param cmd CUDA API call expression
 * @throws EPException if the CUDA call returns an error
 *
 * Example:
 * ```cpp
 * CUDA_CHECK(cudaMalloc(&ptr, size));
 * CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
 * ```
 */
#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                                                                                \
  do {                                                                                                                 \
    cudaError_t e = (cmd);                                                                                             \
    if (e != cudaSuccess) {                                                                                            \
      throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e));                                            \
    }                                                                                                                  \
  } while (0)
#endif

/**
 * @brief Host-side assertion that throws on failure.
 *
 * Use this for runtime checks in host code. If the condition is false,
 * an EPException is thrown with the condition as the error message.
 *
 * @param cond Condition to check (must be true)
 * @throws EPException if condition is false
 *
 * Example:
 * ```cpp
 * EP_HOST_ASSERT(tensor.dim() == 4);
 * EP_HOST_ASSERT(rank >= 0 && rank < world_size);
 * ```
 */
#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                                                                           \
  do {                                                                                                                 \
    if (not(cond)) {                                                                                                   \
      throw EPException("Assertion", __FILE__, __LINE__, #cond);                                                       \
    }                                                                                                                  \
  } while (0)
#endif

/**
 * @brief Device-side assertion that traps on failure.
 *
 * Use this for runtime checks inside CUDA kernels. If the condition is
 * false, prints an error message and executes a trap instruction to
 * halt the GPU.
 *
 * @warning This causes the entire kernel to abort. Use sparingly and
 *          consider removing from release builds for performance.
 *
 * @param cond Condition to check (must be true)
 *
 * Example:
 * ```cpp
 * __global__ void my_kernel(int* data, int size) {
 *     int idx = threadIdx.x + blockIdx.x * blockDim.x;
 *     EP_DEVICE_ASSERT(idx < size);
 *     data[idx] = 42;
 * }
 * ```
 */
#ifndef EP_DEVICE_ASSERT
#define EP_DEVICE_ASSERT(cond)                                                                                         \
  do {                                                                                                                 \
    if (not(cond)) {                                                                                                   \
      printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond);                                   \
      asm("trap;");                                                                                                    \
    }                                                                                                                  \
  } while (0)
#endif
