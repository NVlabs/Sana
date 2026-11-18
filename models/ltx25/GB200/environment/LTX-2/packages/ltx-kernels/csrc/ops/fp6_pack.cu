#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <ATen/ATen.h>
#include <torch/types.h>

// Device function to pack 8-bit to 6-bit
// 8-bit layout: s e_1 e_2 e_3 m_1 m_2 m_3 m_4 (bits 7-0)
// 6-bit layout: s e_3 m_1 m_2 m_3 m_4 (bits 5-0)
// Drop e_1 (bit 6) and e_2 (bit 5)
__device__ __forceinline__ uint8_t pack_8bit_to_6bit(uint8_t input) {
    // Extract the sign bit (bit 7)
    uint8_t sign = (input >> 7) & 0x1;

    // Extract e_3 (bit 4)
    uint8_t e_3 = (input >> 4) & 0x1;

    // Extract mantissa bits (bits 3-0)
    uint8_t mantissa = input & 0x0F;

    // Pack into 6-bit format: s e_3 m_1 m_2 m_3 m_4
    uint8_t result = (sign << 5) | (e_3 << 4) | mantissa;

    return result & 0x3F; // Mask to 6 bits
}

// Device function to pack 4 x 6-bit values into 3 bytes
__device__ __forceinline__ void pack_4x6bit_to_3bytes(const uint8_t* input_6bit, uint8_t* output_3bytes) {
    uint8_t v0 = input_6bit[0] & 0x3F;
    uint8_t v1 = input_6bit[1] & 0x3F;
    uint8_t v2 = input_6bit[2] & 0x3F;
    uint8_t v3 = input_6bit[3] & 0x3F;

    // Pack: [v0: 6 bits][v1: 6 bits][v2: 6 bits][v3: 6 bits] = 24 bits = 3 bytes
    output_3bytes[0] = (v0 << 2) | (v1 >> 4);
    output_3bytes[1] = (v1 << 4) | (v2 >> 2);
    output_3bytes[2] = (v2 << 6) | v3;
}

// CUDA kernel for packing 2D tensor
// Input: [m, n] uint8 tensor
// Output: [m, n*3/4] uint8 tensor
__global__ void fp6_pack_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int m,
    int n,
    int n_packed
) {
    // Each thread processes one row and 4 elements at a time
    int row = blockIdx.x;
    int col_group = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= m) return;

    // Calculate input and output positions
    int input_col = col_group * 4;
    if (input_col >= n) return;

    int output_col = col_group * 3;

    const uint8_t* input_row = input + row * n;
    uint8_t* output_row = output + row * n_packed;

    uint8_t temp_6bit[4];

    // Pack 4 elements
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (input_col + i < n) {
            temp_6bit[i] = pack_8bit_to_6bit(input_row[input_col + i]);
        } else {
            temp_6bit[i] = 0;
        }
    }

    // Write 3 bytes to output
    uint8_t temp_3bytes[3];
    pack_4x6bit_to_3bytes(temp_6bit, temp_3bytes);

    if (output_col < n_packed) output_row[output_col] = temp_3bytes[0];
    if (output_col + 1 < n_packed) output_row[output_col + 1] = temp_3bytes[1];
    if (output_col + 2 < n_packed) output_row[output_col + 2] = temp_3bytes[2];
}

// Device function to unpack 6-bit to 8-bit
__device__ __forceinline__ uint8_t unpack_6bit_to_8bit(uint8_t input) {
    input = input & 0x3F; // Ensure only 6 bits

    uint8_t sign = (input >> 5) & 0x1;
    uint8_t e_3 = (input >> 4) & 0x1;
    uint8_t mantissa = input & 0x0F;

    // Reconstruct 8-bit with e_1 and e_2 set to 0
    uint8_t result = (sign << 7) | (e_3 << 4) | mantissa;

    return result;
}

// Device function to unpack 3 bytes into 4 x 6-bit values
__device__ __forceinline__ void unpack_3bytes_to_4x6bit(const uint8_t* input_3bytes, uint8_t* output_6bit) {
    output_6bit[0] = (input_3bytes[0] >> 2) & 0x3F;
    output_6bit[1] = ((input_3bytes[0] << 4) | (input_3bytes[1] >> 4)) & 0x3F;
    output_6bit[2] = ((input_3bytes[1] << 2) | (input_3bytes[2] >> 6)) & 0x3F;
    output_6bit[3] = input_3bytes[2] & 0x3F;
}

// CUDA kernel for unpacking 2D tensor
// Input: [m, n_packed] uint8 tensor
// Output: [m, n] uint8 tensor
__global__ void fp6_unpack_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int m,
    int n_packed,
    int n
) {
    // Each thread processes one row and 4 elements at a time
    int row = blockIdx.x;
    int col_group = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= m) return;

    // Calculate input and output positions
    int input_col = col_group * 3;
    if (input_col >= n_packed) return;

    int output_col = col_group * 4;

    const uint8_t* input_row = input + row * n_packed;
    uint8_t* output_row = output + row * n;

    // Read 3 bytes
    uint8_t temp_3bytes[3];
    temp_3bytes[0] = (input_col < n_packed) ? input_row[input_col] : 0;
    temp_3bytes[1] = (input_col + 1 < n_packed) ? input_row[input_col + 1] : 0;
    temp_3bytes[2] = (input_col + 2 < n_packed) ? input_row[input_col + 2] : 0;

    // Unpack to 4 x 6-bit values
    uint8_t temp_6bit[4];
    unpack_3bytes_to_4x6bit(temp_3bytes, temp_6bit);

    // Convert to 8-bit and write
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (output_col + i < n) {
            output_row[output_col + i] = unpack_6bit_to_8bit(temp_6bit[i]);
        }
    }
}

// Host function to launch pack kernel
void fp6_pack_cuda(
    at::Tensor& x,
    at::Tensor& out,
    cudaStream_t stream
) {
    int m = x.size(0);
    int n = x.size(1);
    int n_packed = out.size(1);

    const uint8_t* input_ptr = (uint8_t*)x.data_ptr();
    uint8_t* output_ptr = (uint8_t*)out.data_ptr();

    // Each thread handles 4 input elements -> 3 output bytes
    int num_groups = (n + 3) / 4;

    int threads = 256;
    dim3 blocks(m, (num_groups + threads - 1) / threads);

    fp6_pack_kernel<<<blocks, threads, 0, stream>>>(
        input_ptr,
        output_ptr,
        m,
        n,
        n_packed
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Host function to launch unpack kernel
void fp6_unpack_cuda(
    at::Tensor& x,
    at::Tensor& out,
    cudaStream_t stream
) {
    int m = x.size(0);
    int n_packed = x.size(1);
    int n = out.size(1);

    const uint8_t* input_ptr = (uint8_t*)x.data_ptr();
    uint8_t* output_ptr = (uint8_t*)out.data_ptr();

    // Each thread handles 3 input bytes -> 4 output elements
    int num_groups = (n + 3) / 4;

    int threads = 256;
    dim3 blocks(m, (num_groups + threads - 1) / threads);

    fp6_unpack_kernel<<<blocks, threads, 0, stream>>>(
        input_ptr,
        output_ptr,
        m,
        n_packed,
        n
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
