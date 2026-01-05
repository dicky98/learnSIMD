#include "cuda_norm.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA 错误: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            return false; \
        } \
    } while(0)

__global__ void normalizeKernel(const uint8_t* src, int w, int h, float* dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h)
    {
        int idx = y * w + x;
        int src_idx = idx * 3;
        int plane_size = w * h;

        // BGR HWC -> RGB CHW
        uint8_t b = src[src_idx + 0];
        uint8_t g = src[src_idx + 1];
        uint8_t r = src[src_idx + 2];

        dst[idx] = r / 255.0f;              // R plane
        dst[idx + plane_size] = g / 255.0f;      // G plane
        dst[idx + 2 * plane_size] = b / 255.0f;  // B plane
    }
}

bool normalizeCUDA(const uint8_t* src, int w, int h, float* dst, CudaNormPerfStats* stats)
{
    size_t src_size = (size_t)w * h * 3 * sizeof(uint8_t);
    size_t dst_size = (size_t)w * h * 3 * sizeof(float);

    uint8_t* d_src = nullptr;
    float* d_dst = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    CUDA_CHECK(cudaMalloc(&d_dst, dst_size));

    cudaEvent_t start, after_h2d, after_kernel, stop;
    if (stats)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&after_h2d);
        cudaEventCreate(&after_kernel);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    CUDA_CHECK(cudaMemcpy(d_src, src, src_size, cudaMemcpyHostToDevice));

    if (stats) cudaEventRecord(after_h2d);

    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    normalizeKernel<<<gridSize, blockSize>>>(d_src, w, h, d_dst);
    CUDA_CHECK(cudaGetLastError());

    if (stats) cudaEventRecord(after_kernel);
    if (!stats) CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dst, d_dst, dst_size, cudaMemcpyDeviceToHost));

    if (stats)
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&stats->h2d_time_ms, start, after_h2d);
        cudaEventElapsedTime(&stats->kernel_time_ms, after_h2d, after_kernel);
        cudaEventElapsedTime(&stats->d2h_time_ms, after_kernel, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(after_h2d);
        cudaEventDestroy(after_kernel);
        cudaEventDestroy(stop);
    }

    cudaFree(d_src);
    cudaFree(d_dst);

    return true;
}
