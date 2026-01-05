#include "cuda_crop.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA 错误: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            return false; \
        } \
    } while(0)

// CUDA Kernel: 图像裁剪
__global__ void cropKernel(
    const uint8_t* src, int src_w, int src_h,
    uint8_t* dst, int dst_w, int dst_h,
    int crop_x, int crop_y,
    int channels)
{
    // 每个线程处理一个目标图像像素
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < dst_w && dy < dst_h)
    {
        // 映射回源图像坐标
        int sx = dx + crop_x;
        int sy = dy + crop_y;

        // 源像素指针和目标像素指针
        const uint8_t* src_pixel = src + (sy * src_w + sx) * channels;
        uint8_t* dst_pixel = dst + (dy * dst_w + dx) * channels;

        // 通道拷贝
        for (int c = 0; c < channels; ++c)
        {
            dst_pixel[c] = src_pixel[c];
        }
    }
}

bool cropImageCUDA(
    const uint8_t* src, int src_w, int src_h,
    uint8_t* dst,
    int crop_x, int crop_y, int crop_w, int crop_h,
    int channels,
    CudaCropPerfStats* stats)
{
    size_t src_size = (size_t)src_w * src_h * channels;
    size_t dst_size = (size_t)crop_w * crop_h * channels;

    uint8_t *d_src = nullptr, *d_dst = nullptr;

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

    // H2D
    CUDA_CHECK(cudaMemcpy(d_src, src, src_size, cudaMemcpyHostToDevice));

    if (stats) cudaEventRecord(after_h2d);

    // Kernel 配置
    dim3 blockSize(16, 16);
    dim3 gridSize((crop_w + blockSize.x - 1) / blockSize.x, (crop_h + blockSize.y - 1) / blockSize.y);

    cropKernel<<<gridSize, blockSize>>>(d_src, src_w, src_h, d_dst, crop_w, crop_h, crop_x, crop_y, channels);
    CUDA_CHECK(cudaGetLastError());

    if (stats) cudaEventRecord(after_kernel);
    if (!stats) CUDA_CHECK(cudaDeviceSynchronize());

    // D2H
    CUDA_CHECK(cudaMemcpy(dst, d_dst, dst_size, cudaMemcpyDeviceToHost));

    if (stats)
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float h2d, kernel, d2h;
        cudaEventElapsedTime(&h2d, start, after_h2d);
        cudaEventElapsedTime(&kernel, after_h2d, after_kernel);
        cudaEventElapsedTime(&d2h, after_kernel, stop);

        stats->h2d_time_ms = h2d;
        stats->kernel_time_ms = kernel;
        stats->d2h_time_ms = d2h;

        cudaEventDestroy(start);
        cudaEventDestroy(after_h2d);
        cudaEventDestroy(after_kernel);
        cudaEventDestroy(stop);
    }

    cudaFree(d_src);
    cudaFree(d_dst);

    return true;
}
