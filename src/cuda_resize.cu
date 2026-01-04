#include "cuda_resize.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA错误: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            return false; \
        } \
    } while(0)

// CUDA kernel: 双线性插值图像缩放
__global__ void resizeKernel(
    const uint8_t* src,
    int src_width,
    int src_height,
    uint8_t* dst,
    int dst_width,
    int dst_height,
    int channels,
    float scale_x,
    float scale_y
)
{
    // 计算当前线程处理的目标像素坐标
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查
    if (dx >= dst_width || dy >= dst_height)
    {
        return;
    }

    // 计算源图像中的浮点坐标
    float sx = (dx + 0.5f) * scale_x - 0.5f;
    float sy = (dy + 0.5f) * scale_y - 0.5f;

    // 边界处理
    sx = fmaxf(0.0f, fminf(sx, src_width - 1.0f));
    sy = fmaxf(0.0f, fminf(sy, src_height - 1.0f));

    // 计算整数坐标和小数部分
    int sx0 = static_cast<int>(sx);
    int sy0 = static_cast<int>(sy);
    int sx1 = min(sx0 + 1, src_width - 1);
    int sy1 = min(sy0 + 1, src_height - 1);

    float fx = sx - sx0;
    float fy = sy - sy0;

    // 获取四个邻近像素的位置
    const uint8_t* p00 = src + (sy0 * src_width + sx0) * channels;
    const uint8_t* p01 = src + (sy0 * src_width + sx1) * channels;
    const uint8_t* p10 = src + (sy1 * src_width + sx0) * channels;
    const uint8_t* p11 = src + (sy1 * src_width + sx1) * channels;

    // 目标像素位置
    uint8_t* dst_pixel = dst + (dy * dst_width + dx) * channels;

    // 对每个通道进行双线性插值
    for (int c = 0; c < channels; ++c)
    {
        float v00 = static_cast<float>(p00[c]);
        float v01 = static_cast<float>(p01[c]);
        float v10 = static_cast<float>(p10[c]);
        float v11 = static_cast<float>(p11[c]);

        // 双线性插值公式
        float top = v00 * (1.0f - fx) + v01 * fx;
        float bottom = v10 * (1.0f - fx) + v11 * fx;
        float result = top * (1.0f - fy) + bottom * fy;

        // 转换回uint8并写入
        dst_pixel[c] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, result)));
    }
}

// 主机端函数: 调用CUDA kernel进行图像缩放
bool resizeImageCUDA(
    const uint8_t* src,
    int src_width,
    int src_height,
    uint8_t* dst,
    int dst_width,
    int dst_height,
    int channels
)
{
    // 计算图像数据大小
    size_t src_size = src_width * src_height * channels;
    size_t dst_size = dst_width * dst_height * channels;

    // 分配GPU内存
    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    CUDA_CHECK(cudaMalloc(&d_dst, dst_size));

    // 将源图像数据复制到GPU
    CUDA_CHECK(cudaMemcpy(d_src, src, src_size, cudaMemcpyHostToDevice));

    // 计算缩放比例
    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;

    // 配置kernel启动参数
    dim3 blockSize(16, 16);  // 每个block 16x16个线程
    dim3 gridSize(
        (dst_width + blockSize.x - 1) / blockSize.x,
        (dst_height + blockSize.y - 1) / blockSize.y
    );

    // 启动kernel
    resizeKernel<<<gridSize, blockSize>>>(
        d_src, src_width, src_height,
        d_dst, dst_width, dst_height,
        channels, scale_x, scale_y
    );

    // 检查kernel执行错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 将结果复制回主机
    CUDA_CHECK(cudaMemcpy(dst, d_dst, dst_size, cudaMemcpyDeviceToHost));

    // 释放GPU内存
    cudaFree(d_src);
    cudaFree(d_dst);

    return true;
}
