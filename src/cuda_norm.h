#ifndef CUDA_NORM_H
#define CUDA_NORM_H

#include <cstdint>

struct CudaNormPerfStats
{
    float h2d_time_ms;
    float kernel_time_ms;
    float d2h_time_ms;
};

/**
 * @brief CUDA 归一化 (BGR HWC -> RGB CHW)
 * 
 * @param src 源图像 (Host, uint8, HWC)
 * @param w 宽度
 * @param h 高度
 * @param dst 目标数据 (Host, float, CHW)
 * @param stats 性能统计
 */
bool normalizeCUDA(const uint8_t* src, int w, int h, float* dst, CudaNormPerfStats* stats = nullptr);

#endif // CUDA_NORM_H
