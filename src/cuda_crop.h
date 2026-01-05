#ifndef CUDA_CROP_H
#define CUDA_CROP_H

#include <cstdint>

// 性能统计结构体
struct CudaCropPerfStats
{
    float h2d_time_ms;
    float kernel_time_ms;
    float d2h_time_ms;
};

/**
 * @brief 使用 CUDA 进行图像裁剪
 * 
 * @param src 源图像指针 (Host)
 * @param src_w 源图像宽度
 * @param src_h 源图像高度
 * @param dst 目标图像指针 (Host, 预分配)
 * @param crop_x 裁剪起始 x 坐标
 * @param crop_y 裁剪起始 y 坐标
 * @param crop_w 裁剪宽度
 * @param crop_h 裁剪高度
 * @param channels 通道数
 * @param stats 性能统计 (可选)
 * @return true 成功
 * @return false 失败
 */
bool cropImageCUDA(
    const uint8_t* src, int src_w, int src_h,
    uint8_t* dst,
    int crop_x, int crop_y, int crop_w, int crop_h,
    int channels,
    CudaCropPerfStats* stats = nullptr);

#endif // CUDA_CROP_H
