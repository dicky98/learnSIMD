#ifndef CPU_CROP_H
#define CPU_CROP_H

#include <cstdint>

/**
 * @brief 使用 CPU 基础实现图像裁剪 (嵌套循环)
 * 
 * @param src 源图像指针
 * @param src_w 源图像宽度
 * @param src_h 源图像高度
 * @param dst 目标图像指针 (预分配)
 * @param crop_x 裁剪起始 x 坐标
 * @param crop_y 裁剪起始 y 坐标
 * @param crop_w 裁剪宽度
 * @param crop_h 裁剪高度
 * @param channels 通道数
 */
void cropImageCPU_Baseline(
    const uint8_t* src, int src_w, int src_h,
    uint8_t* dst,
    int crop_x, int crop_y, int crop_w, int crop_h,
    int channels);

/**
 * @brief 使用 CPU SIMD (AVX2) 实现图像裁剪
 */
void cropImageCPU_SIMD(
    const uint8_t* src, int src_w, int src_h,
    uint8_t* dst,
    int crop_x, int crop_y, int crop_w, int crop_h,
    int channels);

/**
 * @brief 使用 CPU SIMD (AVX2) + OpenMP 实现图像裁剪
 */
void cropImageCPU_SIMD_OMP(
    const uint8_t* src, int src_w, int src_h,
    uint8_t* dst,
    int crop_x, int crop_y, int crop_w, int crop_h,
    int channels);

#endif // CPU_CROP_H
