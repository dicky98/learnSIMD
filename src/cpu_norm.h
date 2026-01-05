#ifndef CPU_NORM_H
#define CPU_NORM_H

#include <cstdint>

/**
 * @brief CPU 基础归一化 (BGR HWC -> RGB CHW, uint8 -> float32 [0, 1])
 * 
 * @param src 源图像指针 (HWC, BGR)
 * @param src_w 图像宽度
 * @param src_h 图像高度
 * @param dst 目标数据指针 (CHW, RGB, 预分配大小 w*h*3*sizeof(float))
 */
void normalizeCPU_Baseline(const uint8_t* src, int src_w, int src_h, float* dst);

/**
 * @brief CPU SIMD (AVX2) 加速归一化
 */
void normalizeCPU_SIMD(const uint8_t* src, int src_w, int src_h, float* dst);

#endif // CPU_NORM_H
