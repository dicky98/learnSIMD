#ifndef CPU_SIMD_RESIZE_H
#define CPU_SIMD_RESIZE_H

#include <cstdint>

/**
 * 使用CPU SIMD(AVX2)指令进行图像缩放
 * 
 * @param src 源图像数据指针
 * @param src_width 源图像宽度
 * @param src_height 源图像高度
 * @param dst 目标图像数据指针(需要预先分配内存)
 * @param dst_width 目标图像宽度
 * @param dst_height 目标图像高度
 * @param channels 图像通道数(3=RGB, 4=RGBA)
 */
void resizeImageCPU_SIMD(
    const uint8_t* src,
    int src_width,
    int src_height,
    uint8_t* dst,
    int dst_width,
    int dst_height,
    int channels
);

#endif // CPU_SIMD_RESIZE_H
