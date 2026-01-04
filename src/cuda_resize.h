#ifndef CUDA_RESIZE_H
#define CUDA_RESIZE_H

#include <cstdint>

struct CudaResizePerfStats {
    float h2d_time_ms;
    float kernel_time_ms;
    float d2h_time_ms;
};

/**
 * 使用CUDA进行图像缩放
 * 
 * @param src 源图像数据指针(主机内存)
 * @param src_width 源图像宽度
 * @param src_height 源图像高度
 * @param dst 目标图像数据指针(主机内存,需要预先分配)
 * @param dst_width 目标图像宽度
 * @param dst_height 目标图像高度
 * @param channels 图像通道数(3=RGB, 4=RGBA)
 * @param stats 可选的性能统计指针
 * @return true表示成功,false表示失败
 */
bool resizeImageCUDA(
    const uint8_t* src,
    int src_width,
    int src_height,
    uint8_t* dst,
    int dst_width,
    int dst_height,
    int channels,
    CudaResizePerfStats* stats = nullptr
);

#endif // CUDA_RESIZE_H
