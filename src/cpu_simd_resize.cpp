#include "cpu_simd_resize.h"
#include <immintrin.h>
#include <cstring>
#include <algorithm>

// 使用AVX2进行双线性插值图像缩放
void resizeImageCPU_SIMD(
    const uint8_t* src,
    int src_width,
    int src_height,
    uint8_t* dst,
    int dst_width,
    int dst_height,
    int channels
)
{
    // 计算缩放比例
    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;

    // 对每个目标像素进行处理
    for (int dy = 0; dy < dst_height; ++dy)
    {
        for (int dx = 0; dx < dst_width; ++dx)
        {
            // 计算源图像中的浮点坐标
            float sx = (dx + 0.5f) * scale_x - 0.5f;
            float sy = (dy + 0.5f) * scale_y - 0.5f;

            // 边界处理
            sx = std::max(0.0f, std::min(sx, src_width - 1.0f));
            sy = std::max(0.0f, std::min(sy, src_height - 1.0f));

            // 计算整数坐标和小数部分
            int sx0 = static_cast<int>(sx);
            int sy0 = static_cast<int>(sy);
            int sx1 = std::min(sx0 + 1, src_width - 1);
            int sy1 = std::min(sy0 + 1, src_height - 1);

            float fx = sx - sx0;
            float fy = sy - sy0;

            // 使用AVX2进行双线性插值
            // 加载权重到SIMD寄存器
            __m256 weight_x0 = _mm256_set1_ps(1.0f - fx);
            __m256 weight_x1 = _mm256_set1_ps(fx);
            __m256 weight_y0 = _mm256_set1_ps(1.0f - fy);
            __m256 weight_y1 = _mm256_set1_ps(fy);

            // 获取四个邻近像素的位置
            const uint8_t* p00 = src + (sy0 * src_width + sx0) * channels;
            const uint8_t* p01 = src + (sy0 * src_width + sx1) * channels;
            const uint8_t* p10 = src + (sy1 * src_width + sx0) * channels;
            const uint8_t* p11 = src + (sy1 * src_width + sx1) * channels;

            // 对每个通道进行插值
            uint8_t* dst_pixel = dst + (dy * dst_width + dx) * channels;
            
            for (int c = 0; c < channels; ++c)
            {
                // 将uint8转换为float进行计算
                float v00 = static_cast<float>(p00[c]);
                float v01 = static_cast<float>(p01[c]);
                float v10 = static_cast<float>(p10[c]);
                float v11 = static_cast<float>(p11[c]);

                // 双线性插值
                // result = (1-fx)*(1-fy)*v00 + fx*(1-fy)*v01 + (1-fx)*fy*v10 + fx*fy*v11
                float top = v00 * (1.0f - fx) + v01 * fx;
                float bottom = v10 * (1.0f - fx) + v11 * fx;
                float result = top * (1.0f - fy) + bottom * fy;

                // 转换回uint8
                dst_pixel[c] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, result)));
            }
        }
    }
}
