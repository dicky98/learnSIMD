#include "cpu_crop.h"
#include <immintrin.h>
#include <cstring>
#include <omp.h>
#include <algorithm>

// Baseline 实现：标准的嵌套循环 + memcpy
void cropImageCPU_Baseline(
    const uint8_t* src, int src_w, int src_h,
    uint8_t* dst,
    int crop_x, int crop_y, int crop_w, int crop_h,
    int channels)
{
    int row_bytes = crop_w * channels;
    for (int y = 0; y < crop_h; ++y)
    {
        const uint8_t* src_row = src + ((crop_y + y) * src_w + crop_x) * channels;
        uint8_t* dst_row = dst + y * row_bytes;
        std::memcpy(dst_row, src_row, row_bytes);
    }
}

// SIMD 实现：针对每一行使用 AVX2 进行拷贝
// 注意：如果 row_bytes 不是 32 的倍数，需要处理尾部
void cropImageCPU_SIMD(
    const uint8_t* src, int src_w, int src_h,
    uint8_t* dst,
    int crop_x, int crop_y, int crop_w, int crop_h,
    int channels)
{
    int row_bytes = crop_w * channels;
    int simd_width = 32; // AVX2 256-bit = 32 bytes
    int vector_limit = (row_bytes / simd_width) * simd_width;

    for (int y = 0; y < crop_h; ++y)
    {
        const uint8_t* src_ptr = src + ((crop_y + y) * src_w + crop_x) * channels;
        uint8_t* dst_ptr = dst + y * row_bytes;

        int x = 0;
        // 使用 SIMD 指令进行按行的大块拷贝
        for (; x < vector_limit; x += simd_width)
        {
            __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + x));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + x), data);
        }

        // 处理剩余字节
        for (; x < row_bytes; ++x)
        {
            dst_ptr[x] = src_ptr[x];
        }
    }
}

// SIMD + OpenMP 实现
void cropImageCPU_SIMD_OMP(
    const uint8_t* src, int src_w, int src_h,
    uint8_t* dst,
    int crop_x, int crop_y, int crop_w, int crop_h,
    int channels)
{
    int row_bytes = crop_w * channels;
    int simd_width = 32;
    int vector_limit = (row_bytes / simd_width) * simd_width;

    #pragma omp parallel for
    for (int y = 0; y < crop_h; ++y)
    {
        const uint8_t* src_ptr = src + ((crop_y + y) * src_w + crop_x) * channels;
        uint8_t* dst_ptr = dst + y * row_bytes;

        int x = 0;
        for (; x < vector_limit; x += simd_width)
        {
            __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + x));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + x), data);
        }

        for (; x < row_bytes; ++x)
        {
            dst_ptr[x] = src_ptr[x];
        }
    }
}
