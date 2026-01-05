#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include "cpu_simd_resize.h"
#include "cuda_resize.h"
#include "cpu_crop.h"
#include "cuda_crop.h"

// 性能计时辅助类
class Timer
{
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    // 返回经过的毫秒数
    double elapsed() const
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    void reset()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// 生成测试图像
cv::Mat generateTestImage(int width, int height)
{
    cv::Mat img(height, width, CV_8UC3);
    
    // 生成渐变和图案
    #pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            img.at<cv::Vec3b>(y, x)[0] = (x * 255) / width;           // B通道
            img.at<cv::Vec3b>(y, x)[1] = (y * 255) / height;          // G通道
            img.at<cv::Vec3b>(y, x)[2] = ((x + y) * 255) / (width + height); // R通道
        }
    }
    
    return img;
}

// 打印测试分割线
void printSeparator() {
    std::cout << "----------------------------------------\n";
}

// 测试图像缩放性能 (保留原有功能)
void benchmarkResize(const std::string& name, int src_width, int src_height, int iterations = 100)
{
    std::cout << "\n========================================\n";
    std::cout << "测试 [Resize]: " << name << "\n";
    std::cout << "源图像尺寸: " << src_width << "x" << src_height << "\n";
    std::cout << "目标尺寸: 640x640\n";
    std::cout << "迭代次数: " << iterations << "\n";
    std::cout << "========================================\n";

    cv::Mat src_img = generateTestImage(src_width, src_height);
    int dst_width = 640;
    int dst_height = 640;
    int channels = 3;

    cv::Mat dst_cpu(dst_height, dst_width, CV_8UC3);
    cv::Mat dst_cuda(dst_height, dst_width, CV_8UC3);
    cv::Mat dst_opencv(dst_height, dst_width, CV_8UC3);

    printSeparator();
    std::cout << "性能测试结果:\n";
    printSeparator();

    // 1. 测试CPU SIMD (单线程)
    {
        resizeImageCPU_SIMD(src_img.data, src_width, src_height, 
                           dst_cpu.data, dst_width, dst_height, channels);
        Timer timer;
        for (int i = 0; i < iterations; ++i)
            resizeImageCPU_SIMD(src_img.data, src_width, src_height, dst_cpu.data, dst_width, dst_height, channels);
        std::cout << "1. CPU AVX2 (单线程): " << timer.elapsed() / iterations << " ms\n";
    }

    // 2. 测试CPU SIMD (OpenMP)
    {
        resizeImageCPU_SIMD_OMP(src_img.data, src_width, src_height, 
                               dst_cpu.data, dst_width, dst_height, channels);
        Timer timer;
        for (int i = 0; i < iterations; ++i)
            resizeImageCPU_SIMD_OMP(src_img.data, src_width, src_height, dst_cpu.data, dst_width, dst_height, channels);
        std::cout << "2. CPU AVX2 + OpenMP: " << timer.elapsed() / iterations << " ms\n";
    }

    // 3. CUDA
    {
        resizeImageCUDA(src_img.data, src_width, src_height, dst_cuda.data, dst_width, dst_height, channels);
        CudaResizePerfStats total_stats = {0, 0, 0};
        for (int i = 0; i < iterations; ++i) {
            CudaResizePerfStats stats;
            resizeImageCUDA(src_img.data, src_width, src_height, dst_cuda.data, dst_width, dst_height, channels, &stats);
            total_stats.h2d_time_ms += stats.h2d_time_ms;
            total_stats.kernel_time_ms += stats.kernel_time_ms;
            total_stats.d2h_time_ms += stats.d2h_time_ms;
        }
        double avg_total = (total_stats.h2d_time_ms + total_stats.kernel_time_ms + total_stats.d2h_time_ms) / iterations;
        std::cout << "3. CUDA (Total): " << avg_total << " ms (Kernel: " << total_stats.kernel_time_ms / iterations << " ms)\n";
    }

    // 4. OpenCV
    {
        cv::setNumThreads(-1);
        cv::resize(src_img, dst_opencv, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_LINEAR);
        Timer timer;
        for (int i = 0; i < iterations; ++i)
            cv::resize(src_img, dst_opencv, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_LINEAR);
        std::cout << "4. OpenCV (Multi): " << timer.elapsed() / iterations << " ms\n";
    }
}

// 测试图像裁剪性能
void benchmarkCrop(const std::string& name, int src_width, int src_height, int crop_w, int crop_h, int iterations = 100)
{
    std::cout << "\n========================================\n";
    std::cout << "测试 [Crop]: " << name << "\n";
    std::cout << "源图像尺寸: " << src_width << "x" << src_height << "\n";
    std::cout << "裁剪尺寸: " << crop_w << "x" << crop_h << "\n";
    std::cout << "迭代次数: " << iterations << "\n";
    std::cout << "========================================\n";

    cv::Mat src_img = generateTestImage(src_width, src_height);
    int crop_x = (src_width - crop_w) / 2;
    int crop_y = (src_height - crop_h) / 2;
    int channels = 3;

    cv::Mat dst_cpu(crop_h, crop_w, CV_8UC3);
    cv::Mat dst_cuda(crop_h, crop_w, CV_8UC3);
    cv::Mat dst_opencv(crop_h, crop_w, CV_8UC3);

    printSeparator();
    std::cout << "性能测试结果:\n";
    printSeparator();

    // 1. Baseline
    {
        cropImageCPU_Baseline(src_img.data, src_width, src_height, dst_cpu.data, crop_x, crop_y, crop_w, crop_h, channels);
        Timer timer;
        for (int i = 0; i < iterations; ++i)
            cropImageCPU_Baseline(src_img.data, src_width, src_height, dst_cpu.data, crop_x, crop_y, crop_w, crop_h, channels);
        std::cout << "1. CPU Baseline (memcpy): " << timer.elapsed() / iterations << " ms\n";
    }

    // 2. SIMD
    {
        cropImageCPU_SIMD(src_img.data, src_width, src_height, dst_cpu.data, crop_x, crop_y, crop_w, crop_h, channels);
        Timer timer;
        for (int i = 0; i < iterations; ++i)
            cropImageCPU_SIMD(src_img.data, src_width, src_height, dst_cpu.data, crop_x, crop_y, crop_w, crop_h, channels);
        std::cout << "2. CPU SIMD (AVX2):      " << timer.elapsed() / iterations << " ms\n";
    }

    // 3. SIMD + OpenMP
    {
        cropImageCPU_SIMD_OMP(src_img.data, src_width, src_height, dst_cpu.data, crop_x, crop_y, crop_w, crop_h, channels);
        Timer timer;
        for (int i = 0; i < iterations; ++i)
            cropImageCPU_SIMD_OMP(src_img.data, src_width, src_height, dst_cpu.data, crop_x, crop_y, crop_w, crop_h, channels);
        std::cout << "3. CPU SIMD + OpenMP:    " << timer.elapsed() / iterations << " ms\n";
    }

    // 4. CUDA
    {
        cropImageCUDA(src_img.data, src_width, src_height, dst_cuda.data, crop_x, crop_y, crop_w, crop_h, channels);
        CudaCropPerfStats total_stats = {0, 0, 0};
        for (int i = 0; i < iterations; ++i) {
            CudaCropPerfStats stats;
            cropImageCUDA(src_img.data, src_width, src_height, dst_cuda.data, crop_x, crop_y, crop_w, crop_h, channels, &stats);
            total_stats.h2d_time_ms += stats.h2d_time_ms;
            total_stats.kernel_time_ms += stats.kernel_time_ms;
            total_stats.d2h_time_ms += stats.d2h_time_ms;
        }
        double avg_h2d = total_stats.h2d_time_ms / iterations;
        double avg_kernel = total_stats.kernel_time_ms / iterations;
        double avg_d2h = total_stats.d2h_time_ms / iterations;
        double avg_total = avg_h2d + avg_kernel + avg_d2h;
        std::cout << "4. CUDA (Total):         " << avg_total << " ms\n";
        std::cout << "   - H2D: " << avg_h2d << " ms, Kernel: " << avg_kernel << " ms, D2H: " << avg_d2h << " ms\n";
    }

    // 5. OpenCV
    {
        cv::Rect roi(crop_x, crop_y, crop_w, crop_h);
        Timer timer;
        for (int i = 0; i < iterations; ++i)
            dst_opencv = src_img(roi).clone();
        std::cout << "5. OpenCV (ROI clone):   " << timer.elapsed() / iterations << " ms\n";
    }

    printSeparator();
}

int main()
{
    std::cout << "图像处理算法性能对比 (Resize & Crop)\n";
    std::cout << "CPU核心数: " << omp_get_max_threads() << "\n";

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "CUDA设备: " << prop.name << "\n";
    }

    // Crop 测试：从 4K 裁剪出 2K
    benchmarkCrop("4K -> 2K Crop", 3840, 2160, 1920, 1080, 100);

    // Crop 测试：从 4K 裁剪出 640x640
    benchmarkCrop("4K -> 640x640 Crop", 3840, 2160, 640, 640, 100);

    // Resize 测试
    benchmarkResize("4K -> 640x640 Resize", 3840, 2160, 50);

    return 0;
}
