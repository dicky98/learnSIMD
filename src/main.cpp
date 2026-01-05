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
#include "cpu_norm.h"
#include "cuda_norm.h"

// 性能计时辅助类
class Timer
{
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
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
    #pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            img.at<cv::Vec3b>(y, x)[0] = (x * 255) / width;           // B
            img.at<cv::Vec3b>(y, x)[1] = (y * 255) / height;          // G
            img.at<cv::Vec3b>(y, x)[2] = ((x + y) * 255) / (width + height); // R
        }
    }
    return img;
}

void printSeparator() {
    std::cout << "----------------------------------------\n";
}

// 测试归一化性能
void benchmarkNormalize(const std::string& name, int w, int h, int iterations = 100)
{
    std::cout << "\n========================================\n";
    std::cout << "测试 [Normalize]: " << name << "\n";
    std::cout << "图像尺寸: " << w << "x" << h << " (HWC uint8 -> CHW float32)\n";
    std::cout << "迭代次数: " << iterations << "\n";
    std::cout << "========================================\n";

    cv::Mat src_img = generateTestImage(w, h);
    std::vector<float> dst_cpu(w * h * 3);
    std::vector<float> dst_cuda(w * h * 3);

    printSeparator();
    std::cout << "性能测试结果:\n";
    printSeparator();

    // 1. CPU Baseline
    {
        normalizeCPU_Baseline(src_img.data, w, h, dst_cpu.data());
        Timer timer;
        for (int i = 0; i < iterations; ++i)
            normalizeCPU_Baseline(src_img.data, w, h, dst_cpu.data());
        std::cout << "1. CPU Baseline:         " << timer.elapsed() / iterations << " ms\n";
    }

    // 2. CPU SIMD
    {
        normalizeCPU_SIMD(src_img.data, w, h, dst_cpu.data());
        Timer timer;
        for (int i = 0; i < iterations; ++i)
            normalizeCPU_SIMD(src_img.data, w, h, dst_cpu.data());
        std::cout << "2. CPU SIMD (AVX2):      " << timer.elapsed() / iterations << " ms\n";
    }

    // 3. CUDA
    {
        normalizeCUDA(src_img.data, w, h, dst_cuda.data());
        CudaNormPerfStats total_stats = {0, 0, 0};
        for (int i = 0; i < iterations; ++i) {
            CudaNormPerfStats stats;
            normalizeCUDA(src_img.data, w, h, dst_cuda.data(), &stats);
            total_stats.h2d_time_ms += stats.h2d_time_ms;
            total_stats.kernel_time_ms += stats.kernel_time_ms;
            total_stats.d2h_time_ms += stats.d2h_time_ms;
        }
        double avg_h2d = total_stats.h2d_time_ms / iterations;
        double avg_kernel = total_stats.kernel_time_ms / iterations;
        double avg_d2h = total_stats.d2h_time_ms / iterations;
        double avg_total = avg_h2d + avg_kernel + avg_d2h;
        std::cout << "3. CUDA (Total):         " << avg_total << " ms\n";
        std::cout << "   - H2D: " << avg_h2d << " ms, Kernel: " << avg_kernel << " ms, D2H: " << avg_d2h << " ms\n";
    }

    // 4. OpenCV (blobFromImage)
    {
        cv::Mat blob;
        Timer timer;
        for (int i = 0; i < iterations; ++i)
            blob = cv::dnn::blobFromImage(src_img, 1.0/255.0, cv::Size(), cv::Scalar(), true, false);
        std::cout << "4. OpenCV (blobFromImage): " << timer.elapsed() / iterations << " ms\n";
    }

    printSeparator();
}

int main()
{
    std::cout << "图像处理算法性能对比 (Resize & Crop & Normalize)\n";
    std::cout << "CPU核心数: " << omp_get_max_threads() << "\n";

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "CUDA设备: " << prop.name << "\n";
    }

    // Normalize 测试: 640x640 (YOLOv8 标准尺寸)
    benchmarkNormalize("640x640 Normalize", 640, 640, 100);

    // Normalize 测试: 2K
    benchmarkNormalize("2K Normalize", 1920, 1080, 100);

    return 0;
}
