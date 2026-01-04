#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include "cpu_simd_resize.h"
#include "cuda_resize.h"

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

// 测试图像缩放性能
void benchmarkResize(const std::string& name, int src_width, int src_height, int iterations = 100)
{
    std::cout << "\n========================================\n";
    std::cout << "测试: " << name << "\n";
    std::cout << "源图像尺寸: " << src_width << "x" << src_height << "\n";
    std::cout << "目标尺寸: 640x640\n";
    std::cout << "迭代次数: " << iterations << "\n";
    std::cout << "========================================\n";

    // 生成测试图像
    cv::Mat src_img = generateTestImage(src_width, src_height);
    int dst_width = 640;
    int dst_height = 640;
    int channels = 3;

    // 准备目标图像缓冲区
    cv::Mat dst_cpu(dst_height, dst_width, CV_8UC3);
    cv::Mat dst_cuda(dst_height, dst_width, CV_8UC3);
    cv::Mat dst_opencv(dst_height, dst_width, CV_8UC3);

    printSeparator();
    std::cout << "性能测试结果:\n";
    printSeparator();

    // 1. 测试CPU SIMD (单线程)
    {
        // 预热
        resizeImageCPU_SIMD(src_img.data, src_width, src_height, 
                           dst_cpu.data, dst_width, dst_height, channels);
        
        Timer timer;
        for (int i = 0; i < iterations; ++i)
        {
            resizeImageCPU_SIMD(src_img.data, src_width, src_height,
                               dst_cpu.data, dst_width, dst_height, channels);
        }
        double avg_time = timer.elapsed() / iterations;
        std::cout << "1. CPU AVX2 (单线程):\n";
        std::cout << "   平均耗时: " << avg_time << " ms\n";
        std::cout << "   相对基准: 1.0x\n";
    }

    // 2. 测试CPU SIMD (OpenMP多线程)
    {
        // 预热
        resizeImageCPU_SIMD_OMP(src_img.data, src_width, src_height, 
                               dst_cpu.data, dst_width, dst_height, channels);
        
        Timer timer;
        for (int i = 0; i < iterations; ++i)
        {
            resizeImageCPU_SIMD_OMP(src_img.data, src_width, src_height,
                                   dst_cpu.data, dst_width, dst_height, channels);
        }
        double avg_time = timer.elapsed() / iterations;
        std::cout << "2. CPU AVX2 + OpenMP (" << omp_get_max_threads() << " 线程):\n";
        std::cout << "   平均耗时: " << avg_time << " ms\n";
    }

    // 3. 测试CUDA (总耗时和分段耗时)
    {
        // 预热
        resizeImageCUDA(src_img.data, src_width, src_height,
                       dst_cuda.data, dst_width, dst_height, channels);

        CudaResizePerfStats total_stats = {0, 0, 0};
        
        // 运行测试并收集统计
        for (int i = 0; i < iterations; ++i)
        {
            CudaResizePerfStats stats;
            resizeImageCUDA(src_img.data, src_width, src_height,
                          dst_cuda.data, dst_width, dst_height, channels, &stats);
            total_stats.h2d_time_ms += stats.h2d_time_ms;
            total_stats.kernel_time_ms += stats.kernel_time_ms;
            total_stats.d2h_time_ms += stats.d2h_time_ms;
        }

        double avg_h2d = total_stats.h2d_time_ms / iterations;
        double avg_kernel = total_stats.kernel_time_ms / iterations;
        double avg_d2h = total_stats.d2h_time_ms / iterations;
        double avg_total = avg_h2d + avg_kernel + avg_d2h;

        std::cout << "3. CUDA (完整流程):\n";
        std::cout << "   总平均耗时: " << avg_total << " ms\n";
        std::cout << "   [分段详情]\n";
        std::cout << "     传输 (Host->Device): " << avg_h2d << " ms (" << (avg_h2d/avg_total)*100 << "%)\n";
        std::cout << "     计算 (Kernel only) : " << avg_kernel << " ms (" << (avg_kernel/avg_total)*100 << "%)\n";
        std::cout << "     传输 (Device->Host): " << avg_d2h << " ms (" << (avg_d2h/avg_total)*100 << "%)\n";
    }

    // 4. OpenCV (默认多线程)
    {
        cv::setNumThreads(-1); // 启用默认线程数
        // 预热
        cv::resize(src_img, dst_opencv, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_LINEAR);

        Timer timer;
        for (int i = 0; i < iterations; ++i)
        {
            cv::resize(src_img, dst_opencv, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_LINEAR);
        }
        double avg_time = timer.elapsed() / iterations;
        std::cout << "4. OpenCV (默认多线程):\n";
        std::cout << "   平均耗时: " << avg_time << " ms\n";
    }

    // 5. OpenCV (强制单线程)
    {
        cv::setNumThreads(1); // 强制单线程
        // 预热
        cv::resize(src_img, dst_opencv, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_LINEAR);

        Timer timer;
        for (int i = 0; i < iterations; ++i)
        {
            cv::resize(src_img, dst_opencv, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_LINEAR);
        }
        double avg_time = timer.elapsed() / iterations;
        std::cout << "5. OpenCV (强制单线程):\n";
        std::cout << "   平均耗时: " << avg_time << " ms\n";
    }

    printSeparator();
    
    // 恢复OpenCV默认设置
    cv::setNumThreads(-1);
}

int main()
{
    std::cout << "图像缩放性能深度调查\n";
    std::cout << "对比: CPU单线程 vs OpenMP多线程 vs CUDA分段 vs OpenCV\n";
    std::cout << "CPU核心数: " << omp_get_max_threads() << "\n";
    std::cout << "========================================\n";

    // 检查CUDA设备
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        std::cerr << "错误: 未找到CUDA设备!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA设备: " << prop.name << "\n";

    // 测试2K图像
    benchmarkResize("2K图像", 2048, 1080, 50);

    // 测试4K图像
    benchmarkResize("4K图像", 3840, 2160, 50);

    std::cout << "\n测试完成!\n";
    return 0;
}
