#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
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

    // 预热
    resizeImageCPU_SIMD(src_img.data, src_width, src_height, 
                        dst_cpu.data, dst_width, dst_height, channels);
    resizeImageCUDA(src_img.data, src_width, src_height,
                    dst_cuda.data, dst_width, dst_height, channels);

    // 测试CPU SIMD性能
    Timer timer;
    for (int i = 0; i < iterations; ++i)
    {
        resizeImageCPU_SIMD(src_img.data, src_width, src_height,
                           dst_cpu.data, dst_width, dst_height, channels);
    }
    double cpu_time = timer.elapsed();
    double cpu_avg = cpu_time / iterations;

    // 测试CUDA性能
    timer.reset();
    for (int i = 0; i < iterations; ++i)
    {
        resizeImageCUDA(src_img.data, src_width, src_height,
                       dst_cuda.data, dst_width, dst_height, channels);
    }
    double cuda_time = timer.elapsed();
    double cuda_avg = cuda_time / iterations;

    // 测试OpenCV性能(作为参考)
    timer.reset();
    for (int i = 0; i < iterations; ++i)
    {
        cv::resize(src_img, dst_opencv, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_LINEAR);
    }
    double opencv_time = timer.elapsed();
    double opencv_avg = opencv_time / iterations;

    // 输出结果
    std::cout << "\n性能结果:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "CPU SIMD (AVX2):\n";
    std::cout << "  总耗时: " << cpu_time << " ms\n";
    std::cout << "  平均耗时: " << cpu_avg << " ms\n";
    std::cout << "----------------------------------------\n";
    std::cout << "CUDA:\n";
    std::cout << "  总耗时: " << cuda_time << " ms\n";
    std::cout << "  平均耗时: " << cuda_avg << " ms\n";
    std::cout << "  加速比(vs CPU SIMD): " << (cpu_time / cuda_time) << "x\n";
    std::cout << "----------------------------------------\n";
    std::cout << "OpenCV (参考):\n";
    std::cout << "  总耗时: " << opencv_time << " ms\n";
    std::cout << "  平均耗时: " << opencv_avg << " ms\n";
    std::cout << "----------------------------------------\n";

    // 保存结果图像
    std::string prefix = name;
    std::replace(prefix.begin(), prefix.end(), ' ', '_');
    cv::imwrite(prefix + "_cpu_simd.jpg", dst_cpu);
    cv::imwrite(prefix + "_cuda.jpg", dst_cuda);
    cv::imwrite(prefix + "_opencv.jpg", dst_opencv);
    
    std::cout << "\n结果图像已保存:\n";
    std::cout << "  " << prefix << "_cpu_simd.jpg\n";
    std::cout << "  " << prefix << "_cuda.jpg\n";
    std::cout << "  " << prefix << "_opencv.jpg\n";
}

int main()
{
    std::cout << "图像缩放性能对比测试\n";
    std::cout << "对比CPU SIMD(AVX2) vs CUDA\n";
    std::cout << "========================================\n";

    // 检查CUDA设备
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0)
    {
        std::cerr << "错误: 未找到CUDA设备!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA设备: " << prop.name << "\n";
    std::cout << "计算能力: " << prop.major << "." << prop.minor << "\n";

    // 测试2K图像 (2048x1080)
    benchmarkResize("2K图像", 2048, 1080, 100);

    // 测试4K图像 (3840x2160)
    benchmarkResize("4K图像", 3840, 2160, 100);

    std::cout << "\n测试完成!\n";
    return 0;
}
