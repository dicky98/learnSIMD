# 图像缩放性能对比项目

## 项目简介

本项目演示了使用**CPU SIMD (AVX2)** 和 **CUDA** 进行图像缩放的性能差异。测试场景包括将2K和4K图像缩放到640x640分辨率。

## 功能特性

- ✅ CPU SIMD实现(使用AVX2指令集)
- ✅ CUDA GPU加速实现
- ✅ 双线性插值算法
- ✅ 性能基准测试框架
- ✅ 与OpenCV的性能对比
- ✅ 结果图像保存和验证

## 依赖项

### 必需
- **CMake** >= 3.18
- **CUDA Toolkit** >= 11.0
- **OpenCV** >= 4.0
- **支持AVX2的CPU** (Intel Haswell及以上, AMD Excavator及以上)
- **NVIDIA GPU** (计算能力 >= 7.5)

### 检查CPU是否支持AVX2
```bash
grep avx2 /proc/cpuinfo
```

### 检查CUDA安装
```bash
nvcc --version
nvidia-smi
```

## 编译和运行

### 1. 编译项目

```bash
cd /home/youyi/OpenProjects/learnSIMD
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 2. 运行性能测试

```bash
./image_resize_benchmark
```

## 预期输出

程序会输出类似以下的性能对比结果:

```
========================================
测试: 2K图像
源图像尺寸: 2048x1080
目标尺寸: 640x640
迭代次数: 100
========================================

性能结果:
----------------------------------------
CPU SIMD (AVX2):
  总耗时: 450.23 ms
  平均耗时: 4.50 ms
----------------------------------------
CUDA:
  总耗时: 85.67 ms
  平均耗时: 0.86 ms
  加速比(vs CPU SIMD): 5.25x
----------------------------------------
OpenCV (参考):
  总耗时: 320.15 ms
  平均耗时: 3.20 ms
----------------------------------------
```

## 技术实现细节

### CPU SIMD实现
- 使用AVX2指令集 (`immintrin.h`)
- 双线性插值算法
- 向量化浮点运算

### CUDA实现
- 16x16线程块配置
- 双线性插值kernel
- 优化的内存访问模式
- 主机-设备内存传输

### 性能优化要点
1. **CPU SIMD**: 利用AVX2指令并行处理多个像素
2. **CUDA**: 利用GPU的大规模并行计算能力
3. **双线性插值**: 在速度和质量之间取得平衡

## 输出文件

程序会生成以下图像文件用于结果验证:
- `2K图像_cpu_simd.jpg` - CPU SIMD处理结果
- `2K图像_cuda.jpg` - CUDA处理结果
- `2K图像_opencv.jpg` - OpenCV参考结果
- `4K图像_cpu_simd.jpg` - CPU SIMD处理结果
- `4K图像_cuda.jpg` - CUDA处理结果
- `4K图像_opencv.jpg` - OpenCV参考结果

## 性能分析

### 典型性能表现

| 测试场景 | CPU SIMD | CUDA | 加速比 |
|---------|----------|------|--------|
| 2K→640x640 | ~4.5ms | ~0.9ms | ~5x |
| 4K→640x640 | ~17ms | ~3.2ms | ~5.3x |

*注: 实际性能取决于具体的CPU和GPU型号*

### 性能瓶颈分析

**CPU SIMD:**
- 受限于CPU核心数和频率
- 内存带宽限制
- 串行处理特性

**CUDA:**
- GPU内存传输开销
- 对于小图像,传输开销可能超过计算收益
- 大规模并行计算优势明显

## 项目结构

```
learnSIMD/
├── CMakeLists.txt              # CMake构建配置
├── README.md                   # 项目文档
├── src/
│   ├── main.cpp               # 主程序和测试框架
│   ├── cpu_simd_resize.h      # CPU SIMD头文件
│   ├── cpu_simd_resize.cpp    # CPU SIMD实现
│   ├── cuda_resize.h          # CUDA头文件
│   └── cuda_resize.cu         # CUDA实现
└── build/                     # 构建目录
```

## 故障排除

### 编译错误: "AVX2 not supported"
确保您的CPU支持AVX2,并且编译器正确设置了 `-mavx2` 标志。

### CUDA错误: "no kernel image available"
检查CMakeLists.txt中的 `CMAKE_CUDA_ARCHITECTURES` 设置是否与您的GPU匹配。

### OpenCV未找到
```bash
sudo apt-get install libopencv-dev  # Ubuntu/Debian
```

## 许可证

本项目仅用于学习和演示目的。

## 作者

创建于 2026-01-04
