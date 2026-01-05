# YOLOv8 归一化性能报告

## 1. 测试环境
- **CPU**: 16 核心 (支持 AVX2)
- **GPU**: NVIDIA GeForce RTX 4070
- **任务**: BGR HWC (uint8) -> RGB CHW (float32), 缩放系数 1/255.0

## 2. 测试结果 (平均耗时)

| 场景 | CPU Baseline | CPU SIMD | CUDA (Total) | OpenCV (blob) |
| :--- | :---: | :---: | :---: | :---: |
| **640x640** | 5.37 ms | 4.26 ms | **1.11 ms** | 1.15 ms |
| **2K (1920x1080)** | 24.29 ms | 22.69 ms | **5.11 ms** | 12.61 ms |

### CUDA 耗时拆解 (2K 场景)
- **Host -> Device (H2D)**: 1.15 ms (22%)
- **Kernel 计算**: 0.08 ms (1.5%)
- **Device -> Host (D2H)**: 3.87 ms (76%)
- **总计**: 5.11 ms

## 3. 深度分析

### 为什么 Normalize 适合 GPU？
1.  **计算量增加**: 相比 Crop 纯拷贝，Normalize 涉及 `uint8 -> float32` 转换和除法运算。虽然对单个像素不重，但对数百万个像素累计起来，GPU 的吞吐量优势开始显现。
2.  **数据类型膨胀**: 输出是 `float32`，数据大小是输入的 4 倍。
    - 在 CPU 上，这意味着更大的内存带宽压力和缓存未命中。
    - 在 GPU 上，D2H 传输时间虽长，但 Kernel 并行处理能力极强，抵消了由于数据变大带来的部分开销。
3.  **OpenCV 的表现**: `cv::dnn::blobFromImage` 在 2K 分辨率下表现一般 (12.6ms)，而 CUDA (5.1ms) 快了一倍以上。

### 多阶段流程建议 (Crop + Normalize)
如果我们将 Crop 和 Normalize 放在一起考虑：
- **CPU 方案**: Crop (memcpy, 0.6ms) + Normalize (Baseline, 24ms) = **24.6ms**
- **GPU 方案**: Crop (Kernel, 0.06ms) + Normalize (Kernel, 0.08ms) + 数据传输 = **~5ms** (假设数据只传输一次)

**核心结论**: 
当处理链路中包含**计算型**操作（如归一化、色域转换）时，即便存在数据传输开销，GPU 依然是更优选择。尤其是如果能将「Resize -> Crop -> Normalize -> Inference」全流程保留在 GPU 上，性能将会有跨越式的提升。

## 4. 结论
- 对于 **YOLOv8 预处理**，强烈建议使用 **CUDA** 实现，或者将图片原本就放在显存中。
- CPU 端的归一化效率较低，在处理高分辨率图像时性能下降明显。
