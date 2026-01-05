cmake_minimum_required(VERSION 3.18)
project(ImageResizeBenchmark LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 查找OpenMP (用于CPU多线程)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma ${OpenMP_CXX_FLAGS}")

# 设置CUDA架构 (根据您的GPU调整)
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# 源文件
set(SOURCES
    src/main.cpp
    src/cpu_simd_resize.cpp
    src/cuda_resize.cu
    src/cpu_crop.cpp
    src/cuda_crop.cu
)

# 创建可执行文件
add_executable(image_resize_benchmark ${SOURCES})

# 链接库
target_link_libraries(image_resize_benchmark
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

# 设置包含目录
target_include_directories(image_resize_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
