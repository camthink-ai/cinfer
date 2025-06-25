# Cinfer GPU Docker环境说明

## GPU环境概述

Cinfer的GPU容器环境基于NVIDIA CUDA技术栈构建，提供高性能深度学习推理能力。容器镜像大小约为40GB，主要由以下组件构成。

## 基础环境

| 组件 | 版本 | 说明 |
|------|------|------|
| 基础镜像 | nvidia/cuda:12.4.0-devel-ubuntu22.04 | NVIDIA官方CUDA开发镜像，约7.22GB |
| 操作系统 | Ubuntu 22.04 LTS | 长期支持版本 |
| CUDA | 12.4.0 | NVIDIA并行计算平台 |
| Python | 3.10 | 主要编程语言环境 |

## 深度学习框架与加速库

| 组件 | 版本 | 大小(估计) | 说明 |
|------|------|------------|------|
| cuDNN | 9.1.0.70 | ~1GB | NVIDIA深度神经网络加速库 |
| TensorRT | 10.0.1.6 | ~1.5GB | NVIDIA高性能深度学习推理优化器 |
| PyTorch | 最新(cu124) | ~2GB | 深度学习框架，使用CUDA 12.4版本 |
| torchvision | 最新(cu124) | ~400MB | PyTorch计算机视觉库 |
| torchaudio | 最新(cu124) | ~200MB | PyTorch音频处理库 |
| ONNX Runtime GPU | 1.22.0 | ~300MB | 跨平台推理加速库GPU版本 |
| ONNX | 1.18.0 | ~50MB | 开放神经网络交换格式 |
| OpenVINO | 2024.6.0 | ~1GB | Intel视觉推理优化库 |
| PaddleOCR | 3.0.0 | ~200MB | 百度飞桨OCR工具库 |
| PaddlePaddle GPU | 3.0.0 | ~2GB | 百度飞桨深度学习框架GPU版本 |
| PyCUDA | 最新 | ~100MB | Python CUDA接口 |

## 本地依赖包

项目中的`backend/backend_apps`目录包含了预下载的cuDNN和TensorRT安装包，大小约为3.0GB。这些包在构建Docker镜像时被复制并安装到容器中。

## 镜像大小分析

总计约40GB的Docker镜像大小主要来源于：

1. **基础CUDA开发环境**: ~7.22GB
   - CUDA工具链、编译器和运行时库

2. **深度学习框架**: ~6GB
   - PyTorch全家桶(CUDA版): ~2.6GB
   - PaddlePaddle GPU版: ~2GB
   - ONNX Runtime GPU: ~300MB
   - 其他框架和工具: ~1.1GB

3. **AI加速库**: ~2.5GB
   - cuDNN: ~1GB
   - TensorRT: ~1.5GB

4. **预训练模型和数据**: ~20GB
   - 各种预训练模型文件
   - 示例数据集
   - 中间缓存文件

5. **操作系统和Python环境**: ~4GB
   - Ubuntu基础系统: ~2GB
   - Python及依赖包: ~2GB

## 优化建议

如需减小镜像体积，可考虑以下方案：

1. 使用多阶段构建，分离编译环境和运行环境
2. 移除不必要的预训练模型，改为按需下载
3. 清理构建过程中的缓存文件
4. 使用更轻量级的基础镜像，如nvidia/cuda:12.4.0-runtime而非devel版本

## 版本兼容性

- NVIDIA驱动版本需求: ≥ 535.129.03
- 支持的GPU: 支持CUDA 12.4的NVIDIA GPU (Ampere、Hopper、Ada Lovelace架构优先)
- 主要依赖版本固定，确保环境稳定性和一致性

## 注意事项

1. 镜像包含完整的开发和推理环境，适合生产和开发使用
2. 首次拉取或构建镜像需要较长时间和较大带宽
3. 运行容器需要足够的磁盘空间(至少50GB)和内存(至少16GB)
4. 宿主机必须正确安装NVIDIA驱动和NVIDIA Container Toolkit 