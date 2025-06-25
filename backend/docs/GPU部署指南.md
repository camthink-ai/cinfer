# Cinfer GPU 部署指南

## 目录
- [系统要求](#系统要求)
- [部署前准备](#部署前准备)
- [部署方式](#部署方式)
- [GPU配置选项](#gpu配置选项)
- [性能优化](#性能优化)
- [故障排除](#故障排除)
- [常见问题](#常见问题)

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU，支持CUDA 12.4
- **内存**: 最低16GB，推荐32GB或更高
- **存储**: 最低50GB可用空间，用于Docker镜像和模型文件
- **CPU**: 8核或更高

### 软件要求
- **操作系统**: Ubuntu 20.04/22.04 LTS 或兼容Linux发行版
- **NVIDIA驱动**: 535.129.03 或更高版本
- **Docker**: 24.0.0 或更高版本
- **NVIDIA Container Toolkit**: 最新版本

## 部署前准备

### 1. 安装NVIDIA驱动
```bash
# 添加NVIDIA PPA仓库
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# 安装NVIDIA驱动
sudo apt install nvidia-driver-535

# 重启系统
sudo reboot

# 验证驱动安装
nvidia-smi
```

### 2. 安装Docker和NVIDIA Container Toolkit
```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 重启Docker服务
sudo systemctl restart docker

# 验证NVIDIA Container Toolkit安装
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 3. 下载Cinfer项目
```bash
git clone https://github.com/your-org/cinfer.git
cd cinfer
```

## 部署方式

Cinfer支持两种GPU部署方式：

### 1. 分离部署模式
前端和后端分别部署在不同的容器中，后端容器使用GPU。

```bash
./deploy.sh --mode separate --gpu yes --backend-port 8000 --frontend-port 3000 --name cinfer-gpu
```

### 2. 集成部署模式
前端和后端部署在同一个容器中，该容器使用GPU。

```bash
./deploy.sh --mode integrated --gpu yes --integrated-port 8000 --name cinfer-gpu-integrated
```

## GPU配置选项

### 基本GPU部署命令
```bash
./deploy.sh --gpu yes [其他选项]
```

### 完整GPU部署示例
```bash
# 分离部署示例
./deploy.sh --mode separate --gpu yes --backend-port 8000 --frontend-port 3000 --name prod-gpu --rebuild yes

# 集成部署示例
./deploy.sh --mode integrated --gpu yes --integrated-port 8080 --name prod-gpu-integrated --rebuild yes
```

### 重要参数说明
- `--gpu yes`: 启用GPU支持
- `--mode`: 部署模式，可选 `separate` 或 `integrated`
- `--name`: 实例名称，用于区分多个部署
- `--rebuild`: 是否重新构建镜像，首次部署或更新后设为 `yes`

## 性能优化

### 1. 环境变量配置
在 `backend/docker/prod.env` 文件中添加或修改以下环境变量：

```
# GPU内存使用限制（MB）
GPU_MEMORY_LIMIT=4096

# 启用TensorRT优化
ENABLE_TENSORRT=true

# 设置CUDA设备ID（多GPU环境）
CUDA_VISIBLE_DEVICES=0

# 模型优化级别
MODEL_OPTIMIZATION_LEVEL=3
```

### 2. 多GPU配置
如果您的系统有多个GPU，可以通过环境变量指定使用特定GPU：

```
# 在prod.env中设置
CUDA_VISIBLE_DEVICES=0,1  # 使用GPU 0和1
```

### 3. 模型量化和优化
在GPU模式下，系统会自动使用TensorRT进行模型优化。您可以通过API或Web界面进一步配置模型量化选项。

## 故障排除

### 1. 检查GPU可见性
如果容器内无法检测到GPU，请执行以下检查：

```bash
# 在宿主机上验证GPU状态
nvidia-smi

# 检查Docker中的GPU支持
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 检查Cinfer容器中的GPU支持
docker exec -it backend_cinfer-gpu nvidia-smi
```

### 2. CUDA相关错误
如果遇到CUDA错误，可能是驱动版本与CUDA版本不兼容：

```bash
# 检查NVIDIA驱动版本
nvidia-smi | grep "Driver Version"

# 检查容器内CUDA版本
docker exec -it backend_cinfer-gpu nvcc --version
```

确保您的NVIDIA驱动支持CUDA 12.4。

### 3. 内存不足错误
GPU内存不足时，可以调整模型配置或限制内存使用：

```bash
# 编辑环境变量文件
nano backend/docker/prod.env

# 添加或修改GPU内存限制
GPU_MEMORY_LIMIT=2048  # 限制为2GB
```

### 4. 检查日志
查看容器日志以获取更详细的错误信息：

```bash
./deploy.sh --name cinfer-gpu --action logs
```

## 常见问题

### Q: 如何确认GPU正在被使用？
**A**: 在容器内运行 `nvidia-smi` 命令，查看GPU使用率和内存占用情况。或者通过Web界面的系统监控页面查看GPU利用率。

### Q: 部署后模型推理速度没有明显提升？
**A**: 确认以下几点：
1. 模型是否支持GPU加速
2. 是否已启用TensorRT优化
3. 检查GPU利用率，确认是否真正使用了GPU

### Q: 如何在多个GPU之间分配负载？
**A**: 目前系统默认使用所有可见GPU。如需指定特定GPU，请在环境变量中设置 `CUDA_VISIBLE_DEVICES`。

### Q: 如何监控GPU使用情况？
**A**: 可以使用以下方法：
1. 宿主机上运行 `nvidia-smi -l 1` 实时监控
2. 使用Cinfer内置的监控面板
3. 集成Prometheus和Grafana进行高级监控

### Q: 如何更新GPU驱动或CUDA版本？
**A**: 更新驱动后需要重新构建容器：
```bash
./deploy.sh --name cinfer-gpu --action down
./deploy.sh --mode separate --gpu yes --name cinfer-gpu --rebuild yes
```

---

如需更多帮助，请参考完整文档或联系技术支持团队。 