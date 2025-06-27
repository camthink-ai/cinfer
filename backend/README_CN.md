# Cinfer - 视觉AI推理服务

Cinfer是一个灵活、可扩展的AI推理服务平台，用于高效部署和管理视觉AI模型。

## 功能特点

- **多引擎支持**：TensorRT、ONNX Runtime、PyTorch
- **动态模型加载**：无需中断服务即可热插拔模型
- **灵活API**：用于管理和推理的RESTful API
- **访问控制**：基于令牌的身份验证和权限管理
- **资源管理**：请求队列、速率限制、IP白名单
- **监控**：CPU、GPU和内存使用的系统指标

## 快速开始

### 前提条件
- Python 3.10+
- CUDA（用于GPU支持）
- Docker（可选）

### 安装
```bash
git clone https://github.com/yourusername/cinfer.git
cd cinfer
pip install -r requirements.txt
python run.py
```

### 开发模式
```bash
python run.py
```

### 生产模式
```bash
python run.py --prod --workers 4
```

## API端点

### 管理API
- `/api/v1/internal/auth`：身份验证管理
- `/api/v1/internal/system`：系统监控和管理
- `/api/v1/internal/models`：模型管理
- `/api/v1/internal/tokens`：API令牌管理

### 推理API
- `GET /api/v1/models`：列出可用模型
- `POST /api/v1/inference/{model_id}`：运行推理

## 身份验证

- **管理API**：使用登录端点获取的X-Auth-Token
- **推理API**：使用管理员提供的X-Access-Token

## Docker部署

```bash
# 构建（CPU/GPU）
./scripts/docker-build.sh [--gpu]

# 运行
./scripts/docker-run.sh [--gpu]
```

访问API：http://localhost:8000

## 许可证

[MIT许可证](LICENSE) 