# Cinfer - Docker部署指南

Cinfer是一个基于FastAPI和React的AI推理平台，支持模型管理、推理请求处理和监控功能。本文档提供Docker部署指南。

## 系统架构

系统由以下主要组件构成：

- **后端服务**：基于FastAPI的API服务，提供模型管理、推理和监控功能
- **前端界面**：基于React的Web界面，提供用户交互
- **数据存储**：支持SQLite（默认）或PostgreSQL数据库

## 部署选项

Cinfer提供以下部署方式：

1. **分离部署**：前端和后端分别部署在不同容器中
2. **智能 GPU 支持**：可选择是否启用 GPU 加速。启用后，脚本能**自动检测** `x86_64` (标准PC) 和 `jetson` (Jetson/ARM) 平台，并选用对应的 Dockerfile 进行构建。

## 快速开始

### 先决条件

- Docker Engine (20.10.0+)
- Docker Compose (v2.0.0+)
- 如使用GPU，需要安装NVIDIA Docker支持

### 使用部署脚本

项目提供了一个灵活的部署脚本`deploy.sh`，支持多种部署方式和配置选项。

#### 基本用法

```bash
# 使用默认配置部署（分离部署，前端端口3000，后端端口8000）
./deploy.sh

# 使用GPU部署
./deploy.sh --gpu yes

# 自定义端口
./deploy.sh --backend-port 8001 --frontend-port 3001

# 指定后端主机
./deploy.sh --host 192.168.100.2
```

#### 重新构建镜像

如果修改了Dockerfile或源代码，可以使用以下命令强制重新构建镜像：

```bash
# 重新构建并启动
./deploy.sh --rebuild yes

# 仅重新构建镜像不启动
./deploy.sh --action build

# 重新构建特定实例
./deploy.sh --name prod --rebuild yes
```

#### 多实例部署

```bash
# 部署生产环境
./deploy.sh --name prod --gpu yes

# 同时部署开发环境
./deploy.sh --name dev --backend-port 8001 --frontend-port 3001
```

#### 管理实例

```bash
# 停止实例
./deploy.sh --name prod --action down

# 重启实例
./deploy.sh --name dev --action restart

# 查看日志
./deploy.sh --name prod --action logs
```

### 部署脚本选项

| 选项 | 长选项            | 说明                                                         | 默认值   |
| ---- | ----------------- | ------------------------------------------------------------ | -------- |
|      | --arch            | 指定 GPU 架构: `x86_64` 或 `jetson`。                        | 自动检测 |
| `-g` | `--gpu`           | 是否使用GPU: yes 或 no                                       | no       |
| `-b` | `--backend-port`  | 后端服务端口                                                 | 8000     |
| `-f` | `--frontend-port` | 前端服务端口                                                 | 3000     |
| `-n` | `--name`          | 实例名称                                                     | default  |
| `-a` | `--action`        | 操作: up(启动), down(停止), restart(重启), logs(查看日志), build(构建) | up       |
| `-h` | `--host`          | 后端主机名或IP地址                                           | backend  |
| `-r` | `--rebuild`       | 是否重新构建镜像: yes 或 no                                  | no       |

## 手动配置

如果需要手动配置部署，可以参考以下步骤：

### 1. 后端配置

后端环境变量在`backend/docker/prod.env`文件中配置，主要包括：

- 服务器配置（主机、端口、工作进程数）
- 数据库配置（类型、连接信息）
- 日志配置
- 安全配置（JWT密钥等）
- 模型存储配置

### 2. 前端配置

前端通过Nginx配置API代理，配置文件位于`web/nginx.conf`。

## 数据持久化

数据存储在以下位置：

- **后端数据**：挂载到容器的`/app/data`目录
- **数据库**：默认使用SQLite，存储在`/app/data/cinfer.db`
- **模型文件**：存储在`/app/data/models`目录

## 访问服务

部署完成后，可通过以下地址访问服务：

- **分离部署**：
  - 前端：`http://<主机IP>:<前端端口>`（默认为3000）
  - 后端API：`http://<主机IP>:<后端端口>/api`（默认为8000）
  - Swagger文档：`http://<主机IP>:<后端端口>/docs`

- **集成部署**：
  - 应用：`http://<主机IP>:<集成端口>`（默认为8080）
  - API：`http://<主机IP>:<集成端口>/api`
  - Swagger文档：`http://<主机IP>:<集成端口>/docs`

## 常见问题

### 1. 容器无法启动

检查Docker日志以获取详细错误信息：

```bash
docker logs <容器ID或名称>
```

### 2. 前端无法连接后端API

确保Nginx配置中的`proxy_pass`指向正确的后端地址和端口。

### 3. GPU支持问题

确保主机已正确安装NVIDIA Docker支持，并且GPU驱动版本兼容。

### 4. 修改代码后需要重新构建

如果修改了Dockerfile或源代码，需要使用`--rebuild yes`参数重新构建镜像：

```bash
./deploy.sh --rebuild yes
```

## 技术栈

- **后端**：Python, FastAPI, SQLAlchemy, ONNX Runtime
- **前端**：React, TypeScript
- **容器化**：Docker, Docker Compose
- **Web服务器**：Nginx

## 许可证

MIT License 