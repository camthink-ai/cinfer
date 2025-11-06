# Cinfer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[English](./README.md) | [简体中文](./README_zh.md) 

Cinfer is a general-purpose Vision AI inference software by CamThink, open to the developer community. It supports deploying Vision AI inference services, provides model management, and exposes standard OpenAPI services. It is suitable for integrating as an image recognition AI component in vision application development, allowing you to call various models for image recognition. It is lightweight and simple, making it ideal for quickly adding models for deployment and project integration.


## Documentation
[Quick Start](https://wiki.camthink.ai/docs/ai-application/cinfer-ai-Inference-service/quick-start)

[User Guide](https://wiki.camthink.ai/docs/ai-application/cinfer-ai-Inference-service/quick-start)

## System Architecture

The system consists of the following main components:

- **Backend Service**: API service based on FastAPI, providing model management, inference, and monitoring functionality
- **Frontend Interface**: Web interface based on React, offering user interactions
- **Data Storage**: Supports SQLite (default) or PostgreSQL databases

## Models Directory

The repository ships with a `models` directory that contains sample ONNX packages ready for import. Each model folder includes the exported weights and a `params.yaml` or `params.yml` file describing how the backend should serve it, so you can copy the folder into your persistent model storage (`/app/data/models` by default) or register it through the API/UI without additional setup. For more details, see the [Model Samples guide](models/README.md).

- `models/object-detection`: COCO-style 80-class detector using `80-target-detection-yolo-11n.onnx` with `params.yml`.
- `models/generic-target-detection`: Lightweight digit detector (`0`–`9`) packaged as `meter_v5i-6.onnx` with `params.yaml`.
- `models/pose-estimation`: Human pose estimation sample (`yolov8n-pose.onnx`) with keypoint outputs defined in `params.yaml`.
- `models/instance-seg`: Pointer-gauge instance segmentation model (`yolov8s-seg-20240816.onnx`) relying on the `pointer` post-processing strategy and metadata such as `scale_min`/`scale_max`.

## Deployment Options

Cinfer provides the following deployment methods:

1. **Separated Deployment**: Frontend and backend are deployed in separate containers
2. **Intelligent GPU Support**: You can choose whether to enable GPU acceleration. When enabled, the script will **automatically detect** the `x86_64` (Standard PC) or `jetson` (Jetson/ARM) platforms and select the corresponding Dockerfile for building.

## Quick Start

### Prerequisites

- Docker Engine (20.10.0+)
- Docker Compose (v2.0.0+)
- If using GPU, NVIDIA Docker support is required

### Using the Deployment Script

The project provides a flexible deployment script `deploy.sh`, supporting multiple deployment methods and configuration options.

#### Basic Usage

```bash
# Deploy with default configuration (separated deployment, frontend port 3000, backend port 8000)
./deploy.sh

# Deploy with GPU support
./deploy.sh --gpu yes

# Customize ports
./deploy.sh --backend-port 8001 --frontend-port 3001

# Specify backend host
./deploy.sh --host 192.168.100.2
```

#### Rebuild Image

If you have modified the Dockerfile or source code, you can force rebuilding the image using the following commands:

```bash
# Rebuild and start
./deploy.sh --rebuild yes

# Only rebuild the image, do not start
./deploy.sh --action build

# Rebuild a specific instance
./deploy.sh --name prod --rebuild yes
```

#### Multi-instance Deployment

```bash
# Deploy production environment
./deploy.sh --name prod --gpu yes

# Deploy development environment simultaneously
./deploy.sh --name dev --backend-port 8001 --frontend-port 3001
```

#### Manage Instances

```bash
# Stop an instance
./deploy.sh --name prod --action down

# Restart an instance
./deploy.sh --name dev --action restart

# View logs
./deploy.sh --name prod --action logs
```

### Deployment Script Options

| Option | Long Option        | Description                                                                                            | Default   |
| ------ | ----------------- | ------------------------------------------------------------------------------------------------------ | --------- |
|        | --arch            | Specify GPU architecture: `x86_64` or `jetson`.                                                        | Auto      |
| `-g`   | `--gpu`           | Use GPU: yes or no                                                                                     | no        |
| `-b`   | `--backend-port`  | Backend service port                                                                                   | 8000      |
| `-f`   | `--frontend-port` | Frontend service port                                                                                  | 3000      |
| `-n`   | `--name`          | Instance name                                                                                          | default   |
| `-a`   | `--action`        | Action: up (start), down (stop), restart (restart), logs (view logs), build (build)                    | up        |
| `-h`   | `--host`          | Backend hostname or IP address                                                                         | backend   |
| `-r`   | `--rebuild`       | Rebuild image: yes or no                                                                               | no        |

## Manual Configuration

If you need to manually configure deployment, you can follow the steps below:

### 1. Backend Configuration

Backend environment variables are configured in the `backend/docker/prod.env` file, mainly including:

- Server configuration (host, port, number of worker processes)
- Database configuration (type, connection info)
- Logging configuration
- Security configuration (JWT keys, etc.)
- Model storage configuration

### 2. Frontend Configuration

The frontend configures API proxying through Nginx, with the configuration file at `web/nginx.conf`.

## Data Persistence

Data is stored in the following locations:

- **Backend Data**: Mounted to the container’s `/app/data` directory
- **Database**: SQLite is used by default, stored at `/app/data/cinfer.db`
- **Model Files**: Stored in the `/app/data/models` directory

## Accessing the Services

After deployment, you can access the services at the following addresses:

- **Separated Deployment**:
  - Frontend: `http://<host-ip>:<frontend-port>` (default: 3000)
  - Backend API: `http://<host-ip>:<backend-port>/api` (default: 8000)
  - Swagger Docs: `http://<host-ip>:<backend-port>/docs`

- **Integrated Deployment**:
  - App: `http://<host-ip>:<integrated-port>` (default: 8080)
  - API: `http://<host-ip>:<integrated-port>/api`
  - Swagger Docs: `http://<host-ip>:<integrated-port>/docs`

## FAQ

### 1. Container Cannot Start

Check the Docker logs for detailed error information:

```bash
docker logs <container-id-or-name>
```

### 2. Frontend Cannot Connect to Backend API

Ensure that the `proxy_pass` in the Nginx configuration points to the correct backend address and port.

### 3. GPU Support Issues

Make sure NVIDIA Docker support is correctly installed and your GPU driver versions are compatible.

### 4. Need to Rebuild After Source Code Changes

If you have modified the Dockerfile or source code, use the `--rebuild yes` parameter to rebuild the image:

```bash
./deploy.sh --rebuild yes
```

## Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy, ONNX Runtime
- **Frontend**: React, TypeScript
- **Containerization**: Docker, Docker Compose
- **Web Server**: Nginx

## License

MIT License
