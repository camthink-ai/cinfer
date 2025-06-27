# Cinfer - Vision AI Inference Service

Cinfer is a flexible, scalable AI inference service platform for efficient deployment and management of vision AI models.

## Features

- **Multi-Engine Support**: TensorRT, ONNX Runtime, PyTorch
- **Dynamic Model Loading**: Hot-swap models without service interruption
- **Flexible APIs**: RESTful APIs for management and inference
- **Access Control**: Token-based authentication with permissions
- **Resource Management**: Request queuing, rate limiting, IP whitelisting
- **Monitoring**: System metrics for CPU, GPU, memory usage

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA (for GPU support)
- Docker (optional)

### Installation
```bash
git clone https://github.com/yourusername/cinfer.git
cd cinfer
pip install -r requirements.txt
python run.py
```

### Development Mode
```bash
python run.py
```

### Production Mode
```bash
python run.py --prod --workers 4
```

## API Endpoints

### Management APIs
- `/api/v1/internal/auth`: Authentication management
- `/api/v1/internal/system`: System monitoring and management
- `/api/v1/internal/models`: Model management
- `/api/v1/internal/tokens`: API token management

### Inference APIs
- `GET /api/v1/models`: List available models
- `POST /api/v1/inference/{model_id}`: Run inference

## Authentication

- **Management API**: Use X-Auth-Token from login endpoint
- **Inference API**: Use X-Access-Token provided by administrator

## Docker Deployment

```bash
# Build (CPU/GPU)
./scripts/docker-build.sh [--gpu]

# Run
./scripts/docker-run.sh [--gpu]
```

Access the API at: http://localhost:8000

## License

[MIT License](LICENSE)
