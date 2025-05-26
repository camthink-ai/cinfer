# Cinfer - Vision AI Inference Service

Cinfer is a flexible, scalable AI inference service platform designed for efficient deployment and management of vision AI models. It provides a unified interface for model deployment, management, and inference across multiple deep learning frameworks.

## Features

- **Multi-Engine Support**: TensorRT, ONNX Runtime, and PyTorch inference engines
- **Dynamic Model Loading**: Hot-swapping models without service interruption
- **Flexible API System**: RESTful APIs for both internal management and external inference services
- **Access Control**: Token-based authentication with permission management
- **Resource Management**: Request queuing, rate limiting, and IP whitelisting
- **Monitoring**: System metrics for CPU, GPU, and memory usage

## System Architecture

```
┌─────────────┐    ┌───────────────┐    ┌────────────────┐
│   API Layer │────│  Core System  │────│ Model Engines  │
└─────────────┘    └───────────────┘    └────────────────┘
       │                   │                    │
       │                   │                    │
┌─────────────┐    ┌───────────────┐    ┌────────────────┐
│  Auth/Perm  │    │ Request Queue │    │  Model Store   │
└─────────────┘    └───────────────┘    └────────────────┘
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA (for GPU support)
- Docker (optional)

### Using Docker
```bash
docker-compose up -d
```

### Manual Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/cinfer.git
cd cinfer

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

## Development Usage

### Development Mode
```bash
# Run with auto-reload enabled
python run.py
```

### Production Mode
```bash
# Run with multiple workers, no auto-reload
python run.py --prod --workers 4
```

### Command Line Options
- `--prod`: Run in production mode
- `--host`: Override host from config (default: 127.0.0.1)
- `--port`: Override port from config (default: 8000)
- `--workers`: Set number of worker processes for production mode

## API Endpoints

### Internal APIs (Admin)
- `GET /api/v1/internal/auth`: Authentication management
- `GET /api/v1/internal/system`: System monitoring and management
- `GET /api/v1/internal/models`: Model management (upload, deploy)
- `GET /api/v1/internal/tokens`: API token management

### External APIs (Inference)
- `GET /api/v1/models`: List available models
- `POST /api/v1/inference/{model_id}`: Run inference on a model


## Authentication

- **Internal Management API**:
  - Obtain X-Auth-Token through the login endpoint.
  - Include X-Auth-Token in request headers for all subsequent requests.
- **OpenAPI**:
  - Obtain X-Access-Token from an administrator.
  - Include X-Access-Token in request headers for all API requests.

### API Usage Example
```python
import requests

# Get available models
response = requests.get("http://localhost:8000/api/v1/models", headers=headers)
models = response.json()

# Run inference
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post(
    "http://localhost:8000/api/v1/inference/model_id",
    headers=headers,
    files=files
)
result = response.json()
```

## Project Structure

```
cinfer/
├── api/                   # API interfaces (FastAPI)
│   ├── internal/          # Admin API endpoints
│   └── openapi/           # External inference API endpoints
├── core/                  # Core system components
│   ├── auth/              # Authentication and authorization
│   ├── engine/            # Inference engine implementations
│   ├── model/             # Model management
│   ├── request/           # Request processing and queuing
│   ├── config.py          # Configuration management
│   ├── logging.py         # Logging setup
│   └── database.py        # Database service
├── schemas/               # Data schemas and validation
├── utils/                 # Utility functions and error handling
├── main.py                # FastAPI application entrypoint
├── run.py                 # Application runner script
└── requirements.txt       # Project dependencies
```

## Core Components

- **ConfigManager**: Manages application configuration from files/env vars
- **EngineService**: Handles different inference engines (ONNX, TensorRT, PyTorch)
- **ModelManager**: Manages model lifecycle (upload, validation, deployment)
- **AuthService**: Handles authentication, rate limiting, and IP filtering
- **RequestProcessor**: Processes inference requests through the queue system
- **QueueManager**: Manages request queues for different models

## Configuration

Configuration is managed through environment variables or a configuration file. Key settings include:

- `server.host`: Host to bind the server to (default: 127.0.0.1)
- `server.port`: Port to bind the server to (default: 8000)
- `server.workers`: Number of worker processes (default: 1)
- `database.type`: Database type (default: sqlite)
- `database.path`: Database path (default: data/cinfer.db)

## Development

### Adding a New Engine
1. Create a new engine implementation in `core/engine/`
2. Implement the required interface from `BaseEngine`
3. Register the engine in the `EngineRegistry`

### Testing
```bash
# Run tests
python -m pytest tests/
```

## License

[MIT License](LICENSE)
