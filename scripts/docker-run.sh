#!/bin/bash
set -e

# Set default variables
USE_GPU=false
CONTAINER_NAME="cinfer-service"
IMAGE_NAME="cinfer-ai"
TAG="latest"
PORT=8000
DATA_DIR="$(pwd)/data"
CONFIG_DIR="$(pwd)/config"
SERVER_IP="0.0.0.0"  # default bind to all network interfaces

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      USE_GPU=true
      TAG="latest-gpu"
      CONTAINER_NAME="cinfer-service-gpu"
      shift
      ;;
    --name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --image)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --ip)
      SERVER_IP="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --config-dir)
      CONFIG_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create directories if they don't exist
mkdir -p "${DATA_DIR}"
mkdir -p "${CONFIG_DIR}"

# Stop and remove the container if it exists
if docker ps -a | grep -q ${CONTAINER_NAME}; then
  echo "Stopping and removing existing container: ${CONTAINER_NAME}"
  docker stop ${CONTAINER_NAME} || true
  docker rm ${CONTAINER_NAME} || true
fi

# Run the container with GPU support if requested
if [ "$USE_GPU" = true ]; then
  echo "Starting container with GPU support: ${CONTAINER_NAME}"
  docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    -p ${PORT}:8000 \
    -v ${DATA_DIR}:/app/data \
    -v ${CONFIG_DIR}:/app/config \
    -e SERVER_HOST=${SERVER_IP} \
    --restart unless-stopped \
    ${IMAGE_NAME}:${TAG}
else
  echo "Starting container without GPU support: ${CONTAINER_NAME}"
  docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${PORT}:8000 \
    -v ${DATA_DIR}:/app/data \
    -v ${CONFIG_DIR}:/app/config \
    -e SERVER_HOST=${SERVER_IP} \
    --restart unless-stopped \
    ${IMAGE_NAME}:${TAG}
fi

# Get the server's IP address
SERVER_PUBLIC_IP=$(hostname -I | awk '{print $1}')
echo "Container ${CONTAINER_NAME} started successfully!"
echo "API is available at: http://localhost:${PORT} (local access)"
echo "API is available at: http://${SERVER_PUBLIC_IP}:${PORT} (remote access)" 