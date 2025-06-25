#!/bin/bash
# This script builds Docker images with frontend built inside the container

set -e

# Set default variables
IMAGE_NAME="cinfer-ai"
TAG="latest"
BUILD_GPU=false
BUILD_CPU=true
PUSH=false

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      BUILD_GPU=true
      shift
      ;;
    --no-cpu)
      BUILD_CPU=false
      shift
      ;;
    --push)
      PUSH=true
      shift
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Project root directory
PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

echo "=== Starting all-in-one Docker build (frontend + backend) ==="
echo "Project root: $PROJECT_ROOT"

# Build CPU image with frontend
if [ "$BUILD_CPU" = true ]; then
  echo "=== Building CPU image with integrated frontend ==="
  echo "Building image: ${IMAGE_NAME}:${TAG}"
  docker build -t ${IMAGE_NAME}:${TAG} -f Dockerfile.with-frontend .
  
  if [ "$PUSH" = true ]; then
    echo "Pushing image: ${IMAGE_NAME}:${TAG}"
    docker push ${IMAGE_NAME}:${TAG}
  fi
fi

# Build GPU image with frontend
if [ "$BUILD_GPU" = true ]; then
  echo "=== Building GPU image with integrated frontend ==="
  echo "Building image: ${IMAGE_NAME}:${TAG}-gpu"
  docker build -t ${IMAGE_NAME}:${TAG}-gpu -f Dockerfile.gpu-with-frontend .
  
  if [ "$PUSH" = true ]; then
    echo "Pushing image: ${IMAGE_NAME}:${TAG}-gpu"
    docker push ${IMAGE_NAME}:${TAG}-gpu
  fi
fi

echo "=== Build completed! ==="
echo "You can now run the container with the following command:"
echo "  ./scripts/docker-run.sh"
echo "Or use GPU support:"
echo "  ./scripts/docker-run.sh --gpu" 