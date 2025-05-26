#!/bin/bash
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

# Build CPU image
if [ "$BUILD_CPU" = true ]; then
  echo "Building CPU image: ${IMAGE_NAME}:${TAG}"
  docker build -t ${IMAGE_NAME}:${TAG} -f Dockerfile .
  
  if [ "$PUSH" = true ]; then
    echo "Pushing CPU image: ${IMAGE_NAME}:${TAG}"
    docker push ${IMAGE_NAME}:${TAG}
  fi
fi

# Build GPU image
if [ "$BUILD_GPU" = true ]; then
  echo "Building GPU image: ${IMAGE_NAME}:${TAG}-gpu"
  docker build -t ${IMAGE_NAME}:${TAG}-gpu -f Dockerfile.gpu .
  
  if [ "$PUSH" = true ]; then
    echo "Pushing GPU image: ${IMAGE_NAME}:${TAG}-gpu"
    docker push ${IMAGE_NAME}:${TAG}-gpu
  fi
fi

echo "Build completed successfully!" 