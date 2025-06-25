#!/bin/bash
# This script is used to build the Docker image for the Cinfer AI application.

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
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

# Project root directory
PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
FRONTEND_DIR="$PROJECT_ROOT/web"
BACKEND_STATIC_DIR="$PROJECT_ROOT/static"

echo "=== Start building the Docker image for the Cinfer AI application  ==="
echo "Project root: $PROJECT_ROOT"

# Check if the frontend directory exists
if [ ! -d "$FRONTEND_DIR" ]; then
  echo "Error: Frontend directory does not exist: $FRONTEND_DIR"
  exit 1
fi

# 1. First build the frontend
echo "=== 1. Build the frontend ==="
cd "$FRONTEND_DIR"

# Install dependencies
echo "Install frontend dependencies..."
pnpm install || { echo "Error: Failed to install dependencies"; exit 1; }

# Build the frontend project
echo "Build the frontend project..."
pnpm run build || { echo "Error: Failed to build the frontend project"; exit 1; }

# Check if the dist directory exists
DIST_DIR="$FRONTEND_DIR/apps/web/dist"
if [ ! -d "$DIST_DIR" ]; then
  echo "Error: The dist directory does not exist: $DIST_DIR"
  exit 1
fi

# 2. Prepare the static file directory
echo "=== 2. Prepare the static file directory ==="
cd "$PROJECT_ROOT"
mkdir -p "$BACKEND_STATIC_DIR"
rm -rf "$BACKEND_STATIC_DIR"/*
cp -r "$DIST_DIR"/* "$BACKEND_STATIC_DIR/" || { echo "Error: Failed to copy the frontend build artifacts"; exit 1; }

# 3. Modify main.py to add static file support
echo "=== 3. Ensure main.py supports static files ==="
if ! grep -q "StaticFiles" "$PROJECT_ROOT/main.py"; then
  # Create a temporary file
  TEMP_FILE=$(mktemp)
  # Add static file configuration
  cat "$PROJECT_ROOT/main.py" | awk '
  /from fastapi import FastAPI/ {
    print $0;
    print "from fastapi.staticfiles import StaticFiles";
    next;
  }
  /app = FastAPI/ {
    print $0;
    next;
  }
  /# --- Root Endpoint ---/ {
    print "# --- Static Files ---";
    print "app.mount(\"/\", StaticFiles(directory=\"static\", html=True), name=\"static\")";
    print "";
    print $0;
    next;
  }
  { print $0; }
  ' > "$TEMP_FILE"
  
  # Backup the original file
  cp "$PROJECT_ROOT/main.py" "$PROJECT_ROOT/main.py.bak"
  # Replace the original file with the new content
  mv "$TEMP_FILE" "$PROJECT_ROOT/main.py"
  echo "Added static file configuration to main.py"
else
  echo "main.py already contains static file configuration, no need to modify"
fi

# 4. Build the Docker image
echo "=== 4. Build the Docker image ==="

# Build the CPU image
if [ "$BUILD_CPU" = true ]; then
  echo "Build the CPU image: ${IMAGE_NAME}:${TAG}"
  docker build -t ${IMAGE_NAME}:${TAG} -f Dockerfile .
  
  if [ "$PUSH" = true ]; then
    echo "Push the CPU image: ${IMAGE_NAME}:${TAG}"
    docker push ${IMAGE_NAME}:${TAG}
  fi
fi

# Build the GPU image
if [ "$BUILD_GPU" = true ]; then
  echo "Build the GPU image: ${IMAGE_NAME}:${TAG}-gpu"
  docker build -t ${IMAGE_NAME}:${TAG}-gpu -f Dockerfile.gpu .
  
  if [ "$PUSH" = true ]; then
    echo "Push the GPU image: ${IMAGE_NAME}:${TAG}-gpu"
    docker push ${IMAGE_NAME}:${TAG}-gpu
  fi
fi

echo "=== Build completed! ==="
echo "You can now run the container with the following command:"
echo "  ./scripts/docker-run.sh"
echo "Or use GPU support:"
echo "  ./scripts/docker-run.sh --gpu" 