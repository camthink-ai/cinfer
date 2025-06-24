#!/bin/bash
# scripts/deploy_frontend.sh
# Frontend deployment script - build and integrate frontend into backend service

set -e

# Project root directory
PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
FRONTEND_DIR="$PROJECT_ROOT/web"
BACKEND_STATIC_DIR="$PROJECT_ROOT/static"

echo "=== Frontend deployment script ==="
echo "Project root directory: $PROJECT_ROOT"
echo "Frontend directory: $FRONTEND_DIR"
echo "Backend static file directory: $BACKEND_STATIC_DIR"

# Check if the frontend directory exists
if [ ! -d "$FRONTEND_DIR" ]; then
  echo "Error: The frontend directory does not exist: $FRONTEND_DIR"
  exit 1
fi

# Enter the frontend project directory
echo "=== Enter the frontend project directory ==="
cd "$FRONTEND_DIR"

# Install dependencies
echo "=== Install dependencies ==="
pnpm install || { echo "Error: Failed to install dependencies"; exit 1; }

# Build the frontend project
echo "=== Build the frontend project ==="
pnpm run build || { echo "Error: Failed to build the frontend project"; exit 1; }

# Check if the dist directory exists
DIST_DIR="$FRONTEND_DIR/apps/web/dist"
if [ ! -d "$DIST_DIR" ]; then
  echo "Error: The dist directory does not exist: $DIST_DIR"
  exit 1
fi

# Create the backend static file directory
echo "=== Create the backend static file directory ==="
mkdir -p "$BACKEND_STATIC_DIR"

# Clear the previous static files
echo "=== Clear the previous static files ==="
rm -rf "$BACKEND_STATIC_DIR"/*

# Copy the frontend build artifacts to the backend static file directory
echo "=== Copy the frontend build artifacts to the backend static file directory ==="
cp -r "$DIST_DIR"/* "$BACKEND_STATIC_DIR/" || { echo "Error: Failed to copy the frontend build artifacts"; exit 1; }

echo "=== Configure FastAPI static file support ==="
# Check if main.py already contains static file configuration
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

echo "=== Frontend deployment completed ==="
echo "Frontend has been successfully deployed to the backend static file directory"
echo "Example: http://localhost:8000/" 