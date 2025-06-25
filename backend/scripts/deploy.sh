#!/bin/bash
# scripts/deploy.sh - 一键部署脚本，支持配置端口和容器名

set -e

# 默认值
PORT=8000
CONTAINER_NAME="cinfer-service"
USE_GPU=false
BUILD_IMAGE=true
IMAGE_TAG="latest"

# 帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  --port PORT           指定容器映射端口 (默认: 8000)"
    echo "  --name NAME           指定容器名称 (默认: cinfer-service)"
    echo "  --gpu                 使用GPU版本"
    echo "  --no-build            跳过构建镜像步骤，直接运行容器"
    echo "  --tag TAG             指定镜像标签 (默认: latest)"
    echo "  --help                显示此帮助信息"
    exit 0
}

# 处理命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --no-build)
            BUILD_IMAGE=false
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "未知选项: $1"
            show_help
            ;;
    esac
done

echo "============================================"
echo "🚀 开始部署 CInfer AI 推理服务"
echo "============================================"
echo "配置信息:"
echo "- 端口: $PORT"
echo "- 容器名: $CONTAINER_NAME"
echo "- GPU支持: $USE_GPU"
echo "- 构建镜像: $BUILD_IMAGE"
echo "- 镜像标签: $IMAGE_TAG"
echo "============================================"

# 构建镜像
if [ "$BUILD_IMAGE" = true ]; then
    echo "📦 步骤1: 构建Docker镜像..."
    
    # 构建参数
    BUILD_ARGS=""
    if [ "$USE_GPU" = true ]; then
        BUILD_ARGS="--gpu"
    fi
    
    # 执行构建
    ./scripts/docker-build-all-in-one.sh $BUILD_ARGS --tag $IMAGE_TAG
    
    echo "✅ 镜像构建完成"
else
    echo "📦 跳过镜像构建步骤"
fi

echo "🚢 步骤2: 启动容器..."

# 运行参数
RUN_ARGS="--port $PORT --name $CONTAINER_NAME"
if [ "$USE_GPU" = true ]; then
    RUN_ARGS="$RUN_ARGS --gpu"
fi
if [ "$IMAGE_TAG" != "latest" ]; then
    RUN_ARGS="$RUN_ARGS --tag $IMAGE_TAG"
fi

# 执行运行
./scripts/docker-run.sh $RUN_ARGS

echo "✅ 容器启动完成"
echo "============================================"
echo "🎉 部署完成! 您可以通过以下地址访问服务:"
echo "- 本地访问: http://localhost:$PORT"
SERVER_IP=$(hostname -I | awk '{print $1}')
echo "- 远程访问: http://$SERVER_IP:$PORT"
echo "============================================" 