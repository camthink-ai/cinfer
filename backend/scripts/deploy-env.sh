#!/bin/bash
# scripts/deploy-env.sh - 环境部署脚本，支持不同环境的部署

set -e

# 默认值
ENV="dev"
BUILD=true
DETACH=true

# 帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  --env ENV             指定环境 (dev|test|test-gpu) (默认: dev)"
    echo "  --no-build            跳过构建镜像步骤，直接运行容器"
    echo "  --no-detach           前台运行容器（不分离）"
    echo "  --help                显示此帮助信息"
    exit 0
}

# 处理命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --no-build)
            BUILD=false
            shift
            ;;
        --no-detach)
            DETACH=false
            shift
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

# 验证环境参数
if [[ "$ENV" != "dev" && "$ENV" != "test" && "$ENV" != "test-gpu" ]]; then
    echo "错误: 环境参数必须是 dev、test 或 test-gpu"
    exit 1
fi

# 获取服务端口
if [[ "$ENV" == "dev" ]]; then
    PORT=8000
elif [[ "$ENV" == "test" ]]; then
    PORT=8008
elif [[ "$ENV" == "test-gpu" ]]; then
    PORT=8009
fi

echo "============================================"
echo "🚀 开始部署 CInfer AI 推理服务 - $ENV 环境"
echo "============================================"
echo "配置信息:"
echo "- 环境: $ENV"
echo "- 端口: $PORT"
echo "- 构建镜像: $BUILD"
echo "- 后台运行: $DETACH"
echo "============================================"

# 确保目录存在
mkdir -p ./data/models
mkdir -p ./data/logs
mkdir -p ./data/db

# 构建和启动容器
if [ "$BUILD" = true ]; then
    echo "📦 步骤1: 构建 Docker 镜像..."
    
    # 根据环境选择服务
    SERVICE="cinfer-$ENV"
    
    echo "🔨 构建服务: $SERVICE"
    docker-compose build $SERVICE
    
    echo "✅ 镜像构建完成"
else
    echo "📦 跳过镜像构建步骤"
fi

echo "🚢 步骤2: 启动容器..."

# 启动参数
RUN_ARGS=""
if [ "$DETACH" = true ]; then
    RUN_ARGS="-d"
fi

# 根据环境选择服务
SERVICE="cinfer-$ENV"

echo "🚀 启动服务: $SERVICE"
docker-compose up $RUN_ARGS $SERVICE

echo "✅ 容器启动完成"
echo "============================================"
echo "🎉 部署完成! 您可以通过以下地址访问服务:"
echo "- 本地访问: http://localhost:$PORT"
SERVER_IP=$(hostname -I | awk '{print $1}')
echo "- 远程访问: http://$SERVER_IP:$PORT"
echo "============================================" 