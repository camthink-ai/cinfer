#!/bin/bash
# scripts/stop-env.sh - 停止指定环境的容器

set -e

# 默认值
ENV="dev"
REMOVE=false

# 帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  --env ENV             指定环境 (dev|test|test-gpu|all) (默认: dev)"
    echo "  --remove              停止后移除容器"
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
        --remove)
            REMOVE=true
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
if [[ "$ENV" != "dev" && "$ENV" != "test" && "$ENV" != "test-gpu" && "$ENV" != "all" ]]; then
    echo "错误: 环境参数必须是 dev、test、test-gpu 或 all"
    exit 1
fi

echo "============================================"
echo "🛑 停止 CInfer AI 推理服务 - $ENV 环境"
echo "============================================"
echo "配置信息:"
echo "- 环境: $ENV"
echo "- 移除容器: $REMOVE"
echo "============================================"

# 停止容器
if [[ "$ENV" == "all" ]]; then
    echo "🛑 停止所有环境的容器..."
    
    if [ "$REMOVE" = true ]; then
        docker-compose down
        echo "✅ 所有容器已停止并移除"
    else
        docker-compose stop
        echo "✅ 所有容器已停止"
    fi
else
    # 根据环境选择服务
    SERVICE="cinfer-$ENV"
    
    echo "🛑 停止服务: $SERVICE"
    
    if [ "$REMOVE" = true ]; then
        docker-compose rm -sf $SERVICE
        echo "✅ 容器 $SERVICE 已停止并移除"
    else
        docker-compose stop $SERVICE
        echo "✅ 容器 $SERVICE 已停止"
    fi
fi

echo "============================================" 