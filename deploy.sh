#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 默认参数
DEPLOYMENT_MODE="separate"
USE_GPU="no"
BACKEND_PORT=8000
FRONTEND_PORT=3000
INTEGRATED_PORT=8000
INSTANCE_NAME="default"
ACTION="up"
BACKEND_HOST="backend"  # 默认后端主机名
REBUILD="no"  # 默认不重新构建

echo "DEPLOYMENT_MODE: $DEPLOYMENT_MODE"
echo "USE_GPU: $USE_GPU"
echo "ACTION: $ACTION"
echo "INSTANCE_NAME: $INSTANCE_NAME"
echo "BACKEND_HOST: $BACKEND_HOST"
echo "BACKEND_PORT: $BACKEND_PORT"
echo "FRONTEND_PORT: $FRONTEND_PORT"
# 显示帮助信息
show_help() {
    echo -e "${BLUE}Cinfer 部署脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --mode         部署模式: separate(分离部署) 或 integrated(集成部署) (默认: separate)"
    echo "  -g, --gpu          是否使用GPU: yes 或 no (默认: no)"
    echo "  -b, --backend-port 后端服务端口 (默认: 8000)"
    echo "  -f, --frontend-port 前端服务端口 (默认: 3000)"
    echo "  -i, --integrated-port 集成服务端口 (默认: 8080)"
    echo "  -n, --name         实例名称 (默认: default)"
    echo "  -a, --action       操作: up(启动), down(停止), restart(重启), logs(查看日志) (默认: up)"
    echo "  -h, --host         后端主机名或IP地址 (默认: backend)"
    echo "  -r, --rebuild      是否重新构建镜像: yes 或 no (默认: no)"
    echo "  --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --mode integrated --gpu yes --integrated-port 8080 --name prod"
    echo "  $0 --mode separate --backend-port 8001 --frontend-port 3001 --name dev"
    echo "  $0 --name prod --action down"
    echo "  $0 --host 192.168.100.2 --backend-port 8000"
    echo "  $0 --rebuild yes  # 强制重新构建镜像"
    echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--mode)
            DEPLOYMENT_MODE="$2"
            shift
            shift
            ;;
        -g|--gpu)
            USE_GPU="$2"
            shift
            shift
            ;;
        -b|--backend-port)
            BACKEND_PORT="$2"
            shift
            shift
            ;;
        -f|--frontend-port)
            FRONTEND_PORT="$2"
            shift
            shift
            ;;
        -i|--integrated-port)
            INTEGRATED_PORT="$2"
            shift
            shift
            ;;
        -n|--name)
            INSTANCE_NAME="$2"
            shift
            shift
            ;;
        -a|--action)
            ACTION="$2"
            shift
            shift
            ;;
        -h|--host)
            BACKEND_HOST="$2"
            shift
            shift
            ;;
        -r|--rebuild)
            REBUILD="$2"
            shift
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 验证参数
if [[ "$DEPLOYMENT_MODE" != "separate" && "$DEPLOYMENT_MODE" != "integrated" ]]; then
    echo -e "${RED}错误: 部署模式必须是 separate 或 integrated${NC}"
    exit 1
fi

if [[ "$USE_GPU" != "yes" && "$USE_GPU" != "no" ]]; then
    echo -e "${RED}错误: GPU选项必须是 yes 或 no${NC}"
    exit 1
fi

if [[ "$ACTION" != "up" && "$ACTION" != "down" && "$ACTION" != "restart" && "$ACTION" != "logs" && "$ACTION" != "build" ]]; then
    echo -e "${RED}错误: 操作必须是 up, down, restart, logs 或 build${NC}"
    exit 1
fi

if [[ "$REBUILD" != "yes" && "$REBUILD" != "no" ]]; then
    echo -e "${RED}错误: 重新构建选项必须是 yes 或 no${NC}"
    exit 1
fi

# 创建临时docker-compose文件
COMPOSE_FILE="docker-compose.${INSTANCE_NAME}.yml"

echo -e "${BLUE}正在创建部署配置...${NC}"

# 为分离部署模式创建自定义nginx配置
if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then
    NGINX_CONF_DIR="./web"
    mkdir -p "$NGINX_CONF_DIR"
    
    # 创建自定义nginx.conf文件
    cat > "${NGINX_CONF_DIR}/nginx.conf" << EOL
server {
    listen 80;
    server_name localhost;
    
    # 前端静态文件
    location / {
        root /usr/share/nginx/html;
        index index.html index.htm;
        try_files \$uri \$uri/ /index.html;
    }
    
    # API代理转发
    location /api/ {
        proxy_pass http://backend_${INSTANCE_NAME}:${BACKEND_PORT}/api/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # 错误页面
    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
    }
}
EOL
    echo -e "${GREEN}已创建自定义Nginx配置: ${NGINX_CONF_DIR}/nginx.conf${NC}"
    echo -e "${BLUE}API代理设置为: http://backend_${INSTANCE_NAME}:${BACKEND_PORT}/api/${NC}"
fi

# 生成docker-compose文件
cat > $COMPOSE_FILE << EOL
services:
EOL

# 根据部署模式添加服务
if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then
    # 添加后端服务
    cat >> $COMPOSE_FILE << EOL
  backend_${INSTANCE_NAME}:
    build:
      context: ./backend
      dockerfile: $(if [[ "$USE_GPU" == "yes" ]]; then echo "Dockerfile.gpu"; else echo "Dockerfile"; fi)
    volumes:
      - ./backend:/app
    env_file:
      - ./backend/docker/prod.env
    ports:
      - "${BACKEND_PORT}:8000"
    restart: unless-stopped
    container_name: backend_${INSTANCE_NAME}
    hostname: backend_${INSTANCE_NAME}

  frontend_${INSTANCE_NAME}:
    build:
      context: ./web
      dockerfile: Dockerfile
    depends_on:
      - backend_${INSTANCE_NAME}
    ports:
      - "${FRONTEND_PORT}:80"
    restart: unless-stopped
    volumes:
      - ${NGINX_CONF_DIR}/nginx.conf:/etc/nginx/conf.d/default.conf
      - ./web:/app
      - /app/node_modules
    environment:
      - API_URL=http://${BACKEND_HOST}:${BACKEND_PORT}
    container_name: frontend_${INSTANCE_NAME}
EOL
else
    # 添加集成服务
    cat >> $COMPOSE_FILE << EOL
  integrated_${INSTANCE_NAME}:
    build:
      context: .
      dockerfile: $(if [[ "$USE_GPU" == "yes" ]]; then echo "Dockerfile.gpu-with-frontend"; else echo "Dockerfile.with-frontend"; fi)
    volumes:
      - ./backend/data:/app/data
      - ./backend/config:/app/config
      - ./backend/core:/app/core
      - ./backend/schemas:/app/schemas
      - ./backend/scripts:/app/scripts
      - ./backend/utils:/app/utils
      - ./backend/api:/app/api
      - ./backend/monitoring:/app/monitoring
      - ./web:/app/web
     
    env_file:
      - ./backend/docker/prod.env
    ports:
      - "${INTEGRATED_PORT}:8000"
    restart: unless-stopped
    container_name: integrated_${INSTANCE_NAME}
EOL
fi

# 添加卷和网络配置
cat >> $COMPOSE_FILE << EOL

volumes:
  backend_data_${INSTANCE_NAME}:

networks:
  default:
    name: cinfer-network-${INSTANCE_NAME}
EOL

# 执行docker-compose命令
echo -e "${GREEN}配置文件已创建: ${COMPOSE_FILE}${NC}"

# 构建选项
BUILD_OPTION=""
if [[ "$REBUILD" == "yes" ]]; then
    BUILD_OPTION="--build --no-cache"
    echo -e "${YELLOW}将强制重新构建镜像（不使用缓存）${NC}"
fi

case $ACTION in
    up)
        echo -e "${YELLOW}正在启动服务...${NC}"
        if [[ "$REBUILD" == "yes" ]]; then
            echo -e "${YELLOW}先构建镜像...${NC}"
            docker compose -f $COMPOSE_FILE build --no-cache
        fi
        docker compose -f $COMPOSE_FILE up -d $BUILD_OPTION
        ;;
    down)
        echo -e "${YELLOW}正在停止服务...${NC}"
        docker compose -f $COMPOSE_FILE down
        ;;
    restart)
        echo -e "${YELLOW}正在重启服务...${NC}"
        if [[ "$REBUILD" == "yes" ]]; then
            echo -e "${YELLOW}先停止服务...${NC}"
            docker compose -f $COMPOSE_FILE down
            echo -e "${YELLOW}重新构建镜像...${NC}"
            docker compose -f $COMPOSE_FILE build --no-cache
            echo -e "${YELLOW}启动服务...${NC}"
            docker compose -f $COMPOSE_FILE up -d
        else
            docker compose -f $COMPOSE_FILE restart
        fi
        ;;
    logs)
        echo -e "${YELLOW}正在查看日志...${NC}"
        docker compose -f $COMPOSE_FILE logs -f
        ;;
    build)
        echo -e "${YELLOW}正在构建镜像...${NC}"
        docker compose -f $COMPOSE_FILE build --no-cache
        ;;
esac

# 获取本机IP地址
get_local_ip() {
    # 尝试获取主要网络接口的IP地址
    local ip=$(hostname -I | awk '{print $1}')
    if [[ -z "$ip" ]]; then
        # 备用方法
        ip=$(ip -4 addr show scope global | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)
    fi
    echo "$ip"
}

LOCAL_IP=$(get_local_ip)

# 显示访问信息
if [[ "$ACTION" == "up" || "$ACTION" == "restart" || "$ACTION" == "build" ]]; then
    echo -e "\n${GREEN}=== 部署完成! ===${NC}"
    echo -e "\n${BLUE}实例信息:${NC}"
    echo -e "  名称: ${YELLOW}${INSTANCE_NAME}${NC}"
    echo -e "  模式: ${YELLOW}$(if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then echo "分离部署"; else echo "集成部署"; fi)${NC}"
    echo -e "  GPU支持: ${YELLOW}$(if [[ "$USE_GPU" == "yes" ]]; then echo "是"; else echo "否"; fi)${NC}"
    
    echo -e "\n${BLUE}容器信息:${NC}"
    if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then
        echo -e "  后端容器: ${YELLOW}backend_${INSTANCE_NAME}${NC}"
        echo -e "  前端容器: ${YELLOW}frontend_${INSTANCE_NAME}${NC}"
    else
        echo -e "  集成容器: ${YELLOW}integrated_${INSTANCE_NAME}${NC}"
    fi
    
    echo -e "\n${BLUE}访问地址:${NC}"
    if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then
        echo -e "  前端 (本地): ${GREEN}http://localhost:${FRONTEND_PORT}${NC}"
        echo -e "  前端 (局域网): ${GREEN}http://${LOCAL_IP}:${FRONTEND_PORT}${NC}"
        echo -e "  后端API (本地): ${GREEN}http://localhost:${BACKEND_PORT}/api${NC}"
        echo -e "  后端API (局域网): ${GREEN}http://${LOCAL_IP}:${BACKEND_PORT}/api${NC}"
        echo -e "  后端API (容器内): ${GREEN}http://backend_${INSTANCE_NAME}:8000/api${NC}"
        echo -e "  Swagger文档: ${GREEN}http://localhost:${BACKEND_PORT}/docs${NC}"
    else
        echo -e "  应用 (本地): ${GREEN}http://localhost:${INTEGRATED_PORT}${NC}"
        echo -e "  应用 (局域网): ${GREEN}http://${LOCAL_IP}:${INTEGRATED_PORT}${NC}"
        echo -e "  API (本地): ${GREEN}http://localhost:${INTEGRATED_PORT}/api${NC}"
        echo -e "  API (局域网): ${GREEN}http://${LOCAL_IP}:${INTEGRATED_PORT}/api${NC}"
        echo -e "  Swagger文档: ${GREEN}http://localhost:${INTEGRATED_PORT}/docs${NC}"
    fi
    
    echo -e "\n${BLUE}管理命令:${NC}"
    echo -e "  查看日志: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --action logs${NC}"
    echo -e "  重启服务: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --action restart${NC}"
    echo -e "  停止服务: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --action down${NC}"
    echo -e "  重新构建: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --rebuild yes${NC}"
    
    echo -e "\n${BLUE}配置文件:${NC}"
    echo -e "  Docker Compose: ${YELLOW}${COMPOSE_FILE}${NC}"
    if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then
        echo -e "  Nginx配置: ${YELLOW}${NGINX_CONF_DIR}/nginx.conf${NC}"
    fi
fi

echo -e "${BLUE}实例名称: ${INSTANCE_NAME}${NC}"
echo -e "${BLUE}使用以下命令管理此实例:${NC}"
echo "  ./$(basename $0) --name ${INSTANCE_NAME} --action [up|down|restart|logs]" 