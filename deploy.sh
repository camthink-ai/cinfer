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
BACKEND_HOST="backend"
REBUILD="no"

# 功能：显示帮助信息
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
    echo "  -i, --integrated-port 集成服务端口 (默认: 8000)"
    echo "  -n, --name         实例名称 (默认: default)"
    echo "  -a, --action       操作: up(启动), down(停止), restart(重启), logs(查看日志), status(查看状态) (默认: up)"
    echo "  -h, --host         后端主机名或IP地址 (默认: backend)"
    echo "  -r, --rebuild      是否重新构建镜像: yes 或 no (默认: no)"
    echo "  --help             显示此帮助信息"
    echo ""
    echo "快速命令:"
    echo "  $0 start           启动默认实例"
    echo "  $0 stop            停止默认实例"
    echo "  $0 restart         重启默认实例"
    echo "  $0 logs            查看默认实例日志"
    echo "  $0 status          查看所有实例状态"
    echo ""
    echo "示例:"
    echo "  $0 --mode integrated --gpu yes --integrated-port 8080 --name prod"
    echo "  $0 --mode separate --backend-port 8001 --frontend-port 3001 --name dev"
    echo "  $0 --name prod --action down"
    echo "  $0 --host 192.168.100.2 --backend-port 8000"
    echo "  $0 --rebuild yes  # 强制重新构建镜像"
    echo ""
}

# 功能：检查必要依赖
check_dependencies() {
    local missing_deps=()
    
    # 检查Docker是否安装
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    # 检查Docker Compose是否安装
    if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    # 如果有缺失依赖，显示错误并退出
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${RED}错误: 缺少必要依赖:${NC}"
        for dep in "${missing_deps[@]}"; do
            echo -e "  - ${dep}"
        done
        echo -e "${YELLOW}请安装缺失的依赖后重试${NC}"
        exit 1
    fi
}

# 功能：解析简短命令
parse_short_command() {
    case "$1" in
        start)
            ACTION="up"
            ;;
        stop)
            ACTION="down"
            ;;
        restart)
            ACTION="restart"
            ;;
        logs)
            ACTION="logs"
            ;;
        status)
            ACTION="status"
            ;;
        *)
            return 1 # 不是简短命令
            ;;
    esac
    return 0 # 是简短命令
}

# 功能：验证输入参数
validate_parameters() {
    if [[ "$DEPLOYMENT_MODE" != "separate" && "$DEPLOYMENT_MODE" != "integrated" ]]; then
        echo -e "${RED}错误: 部署模式必须是 separate 或 integrated${NC}"
        exit 1
    fi

    if [[ "$USE_GPU" != "yes" && "$USE_GPU" != "no" ]]; then
        echo -e "${RED}错误: GPU选项必须是 yes 或 no${NC}"
        exit 1
    fi

    if [[ "$ACTION" != "up" && "$ACTION" != "down" && "$ACTION" != "restart" && "$ACTION" != "logs" && "$ACTION" != "build" && "$ACTION" != "status" ]]; then
        echo -e "${RED}错误: 操作必须是 up, down, restart, logs, build 或 status${NC}"
        exit 1
    fi

    if [[ "$REBUILD" != "yes" && "$REBUILD" != "no" ]]; then
        echo -e "${RED}错误: 重新构建选项必须是 yes 或 no${NC}"
        exit 1
    fi

    # 验证端口是否为数字
    if ! [[ "$BACKEND_PORT" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}错误: 后端端口必须是数字${NC}"
        exit 1
    fi
    
    if ! [[ "$FRONTEND_PORT" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}错误: 前端端口必须是数字${NC}"
        exit 1
    fi
    
    if ! [[ "$INTEGRATED_PORT" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}错误: 集成端口必须是数字${NC}"
        exit 1
    fi
    
    # 验证端口是否已被占用
    local ports_to_check=()
    if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then
        ports_to_check+=($BACKEND_PORT $FRONTEND_PORT)
    else
        ports_to_check+=($INTEGRATED_PORT)
    fi
    
    if [[ "$ACTION" == "up" ]]; then
        for port in "${ports_to_check[@]}"; do
            if netstat -tuln | grep -q ":$port "; then
                echo -e "${YELLOW}警告: 端口 $port 已被占用。继续部署可能会导致冲突。${NC}"
                read -p "是否继续? [y/N] " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo -e "${RED}部署已取消${NC}"
                    exit 1
                fi
            fi
        done
    fi
}

# 功能：生成nginx配置
create_nginx_config() {
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
        client_max_body_size 1024m;
    }
    
    # 错误页面
    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
    }
}
EOL
    echo -e "${GREEN}✓ 已创建自定义Nginx配置: ${NGINX_CONF_DIR}/nginx.conf${NC}"
    echo -e "${BLUE}✓ API代理设置为: http://backend_${INSTANCE_NAME}:${BACKEND_PORT}/api/${NC}"
}

# 功能：生成docker-compose文件
# 功能：生成docker-compose文件
create_compose_file() {
    local COMPOSE_FILE="docker-compose.${INSTANCE_NAME}.yml"

    echo -e "${BLUE}正在创建部署配置...${NC}" >&2

    # 生成docker-compose文件头部
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
    restart: $(if [[ "$USE_GPU" == "yes" ]]; then echo "on-failure"; else echo "unless-stopped"; fi)
    container_name: backend_${INSTANCE_NAME}
    hostname: backend_${INSTANCE_NAME}
EOL
        if [[ "$USE_GPU" == "yes" ]]; then
            cat >> $COMPOSE_FILE <<'EOL'
    deploy:
      resources:
        reservations:
          devices:
             - capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
EOL
        fi
        # 添加前端服务
        cat >> $COMPOSE_FILE << EOL
  frontend_${INSTANCE_NAME}:
    build:
      context: ./web
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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
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

    echo -e "${GREEN}✓ 配置文件已创建: ${COMPOSE_FILE}${NC}" >&2

    echo $COMPOSE_FILE
}

# 功能：获取本机IP地址
get_local_ip() {
    # 尝试获取主要网络接口的IP地址
    local ip=$(hostname -I | awk '{print $1}')
    if [[ -z "$ip" ]]; then
        # 备用方法
        ip=$(ip -4 addr show scope global | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)
    fi
    echo "$ip"
}

# 功能：显示容器状态
show_container_status() {
    local container_name=$1
    local status=$(docker inspect -f '{{.State.Status}}' $container_name 2>/dev/null)
    local health_status=""
    
    if [[ "$status" == "running" ]]; then
        health_status=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}N/A{{end}}' $container_name 2>/dev/null)
        if [[ "$health_status" == "healthy" ]]; then
            echo -e "${GREEN}运行中 \(健康\)${NC}"
        elif [[ "$health_status" == "unhealthy" ]]; then
            echo -e "${RED}运行中 \(不健康\)${NC}"
        else
            echo -e "${YELLOW}运行中${NC}"
        fi
    elif [[ "$status" == "exited" ]]; then
        echo -e "${RED}已停止${NC}"
    elif [[ -z "$status" ]]; then
        echo -e "${BLUE}未创建${NC}"
    else
        echo -e "${YELLOW}$status${NC}"
    fi
}

# 功能：检查并显示所有实例状态
show_status() {
    echo -e "${BLUE}检查实例状态...${NC}"
    
    # 获取所有docker-compose文件
    local compose_files=$(ls docker-compose.*.yml 2>/dev/null)
    
    if [[ -z "$compose_files" ]]; then
        echo -e "${YELLOW}没有找到任何部署实例${NC}"
        return
    fi
    
    echo -e "\n${BLUE}实例状态:${NC}"
    printf "%-15s %-15s %-20s %-20s\n" "实例名称" "部署模式" "容器" "状态"
    echo "---------------------------------------------------------------------"
    
    for file in $compose_files; do
        local instance_name=$(echo $file | sed 's/docker-compose\.\(.*\)\.yml/\1/')
        
        # 检查是分离部署还是集成部署
        if grep -q "integrated_${instance_name}:" "$file"; then
            local mode="integrated"
            printf "%-15s %-15s %-20s %-20s\n" \
                "${instance_name}" \
                "集成部署" \
                "integrated_${instance_name}" \
                "$(show_container_status integrated_${instance_name})"
        else
            local mode="separate"
            printf "%-15s %-15s %-20s %-20s\n" \
                "${instance_name}" \
                "分离部署" \
                "backend_${instance_name}" \
                "$(show_container_status backend_${instance_name})"
            printf "%-15s %-15s %-20s %-20s\n" \
                "" \
                "" \
                "frontend_${instance_name}" \
                "$(show_container_status frontend_${instance_name})"
        fi
    done
    
    echo ""
}

# 功能：显示部署信息
show_deployment_info() {
    local LOCAL_IP=$(get_local_ip)
    
    echo -e "\n${GREEN}=== 部署完成! ===${NC}"
    echo -e "\n${BLUE}实例信息:${NC}"
    echo -e "  名称: ${YELLOW}${INSTANCE_NAME}${NC}"
    echo -e "  模式: ${YELLOW}$(if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then echo "分离部署"; else echo "集成部署"; fi)${NC}"
    echo -e "  GPU支持: ${YELLOW}$(if [[ "$USE_GPU" == "yes" ]]; then echo "是"; else echo "否"; fi)${NC}"
    
    echo -e "\n${BLUE}容器信息:${NC}"
    if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then
        echo -e "  后端容器: ${YELLOW}backend_${INSTANCE_NAME}${NC} - $(show_container_status backend_${INSTANCE_NAME})"
        echo -e "  前端容器: ${YELLOW}frontend_${INSTANCE_NAME}${NC} - $(show_container_status frontend_${INSTANCE_NAME})"
    else
        echo -e "  集成容器: ${YELLOW}integrated_${INSTANCE_NAME}${NC} - $(show_container_status integrated_${INSTANCE_NAME})"
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
    echo -e "  查看状态: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --action status${NC}"
    echo -e "  重新构建: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --rebuild yes${NC}"
    
    echo -e "\n${BLUE}配置文件:${NC}"
    echo -e "  Docker Compose: ${YELLOW}${COMPOSE_FILE}${NC}"
    if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then
        echo -e "  Nginx配置: ${YELLOW}${NGINX_CONF_DIR}/nginx.conf${NC}"
    fi
}

# 功能：执行Docker Compose操作
execute_docker_compose() {
    local action=$1
    local compose_file=$2
    
    # 构建选项
    local BUILD_OPTION=""
    if [[ "$REBUILD" == "yes" ]]; then
        BUILD_OPTION="--build "
    fi
    
    case $action in
        up)
            echo -e "${YELLOW}正在启动服务...${NC}"
            if [[ "$REBUILD" == "yes" ]]; then
                echo -e "${YELLOW}先构建镜像...${NC}"
                docker compose -f $compose_file build --no-cache
            fi
            echo "docker compose -f $compose_file up -d $BUILD_OPTION"
            docker compose -f $compose_file up -d $BUILD_OPTION
            sleep 2 # 等待服务启动
            ;;
        down)
            echo -e "${YELLOW}正在停止服务...${NC}"
            docker compose -f $compose_file down
            ;;
        restart)
            echo -e "${YELLOW}正在重启服务...${NC}"
            if [[ "$REBUILD" == "yes" ]]; then
                echo -e "${YELLOW}先停止服务...${NC}"
                docker compose -f $compose_file down
                echo -e "${YELLOW}重新构建镜像...${NC}"
                docker compose -f $compose_file build --no-cache
                echo -e "${YELLOW}启动服务...${NC}"
                docker compose -f $compose_file up -d
            else
                docker compose -f $compose_file restart
            fi
            sleep 2 # 等待服务启动
            ;;
        logs)
            echo -e "${YELLOW}正在查看日志...${NC}"
            docker compose -f $compose_file logs -f
            ;;
        build)
            echo -e "${YELLOW}正在构建镜像...${NC}"
            docker compose -f $compose_file build --no-cache
            ;;
    esac
}

# 主函数
main() {
    # 检查依赖
    check_dependencies
    
    # 如果没有参数，显示帮助
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    # 如果是简短命令，解析它
    if [[ $# -eq 1 ]]; then
        if parse_short_command "$1"; then
            shift # 移除已处理的参数
        fi
    fi
    
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
    validate_parameters
    
    # 如果是查看状态操作
    if [[ "$ACTION" == "status" ]]; then
        show_status
        exit 0
    fi
    
    # 为分离部署模式创建自定义nginx配置
    if [[ "$DEPLOYMENT_MODE" == "separate" ]]; then
        create_nginx_config
    fi
    
    # 创建Docker Compose文件
    COMPOSE_FILE=$(create_compose_file)
    
    # 执行Docker Compose操作
    execute_docker_compose "$ACTION" "$COMPOSE_FILE"
    
    # 显示部署信息
    if [[ "$ACTION" == "up" || "$ACTION" == "restart" || "$ACTION" == "build" ]]; then
        show_deployment_info
    fi
    
    echo -e "${BLUE}实例名称: ${INSTANCE_NAME}${NC}"
    echo -e "${BLUE}使用以下命令管理此实例:${NC}"
    echo "  ./$(basename $0) --name ${INSTANCE_NAME} --action [up|down|restart|logs|status]" 
}

# 执行主函数
main "$@" 
