#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

# --- Default parameters ---
# Auto-detect architecture (x86_64 or jetson)
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ]; then
    ARCHITECTURE="jetson"
else
    ARCHITECTURE="x86_64"
fi

USE_GPU="no"
BACKEND_PORT=8000
FRONTEND_PORT=3000
INSTANCE_NAME="cinfer"
ACTION="up"
BACKEND_HOST="backend"
REBUILD="no"


COMPOSE_FILE="" 

# Function: Show help information
show_help() {
    echo -e "${BLUE}Cinfer Deployment Script${NC}"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo -e "  --arch             Specify GPU architecture: ${YELLOW}x86_64${NC} or ${YELLOW}jetson${NC} (default: auto-detect as ${GREEN}${ARCHITECTURE}${NC})"
    echo "  -g, --gpu          Use GPU: yes or no (default: no)"
    echo "  -b, --backend-port Backend service port (default: 8000)"
    echo "  -f, --frontend-port Frontend service port (default: 3000)"
    echo "  -n, --name         Instance name (default: cinfer)"
    echo "  -a, --action       Action: up(start), down(stop), restart(restart), logs(view logs), status(view status) (default: up)"
    echo "  -h, --host         Backend hostname or IP address (default: backend)"
    echo "  -r, --rebuild      Rebuild images: yes or no (default: no)"
    echo "  --help             Show this help information"
    echo ""
    echo "Quick commands:"
    echo "  $0 start           Start default instance"
    echo "  $0 stop            Stop default instance"
    echo "  $0 restart         Restart default instance"
    echo "  $0 logs            View default instance logs"
    echo "  $0 status          View all instances status"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu yes  --name prod"
    echo "  $0 --backend-port 8001 --frontend-port 3001 --name dev"
    echo "  $0 --name prod --action down"
    echo "  $0 --host 192.168.100.2 --backend-port 8000"
    echo "  $0 --rebuild yes  # Force rebuild images"
    echo ""
}

# Function: Check required dependencies
check_dependencies() {
    local missing_deps=()
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    # If there are missing dependencies, show error and exit
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${RED}Error: Missing required dependencies:${NC}"
        for dep in "${missing_deps[@]}"; do
            echo -e "  - ${dep}"
        done
        echo -e "${YELLOW}Please install missing dependencies and try again${NC}"
        exit 1
    fi
}

# Function: Parse short commands
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
            return 1 # Not a short command
            ;;
    esac
    return 0 # Is a short command
}

# Function: Validate input parameters
validate_parameters() {

    if [[ "$USE_GPU" != "yes" && "$USE_GPU" != "no" ]]; then
        echo -e "${RED}Error: GPU option must be yes or no${NC}"
        exit 1
    fi

    if [[ "$ACTION" != "up" && "$ACTION" != "down" && "$ACTION" != "restart" && "$ACTION" != "logs" && "$ACTION" != "build" && "$ACTION" != "status" ]]; then
        echo -e "${RED}Error: Action must be up, down, restart, logs, build or status${NC}"
        exit 1
    fi

    if [[ "$REBUILD" != "yes" && "$REBUILD" != "no" ]]; then
        echo -e "${RED}Error: Rebuild option must be yes or no${NC}"
        exit 1
    fi

    # Validate ports are numbers
    if ! [[ "$BACKEND_PORT" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Error: Backend port must be a number${NC}"
        exit 1
    fi
    
    if ! [[ "$FRONTEND_PORT" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Error: Frontend port must be a number${NC}"
        exit 1
    fi
    
    
    # Check if ports are already in use
    local ports_to_check=()
    ports_to_check+=($BACKEND_PORT $FRONTEND_PORT)

    
    if [[ "$ACTION" == "up" ]]; then
        for port in "${ports_to_check[@]}"; do
            if netstat -tuln | grep -q ":$port "; then
                echo -e "${YELLOW}Warning: Port $port is already in use. Continuing deployment may cause conflicts.${NC}"
                read -p "Continue? [y/N] " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo -e "${RED}Deployment cancelled${NC}"
                    exit 1
                fi
            fi
        done
    fi
}


# Function: Generate docker-compose file
create_compose_file() {
    COMPOSE_FILE="docker-compose.${INSTANCE_NAME}.yml"

    echo -e "${BLUE}Creating deployment configuration...${NC}" >&2


     # --- Determine which Dockerfile to use for backend ---
    local backend_dockerfile="Dockerfile" # Default to CPU version
    if [[ "$USE_GPU" == "yes" ]]; then
        if [[ "$ARCHITECTURE" == "x86_64" ]]; then
            backend_dockerfile="Dockerfile.gpu"
        elif [[ "$ARCHITECTURE" == "jetson" ]]; then
            backend_dockerfile="Dockerfile.jetpack6"
        fi
        echo -e "${BLUE}✓ GPU mode detected (${ARCHITECTURE}), will use ${YELLOW}${backend_dockerfile}${NC}" >&2
    fi

    # Generate docker-compose file header
    cat > $COMPOSE_FILE << EOL
services:
  backend_${INSTANCE_NAME}:
    build:
      context: ./backend
      dockerfile: ${backend_dockerfile}
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
            cat >> $COMPOSE_FILE << EOL
    deploy:
      resources:
        reservations:
          devices:
             - capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
EOL
        fi
        # Add frontend service
        cat >> $COMPOSE_FILE << EOL
  frontend_${INSTANCE_NAME}:
    build:
      context: ./web
    ports:
      - "${FRONTEND_PORT}:80"
    restart: unless-stopped
    volumes:
      - ./web:/app
      - /app/node_modules
    environment:
      - API_URL=http://${BACKEND_HOST}:8000
      - BACKEND_HOST=${BACKEND_HOST}
    container_name: frontend_${INSTANCE_NAME}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
EOL
    # Add volumes and network configuration
    cat >> $COMPOSE_FILE << EOL

volumes:
  backend_data_${INSTANCE_NAME}:

networks:
  default:
    name: cinfer-network-${INSTANCE_NAME}
EOL

    echo -e "${GREEN}✓ Configuration file created: ${COMPOSE_FILE}${NC}" >&2

    echo $COMPOSE_FILE
}

# Function: Get local IP address
get_local_ip() {
    # Try to get the primary network interface IP address
    local ip=$(hostname -I | awk '{print $1}')
    if [[ -z "$ip" ]]; then
        # Fallback method
        ip=$(ip -4 addr show scope global | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)
    fi
    echo "$ip"
}

# Function: Show container status
show_container_status() {
    local container_name=$1
    local status=$(docker inspect -f '{{.State.Status}}' $container_name 2>/dev/null)
    local health_status=""
    
    if [[ "$status" == "running" ]]; then
        health_status=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}N/A{{end}}' $container_name 2>/dev/null)
        if [[ "$health_status" == "healthy" ]]; then
            echo -e "${GREEN}Running (Healthy)${NC}"
        elif [[ "$health_status" == "unhealthy" ]]; then
            echo -e "${RED}Running (Unhealthy)${NC}"
        else
            echo -e "${YELLOW}Running${NC}"
        fi
    elif [[ "$status" == "exited" ]]; then
        echo -e "${RED}Stopped${NC}"
    elif [[ -z "$status" ]]; then
        echo -e "${BLUE}Not Created${NC}"
    else
        echo -e "${YELLOW}$status${NC}"
    fi
}

# Function: Check and display all instance status
show_status() {
    echo -e "${BLUE}Checking instance status...${NC}"
    
    # Get all docker-compose files
    local compose_files=$(ls docker-compose.*.yml 2>/dev/null)
    
    if [[ -z "$compose_files" ]]; then
        echo -e "${YELLOW}No deployment instances found${NC}"
        return
    fi
    
    echo -e "\n${BLUE}Instance Status:${NC}"
    printf "%-15s %-15s %-20s %-20s\n" "Instance Name" "Deploy Mode" "Container" "Status"
    echo "---------------------------------------------------------------------"
    
    for file in $compose_files; do
        local instance_name=$(echo $file | sed 's/docker-compose\.\(.*\)\.yml/\1/')
        
        printf "%-15s %-15s %-20s %-20s\n" \
            "${instance_name}" \
            "Separate Deploy" \
            "backend_${instance_name}" \
            "$(show_container_status backend_${instance_name})"
        printf "%-15s %-15s %-20s %-20s\n" \
            "" \
            "" \
            "frontend_${instance_name}" \
            "$(show_container_status frontend_${instance_name})"
        
    done
    
    echo ""
}

# Function: Show deployment information
show_deployment_info() {
    local LOCAL_IP=$(get_local_ip)
    
    echo -e "\n${GREEN}=== Deployment Complete! ===${NC}"
    echo -e "\n${BLUE}Instance Information:${NC}"
    echo -e "  Name: ${YELLOW}${INSTANCE_NAME}${NC}"
    echo -e "  GPU Support: ${YELLOW}$(if [[ "$USE_GPU" == "yes" ]]; then echo "Yes"; else echo "No"; fi)${NC}"
    if [[ "$USE_GPU" == "yes" ]]; then
        echo -e "  GPU Architecture: ${YELLOW}${ARCHITECTURE}${NC}"
    fi
    
    echo -e "\n${BLUE}Container Information:${NC}"
    echo -e "  Backend Container: ${YELLOW}backend_${INSTANCE_NAME}${NC} - $(show_container_status backend_${INSTANCE_NAME})"
    echo -e "  Frontend Container: ${YELLOW}frontend_${INSTANCE_NAME}${NC} - $(show_container_status frontend_${INSTANCE_NAME})"
  
    echo -e "\n${BLUE}Access URLs:${NC}"
    echo -e "  Frontend (Local): ${GREEN}http://localhost:${FRONTEND_PORT}${NC}"
    echo -e "  Frontend (LAN): ${GREEN}http://${LOCAL_IP}:${FRONTEND_PORT}${NC}"
    echo -e "  Backend API (Local): ${GREEN}http://localhost:${BACKEND_PORT}/api${NC}"
    echo -e "  Backend API (LAN): ${GREEN}http://${LOCAL_IP}:${BACKEND_PORT}/api${NC}"
    echo -e "  Backend API (Container): ${GREEN}http://backend_${INSTANCE_NAME}:8000/api${NC}"
    echo -e "  Swagger Documentation: ${GREEN}http://localhost:${BACKEND_PORT}/docs${NC}"

    
    echo -e "\n${BLUE}Management Commands:${NC}"
    echo -e "  View Logs: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --action logs${NC}"
    echo -e "  Restart Service: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --action restart${NC}"
    echo -e "  Stop Service: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --action down${NC}"
    echo -e "  View Status: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --action status${NC}"
    echo -e "  Rebuild: ${YELLOW}./$(basename $0) --name ${INSTANCE_NAME} --rebuild yes${NC}"
    
    echo -e "\n${BLUE}Configuration Files:${NC}"
    echo -e "  Docker Compose: ${YELLOW}${COMPOSE_FILE}${NC}"
    
}

# Function: Execute Docker Compose operations
execute_docker_compose() {
    local action=$1
    local compose_file=$2


    
    # Build options
    local BUILD_OPTION=""
    if [[ "$REBUILD" == "yes" ]]; then
        BUILD_OPTION="--build "
    fi
    
    case $action in
        up)
            echo -e "${YELLOW}Starting services...${NC}"
            if [[ "$REBUILD" == "yes" ]]; then
                echo -e "${YELLOW}Building images first...${NC}"
                docker compose -f $compose_file  build --no-cache 
            fi
            echo "docker compose -f $compose_file  up -d $BUILD_OPTION"
            docker compose -f $compose_file  up -d $BUILD_OPTION
            sleep 2 # Wait for services to start
            ;;
        down)
            echo -e "${YELLOW}Stopping services...${NC}"
            docker compose -f $compose_file  down 
            ;;
        restart)
            echo -e "${YELLOW}Restarting services...${NC}"
            if [[ "$REBUILD" == "yes" ]]; then
                echo -e "${YELLOW}Stopping services first...${NC}"
                docker compose -f $compose_file  down 
                echo -e "${YELLOW}Rebuilding images...${NC}"
                docker compose -f $compose_file  build --no-cache 
                echo -e "${YELLOW}Starting services...${NC}"
                docker compose -f $compose_file  up -d $BUILD_OPTION
            else
                docker compose -f $compose_file  restart 
            fi
            sleep 2 # Wait for services to start
            ;;
        logs)
            echo -e "${YELLOW}Viewing logs...${NC}"
            docker compose -f $compose_file  logs -f 
            ;;
        build)
            echo -e "${YELLOW}Building images...${NC}"
            docker compose -f $compose_file  build --no-cache 
            ;;
    esac
}

# Main function
main() {
    # Check dependencies
    check_dependencies
    
    # If no arguments, show help
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    # If it's a short command, parse it
    if [[ $# -eq 1 ]]; then
        if parse_short_command "$1"; then
            shift # Remove processed parameter
        fi
    fi
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            --arch)
                ARCHITECTURE="$2"
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
                echo -e "${RED}Error: Unknown parameter $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate parameters
    validate_parameters
    
    # If it's a status check operation
    if [[ "$ACTION" == "status" ]]; then
        show_status
        exit 0
    fi
    
    BACKEND_HOST="backend_${INSTANCE_NAME}"

    # Create Docker Compose file
    COMPOSE_FILE=$(create_compose_file)
    
    # Execute Docker Compose operations
    execute_docker_compose "$ACTION" "$COMPOSE_FILE" 
    
    # Show deployment information
    if [[ "$ACTION" == "up" || "$ACTION" == "restart" || "$ACTION" == "build" ]]; then
        show_deployment_info
    fi
    
    echo -e "${BLUE}Instance Name: ${INSTANCE_NAME}${NC}"
    echo -e "${BLUE}Use the following commands to manage this instance:${NC}"
    echo "  ./$(basename $0) --name ${INSTANCE_NAME} --action [up|down|restart|logs|status]" 
}

# Execute main function
main "$@" 
