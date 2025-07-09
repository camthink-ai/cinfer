#!/bin/sh
# set -e # If any command fails, exit immediately. Uncomment this line.

/bin/bash /app/scripts/check_gpu.sh &
GPU_PID=$!

# 启动主程序
echo "Entrypoint: Starting main application..."
"$@" &
MAIN_PID=$!

# 监控 GPU 脚本是否异常退出
wait $GPU_PID
echo "[ERROR] GPU check failed, stopping main application..."
kill $MAIN_PID
exit 1