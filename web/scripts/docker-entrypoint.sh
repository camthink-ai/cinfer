#!/bin/sh
set -e

# 创建配置目录
mkdir -p /etc/nginx/conf.d

# 生成nginx配置文件
envsubst '${BACKEND_HOST}' < /etc/nginx/templates/nginx.template.conf > /etc/nginx/conf.d/default.conf

# 输出配置信息
echo "Nginx backend host: ${BACKEND_HOST}"

# 执行CMD
exec "$@"