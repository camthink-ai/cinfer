#!/bin/sh
# set -e # If any command fails, exit immediately. Uncomment this line.

echo "Entrypoint: Starting GPU check script..."
/bin/bash /app/scripts/check_gpu.sh &

# Execute the main command defined in the Dockerfile CMD directive, or the command passed to docker run
echo "Entrypoint: Starting main application..."
exec "$@"