#!/bin/sh
# set -e # If any command fails, exit immediately. Uncomment this line.

echo "Entrypoint: Executing database initialization script (scripts/init_db.py)..."


python /app/scripts/init_db.py




echo "Entrypoint: Database initialization script executed."

# Execute the main command defined in the Dockerfile CMD directive, or the command passed to docker run
echo "Entrypoint: Starting main application..."
exec "$@"