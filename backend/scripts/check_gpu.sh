#!/bin/bash

while true; do
    nvidia-smi > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "[ERROR] nvidia-smi failed, exiting to trigger container restart..."
        exit 1
    fi
    sleep 60
done