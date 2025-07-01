FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install onnx==1.18.0 && \
    pip install onnxruntime  && \
    pip install opencv-contrib-python && \
    python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data/models data/logs data/db data/temp_uploads 

RUN rm -rf /app/backend_apps

# copy the entrypoint script
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
# give execution permission
RUN chmod +x /usr/local/bin/entrypoint.sh

# set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Expose the API port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "run.py", "--prod", "--host", "0.0.0.0"]
