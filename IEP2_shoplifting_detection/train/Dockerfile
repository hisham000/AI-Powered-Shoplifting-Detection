FROM python:3.12-slim

# Install Git and necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# instead of `pip install opencv-python`
RUN pip install opencv-python-headless

# Copy application files
COPY . .

# Create directories for mounted volumes
RUN mkdir -p /data
RUN mkdir -p /app/mlruns

# Set volume mount points
VOLUME ["/data", "/app/mlruns"]

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Set memory optimization environment variables
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_GPU_THREAD_MODE=gpu_private
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV TF_ENABLE_ONEDNN_OPTS=1

# Command to run when container starts with unbuffered output
CMD ["python", "-u", "main.py"]
