FROM python:3.12-slim

# Install Git and necessary dependencies
RUN apt-get update && apt-get install -y \
  git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file
COPY ./requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# instead of `pip install opencv-python`
RUN pip install opencv-python-headless

# Copy application files
COPY . .

# Create directories for mounted volumes
RUN mkdir -p /data
RUN mkdir -p /app/mlruns

# Set volume mount points
VOLUME ["/data", "/app/mlruns"]

# Change directory to ./train
WORKDIR /app

# Command to run when container starts with unbuffered output
CMD ["python", "-u", "main.py"]
