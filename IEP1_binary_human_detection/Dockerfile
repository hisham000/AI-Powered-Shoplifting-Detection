# Use Intel OpenVINO runtime image
FROM openvino/ubuntu20_runtime:latest

# Switch to root user to install packages
USER root

# Set work directory
WORKDIR /workspace

# Install system dependencies
RUN apt update && apt install -y python3-pip libgl1

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenCV headless
RUN pip install --no-cache-dir opencv-python-headless

# Create model directory structure
RUN mkdir -p /workspace/models

# Copy application code
COPY ./main.py ./main.py

# Define volume for model persistence
VOLUME ["/workspace/models"]

# Expose FastAPI port
EXPOSE 5001

# Switch back to default user (optional, for security)
USER openvino

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
