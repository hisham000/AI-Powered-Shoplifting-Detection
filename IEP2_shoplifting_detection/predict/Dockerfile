FROM python:3.12-slim

WORKDIR /app

# Copy requirements file
COPY ./requirements.txt ./requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y curl unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# instead of `pip install opencv-python`
RUN pip install opencv-python-headless

# Copy application files
COPY . .

# Create directories for mounted volumes
RUN mkdir -p /app/mlruns

# Set volume mount points
VOLUME ["/app/mlruns"]

# Expose port for the FastAPI application
EXPOSE 5002

# Command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5002"]
