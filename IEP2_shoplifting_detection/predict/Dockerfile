FROM python:3.12-slim

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
RUN mkdir -p /data/unsupervised
RUN mkdir -p /app/mlruns

# Set volume mount points
VOLUME ["/data/unsupervised", "/app/mlruns"]

# Expose port for the FastAPI application
EXPOSE 5001

# Command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
