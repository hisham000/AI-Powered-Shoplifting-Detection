FROM python:3.12-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# instead of `pip install opencv-python`
RUN pip install opencv-python-headless

# Set environment variable to indicate we're running in Docker
ENV RUNNING_IN_DOCKER=true

# Copy application files
COPY . .
VOLUME ["/app/data/unconfirmed", "/app/data/confirmed"]

# Expose port for the FastAPI application
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
