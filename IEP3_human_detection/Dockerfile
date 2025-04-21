FROM python:3.12-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# instead of `pip install opencv-python`
RUN pip install opencv-python-headless

# Copy only main.py file
COPY main.py .

# Create directories for mounted volumes
RUN mkdir -p /app/runs

# Set volume mount points
VOLUME ["/app/runs"]

# Expose port for the FastAPI application
EXPOSE 5002

# Command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5002"]
