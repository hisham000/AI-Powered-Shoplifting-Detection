FROM python:3.12-slim

WORKDIR /app

# Copy requirements file
COPY ./requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy application files
COPY . .

# Set volume mount points
VOLUME ["/CCTV"]

# Expose port for the FastAPI application
EXPOSE 9000

# Command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
