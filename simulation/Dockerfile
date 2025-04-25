FROM python:3.12-slim

WORKDIR /app

# Copy requirements file
COPY ./requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy application files
COPY . .

# Set volume mount points
VOLUME ["/CCTV" "/data"]

# Command to run when container starts
CMD ["python", "main.py"]
