# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Create directory structure for unlabeled data
RUN mkdir -p /data-processing/shoplifting/dataset/unlabeled

# Copy requirements file first for better layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# instead of `pip install opencv-python`
RUN pip install opencv-python-headless

# Copy the rest of the application
COPY . .

# Expose port 8000 for the uvicorn server
EXPOSE 8000

# Set volume for unlabeled data
VOLUME ["/data-processing/shoplifting/dataset/unlabeled"]

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
