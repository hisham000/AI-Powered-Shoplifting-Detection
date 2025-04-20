# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# instead of `pip install opencv-python`
RUN pip install opencv-python-headless

# Copy the rest of the application code into the container
COPY . .

# Create directory structure for unsupervised data
RUN mkdir -p /shoplifting-detection/data/unsupervised/0
RUN mkdir -p /shoplifting-detection/data/unsupervised/1
RUN mkdir -p /app/mlruns

# Expose port 8000 for the uvicorn server
EXPOSE 8000

# Set volumes for unsupervised data and mlruns
VOLUME ["/shoplifting-detection/data/unsupervised", "/app/mlruns"]

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
