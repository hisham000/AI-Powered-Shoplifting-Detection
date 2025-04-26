FROM python:3.12-slim

WORKDIR /app

# Copy requirements file
COPY ./requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# instead of `pip install opencv-python`
RUN pip install opencv-python-headless

# Install gdown for downloading the mlruns folder
RUN pip install --no-cache-dir gdown

# Copy application files
COPY . .

# Create directories for mounted volumes
RUN mkdir -p /app/mlruns

# Set volume mount points
VOLUME ["/app/mlruns"]

# Expose port for the FastAPI application
EXPOSE 5002

# Command to run when container starts
CMD ["sh", "-c", "if [ -z \"$(ls -A /app/mlruns)\" ]; then echo 'Downloading mlruns from Google Drive'; gdown 'https://drive.google.com/drive/u/6/folders/1qMPS1xd8Apkf0WJDlUjA5h7kaQDRDG5N' --folder --remaining-ok -O /app/mlruns; fi && uvicorn main:app --host 0.0.0.0 --port 5002"]
