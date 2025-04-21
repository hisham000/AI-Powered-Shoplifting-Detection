# IEP3 Human Detection System

This project implements an advanced human detection system using YOLOv5. It provides a FastAPI service that processes images and videos to detect and localize humans with bounding boxes.

## Project Overview

The Human Detection System uses YOLOv5 (You Only Look Once) object detection models specifically trained for human detection. The system:

1. Accepts image and video uploads through a REST API
2. Processes the media to detect humans and their locations
3. Returns detailed detection results with bounding boxes, confidence scores, and coordinates
4. For videos, provides frame-by-frame analysis with tracking information
5. Offers a training pipeline to fine-tune the detection model on custom datasets

## Project Structure

```
IEP3_human_detection/
│
├── Dockerfile           # Docker configuration for the application
├── main.py              # FastAPI application with human detection API
├── train.py             # Training script for fine-tuning YOLOv5 models
├── requirements.txt     # Python dependencies
├── README.md            # This documentation file
│
├── mlruns/              # MLflow tracking for model training experiments
│
└── runs/                # Training run outputs and artifacts
    └── train/           # Contains training results and model weights
```

## Model Architecture

The project uses YOLOv5, a state-of-the-art real-time object detection model:

- Base Architecture: YOLOv5n/s/m/l (depending on configuration)
- Custom Training: Fine-tuned specifically for human detection
- Input: RGB images (resized to model requirements)
- Output: Bounding box coordinates, confidence scores, and class predictions
- Performance: Optimized for real-time inference

## Requirements

- Docker

## Running the Docker Container

```bash
# Build the Docker image
docker build -t human-detection .

# Run the container with model volumes mounted and port exposed
docker run -p 5002:5002 -v $(pwd)/runs:/app/runs human-detection
```

## API Endpoints

Once the service is running, the following endpoints are available:

### Health Check

```
GET /health
```

Returns the service status.

### Prediction

```
POST /predict
Content-Type: multipart/form-data
form-data: file=@your_image.jpg or file=@your_video.mp4
```

Analyzes the uploaded image or video and returns human detection results.

For each detection, the API returns:

```json
{
  "predictions": [
    {
      "frame": 0,
      "xmin": 120,
      "ymin": 45,
      "xmax": 320,
      "ymax": 510,
      "confidence": 0.95,
      "class": 0
    }
    // Additional detections...
  ]
}
```

## Training

The project includes a training pipeline for fine-tuning YOLOv5 models on custom datasets:

```bash
# Run training in Docker
docker run --gpus all -v $(pwd)/dataset:/app/dataset -v $(pwd)/runs:/app/runs human-detection python train.py --data /app/dataset/data.yaml --epochs 100
```

Training metrics and model artifacts are tracked with MLflow and saved to the `runs` directory.

## Example: Testing the API

Once the service is running, you can test it with curl:

### For image detection:

```bash
curl -X POST http://localhost:5002/predict \
  -F "file=@/path/to/your/image.jpg" \
  -H "Content-Type: multipart/form-data"
```

### For video detection:

```bash
curl -X POST http://localhost:5002/predict \
  -F "file=@/path/to/your/video.mp4" \
  -H "Content-Type: multipart/form-data"
```

Or use tools like Postman to make API requests.

## Supported Media Formats

- Images: JPG/JPEG, PNG
- Videos: MP4, AVI, MOV, MKV
