# IEP1 Binary Human Detection System

This project implements a machine learning-based human detection system using OpenVINO. It provides a FastAPI service that processes images and videos to detect the presence of humans.

## Project Overview

The Binary Human Detection System uses Intel's OpenVINO and the pre-trained person-detection-0202 model to detect humans in images and videos. The system:

1. Accepts image and video uploads through a REST API
2. Processes the media to detect human presence
3. Returns prediction results with confidence scores
4. For videos, provides frame-by-frame analysis and a summary

## Project Structure

```
IEP1_binary_human_detection/
│
├── Dockerfile           # Docker configuration for the application
├── main.py              # FastAPI application with human detection logic
├── requirements.txt     # Python dependencies
├── README.md            # This documentation file
└── models/              # Directory for model files (downloaded automatically)
    └── intel/
        └── person-detection-0202/
            └── FP16/    # Contains model weights and configuration
```

## Requirements

- Docker

## Running the Docker Container

```bash
# Build the Docker image
docker build -t binary-human-detection .

# Run the container with model volume mounted and port exposed
docker run -p 5003:5003 -v $(pwd)/models:/workspace/models binary-human-detection
```

On first run, if the model is not present in the mounted volume, it will be automatically downloaded.

## API Endpoints

Once the service is running, the following endpoints are available:

### Health Check

```
GET /health
```

Returns the service status and model path information.

### Check Model

```
POST /check-model
```

Manually triggers the model check and download process.

### Prediction

```
POST /predict
Content-Type: multipart/form-data
form-data: file=@your_image.jpg or file=@your_video.mp4
query parameters: max_frames (optional, for limiting video frame processing)
```

Analyzes the uploaded image or video for human presence.

For images, returns:

```json
{
  "human_detected": true,
  "confidence": 0.95
}
```

For videos, returns frame-by-frame results and a summary:

```json
{
  "results": [
    {
      "frame": 0,
      "human_detected": true,
      "confidence": 0.92
    },
    {
      "frame": 1,
      "human_detected": false,
      "confidence": 0.0
    }
    // Additional frames...
  ],
  "summary": {
    "human_detected": true,
    "highest_confidence": 0.92,
    "total_frames": 30
  }
}
```

## Example: Testing the API

Once the service is running, you can test it with curl:

### For image detection:

```bash
curl -X POST http://localhost:5003/predict \
  -F "file=@/path/to/your/image.jpg" \
  -H "Content-Type: multipart/form-data"
```

### For video detection:

```bash
curl -X POST http://localhost:5003/predict?max_frames=100 \
  -F "file=@/path/to/your/video.mp4" \
  -H "Content-Type: multipart/form-data"
```

Or use tools like Postman to make API requests.

## Supported Media Formats

- Images: JPG/JPEG, PNG
- Videos: MP4, AVI, MOV, MKV
