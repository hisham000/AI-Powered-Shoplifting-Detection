# IEP2 Shoplifting Detection System

This project implements a video-based shoplifting detection system using deep learning techniques. It consists of two main components:

1. A training pipeline that builds and trains a ConvLSTM-based model for video classification
2. A prediction service that exposes the trained model via a RESTful API

## Project Structure

```
IEP2_shoplifting_detection/
│
├── data/                  # Dataset directory
│   ├── 0/                 # Non-shoplifting videos (class 0)
│   ├── 1/                 # Shoplifting videos (class 1)
│
├── train/                 # Training component
│   ├── config.py          # Training configuration
│   ├── Dockerfile         # Docker configuration for training
│   ├── load_data.py       # Data loading utilities
│   ├── main.py            # Training script
│   └── requirements.txt   # Python dependencies for training
│
├── predict/               # Prediction service
│   ├── config.py          # Prediction service configuration
│   ├── Dockerfile         # Docker configuration for prediction service
│   ├── main.py            # FastAPI application
│   └── requirements.txt   # Python dependencies for prediction service
│
└── mlruns/                # MLflow tracking directory (created during training)
```

## Model Architecture

The project uses a ConvLSTM2D-based neural network architecture for video classification:

- Input: Sequence of video frames (extracted at regular intervals)
- Multiple ConvLSTM2D layers with MaxPooling3D and Dropout for spatiotemporal feature extraction
- Dense output layer with softmax activation for binary classification:
  - Class 0: Normal activity
  - Class 1: Shoplifting activity

## Requirements

- Docker

## Running the Docker Containers

### 1. Training Container

The training container processes video data and trains the shoplifting detection model. All models are tracked with MLflow.

```bash
# Build the training container
docker build -t shoplifting-detection-training -f Dockerfile .

# Run the container with mounted volumes for data and MLflow tracking
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/mlruns:/app/mlruns \
  shoplifting-detection-training
```

### 2. Prediction Container

The prediction container serves the trained model through a RESTful API endpoint.

```bash
# Build the prediction container
docker build -t shoplifting-detection-predicting -f Dockerfile .

# Run the container with mounted volumes and exposed port
docker run --rm \
  -p 5002:5002 \
  -v $(pwd)/mlruns:/app/mlruns \
  shoplifting-detection-predicting
```

## Running the prediction container with Docker from Docker Hub

Pull the Docker image from Docker Hub:

```bash
docker pull jadshaker/iep1-predict:latest
```

Run the container:

```bash
docker run --rm -p 5001:5001 jadshaker/iep1-predict:latest
```

## API Endpoints

Once the prediction service is running, the following endpoints are available:

### Health Check

```
GET /health
```

Returns the service status.

### Prediction

```
POST /predict
Content-Type: multipart/form-data
form-data: file=@your_video.mp4
```

Analyzes the uploaded video and returns a prediction:

```json
{
  "prediction": 0 // 0 for normal behavior, 1 for shoplifting
}
```

Processed videos are automatically saved to the appropriate subdirectory in `data/unsupervised` based on their classification.

## Example: Testing the API

Once the prediction service is running, you can test it using curl:

```bash
curl -X POST \
  http://localhost:5002/predict \
  -F "file=@/path/to/your/testvideo.mp4" \
  -H "Content-Type: multipart/form-data"
```

Or use a tool like Postman to make the API request.
