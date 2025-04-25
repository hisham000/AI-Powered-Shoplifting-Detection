import os

# Set environment variable to silence Git warnings from MLflow
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import tempfile
from contextlib import asynccontextmanager

import cv2
import mlflow
import mlflow.keras
import numpy as np
from config import (
    EXPERIMENT_NAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEQUENCE_LENGTH,
    TRACKING_URI,
)
from fastapi import FastAPI, File, HTTPException, UploadFile
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram, Summary
from prometheus_fastapi_instrumentator import Instrumentator

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "iep2_predictions_total", "Number of shoplifting prediction requests processed"
)
POSITIVE_DETECTION_COUNTER = Counter(
    "iep2_shoplifting_detections_total", "Number of positive shoplifting detections"
)
NEGATIVE_DETECTION_COUNTER = Counter(
    "iep2_normal_behavior_detections_total", "Number of normal behavior detections"
)
INFERENCE_TIME = Histogram(
    "iep2_inference_seconds",
    "Time spent on inference",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)
# Summary for shoplifting confidence distribution
CONFIDENCE_SUMMARY = Summary(
    "iep2_shoplifting_confidence",
    "Distribution of shoplifting prediction confidence scores",
)
MODEL_LOADING_TIME = Summary(
    "iep2_model_loading_seconds", "Time taken to load the model"
)
# Health check metrics
IEP2_HEALTH_REQUESTS = Counter(
    "iep2_health_requests_total", "Number of /health requests received"
)
IEP2_HEALTH_FAILURES = Counter(
    "iep2_health_failures_total", "Number of /health requests with error status"
)


# Function to retrieve the best model from MLflow based on test accuracy
def get_best_model():
    # Start timing model loading
    with MODEL_LOADING_TIME.time():
        # Set tracking URI to work both locally and in Docker
        tracking_uri = f"file://{TRACKING_URI}"
        mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["metrics.test_accuracy DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("No runs found for experiment")

        # Get model from the best run
        best_run_id = runs[0].info.run_id

        direct_path = os.path.join(
            TRACKING_URI, exp.experiment_id, best_run_id, "artifacts/model"
        )

        return mlflow.keras.load_model(direct_path)


# Utility: extract a fixed number of frames from video bytes
def extract_frames_from_video(video_bytes: bytes) -> np.ndarray:
    # save to temp file for cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="Cannot read video frames")
    step = max(total_frames // SEQUENCE_LENGTH, 1)
    frames = []
    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frames.append(frame / 255.0)
    cap.release()
    os.remove(tmp_path)
    if len(frames) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {SEQUENCE_LENGTH} frames, but extracted {len(frames)}",
        )
    arr = np.stack(frames, axis=0)
    return np.expand_dims(arr, axis=0)


# Lifespan event handler to load model once
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.target_model = get_best_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model on startup: {e}")
    yield


# Create FastAPI instance
app = FastAPI(lifespan=lifespan)

# Set up Prometheus instrumentation
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health_check() -> dict[str, str]:
    IEP2_HEALTH_REQUESTS.inc()
    try:
        status = {"status": "healthy"}
        return status
    except Exception:
        IEP2_HEALTH_FAILURES.inc()
        raise


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, int]:
    # Increment prediction counter
    PREDICTION_COUNTER.inc()

    model = app.state.target_model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    content_type = file.content_type or ""

    # Only allow video uploads
    if not content_type.startswith("video/"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type. Only video uploads are allowed.",
        )

    data = await file.read()

    # Process video upload
    try:
        arr = extract_frames_from_video(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing video: {str(e)}")

    # Run prediction with timing
    try:
        with INFERENCE_TIME.time():
            preds = model.predict(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    label = int(np.argmax(preds, axis=1)[0])
    confidence = float(preds[0][label])  # Extract confidence score for the prediction

    # Update metrics based on prediction
    if label == 1:  # Shoplifting detected
        POSITIVE_DETECTION_COUNTER.inc()
    else:  # Normal behavior detected
        NEGATIVE_DETECTION_COUNTER.inc()

    # Update confidence summary for average
    CONFIDENCE_SUMMARY.observe(confidence)

    return {"prediction": label}
