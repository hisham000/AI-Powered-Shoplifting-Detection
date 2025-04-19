import io
import os
import tempfile
from contextlib import asynccontextmanager

import cv2
import mlflow
import mlflow.keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from mlflow.tracking import MlflowClient
from PIL import Image, UnidentifiedImageError

from config import (
    EXPERIMENT_NAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEQUENCE_LENGTH,
    TRACKING_URI,
)


# Function to retrieve the best model from MLflow based on test accuracy
def get_best_model():
    mlflow.set_tracking_uri(TRACKING_URI)
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
    best_run_id = runs[0].info.run_id
    model_uri = f"runs:/{best_run_id}/model"
    return mlflow.keras.load_model(model_uri)


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


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, int]:
    model = app.state.target_model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    content_type = file.content_type or ""
    data = await file.read()

    # Handle image uploads
    if content_type.startswith("image/"):
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        arr = np.array(image) / 255.0
        arr = np.expand_dims(arr, axis=0)

    # Handle video uploads
    elif content_type.startswith("video/"):
        arr = extract_frames_from_video(data)

    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type. Upload an image or short video.",
        )

    # Run prediction
    try:
        preds = model.predict(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    label = int(np.argmax(preds, axis=1)[0])
    return {"prediction": label}
