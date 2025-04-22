import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openvino import Core  # type: ignore[import]
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_fastapi_instrumentator import Instrumentator

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "iep1_predictions_total", "Number of prediction requests processed"
)
HUMAN_DETECTION_COUNTER = Counter(
    "iep1_human_detections_total", "Number of human detections made"
)
INFERENCE_TIME = Histogram(
    "iep1_inference_seconds",
    "Time spent on inference",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
CONFIDENCE_GAUGE = Gauge("iep1_detection_confidence", "Last detection confidence score")
MODEL_LOADING_TIME = Summary(
    "iep1_model_loading_seconds", "Time taken to load the model"
)

# Path to model (inside container)
MODEL_DIR = "models"
MODEL_NAME = "person-detection-0202"
MODEL_PRECISION = "FP16"
MODEL_XML = f"{MODEL_DIR}/intel/{MODEL_NAME}/{MODEL_PRECISION}/{MODEL_NAME}.xml"

# Global variables for the model
compiled_model = None
input_layer = None


def check_and_download_model():
    """
    Check if the model exists and download it if it doesn't.
    Returns True if the model exists or was successfully downloaded, False otherwise.
    """
    model_path = Path(MODEL_XML)

    # If the model already exists, return True
    if model_path.exists():
        print(f"Model found at {model_path}")
        return True

    print(f"Model not found at {model_path}, attempting to download...")

    try:
        # Create the model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Check if openvino-dev is installed
        try:
            pass

            print("openvino-dev is already installed")
        except ImportError:
            print("Installing openvino-dev...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "openvino-dev"]
            )

        # Download the model using Open Model Zoo downloader
        print(f"Downloading {MODEL_NAME} with precision {MODEL_PRECISION}...")
        result = subprocess.run(
            [
                "omz_downloader",
                "--name",
                MODEL_NAME,
                "--precision",
                MODEL_PRECISION,
                "--output_dir",
                MODEL_DIR,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error downloading the model: {result.stderr}")
            return False

        print(f"Model downloaded successfully to {MODEL_DIR}")
        return model_path.exists()

    except Exception as e:
        print(f"Error downloading the model: {str(e)}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check and download model if needed
    global compiled_model, input_layer

    model_available = check_and_download_model()
    if not model_available:
        raise RuntimeError(
            f"Model file not found and could not be downloaded: {MODEL_XML}"
        )

    # Load model with timing
    with MODEL_LOADING_TIME.time():
        print(f"Loading model from {MODEL_XML}...")
        ie = Core()
        model = ie.read_model(model=MODEL_XML)
        compiled_model = ie.compile_model(model=model, device_name="CPU")
        input_layer = compiled_model.input(0)
        print("Model loaded successfully")

    yield
    # Cleanup operations can go here (if needed when app shuts down)


app = FastAPI(title="Human Detection API", lifespan=lifespan)

# Set up Prometheus instrumentation
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_path": MODEL_XML,
        "model_exists": os.path.exists(MODEL_XML),
    }


@app.post("/check-model")
def check_model():
    """Manually trigger model check and download if needed"""
    model_exists = os.path.exists(MODEL_XML)

    if model_exists:
        return {"status": "success", "message": f"Model already exists at {MODEL_XML}"}

    result = check_and_download_model()
    if result:
        return {
            "status": "success",
            "message": f"Model downloaded successfully to {MODEL_XML}",
        }
    else:
        return {"status": "error", "message": "Failed to download the model"}


def process_frame(
    frame, input_h: int, input_w: int, threshold: float = 0.5
) -> Dict[str, Any]:
    """Process a single frame and detect humans"""
    # Preprocess
    input_blob = cv2.resize(frame, (input_w, input_h))
    input_blob = input_blob.transpose(2, 0, 1)[np.newaxis, :]
    input_blob = input_blob.astype(np.float32)

    # Inference with timing
    with INFERENCE_TIME.time():
        results = compiled_model([input_blob])[compiled_model.output(0)]

    # Post-process
    human_detected = False
    highest_conf = 0.0

    for detection in results[0][0]:
        conf = float(detection[2])
        if conf > threshold:
            human_detected = True
            highest_conf = max(highest_conf, conf)

    # Update metrics
    if human_detected:
        HUMAN_DETECTION_COUNTER.inc()
        CONFIDENCE_GAUGE.set(highest_conf)

    return {
        "human_detected": human_detected,
        "confidence": highest_conf if human_detected else 0.0,
    }


@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    """
    Process a video and return whether at least one frame contains a human.
    Returns: {"human_detected": bool}
    """
    PREDICTION_COUNTER.inc()
    # Save upload to a temp mp4 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        any_human = False
        # Process each frame until detection
        h, w = input_layer.shape[2:]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = process_frame(frame, h, w)
            if result["human_detected"]:
                any_human = True
                break
        cap.release()
        return JSONResponse({"human_detected": any_human})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
