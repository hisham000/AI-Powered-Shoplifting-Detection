import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO  # type: ignore[import]

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "iep3_predictions_total", "Number of YOLO prediction requests processed"
)
DETECTION_COUNTER = Counter(
    "iep3_detections_total", "Number of objects detected across all frames"
)
INFERENCE_TIME = Histogram(
    "iep3_inference_seconds",
    "Time spent on inference",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
CONFIDENCE_GAUGE = Gauge(
    "iep3_detection_confidence_avg", "Average confidence score of last detection batch"
)
MODEL_LOADING_TIME = Summary(
    "iep3_model_loading_seconds", "Time taken to load the model"
)
FRAME_COUNTER = Counter(
    "iep3_frames_processed_total", "Total number of video frames processed"
)

# Global variable for the model
yolo_model: YOLO = None

# Path to your best checkpoint
MODEL_PATH = (
    Path(__file__).parent / "runs" / "train" / "human-detect12" / "weights" / "best.pt"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model at startup
    global yolo_model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH!r}")

    # Time model loading
    with MODEL_LOADING_TIME.time():
        yolo_model = YOLO(str(MODEL_PATH))
    yield
    # Cleanup operations can go here (if needed when app shuts down)


app = FastAPI(title="YOLOv5 Human Detect API", lifespan=lifespan)

# Set up Prometheus instrumentation
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a video (or image) file, runs YOLO inference,
    and returns a list of detections across all frames.
    """
    # Increment prediction counter
    PREDICTION_COUNTER.inc()

    # Save upload to a temp file
    if file.filename:
        suffix = Path(file.filename).suffix or ".mp4"
    else:
        suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        # Run inference with timing
        with INFERENCE_TIME.time():
            results = yolo_model.predict(source=tmp_path, imgsz=416)

        # Assemble detections
        output: List[dict] = []
        total_confidence = 0.0
        total_detections = 0

        # Update frame counter
        FRAME_COUNTER.inc(len(results))

        for frame_idx, frame in enumerate(results):
            for box in frame.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                output.append(
                    {
                        "frame": frame_idx,
                        "xmin": int(x1),
                        "ymin": int(y1),
                        "xmax": int(x2),
                        "ymax": int(y2),
                        "confidence": conf,
                        "class": cls,
                    }
                )
                # Track metrics
                total_confidence += conf
                total_detections += 1

        # Update detection counter
        DETECTION_COUNTER.inc(total_detections)

        # Update confidence gauge with average confidence
        if total_detections > 0:
            avg_confidence = total_confidence / total_detections
            CONFIDENCE_GAUGE.set(avg_confidence)

        return JSONResponse({"predictions": output})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        os.remove(tmp_path)
