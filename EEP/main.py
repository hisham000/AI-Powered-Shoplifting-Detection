import datetime
import os
import platform
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# Prometheus metrics
VIDEO_PROCESSING_COUNTER = Counter(
    "eep_video_processed_total", "Number of videos processed"
)
SEGMENT_PROCESSING_COUNTER = Counter(
    "eep_segment_processed_total", "Number of segments processed"
)
HUMAN_DETECTION_COUNTER = Counter(
    "eep_human_detected_total", "Number of segments with humans detected"
)
SHOPLIFTING_DETECTION_COUNTER = Counter(
    "eep_shoplifting_detected_total", "Number of segments with shoplifting detected"
)
PROCESSING_TIME = Histogram(
    "eep_video_processing_seconds",
    "Time spent processing videos",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)
SAMPLE_FPS_GAUGE = Gauge("eep_sample_fps", "Sampling FPS used for IEP1 requests")

# Determine if running inside Docker by checking for the DOCKER environment variable
# or by checking if a /.dockerenv file exists (a common Docker indicator)
RUNNING_IN_DOCKER = os.environ.get(
    "RUNNING_IN_DOCKER", "False"
).lower() == "true" or os.path.exists("/.dockerenv")

# If running in Docker, use host.docker.internal to access the host machine (for Mac/Windows)
# or host IP for Linux containers
if RUNNING_IN_DOCKER:
    # For macOS and Windows, host.docker.internal points to the host
    # For Linux, we might need to use the gateway address or set it in docker-compose
    if platform.system() == "Linux":
        # In a production environment, these would be service names from docker-compose
        HOST_PREFIX = os.environ.get("HOST_PREFIX", "host.docker.internal")
    else:
        HOST_PREFIX = "host.docker.internal"

    # If you're using docker-compose with service names, override these
    IEP1_URL = os.environ.get("IEP1_URL", f"http://{HOST_PREFIX}:5001")
    IEP2_URL = os.environ.get("IEP2_URL", f"http://{HOST_PREFIX}:5002")
else:
    # When running locally/outside Docker
    IEP1_URL = os.environ.get("IEP1_URL", "http://localhost:5001")
    IEP2_URL = os.environ.get("IEP2_URL", "http://localhost:5002")

# Print out the configured URLs for debugging
print(f"IEP1 URL: {IEP1_URL}")
print(f"IEP2 URL: {IEP2_URL}")


# Lifespan event to initialize resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    print(f"Starting EEP service... Running in Docker: {RUNNING_IN_DOCKER}")
    yield
    # Cleanup
    print("Shutting down EEP service...")


app = FastAPI(title="External Endpoint (EEP)", lifespan=lifespan)

# Set up Prometheus instrumentation
Instrumentator().instrument(app).expose(app)


class ProcessVideoResult(BaseModel):
    human_detected: bool
    shoplifting_detected: bool


@app.get("/health")
async def health_check():
    """Health check endpoint for all services"""
    # Initialize services status
    services_status = {
        "EEP": {"status": "healthy"},
        "IEP1": {"status": "unknown"},
        "IEP2": {"status": "unknown"},
    }

    overall_status = "healthy"

    # Check EEP health - this service
    services_status["EEP"] = {"status": "healthy", "service": "EEP"}

    # Check IEP1 health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{IEP1_URL}/health", timeout=5.0)
            if response.status_code == 200:
                services_status["IEP1"] = {
                    "status": "healthy",
                    "details": response.json(),
                }
            else:
                services_status["IEP1"] = {
                    "status": "unhealthy",
                    "details": response.json(),
                }
                overall_status = "degraded"
    except Exception as e:
        services_status["IEP1"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "degraded"

    # Check IEP2 health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{IEP2_URL}/health", timeout=5.0)
            if response.status_code == 200:
                services_status["IEP2"] = {
                    "status": "healthy",
                    "details": response.json(),
                }
            else:
                services_status["IEP2"] = {
                    "status": "unhealthy",
                    "details": response.json(),
                }
                overall_status = "degraded"
    except Exception as e:
        services_status["IEP2"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": str(datetime.datetime.now()),
        "services": services_status,
    }


@app.get("/health/EEP")
def eep_health():
    """Health check endpoint for EEP service only"""
    return {
        "status": "healthy",
        "service": "EEP",
        "version": "1.0.0",
        "timestamp": str(datetime.datetime.now()),
    }


@app.get("/health/IEP1")
async def iep1_health():
    """Health check for IEP1"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{IEP1_URL}/health", timeout=5.0)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "service": "IEP1",
                    "details": response.json(),
                }
            return {
                "status": "unhealthy",
                "service": "IEP1",
                "details": response.json(),
            }
    except Exception as e:
        return {"status": "unhealthy", "service": "IEP1", "error": str(e)}


@app.get("/health/IEP2")
async def iep2_health():
    """Health check for IEP2"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{IEP2_URL}/health", timeout=5.0)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "service": "IEP2",
                    "details": response.json(),
                }
            return {
                "status": "unhealthy",
                "service": "IEP2",
                "details": response.json(),
            }
    except Exception as e:
        return {"status": "unhealthy", "service": "IEP2", "error": str(e)}


def split_video_into_segments(video_path: str, segment_duration: int = 3) -> list[str]:
    """Split a video into segments of specified duration"""
    temp_dir = tempfile.mkdtemp()
    segments = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frames per segment
        frames_per_segment = int(fps * segment_duration)

        # Calculate total number of segments
        num_segments = max(1, total_frames // frames_per_segment)

        # Set up video writer properties
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for i in range(num_segments):
            segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
            out = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))

            # Write frames to segment
            for _ in range(frames_per_segment):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()
            segments.append(segment_path)

        cap.release()
        return segments
    except Exception as e:
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


async def process_with_iep1(file_path: str) -> dict[str, Any]:
    """Process a video segment with IEP1 (human detection)"""
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "video/mp4")}
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{IEP1_URL}/predict", files=files)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"IEP1 error: {response.text}",
                )
            return response.json()


async def process_with_iep2(file_path: str) -> dict[str, Any]:
    """Process a video segment with IEP2 (shoplifting detection)"""
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "video/mp4")}
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{IEP2_URL}/predict", files=files)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"IEP2 error: {response.text}",
                )
            return response.json()


@app.post("/process-video", response_model=ProcessVideoResult)
async def process_video(
    file: UploadFile = File(...),
    sample_fps: int = Query(
        4, ge=1, description="Frames per second to sample for IEP1"
    ),
):
    """
    Send 1fps video to IEP1; if human detected, send original video to IEP2.
    Returns human_detected and optional shoplifting prediction.
    """
    VIDEO_PROCESSING_COUNTER.inc()
    # Export sampling fps and start timer for monitoring
    SAMPLE_FPS_GAUGE.set(sample_fps)
    start_time = time.time()
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)
    try:
        # Create a 1fps version of the video
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp1:
            tmp_1fps_path = tmp1.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_1fps_path, fourcc, sample_fps, (width, height))
        cap = cv2.VideoCapture(tmp_path)
        idx = 0
        step = max(int(fps / sample_fps), 1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                out.write(frame)
            idx += 1
        cap.release()
        out.release()

        # Human detection at 1fps
        iep1_res = await process_with_iep1(tmp_1fps_path)
        human_detected = iep1_res.get("human_detected", False)
        if human_detected:
            HUMAN_DETECTION_COUNTER.inc()

        # Cleanup 1fps file
        if os.path.exists(tmp_1fps_path):
            os.remove(tmp_1fps_path)

        if not human_detected:
            elapsed = time.time() - start_time
            PROCESSING_TIME.observe(elapsed)
            return {"human_detected": False, "shoplifting_detected": False}

        # Send original video to IEP2
        iep2_res = await process_with_iep2(tmp_path)
        pred = iep2_res.get("prediction")
        if pred == 1:
            SHOPLIFTING_DETECTION_COUNTER.inc()
        elapsed = time.time() - start_time
        PROCESSING_TIME.observe(elapsed)
        return {"human_detected": True, "shoplifting_detected": pred == 1}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
