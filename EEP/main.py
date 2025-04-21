import datetime
import os
import platform
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Any

import cv2
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Counter, Histogram
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
    IEP3_URL = os.environ.get("IEP3_URL", f"http://{HOST_PREFIX}:5003")
else:
    # When running locally/outside Docker
    IEP1_URL = os.environ.get("IEP1_URL", "http://localhost:5001")
    IEP2_URL = os.environ.get("IEP2_URL", "http://localhost:5002")
    IEP3_URL = os.environ.get("IEP3_URL", "http://localhost:5003")

# Print out the configured URLs for debugging
print(f"IEP1 URL: {IEP1_URL}")
print(f"IEP2 URL: {IEP2_URL}")
print(f"IEP3 URL: {IEP3_URL}")


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


class ProcessingResult(BaseModel):
    video_id: str
    segments: list[dict[str, Any]]
    summary: dict[str, Any]


@app.get("/health")
async def health_check():
    """Health check endpoint for all services"""
    # Initialize services status
    services_status = {
        "EEP": {"status": "healthy"},
        "IEP1": {"status": "unknown"},
        "IEP2": {"status": "unknown"},
        "IEP3": {"status": "unknown"},
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

    # Check IEP3 health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{IEP3_URL}/health", timeout=5.0)
            if response.status_code == 200:
                services_status["IEP3"] = {
                    "status": "healthy",
                    "details": response.json(),
                }
            else:
                services_status["IEP3"] = {
                    "status": "unhealthy",
                    "details": response.json(),
                }
                overall_status = "degraded"
    except Exception as e:
        services_status["IEP3"] = {"status": "unhealthy", "error": str(e)}
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


@app.get("/health/IEP3")
async def iep3_health():
    """Health check for IEP3"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{IEP3_URL}/health", timeout=5.0)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "service": "IEP3",
                    "details": response.json(),
                }
            return {
                "status": "unhealthy",
                "service": "IEP3",
                "details": response.json(),
            }
    except Exception as e:
        return {"status": "unhealthy", "service": "IEP3", "error": str(e)}


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


async def process_with_iep3(file_path: str) -> dict[str, Any]:
    """Process a video segment with IEP3 (detailed human detection)"""
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "video/mp4")}
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{IEP3_URL}/predict", files=files)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"IEP3 error: {response.text}",
                )
            return response.json()


@app.post("/process-video", response_model=ProcessingResult)
async def process_video(file: UploadFile = File(...)):
    """
    Process a video file:
    1. Split into ~3 second segments
    2. Pass each segment through IEP1
    3. If human detected, pass to IEP2
    4. If shoplifting detected (prediction=1), pass to IEP3
    5. Return consolidated results
    """
    # Increment video processing counter
    VIDEO_PROCESSING_COUNTER.inc()

    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    temp_segments = []
    result_segments = []

    try:
        # Start timing the processing
        with PROCESSING_TIME.time():
            # Split video into segments
            temp_segments = split_video_into_segments(tmp_path)

            # Track overall results
            total_segments = len(temp_segments)
            segments_with_humans = 0
            segments_with_shoplifting = 0
            segments_with_detailed_detection = 0

            # Increment segment counter
            SEGMENT_PROCESSING_COUNTER.inc(total_segments)

            # Process each segment through the pipeline
            for i, segment_path in enumerate(temp_segments):
                segment_result: dict[str, Any] = {
                    "segment_id": i,
                    "iep1_result": None,
                    "iep2_result": None,
                    "iep3_result": None,
                }

                # Step 1: IEP1 - Human Detection
                iep1_result = await process_with_iep1(segment_path)
                segment_result["iep1_result"] = iep1_result

                # Check if human detected
                human_detected = iep1_result.get("summary", {}).get(
                    "human_detected", False
                )

                if human_detected:
                    segments_with_humans += 1
                    # Increment human detection counter
                    HUMAN_DETECTION_COUNTER.inc()

                    # Step 2: IEP2 - Shoplifting Detection (only if human detected)
                    iep2_result = await process_with_iep2(segment_path)
                    segment_result["iep2_result"] = iep2_result

                    # Check if shoplifting detected (prediction = 1)
                    shoplifting_detected = iep2_result.get("prediction") == 1

                    if shoplifting_detected:
                        segments_with_shoplifting += 1
                        # Increment shoplifting detection counter
                        SHOPLIFTING_DETECTION_COUNTER.inc()

                        # Step 3: IEP3 - Detailed Human Detection (only if shoplifting detected)
                        iep3_result = await process_with_iep3(segment_path)
                        segment_result["iep3_result"] = iep3_result
                        segments_with_detailed_detection += 1

                result_segments.append(segment_result)

            # Create summary
            summary = {
                "total_segments": total_segments,
                "segments_with_humans": segments_with_humans,
                "segments_with_shoplifting": segments_with_shoplifting,
                "segments_with_detailed_detection": segments_with_detailed_detection,
                "human_detected": segments_with_humans > 0,
                "shoplifting_detected": segments_with_shoplifting > 0,
                "alert_level": "high" if segments_with_shoplifting > 0 else "low",
            }

            return {
                "video_id": file.filename or "uploaded_video",
                "segments": result_segments,
                "summary": summary,
            }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

    finally:
        # Clean up temporary files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # Clean up segment files
        for segment in temp_segments:
            if os.path.exists(segment):
                os.remove(segment)

        # Clean up parent directory of segments if it exists
        segment_dir = os.path.dirname(temp_segments[0]) if temp_segments else None
        if segment_dir and os.path.exists(segment_dir):
            shutil.rmtree(segment_dir, ignore_errors=True)
