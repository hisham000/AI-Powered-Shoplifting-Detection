# main.py

import os
import shutil
import tempfile
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# Global variable for the model
yolo_model = None

# Path to your best checkpoint
MODEL_PATH = Path(__file__).parent / "runs" / "train" / "human-detect12" / "weights" / "best.pt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model at startup
    global yolo_model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH!r}")
    
    yolo_model = YOLO(str(MODEL_PATH))
    yield
    # Cleanup operations can go here (if needed when app shuts down)

app = FastAPI(title="YOLOv5 Human Detect API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a video (or image) file, runs YOLO inference,
    and returns a list of detections across all frames.
    """
    # Save upload to a temp file
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        # Run inference; you can tweak imgsz or other parameters here
        results = yolo_model.predict(source=tmp_path, imgsz=416)

        # Assemble detections
        output: List[dict] = []
        for frame_idx, frame in enumerate(results):
            for box in frame.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                output.append({
                    "frame":      frame_idx,
                    "xmin":       int(x1),
                    "ymin":       int(y1),
                    "xmax":       int(x2),
                    "ymax":       int(y2),
                    "confidence": conf,
                    "class":      cls
                })

        return JSONResponse({"predictions": output})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)