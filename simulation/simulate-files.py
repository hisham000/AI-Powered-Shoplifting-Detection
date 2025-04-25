import glob
import os
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(
    title="File Server API",
    description="API for monitoring folders and accessing video files for CCTV monitoring",
    version="1.0.0",
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper function to get file modification time
def get_file_modified_time(file_path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(file_path))
    except Exception as e:
        print(f"Error getting modified time for {file_path}: {str(e)}")
        return datetime.now()


# Convert path from Windows-style to Unix-style if needed
def normalize_path(path):
    return path.replace("\\", "/")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "UP",
        "service": "file-server",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/readDirectory")
async def read_directory(
    path: str = Query(..., description="Path to directory to read")
):
    """Read all video files from a directory"""
    if not path:
        raise HTTPException(status_code=400, detail="No path specified")

    # Convert path format if needed
    path = normalize_path(path)

    print(f"Reading directory: {path}")

    try:
        # Check if directory exists
        if not os.path.isdir(path):
            print(f"Directory does not exist: {path}")
            return {"files": [], "warning": "Directory does not exist"}

        # Look for video files (mp4, avi, etc.)
        video_files = []
        for ext in ["mp4", "avi", "mov", "mkv"]:
            pattern = os.path.join(path, f"**/*.{ext}")
            video_files.extend(glob.glob(pattern, recursive=True))

        # Normalize paths for consistency
        video_files = [normalize_path(f) for f in video_files]

        print(f"Found {len(video_files)} video files in {path}")
        return {"files": video_files}
    except Exception as e:
        print(f"Error reading directory {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/getFile")
async def get_file(path: str = Query(..., description="Path to file to get")):
    """Get a specific file"""
    if not path:
        raise HTTPException(status_code=400, detail="No path specified")

    # Convert path format if needed
    path = normalize_path(path)

    print(f"Getting file: {path}")

    try:
        # Check if file exists
        if not os.path.isfile(path):
            print(f"File does not exist: {path}")
            raise HTTPException(status_code=404, detail="File not found")

        # Return the file
        return FileResponse(path, filename=os.path.basename(path))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting file {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/getChanges")
async def get_changes(
    path: str = Query(..., description="Path to directory to check for changes"),
    since: str = Query(..., description="ISO timestamp to check changes since"),
):
    """Get files changed since a specific timestamp"""
    if not path:
        raise HTTPException(status_code=400, detail="No path specified")
    if not since:
        raise HTTPException(status_code=400, detail="No timestamp specified")

    # Convert path format if needed
    path = normalize_path(path)

    print(f"Checking for changes in: {path} since {since}")

    try:
        # Parse the timestamp
        since_time = datetime.fromisoformat(since.replace("Z", "+00:00"))

        # Check if directory exists
        if not os.path.isdir(path):
            print(f"Directory does not exist: {path}")
            return {"changes": [], "warning": "Directory does not exist"}

        # Look for video files (mp4, avi, etc.)
        video_files = []
        for ext in ["mp4", "avi", "mov", "mkv"]:
            pattern = os.path.join(path, f"**/*.{ext}")
            video_files.extend(glob.glob(pattern, recursive=True))

        # Filter files modified after the timestamp
        changed_files = []
        for file_path in video_files:
            mod_time = get_file_modified_time(file_path)
            if mod_time > since_time:
                changed_files.append(normalize_path(file_path))

        print(f"Found {len(changed_files)} changed files in {path}")
        return {"changes": changed_files}
    except ValueError as e:
        print(f"Invalid timestamp format: {since}")
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
    except Exception as e:
        print(f"Error checking for changes in {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get port from environment variable or use default 9000
    port = int(os.environ.get("PORT", 9000))

    # Run the app with uvicorn
    print(f"Starting file server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
