#!/bin/bash

# Video directory on your local machine
VIDEO_DIR="/home/hisham/Downloads/dcsass-dataset/dcsass dataset/DCSASS Dataset/Shoplifting"
API_URL="http://localhost:8000/process-video?sample_fps=1"
MAX_PARALLEL_REQUESTS=10  # Number of concurrent requests
MAX_FILES_PER_DIR=10      # Number of files to process per directory

# Stats tracking variables
TOTAL_REQUESTS=0
SUCCESS_COUNT=0
FAIL_COUNT=0
START_TIME=$(date +%s)

# Check if the directory exists
if [ ! -d "$VIDEO_DIR" ]; then
  echo "Error: Video directory not found: $VIDEO_DIR"
  exit 1
fi

# Ensure Kubernetes port-forwarding is set up
echo "Checking if port-forwarding is active for EEP service..."
if ! nc -z localhost 8000 &>/dev/null; then
  echo "Setting up port-forwarding for EEP service..."
  # Start port-forwarding in the background
  kubectl port-forward -n eece-503n svc/eep 8000:8000 &
  PORT_FORWARD_PID=$!
  # Give it a moment to establish the connection
  sleep 3
  echo "Port-forwarding established (PID: $PORT_FORWARD_PID)"
else
  echo "Port-forwarding is already active on port 8000"
fi

# Create a temporary directory to copy and fix videos if needed
TEMP_DIR=$(mktemp -d)
echo "Creating temporary directory for video processing: $TEMP_DIR"

# Function to clean up when script exits
cleanup() {
  echo "Cleaning up..."
  # Remove temporary directory
  rm -rf "$TEMP_DIR"
  
  # Kill port-forwarding if we started it
  if [ -n "$PORT_FORWARD_PID" ]; then
    kill $PORT_FORWARD_PID
    echo "Port-forwarding stopped"
  fi
  
  # Kill any remaining background curl processes
  echo "Stopping any remaining background curl processes..."
  pkill -f "curl -X POST.*$API_URL"
  
  # Display final statistics
  END_TIME=$(date +%s)
  RUNTIME=$((END_TIME - START_TIME))
  HOURS=$((RUNTIME / 3600))
  MINUTES=$(( (RUNTIME % 3600) / 60 ))
  SECONDS=$((RUNTIME % 60))
  
  echo "=========================================="
  echo "Load test summary:"
  echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
  echo "Total requests: $TOTAL_REQUESTS"
  echo "Successful requests: $SUCCESS_COUNT"
  echo "Failed requests: $FAIL_COUNT"
  if [ $TOTAL_REQUESTS -gt 0 ]; then
    SUCCESS_RATE=$((SUCCESS_COUNT * 100 / TOTAL_REQUESTS))
    echo "Success rate: ${SUCCESS_RATE}%"
    REQUESTS_PER_SECOND=$(bc -l <<< "scale=2; $TOTAL_REQUESTS / $RUNTIME")
    echo "Requests per second: ${REQUESTS_PER_SECOND}"
  fi
  echo "=========================================="
  
  echo "Cleanup complete"
}

# Register the cleanup function to be called on script exit
trap cleanup EXIT

# Function to print status update periodically
print_status() {
  local current_time=$(date +%s)
  local elapsed=$((current_time - START_TIME))
  
  # Only print every 10 seconds
  if [ $((elapsed % 10)) -eq 0 ]; then
    local success_rate=0
    if [ $TOTAL_REQUESTS -gt 0 ]; then
      success_rate=$((SUCCESS_COUNT * 100 / TOTAL_REQUESTS))
    fi
    
    echo "=========================================="
    echo "Running for $elapsed seconds"
    echo "Total requests: $TOTAL_REQUESTS"
    echo "Success rate: ${success_rate}%"
    echo "Checking if Kubernetes has autoscaled..."
    
    # Show HPA status
    kubectl get hpa -n eece-503n
    
    # Show detailed pod information to verify replicas
    echo "Detailed pod information:"
    kubectl get pods -n eece-503n -o wide | grep -E 'eep|iep1|iep2'
    
    # Show metrics for top CPU consuming pods
    echo "Top CPU consuming pods:"
    kubectl top pods -n eece-503n | sort -k2 -nr | head -5
    
    echo "=========================================="
  fi
}

echo "Starting infinite load test... Press Ctrl+C to stop"
echo "Looking for videos in: $VIDEO_DIR"

# Create a file to store results from parallel processes
RESULTS_FILE="$TEMP_DIR/results.txt"
touch "$RESULTS_FILE"

# Find all video files first and store them in a list
echo "Creating list of all video files to process..."
find "$VIDEO_DIR" -type f -name "*.mp4" > "$TEMP_DIR/all_video_files.txt"
TOTAL_FILES=$(wc -l < "$TEMP_DIR/all_video_files.txt")
echo "Found $TOTAL_FILES video files to process in infinite loop"

# Infinite loop until user presses Ctrl+C
while true; do
  # Process each video file in the list
  cat "$TEMP_DIR/all_video_files.txt" | while read -r video_file; do
    # Wait if we have too many parallel requests
    while [ $(jobs -p | wc -l) -ge $MAX_PARALLEL_REQUESTS ]; do 
      sleep 0.1
    done
    
    # Process this file in the background
    (
      # Copy file to temp dir with a unique name to avoid conflicts
      unique_id=$(basename "$video_file" | md5sum | cut -d' ' -f1)
      TEMP_VIDEO="$TEMP_DIR/temp_video_${unique_id}.mp4"
      cp "$video_file" "$TEMP_VIDEO"
      
      # Send request and capture full response
      RESPONSE=$(curl -X POST -s -w "\nHTTP_STATUS:%{http_code}" \
        -F "file=@$TEMP_VIDEO" \
        "${API_URL}" 2>&1)
      
      # Extract HTTP status code
      HTTP_STATUS=$(echo "$RESPONSE" | grep -o "HTTP_STATUS:[0-9]*" | cut -d':' -f2)
      
      # Check if request was successful
      if [ "$HTTP_STATUS" = "200" ]; then
        echo "✅ Success ($HTTP_STATUS) - $(basename "$video_file")"
        echo "SUCCESS" >> "$RESULTS_FILE"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      else
        echo "❌ Failed ($HTTP_STATUS) - $(basename "$video_file")"
        # Show response body for debugging
        echo "Response: $(echo "$RESPONSE" | sed 's/HTTP_STATUS:[0-9]*$//')"
        echo "FAIL" >> "$RESULTS_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
      fi
      
      # Increment the total requests counter
      TOTAL_REQUESTS=$((TOTAL_REQUESTS + 1))
      
      # Print status update
      print_status
      
      # Remove the temporary file
      rm -f "$TEMP_VIDEO"
    ) &  # Run in background
    
    # Small delay to avoid overwhelming the file system
    sleep 0.1
  done
  
  # Wait for batch to finish before starting next cycle
  wait
  
  echo "Completed one full cycle through all videos. Starting again..."
  
  # Every 5 cycles, check on the HPA and deployments status
  if [ $((TOTAL_REQUESTS / TOTAL_FILES % 5)) -eq 0 ]; then
    echo "Checking Kubernetes status after $(TOTAL_REQUESTS) requests..."
    echo "Deployments:"
    kubectl get deployments -n eece-503n
    echo "HPAs:"
    kubectl get hpa -n eece-503n
  fi
done