# External Endpoint (EEP)

The External Endpoint (EEP) is the main coordinator service that connects to the three Intelligent Event Processors (IEPs) to process video data through a pipeline of detection models.

## Architecture

The EEP follows a cascading decision process:

1. Split incoming videos into ~3 second segments
2. Pass each segment to IEP1 (Binary Human Detection)
3. If a human is detected, pass the segment to IEP2 (Shoplifting Detection)
4. If shoplifting is detected, pass the segment to IEP3 (Detailed Human Detection)
5. Return consolidated results including a summary across all segments

## API Endpoints

### Health Check Endpoints

- `GET /health` - Check the health of the EEP service
- `GET /health/IEP1` - Check the health of IEP1 service
- `GET /health/IEP2` - Check the health of IEP2 service
- `GET /health/IEP3` - Check the health of IEP3 service

### Video Processing

- `POST /process-video` - Process a video file through the detection pipeline
  - Request: Form data with video file
  - Response: JSON with results of each segment and overall summary

## Response Format

```json
{
  "video_id": "uploaded_video_name.mp4",
  "segments": [
    {
      "segment_id": 0,
      "iep1_result": {
        /* IEP1 response */
      },
      "iep2_result": {
        /* IEP2 response (if applicable) */
      },
      "iep3_result": {
        /* IEP3 response (if applicable) */
      }
    }
    // More segments...
  ],
  "summary": {
    "total_segments": 10,
    "segments_with_humans": 7,
    "segments_with_shoplifting": 2,
    "segments_with_detailed_detection": 2,
    "human_detected": true,
    "shoplifting_detected": true,
    "alert_level": "high"
  }
}
```

## Environment Variables

The EEP uses the following environment variables to connect to the IEP services:

- `IEP1_URL` - URL for the IEP1 service (default: http://localhost:5001)
- `IEP2_URL` - URL for the IEP2 service (default: http://localhost:5002)
- `IEP3_URL` - URL for the IEP3 service (default: http://localhost:5003)

## Running with Docker

Build the Docker image:

```bash
docker build -t eep-service .
```

Run the container:

```bash
docker run --rm -p 8000:8000 eep-service
```
