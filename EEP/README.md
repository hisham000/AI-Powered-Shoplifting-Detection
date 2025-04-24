# External Endpoint (EEP)

The External Endpoint (EEP) is the main coordinator service that connects to the three Intelligent Event Processors (IEPs) to process video data through a pipeline of detection models.

## Architecture

The EEP follows a cascading decision process:

1. Split incoming videos into ~3 second segments
2. Pass each segment to IEP1 (Binary Human Detection)
3. If a human is detected, pass the segment to IEP2 (Shoplifting Detection)
4. Return consolidated results including a summary across all segments

## API Endpoints

### Health Check Endpoints

- `GET /health` - Check the health of the EEP service
- `GET /health/IEP1` - Check the health of IEP1 service
- `GET /health/IEP2` - Check the health of IEP2 service

### Video Processing

- `POST /process-video` - Process a video file through the detection pipeline
  - Request: Form data with video file
  - Response: JSON with detection results

## Response Format

```json
{
  "human_detected": true,
  "shoplifting_detected": true
}
```

## Environment Variables

The EEP uses the following environment variables to connect to the IEP services:

- `IEP1_URL` - URL for the IEP1 service (default: http://localhost:5001)
- `IEP2_URL` - URL for the IEP2 service (default: http://localhost:5002)

## Running with Docker

Build the Docker image:

```bash
docker build -t eep-service .
```

Run the container:

```bash
docker run --rm -p 8000:8000 eep-service
```

## Running with Docker from Docker Hub

Pull the Docker image from Docker Hub:

```bash
docker pull jadshaker/eece-503n-project-eep:latest
```

Run the container:

```bash
docker run --rm -p 8000:8000 jadshaker/eece-503n-project-eep:latest
```
