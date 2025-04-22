# AI-Powered Shoplifting Detection

This project uses three microservices (EEP, IEP1, IEP2) to detect humans and shoplifting in uploaded videos.
Sample usages:

- Start services: `docker compose up`
- Call EEP: `POST http://localhost:8000/process-video?sample_fps=4` with video file

## Example Response

```json
{
  "human_detected": true,
  "shoplifting_detected": true
}
```
