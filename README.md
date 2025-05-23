# AI-Powered Shoplifting Detection

This project uses three microservices (EEP, IEP1, IEP2) to detect humans and shoplifting in uploaded videos.
Sample usages:

- Start services: `docker compose up`
- Call EEP: `POST http://localhost:8000/process-video` with video file

## Example Response

```json
{
  "shoplifting_detected": true
}
```

## Kubernetes Deployment Using DockerHub

First create the CCTV folder or replace the volume mounted below with your local path of the CCTV camera footage.

```bash
mkdir CCTV
```

Then create the clusters and load them.

```bash
k3d cluster create shoplift \
  --servers 1 \
  --agents 0 \
  --port "30080:30080@server:0" \
  --port "32000:32000@server:0" \
  --port "31000:31000@server:0" \
  --port "30090:30090@server:0" \
  --volume "$(pwd)/CCTV:/CCTV@server:0"
kubectl config use-context k3d-shoplift
kubectl create namespace shoplift
kubectl apply -f k3s/monitoring-pvc.yaml
kubectl apply -f k3s/data-bootstrap-job.yaml
kubectl -n shoplift wait --for=condition=complete job/data-bootstrap --timeout=300s
kubectl apply -k k3s
kubectl -n shoplift get pods -w
```

Once everything is up and running you can access the UI [here](http://localhost:30090).

## Testing the System

For testing purposes, you can run the simulation script to generate random CCTV footage.

```bash
docker pull jadshaker/simulation:latest
docker run --rm -v $(pwd)/CCTV:/CCTV -v $(pwd)/data:/data jadshaker/simulation:latest
```
