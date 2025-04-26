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

## Kubernetes Deployment

This project can be deployed to a Kubernetes cluster. The manifests are located in the `k8s-manifests` directory.

### Prerequisites

- `kubectl` command-line tool installed and configured to communicate with your cluster.
- Access to the Kubernetes node(s) might be required for setup, especially if using `local` storage or `hostPath`.

### Deploying ML Models with Persistent Volumes

The IEP2 service requires ML model artifacts to function. These are stored using a `local` Persistent Volume, which means the storage is tied to a specific node in your cluster.

1.  **Prepare the Node and Model Artifacts:**
    - Ensure the directory specified in `mlruns-pv.yaml` (default: `/opt/ml/models`) exists on the target Kubernetes node where the PV will be created.
    - Run the setup script from the `k8s-manifests` directory. **Note:** This script likely needs to be run *on the target Kubernetes node* or requires `sudo` access to its filesystem.
    ```bash
    cd k8s-manifests
    ./setup-mlruns-pv.sh
    ```
    This script will:
    - Copy ML model artifacts from `./IEP2_shoplifting_detection/train/mlruns` (relative to the project root) to the PV host path (`/opt/ml/models`).
    - Attempt to automatically detect your Kubernetes node hostname and update `mlruns-pv.yaml`. Verify the hostname is correct.
    - Set necessary permissions on the host path.

2.  **Apply the Kubernetes Manifests:** Apply them in the specified order:
    ```bash
    kubectl apply -f 00-namespace.yaml
    kubectl apply -f mlruns-pv.yaml       # Creates the Persistent Volume
    kubectl apply -f 02-iep2.yaml       # Creates PVC and Deployment (needs PV)
    kubectl apply -f 01-iep1.yaml       # Order between IEP1/EEP/Monitoring doesn't strictly matter after PV/PVC
    kubectl apply -f 03-eep.yaml
    kubectl apply -f 04-prometheus.yaml
    kubectl apply -f 05-grafana.yaml
    ```
    *Note:* The EEP and Grafana deployments use `hostPath` volumes, meaning they also rely on specific paths existing on the node where their pods are scheduled.

3.  **Verify the Deployment:** Use the verification script:
    ```bash
    ./verify-deployment.sh
    ```
    This script will:
    - Check if all resources (Deployments, Pods, Services) are running correctly.
    - Validate PV/PVC binding.
    - Verify IEP2 can access the model files within its container (`/app/mlruns`).
    - Show instructions for accessing services (NodePort and port-forwarding).

4.  **Access the Service:**
    You can access the EEP service using either NodePort (if your cluster setup allows) or port-forwarding:
    - **Port-Forwarding (Recommended for local testing):**
      ```bash
      kubectl port-forward -n eece-503n service/eep 8000:8000
      ```
      Then access `http://localhost:8000`.
    - **NodePort:** Find the assigned NodePort:
      ```bash
      kubectl get svc eep -n eece-503n -o=jsonpath='{.spec.ports[0].nodePort}'
      ```
      Then access `http://<your-node-ip>:<node-port>`. Similarly, Prometheus and Grafana services are also exposed via NodePort. Check `verify-deployment.sh` output for details.

### Troubleshooting

If the IEP2 pod can't access model files (`/app/mlruns`), check:
1.  The PV is `Bound` to the PVC (`kubectl get pv,pvc -n eece-503n`).
2.  The model files exist in the correct host path on the node (`ls -la /opt/ml/models` on the node).
3.  The `nodeAffinity` selector in `mlruns-pv.yaml` matches the actual hostname of the node where the files exist and where the IEP2 pod is scheduled.
4.  The IEP2 pod is scheduled on the *same node* specified in the PV's `nodeAffinity`. Use `kubectl get pod -n eece-503n -o wide` to check.
