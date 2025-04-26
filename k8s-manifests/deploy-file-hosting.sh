#!/bin/bash

# Script to deploy the file-hosting service with the correct CCTV footage path
echo "File Hosting Service Deployment"
echo "==============================="

# Get the CCTV footage directory path from the user
read -p "Enter the full path to your CCTV footage directory: " CCTV_FOOTAGE_DIR

# Validate the path exists
if [ ! -d "$CCTV_FOOTAGE_DIR" ]; then
    echo "Error: Directory '$CCTV_FOOTAGE_DIR' does not exist."
    echo "Please provide a valid directory path."
    exit 1
fi

echo "Using CCTV footage directory: $CCTV_FOOTAGE_DIR"

# Get the node hostname
NODE_HOSTNAME=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
echo "Using node hostname: $NODE_HOSTNAME"

# Get your image registry - using a default if not specified
read -p "Enter your Docker image registry (or press Enter to use 'jadshaker'): " DOCKER_REGISTRY
DOCKER_REGISTRY=${DOCKER_REGISTRY:-jadshaker}

# Create temporary files
TMP_PV=$(mktemp)
TMP_DEPLOY=$(mktemp)

# Replace placeholders in the PV definition
cat cctv-pv.yaml | sed "s|\${CCTV_FOOTAGE_DIR}|$CCTV_FOOTAGE_DIR|g" | sed "s|\${NODE_HOSTNAME}|$NODE_HOSTNAME|g" > $TMP_PV

# Replace placeholders in the deployment
cat 09-file-hosting.yaml | sed "s|\${YOUR_REGISTRY}|$DOCKER_REGISTRY|g" > $TMP_DEPLOY

echo "Creating file-hosting PersistentVolume..."
kubectl apply -f $TMP_PV

echo "Creating file-hosting PersistentVolumeClaim..."
kubectl apply -f cctv-pvc.yaml

echo "Deploying file-hosting service..."
kubectl apply -f $TMP_DEPLOY

# Clean up temporary files
rm $TMP_PV $TMP_DEPLOY

echo "Deployment complete!"
echo "Verifying deployment..."
kubectl get pv cctv-pv
kubectl get pvc cctv-pvc -n eece-503n
kubectl get deployment file-hosting -n eece-503n
kubectl get service file-hosting -n eece-503n

echo "You can now access the file-hosting service at: http://[cluster-ip]:9000"