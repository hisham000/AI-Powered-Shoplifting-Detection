#!/bin/bash
# Script to set up the ML model artifacts in the Kubernetes Persistent Volume

# Exit on any error
set -e

echo "Setting up ML model artifacts for IEP2 in Kubernetes PV"

# Directory where ML model artifacts are stored on the host
SOURCE_DIR="/home/hisham/Desktop/503n/EECE-503N-Project/IEP2_shoplifting_detection/train/mlruns"

# Directory specified in the PV definition
TARGET_DIR="/opt/ml/models"

# Verify source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "ERROR: Source directory $SOURCE_DIR does not exist!"
  echo "Please make sure your ML model artifacts are available."
  exit 1
fi

# Create target directory if it doesn't exist (needs sudo)
echo "Creating target directory at $TARGET_DIR..."
sudo mkdir -p $TARGET_DIR

# Copy ML model artifacts to the PV location
echo "Copying ML model artifacts from $SOURCE_DIR to $TARGET_DIR..."
sudo cp -r $SOURCE_DIR/* $TARGET_DIR/

# Set proper permissions
echo "Setting permissions..."
sudo chmod -R 755 $TARGET_DIR
sudo chown -R $(id -u):$(id -g) $TARGET_DIR

echo "Setup complete! ML model artifacts are now available in the Persistent Volume at $TARGET_DIR"

# Detect Kubernetes node hostname for PV configuration
if command -v kubectl &> /dev/null; then
  echo "Detecting Kubernetes node hostname..."
  NODE_HOSTNAME=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
  if [ ! -z "$NODE_HOSTNAME" ]; then
    echo "Found node hostname: $NODE_HOSTNAME"
    echo "Please update mlruns-pv.yaml with this hostname if it's not already set."
    sed -i "s/- minikube  # Update this with your actual node name/- $NODE_HOSTNAME  # Automatically updated/" mlruns-pv.yaml
    echo "Updated mlruns-pv.yaml with detected node hostname."
  else
    echo "Unable to determine node hostname. Please manually update mlruns-pv.yaml."
  fi
else
  echo "kubectl not found. Please manually update the node hostname in mlruns-pv.yaml."
fi

echo "Next steps:"
echo "1. Apply the PV: kubectl apply -f mlruns-pv.yaml"
echo "2. Apply the IEP2 deployment with PVC: kubectl apply -f 02-iep2.yaml"