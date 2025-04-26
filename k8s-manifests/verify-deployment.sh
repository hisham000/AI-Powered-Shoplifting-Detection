#!/bin/bash
# Script to verify the Kubernetes deployment

# Set namespace
NAMESPACE="eece-503n"

echo "===== Verifying Kubernetes Deployment ====="

# Check if namespace exists
echo -e "\n[ Checking namespace ]"
if kubectl get namespace $NAMESPACE &> /dev/null; then
  echo "✅ Namespace $NAMESPACE exists"
else
  echo "❌ Namespace $NAMESPACE does not exist. Please apply 00-namespace.yaml first."
  exit 1
fi

# Check Persistent Volume and PVC
echo -e "\n[ Checking storage resources ]"
echo "Persistent Volumes:"
kubectl get pv | grep mlruns-pv
PV_STATUS=$(kubectl get pv mlruns-pv -o jsonpath='{.status.phase}')
if [ "$PV_STATUS" == "Available" ] || [ "$PV_STATUS" == "Bound" ]; then
  echo "✅ PV mlruns-pv status: $PV_STATUS"
else
  echo "❌ PV mlruns-pv status: $PV_STATUS (Expected: Available or Bound)"
fi

echo -e "\nPersistent Volume Claims:"
kubectl get pvc -n $NAMESPACE
PVC_STATUS=$(kubectl get pvc mlruns-pvc -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
if [ "$PVC_STATUS" == "Bound" ]; then
  echo "✅ PVC mlruns-pvc is bound"
else
  echo "❌ PVC mlruns-pvc is not bound (status: $PVC_STATUS)"
fi

# Check deployments
echo -e "\n[ Checking deployments ]"
kubectl get deployments -n $NAMESPACE
for DEPLOYMENT in eep iep1 iep2; do
  READY=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
  DESIRED=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
  
  if [ "$READY" == "$DESIRED" ] && [ "$READY" != "0" ]; then
    echo "✅ Deployment $DEPLOYMENT is ready ($READY/$DESIRED)"
  else
    echo "❌ Deployment $DEPLOYMENT is not ready ($READY/$DESIRED)"
  fi
done

# Check pods
echo -e "\n[ Checking pods ]"
kubectl get pods -n $NAMESPACE
echo -e "\nDetailed pod status:"
for POD in $(kubectl get pods -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}'); do
  STATUS=$(kubectl get pod $POD -n $NAMESPACE -o jsonpath='{.status.phase}')
  if [ "$STATUS" == "Running" ]; then
    READY=$(kubectl get pod $POD -n $NAMESPACE -o jsonpath='{.status.containerStatuses[0].ready}')
    if [ "$READY" == "true" ]; then
      echo "✅ Pod $POD is running and ready"
    else
      echo "❌ Pod $POD is running but not ready"
    fi
  else
    echo "❌ Pod $POD status: $STATUS (Expected: Running)"
    echo "   Pod details:"
    kubectl describe pod $POD -n $NAMESPACE | grep -A 5 "State:"
  fi
done

# Check IEP2 volume mount and access to models
echo -e "\n[ Checking IEP2 volume mount ]"
IEP2_POD=$(kubectl get pod -n $NAMESPACE -l app=iep2 -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [ -n "$IEP2_POD" ]; then
  echo "Checking if models are accessible in the container:"
  if kubectl exec -n $NAMESPACE $IEP2_POD -- ls -la /app/mlruns &> /dev/null; then
    echo "✅ IEP2 pod can access /app/mlruns directory"
    # Check if there are actual model files
    MODEL_FILES=$(kubectl exec -n $NAMESPACE $IEP2_POD -- find /app/mlruns -name "*.keras" 2>/dev/null | wc -l)
    if [ "$MODEL_FILES" -gt 0 ]; then
      echo "✅ Found $MODEL_FILES model files in mlruns directory"
    else
      echo "❌ No model files (*.keras) found in mlruns directory"
    fi
  else
    echo "❌ IEP2 pod cannot access /app/mlruns directory"
  fi
else
  echo "❌ No IEP2 pod found"
fi

# Check services and endpoints
echo -e "\n[ Checking services ]"
kubectl get services -n $NAMESPACE

# Final summary
echo -e "\n===== Deployment Verification Summary ====="
DEPLOYMENTS_READY=$(kubectl get deployments -n $NAMESPACE -o jsonpath='{.items[?(@.status.readyReplicas==@.spec.replicas)].metadata.name}' | wc -w)
TOTAL_DEPLOYMENTS=$(kubectl get deployments -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}' | wc -w)
PODS_RUNNING=$(kubectl get pods -n $NAMESPACE -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)
TOTAL_PODS=$(kubectl get pods -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}' | wc -w)

if [ "$DEPLOYMENTS_READY" -eq "$TOTAL_DEPLOYMENTS" ] && [ "$PODS_RUNNING" -eq "$TOTAL_PODS" ] && [ "$PVC_STATUS" == "Bound" ]; then
  echo "✅ All resources appear to be running correctly!"
  
  # Display access instructions
  EEP_NODEPORT=$(kubectl get svc eep -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "")
  if [ -n "$EEP_NODEPORT" ]; then
    echo -e "\nTo access EEP service:"
    echo "URL: http://localhost:$EEP_NODEPORT"
    echo "or use port-forwarding:"
    echo "kubectl port-forward -n $NAMESPACE service/eep 8000:8000"
  else 
    echo -e "\nTo access EEP service, use port-forwarding:"
    echo "kubectl port-forward -n $NAMESPACE service/eep 8000:8000"
  fi
  
  # Check Prometheus and Grafana
  if kubectl get svc -n $NAMESPACE | grep -q prometheus; then
    PROM_PORT=$(kubectl get svc prometheus -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "")
    if [ -n "$PROM_PORT" ]; then
      echo -e "\nPrometheus UI available at: http://localhost:$PROM_PORT"
    else
      echo -e "\nAccess Prometheus with: kubectl port-forward -n $NAMESPACE service/prometheus 9090:9090"
    fi
  fi
  
  if kubectl get svc -n $NAMESPACE | grep -q grafana; then
    GRAFANA_PORT=$(kubectl get svc grafana -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "")
    if [ -n "$GRAFANA_PORT" ]; then
      echo -e "Grafana UI available at: http://localhost:$GRAFANA_PORT (default credentials: admin/admin)"
    else
      echo -e "Access Grafana with: kubectl port-forward -n $NAMESPACE service/grafana 3000:3000 (default credentials: admin/admin)"
    fi
  fi
else
  echo "❌ Some resources are not running correctly. Please check the details above."
fi