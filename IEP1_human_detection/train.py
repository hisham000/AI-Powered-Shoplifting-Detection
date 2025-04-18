# train.py
import torch
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
from pathlib import Path

# resolve root of this script
ROOT = Path(__file__).parent.resolve()
DATA_YAML = ROOT / "Human.v1-huge-1.yolov5pytorch" / "data.yaml"

def main():
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("yolov5-human-detect")
    with mlflow.start_run():
        # 1. Log params
        params = {
            "model":  "yolov5n.pt",
            "data":   str(DATA_YAML),   # ← use the Path here
            "epochs": 10,
            "imgsz":  416,
            "batch":  12
        }
        mlflow.log_params(params)

        # 2. Load & train via the ultralytics API
        model = YOLO(params["model"])
        results = model.train(
            data=params["data"],
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            project="runs/train",
            name="human-detect"        
            )

        # 3. Log metrics
        try:
            mlflow.log_metric("mAP_0.5",     results.box.map50)
            mlflow.log_metric("mAP_0.5-0.95", results.box.map)
        except Exception as e:
            print("Failed to log metrics:", e)


        # 4. Log the best checkpoint
        ckpt = ROOT / "runs" / "train" / "human-detect" / "weights" / "best.pt"
        mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")

        # 5. Register the model in MLflow
        #    `model.model` is the underlying torch.nn.Module
        mlflow.pytorch.log_model(
            pytorch_model=model.model,  # Changed from torch_model to pytorch_model
            artifact_path="yolov5_model",
            registered_model_name="YOLOv5-Human"
        )

if __name__ == "__main__":
    main()
