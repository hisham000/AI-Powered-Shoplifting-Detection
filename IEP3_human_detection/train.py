from pathlib import Path

import mlflow
import mlflow.pytorch
from ultralytics import YOLO, settings  # type: ignore[import]

# Turn off Ultralytics' own mlflow callbacks so we can control logging manually
settings.update({"mlflow": False})

# Resolve root and data paths
ROOT = Path(__file__).parent.resolve()
DATA_YAML = ROOT / "Human.v1-huge-1.yolov5pytorch" / "data.yaml"


def main():
    # Point MLflow at a local ./mlruns folder inside this project
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("yolov5-human-detect-test-run")

    with mlflow.start_run():
        # 1. Log our hyperparameters
        params = {
            "model": "yolov5n.pt",
            "data": str(DATA_YAML),
            "epochs": 2,
            "patience": 4,
            "imgsz": 416,
            "batch": 14,
        }
        mlflow.log_params(params)

        # 2. Train — note: we do NOT pass exist_ok=True, so each run auto‑increments
        model = YOLO(params["model"])
        results = model.train(
            data=params["data"],
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            patience=params["patience"],
            batch=params["batch"],
            project="runs/train",
            name="human-detect",
        )

        # 3. Log metrics from this run
        try:
            mlflow.log_metric("mAP_0.5", results.box.map50)
            mlflow.log_metric("mAP_0.5-0.95", results.box.map)
        except Exception as e:
            print("⚠️ Failed to log metrics:", e)

        # 4. Artifact logging — use results.path to find the exact run folder
        run_dir = results.path  # e.g. Path("runs/train/human-detect12")
        best_ckpt = run_dir / "weights" / "best.pt"
        if best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt), artifact_path="checkpoints")
        else:
            print(f"⚠️ No best.pt found at {best_ckpt}")

        # 5. Finally register the model
        mlflow.pytorch.log_model(
            pytorch_model=model.model,
            artifact_path="yolov5_model",
            registered_model_name="YOLOv5-Human-test-run",
        )


if __name__ == "__main__":
    main()
