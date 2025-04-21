import gc
import os
import resource
import shutil
import sys
import traceback

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

import cv2
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import seaborn as sns
from config import (
    BATCH_SIZE,
    CLASSES_LIST,
    DATA_ROOT,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    EXPERIMENT_NAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SEQUENCE_LENGTH,
    TRACKING_URI,
)
from load_data import load_data
from sklearn.metrics import confusion_matrix  # type: ignore[import]
from sklearn.model_selection import train_test_split  # type: ignore[import]
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    ConvLSTM2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling3D,
    TimeDistributed,
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore


def print_memory_usage(message=""):
    """Print current memory usage."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"Memory usage {message}: {usage.ru_maxrss / 1024} MB", flush=True)


# Verify data directory exists
if not os.path.exists(f"{DATA_ROOT}/0"):
    print("Loading data...", flush=True)
    load_data()

# Check the classes directories
for cls in CLASSES_LIST:
    cls_dir = os.path.join(DATA_ROOT, cls)
    if not os.path.exists(cls_dir):
        print(f"ERROR: Class directory not found: {cls_dir}", flush=True)
        print(
            f"Available directories in {DATA_ROOT}:", os.listdir(DATA_ROOT), flush=True
        )
        sys.exit(1)

    # Check for video files
    files = [f for f in os.listdir(cls_dir) if f.endswith((".mp4", ".avi", ".mov"))]
    if len(files) == 0:
        print(f"WARNING: No video files found in {cls_dir}", flush=True)
        print(f"Contains: {os.listdir(cls_dir)}", flush=True)

print(f"Found data directory: {DATA_ROOT}", flush=True)
print(f"Classes: {CLASSES_LIST}", flush=True)
for cls in CLASSES_LIST:
    cls_dir = os.path.join(DATA_ROOT, cls)
    files = [f for f in os.listdir(cls_dir) if f.endswith((".mp4", ".avi", ".mov"))]
    print(f"Class {cls}: {len(files)} videos", flush=True)

# MLflow setup
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def extract_frames(video_path):
    try:
        frames = []
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(int(total / SEQUENCE_LENGTH), 1)
        for i in range(SEQUENCE_LENGTH):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            frames.append(frame / 255.0)
        cap.release()
        return frames
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}", flush=True)
        traceback.print_exc()
        return []


def create_dataset():
    print("Starting dataset creation...", flush=True)
    print_memory_usage("before dataset creation")
    X, y = [], []
    try:
        total_videos = 0
        for idx, cls in enumerate(CLASSES_LIST):
            cls_dir = os.path.join(DATA_ROOT, cls)
            files = os.listdir(cls_dir)
            print(f"Processing {len(files)} videos from class {cls}...", flush=True)
            for i, fname in enumerate(files):
                if i % 50 == 0:
                    print(
                        f"Processed {i}/{len(files)} videos for class {cls}", flush=True
                    )
                    print_memory_usage(f"after processing {i} videos")

                path = os.path.join(cls_dir, fname)
                seq = extract_frames(path)
                if len(seq) == SEQUENCE_LENGTH:
                    X.append(seq)
                    y.append(idx)
                    total_videos += 1

                # Force garbage collection periodically
                if i % 100 == 0:
                    gc.collect()

            print(f"Completed processing class {cls}", flush=True)

        print(f"Dataset creation complete with {total_videos} videos", flush=True)
        print_memory_usage("after dataset creation")

        # Convert to numpy arrays
        print("Converting to numpy arrays...", flush=True)
        X_np = np.array(X)
        y_np = to_categorical(y)
        print(f"X shape: {X_np.shape}, y shape: {y_np.shape}", flush=True)
        print_memory_usage("after numpy conversion")

        return X_np, y_np
    except Exception as e:
        print(f"Error creating dataset: {e}", flush=True)
        traceback.print_exc()
        if X and y:
            print(f"Returning partial dataset with {len(X)} samples", flush=True)
            return np.array(X), to_categorical(y)
        else:
            sys.exit(1)


def cleanup_mlruns(experiment_name: str) -> None:
    """
    Delete all models except the one with the highest accuracy.

    Args:
        experiment_name: Name of the experiment to clean up
    """
    print("Starting mlruns cleanup process...", flush=True)
    try:
        # Get the experiment ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(
                f"Experiment {experiment_name} not found, skipping cleanup", flush=True
            )
            return

        experiment_id = experiment.experiment_id
        print(f"Found experiment with ID: {experiment_id}", flush=True)

        # Get all runs for this experiment
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        if runs.empty:
            print("No runs found for the experiment, skipping cleanup", flush=True)
            return

        # Find the run with the highest accuracy
        if "metrics.test_accuracy" in runs.columns:
            accuracy_col = "metrics.test_accuracy"
        elif "metrics.val_accuracy" in runs.columns:
            accuracy_col = "metrics.val_accuracy"
        elif "metrics.accuracy" in runs.columns:
            accuracy_col = "metrics.accuracy"
        else:
            print("No accuracy metric found in runs, skipping cleanup", flush=True)
            return

        # Sort by accuracy (descending) and get the best run
        runs_sorted = runs.sort_values(by=accuracy_col, ascending=False)
        best_run_id = runs_sorted.iloc[0]["run_id"]
        best_accuracy = runs_sorted.iloc[0][accuracy_col]

        print(
            f"Best run ID: {best_run_id} with accuracy: {best_accuracy:.4f}", flush=True
        )

        # Get mlruns folder path
        mlruns_dir = os.path.join(os.getcwd(), "mlruns")
        if not os.path.exists(mlruns_dir):
            print(
                f"MLflow runs directory not found at {mlruns_dir}, skipping cleanup",
                flush=True,
            )
            return

        # Get experiment directory
        exp_dir = os.path.join(mlruns_dir, experiment_id)
        if not os.path.exists(exp_dir):
            print(
                f"Experiment directory not found at {exp_dir}, skipping cleanup",
                flush=True,
            )
            return

        # Count how many runs we'll delete
        delete_count = 0
        total_size = 0

        for run_folder in os.listdir(exp_dir):
            run_path = os.path.join(exp_dir, run_folder)
            # Skip non-directories and the best run
            if not os.path.isdir(run_path) or run_folder == best_run_id:
                continue

            # Calculate size before deletion
            folder_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(run_path)
                for filename in filenames
            )
            total_size += folder_size
            delete_count += 1

            # Delete the run directory
            shutil.rmtree(run_path)
            print(
                f"Deleted run {run_folder} ({folder_size / (1024*1024):.2f} MB)",
                flush=True,
            )

        print(
            f"Cleanup complete. Deleted {delete_count} runs, saved {total_size / (1024*1024):.2f} MB",
            flush=True,
        )
        print(
            f"Kept best run {best_run_id} with accuracy {best_accuracy:.4f}", flush=True
        )

    except Exception as e:
        print(f"Error during cleanup: {e}", flush=True)
        traceback.print_exc()


def main():
    try:
        print("Starting main process...", flush=True)
        print_memory_usage("at start")

        with mlflow.start_run():
            # Log hyperparameters
            params = {
                "sequence_length": SEQUENCE_LENGTH,
                "image_height": IMAGE_HEIGHT,
                "image_width": IMAGE_WIDTH,
                "num_classes": len(CLASSES_LIST),
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "early_stop_patience": EARLY_STOP_PATIENCE,
            }
            mlflow.log_params(params)

            # Prepare data
            print("Preparing data...", flush=True)
            X, y = create_dataset()
            print_memory_usage("after data preparation")

            print(f"Dataset loaded with shape X: {X.shape}, y: {y.shape}", flush=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, shuffle=True
            )
            print(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}", flush=True)
            print(f"Test shapes - X: {X_test.shape}, y: {y_test.shape}", flush=True)
            print_memory_usage("after train/test split")

            # Reduce memory usage by deleting original arrays
            del X, y
            gc.collect()
            print_memory_usage("after deleting original arrays")

            print("Data preparation complete.", flush=True)

            # Build model
            print("Building model...", flush=True)
            inp = Input((SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            x = ConvLSTM2D(
                4,
                (3, 3),
                activation="tanh",
                recurrent_dropout=0.2,
                return_sequences=True,
            )(inp)
            x = MaxPooling3D((1, 2, 2), padding="same")(x)
            x = TimeDistributed(Dropout(0.2))(x)
            x = ConvLSTM2D(
                8,
                (3, 3),
                activation="tanh",
                recurrent_dropout=0.2,
                return_sequences=True,
            )(x)
            x = MaxPooling3D((1, 2, 2), padding="same")(x)
            x = TimeDistributed(Dropout(0.2))(x)
            x = ConvLSTM2D(
                16,
                (3, 3),
                activation="tanh",
                recurrent_dropout=0.2,
                return_sequences=True,
            )(x)
            x = MaxPooling3D((1, 2, 2), padding="same")(x)
            x = TimeDistributed(Dropout(0.2))(x)
            x = ConvLSTM2D(
                32,
                (3, 3),
                activation="tanh",
                recurrent_dropout=0.2,
                return_sequences=True,
            )(x)
            x = MaxPooling3D((1, 2, 2), padding="same")(x)
            x = Flatten()(x)
            out = Dense(len(CLASSES_LIST), activation="softmax")(x)
            model = Model(inp, out)
            print("Model built.", flush=True)

            # Compile
            print("Compiling model...", flush=True)
            model.compile(
                loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
            print("Model compiled.", flush=True)

            # Early stopping
            es = EarlyStopping(
                monitor="val_loss",
                patience=params["early_stop_patience"],
                mode="min",
                restore_best_weights=True,
            )

            # Train
            print("Starting training...", flush=True)
            print_memory_usage("before training")
            history = model.fit(
                X_train,
                y_train,
                validation_split=0.2,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                shuffle=True,
                callbacks=[es],
                verbose=1,  # Make sure training progress is displayed
            )
            print_memory_usage("after training")
            print("Training complete.", flush=True)

            # Log metrics per epoch
            print("Logging metrics...", flush=True)
            for step, (l, vl, acc, vac) in enumerate(
                zip(
                    history.history["loss"],
                    history.history["val_loss"],
                    history.history["accuracy"],
                    history.history["val_accuracy"],
                )
            ):
                mlflow.log_metric("loss", l, step=step)
                mlflow.log_metric("val_loss", vl, step=step)
                mlflow.log_metric("accuracy", acc, step=step)
                mlflow.log_metric("val_accuracy", vac, step=step)
            print("Metrics logged.", flush=True)

            # Evaluate on test set
            print("Evaluating on test set...", flush=True)
            test_loss, test_acc = model.evaluate(X_test, y_test)
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_acc)
            print(
                f"Test evaluation complete. Loss: {test_loss}, Accuracy: {test_acc}",
                flush=True,
            )

            # Log the model (includes weights)
            print("Logging model...", flush=True)
            mlflow.keras.log_model(model, artifact_path="model")
            print("Model logged.", flush=True)

            # Log loss/validation curves as figures
            print("Logging figures...", flush=True)
            plt.figure()
            plt.plot(history.history["loss"], label="loss")
            plt.plot(history.history["val_loss"], label="val_loss")
            plt.title("Loss vs Val Loss")
            plt.legend()
            mlflow.log_figure(plt.gcf(), "loss_vs_val_loss.png")
            plt.close()

            plt.figure()
            plt.plot(history.history["accuracy"], label="accuracy")
            plt.plot(history.history["val_accuracy"], label="val_accuracy")
            plt.title("Accuracy vs Val Accuracy")
            plt.legend()
            mlflow.log_figure(plt.gcf(), "accuracy_vs_val_accuracy.png")
            plt.close()

            # Predictions & confusion matrix
            print("Generating predictions and confusion matrix...", flush=True)
            preds = model.predict(X_test)
            pred_labels = np.argmax(preds, axis=1)
            true_labels = np.argmax(y_test, axis=1)
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            plt.close()
            print("Confusion matrix logged.", flush=True)

        # Clean up old models after the run is complete
        print("Training run completed, starting cleanup...", flush=True)
        cleanup_mlruns(EXPERIMENT_NAME)

    except Exception as e:
        print(f"Error in main process: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
