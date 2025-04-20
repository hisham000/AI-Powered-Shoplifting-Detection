import os

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

# MLflow setup
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def extract_frames(video_path):
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


def create_dataset():
    X, y = [], []
    for idx, cls in enumerate(CLASSES_LIST):
        cls_dir = os.path.join(DATA_ROOT, cls)
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)
            seq = extract_frames(path)
            if len(seq) == SEQUENCE_LENGTH:
                X.append(seq)
                y.append(idx)
    return np.array(X), to_categorical(y)


def main():
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
        X, y = create_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True
        )

        # Build model
        inp = Input((SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        x = ConvLSTM2D(
            4, (3, 3), activation="tanh", recurrent_dropout=0.2, return_sequences=True
        )(inp)
        x = MaxPooling3D((1, 2, 2), padding="same")(x)
        x = TimeDistributed(Dropout(0.2))(x)
        x = ConvLSTM2D(
            8, (3, 3), activation="tanh", recurrent_dropout=0.2, return_sequences=True
        )(x)
        x = MaxPooling3D((1, 2, 2), padding="same")(x)
        x = TimeDistributed(Dropout(0.2))(x)
        x = ConvLSTM2D(
            16, (3, 3), activation="tanh", recurrent_dropout=0.2, return_sequences=True
        )(x)
        x = MaxPooling3D((1, 2, 2), padding="same")(x)
        x = TimeDistributed(Dropout(0.2))(x)
        x = ConvLSTM2D(
            32, (3, 3), activation="tanh", recurrent_dropout=0.2, return_sequences=True
        )(x)
        x = MaxPooling3D((1, 2, 2), padding="same")(x)
        x = Flatten()(x)
        out = Dense(len(CLASSES_LIST), activation="softmax")(x)
        model = Model(inp, out)

        # Compile
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # Early stopping
        es = EarlyStopping(
            monitor="val_loss",
            patience=params["early_stop_patience"],
            mode="min",
            restore_best_weights=True,
        )

        # Train
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            shuffle=True,
            callbacks=[es],
        )

        # Log metrics per epoch
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

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        # Log the model (includes weights)
        mlflow.keras.log_model(model, artifact_path="model")

        # Log loss/validation curves as figures
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


if __name__ == "__main__":
    main()
