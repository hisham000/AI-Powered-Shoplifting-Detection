# Standard libraries
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# TensorFlow and Keras
from sklearn.metrics import (  # type: ignore[import]
    confusion_matrix,
    precision_recall_fscore_support,
)

# Scikit-learn
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

dim = 64

IMAGE_HEIGHT = dim
IMAGE_WIDTH = dim
DESTINATION_ROOT = "data-processing/dataset"

# Specify the number of frames that will be fed to the Neural Network
SEQUENCE_LENGTH = 30
CLASSES_LIST = ["0", "1"]


def frame_extraction(video_path):

    frame_list = []

    video_capture = cv2.VideoCapture(video_path)
    video_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frame_window = max(int(video_frame_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):

        # set the current frame position of the video
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frame_window)

        success, frame = video_capture.read()

        # check if the frame is successfully setup or not
        if not success:
            break

        # Resize the frame into fixed size height and width
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalise the given frame
        normalized_frame = resized_frame / 255

        # Append the normalised frame into  frame list
        frame_list.append(normalized_frame)

        # release the video capture object,
    video_capture.release()

    return frame_list


def create_dataset():
    features = []
    labels = []
    video_file_paths = []
    for cls_index, cls in enumerate(CLASSES_LIST):
        # Get the names of list of video files present in specific class name directory
        file_list = os.listdir(os.path.join(DESTINATION_ROOT, cls))

        for file in file_list:
            video_file_path = os.path.join(DESTINATION_ROOT, cls, file)
            frames = frame_extraction(video_file_path)
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(cls_index)
                video_file_paths.append(video_file_path)

    # Converting list into numpy array

    features = np.asarray(features)
    labels = np.array(labels)
    return features, labels, video_file_paths


# create a dataset
features, labels, video_file_paths = create_dataset()

one_hot_encoded_labels = to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(
    features, one_hot_encoded_labels, test_size=0.25, shuffle=True
)

# Creating Neural Network

input_layer = Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

convlstm_1 = ConvLSTM2D(
    filters=4,
    kernel_size=(3, 3),
    activation="tanh",
    data_format="channels_last",
    recurrent_dropout=0.2,
    return_sequences=True,
)(input_layer)
pool1 = MaxPooling3D(pool_size=(1, 2, 2), padding="same", data_format="channels_last")(
    convlstm_1
)
timedistributed_1 = TimeDistributed(Dropout(0.2))(pool1)

convlstm_2 = ConvLSTM2D(
    filters=8,
    kernel_size=(3, 3),
    activation="tanh",
    data_format="channels_last",
    recurrent_dropout=0.2,
    return_sequences=True,
)(timedistributed_1)
pool2 = MaxPooling3D(pool_size=(1, 2, 2), padding="same", data_format="channels_last")(
    convlstm_2
)
timedistributed_2 = TimeDistributed(Dropout(0.2))(pool2)

convlstm_3 = ConvLSTM2D(
    filters=16,
    kernel_size=(3, 3),
    activation="tanh",
    data_format="channels_last",
    recurrent_dropout=0.2,
    return_sequences=True,
)(timedistributed_2)
pool3 = MaxPooling3D(pool_size=(1, 2, 2), padding="same", data_format="channels_last")(
    convlstm_3
)
timedistributed_3 = TimeDistributed(Dropout(0.2))(pool3)

convlstm_4 = ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    activation="tanh",
    data_format="channels_last",
    recurrent_dropout=0.2,
    return_sequences=True,
)(timedistributed_3)
pool4 = MaxPooling3D(pool_size=(1, 2, 2), padding="same", data_format="channels_last")(
    convlstm_4
)

flatten = Flatten()(pool4)

output = Dense(units=len(CLASSES_LIST), activation="softmax")(flatten)


model: Model = Model(input_layer, output)

model.summary()

early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=10, mode="min", restore_best_weights=True
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=4,
    shuffle=True,
    validation_split=0.2,
    callbacks=[early_stopping_callback],
)

model.save("train/models/model.h5")

loss, accuracy = model.evaluate(x_test, y_test)

print("Loss : ", loss)
print("Accuracy : ", accuracy)

# Plotting loss curve for training and validation set


def save_curve(model_training_history, metric_name_1, metric_name_2, plot_name):

    metric1 = model_training_history.history[metric_name_1]
    metric2 = model_training_history.history[metric_name_2]
    plt.plot(metric1, color="blue", label=metric_name_1)
    plt.plot(metric2, color="red", label=metric_name_2)
    plt.title(str(plot_name))
    plt.legend()
    plt.savefig(f"train/plots/{plot_name}.png")


save_curve(history, "loss", "val_loss", "Total loss vs Total validation loss")

predictions = model.predict(x_test)

# Assuming predictions are in probability form and you need to convert them to binary labels
binary_predictions = (predictions > 0.5).astype("int32")

# Calculate precision, recall, and F1 score
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, binary_predictions, average=None
)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)  # Assuming test_Y is one-hot encoded
conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.savefig("train/plots/confusion_matrix.png")
