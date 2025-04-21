import os
import shutil

import kagglehub  # type: ignore[import]
import pandas as pd


def load_data(destination_root="/data"):
    """
    Load and prepare the shoplifting detection dataset.

    Args:
        destination_root (str): Root directory to store the dataset. Default is "/data".

    Returns:
        None
    """
    # Set up paths based on environment
    DESTINATION_ROOT = destination_root

    # Download latest version
    path = kagglehub.dataset_download("mateohervas/dcsass-dataset")

    print("Path to dataset files:", path)

    ROOT_DIR = f"{path}/DCSASS Dataset/Shoplifting"
    DESTINATION_PATH_0 = f"{DESTINATION_ROOT}/0"
    DESTINATION_PATH_1 = f"{DESTINATION_ROOT}/1"
    FOLDER_NAMES = ["0", "1"]

    # Create destination directories if they don't exist
    if not os.path.exists(DESTINATION_ROOT):
        os.makedirs(DESTINATION_ROOT, exist_ok=True)
        print(f"Created directory: {DESTINATION_ROOT}")

    for folder_name in FOLDER_NAMES:
        folder_path = os.path.join(DESTINATION_ROOT, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created directory: {folder_path}")

    # Check if directories are already populated
    if (
        os.path.exists(DESTINATION_PATH_0)
        and os.path.exists(DESTINATION_PATH_1)
        and len(os.listdir(DESTINATION_PATH_0)) > 0
        and len(os.listdir(DESTINATION_PATH_1)) > 0
    ):
        print("Data directories already populated. Skipping data download.")
        print(f"Files in {DESTINATION_PATH_0}: {len(os.listdir(DESTINATION_PATH_0))}")
        print(f"Files in {DESTINATION_PATH_1}: {len(os.listdir(DESTINATION_PATH_1))}")
        return  # Return from function instead of sys.exit(0)

    # Continue with dataset processing
    dataset = pd.read_csv(f"{path}/DCSASS Dataset/Labels/Shoplifting.csv")

    # the datasets column names are also the part of the dataset
    # first we will append that data into our dataframe and then rename the columns
    data = [dataset.columns[0], dataset.columns[1], int(dataset.columns[2])]

    dataset.loc[len(dataset)] = data

    dataset.rename(
        columns={
            "Shoplifting001_x264_0": "clipname",
            "Shoplifting": "Shoplifting",
            "0": "Action",
        },
        inplace=True,
    )

    directories = os.listdir(ROOT_DIR)

    # Copy files to their respective directories
    for dir in directories:
        for d in os.listdir(os.path.join(ROOT_DIR, dir)):
            row = dataset.loc[dataset["clipname"] == d[:-4]]
            if row["Action"].iloc[0] == 0:
                shutil.copy(
                    os.path.join(ROOT_DIR, dir, d), os.path.join(DESTINATION_PATH_0, d)
                )
            else:
                shutil.copy(
                    os.path.join(ROOT_DIR, dir, d), os.path.join(DESTINATION_PATH_1, d)
                )

    print("Files copied successfully.")
