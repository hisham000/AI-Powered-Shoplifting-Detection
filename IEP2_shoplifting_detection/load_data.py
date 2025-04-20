import os
import shutil

import kagglehub  # type: ignore[import]
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("mateohervas/dcsass-dataset")

print("Path to dataset files:", path)


ROOT_DIR = f"{path}/DCSASS Dataset/Shoplifting"
DESTINATION_ROOT = "IEP2_shoplifting_detection/data"
DESTINATION_PATH_0 = f"{DESTINATION_ROOT}/0"
DESTINATION_PATH_1 = f"{DESTINATION_ROOT}/1"
FOLDER_NAMES = ["0", "1"]


if not os.path.exists(DESTINATION_ROOT):
    os.mkdir(DESTINATION_ROOT)
else:
    print("Folder already exists...")
    exit(1)

for folder_name in FOLDER_NAMES:
    if not os.path.exists(os.path.join(DESTINATION_ROOT, folder_name)):
        os.mkdir(os.path.join(DESTINATION_ROOT, folder_name))
    else:
        print("folder already exists...")

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

for dir in directories:
    for d in os.listdir(os.path.join(ROOT_DIR, dir)):
        row = dataset.loc[dataset["clipname"] == d[:-4]]
        if row["Action"].iloc[0] == 0:
            print(ROOT_DIR)
            shutil.copy(
                os.path.join(ROOT_DIR, dir, d), os.path.join(DESTINATION_PATH_0, d)
            )
        else:
            shutil.copy(
                os.path.join(ROOT_DIR, dir, d), os.path.join(DESTINATION_PATH_1, d)
            )

print("Count of number of video clips with 0 and 1 :-")
print(dataset["Action"].value_counts())
print(
    "---------------------------------------------------------------------------------------"
)
print("Video clips present in 0 and 1 :-")
print("no shoplifting count : ", len(os.listdir(DESTINATION_PATH_0)))
print("shoplifting count : ", len(os.listdir(DESTINATION_PATH_1)))
