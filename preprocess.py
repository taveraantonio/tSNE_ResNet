import os
import numpy as np
from idda_utils import load_dataset
import cv2
import seaborn as sns


sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
RS = 123

DATASET_PATH = '/media/tavera/vandal-hd1/IDDA'
SCENARIO_PATH = '/media/tavera/vandal-hd1/Scenarios/'
# SCENARIOS = ["T01_HRN_A", "T07_HRN_A", "T01_HRN_B"]  #1 or more than 1 scenarios
SCENARIOS = os.listdir(SCENARIO_PATH)  # uncomment this if you want to analyze all the scenarios in the scenario path folder
NUM_IMAGES = 2000
RESIZE_X = 128
RESIZE_Y = 72
FIGSIZE = 30
PRINT_TIME = 100
USE_PCA = True  # true for using first pca to extract relevant feature and than tsne
SAVE_NAME = "allscenarios_pca"
FILE = "./result/Scenarios_PCA.txt"
SAVE_PATH = "/media/tavera/vandal-hd1/Resize_IDDA/"


def preprocess():
    images = []
    labels = np.empty(len(SCENARIOS) * NUM_IMAGES, dtype=np.uint8)
    added = 0

    for label, scenario in enumerate(SCENARIOS):
        scenario_path = os.path.join(SCENARIO_PATH, scenario)
        train_set, _, _ = load_dataset(split_dir=scenario_path)
        # select only the first num_images
        train_set = train_set[:NUM_IMAGES]
        for i, image in enumerate(train_set):
            file_path = os.path.join(DATASET_PATH, "RGB", image.rstrip())
            if os.path.isfile(file_path) and ".png" in file_path:
                train_path = os.path.join(DATASET_PATH, "RGB", image.rstrip())
                image_read = cv2.imread(train_path, cv2.IMREAD_COLOR)
                image_read = cv2.resize(image_read, (RESIZE_X, RESIZE_Y), interpolation=cv2.INTER_NEAREST).reshape(-1)
                save_p = SAVE_PATH + str(label) + "_" + image.rstrip()
                cv2.imwrite(save_p, image_read)
                # images = images + image
            else:
                print("error")

    return

# MAIN
preprocess()
