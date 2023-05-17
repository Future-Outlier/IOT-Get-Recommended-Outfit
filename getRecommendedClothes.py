import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import cv2
from sklearn.metrics.pairwise import cosine_similarity

# Read the configuration file
config_file = "config.json"  # Adjust the filename if needed
with open(config_file) as file:
    config = json.load(file)

# Retrieve the dataset path and combination JSON path from the config
DATASET_PATH = config["DATASET_PATH"]
COMBINATION_JSON_PATH = config["COMBINATION_JSON_PATH"]

# Read the combination JSON file
with open(COMBINATION_JSON_PATH) as file:
    data = json.load(file)

# Read the data.csv file
df = pd.read_csv(os.path.join(DATASET_PATH, "preprocessedData", "data.csv"))
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)

# Define the plot_figures function
def plot_figures(figures, nrows=1, ncols=1, figsize=(12, 12)):
    """Plot a dictionary of figures."""
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()

# Define the img_path function
def img_path(img):
    return os.path.join(DATASET_PATH, "images", img)

# Define the load_image function
def load_image(img, resized_fac=0.1):
    img = cv2.imread(img_path(img))
    w, h, _ = img.shape
    resized = cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_AREA)
    return resized

# Read the embeddings CSV file
df_embs = pd.read_csv(os.path.join(DATASET_PATH, "preprocessedData", "embedding.csv"))
embs_list = df_embs.values.tolist()

# Define the getSimilarList function
def getSimilarList(pickIndex=5, topN=5):
    scoreAndId = []
    for i in range(len(embs_list)):
        if i == pickIndex:
            continue
        scoreAndId.append([cosine_similarity([embs_list[pickIndex]], [embs_list[i]]), i])
    sorted_list = sorted(scoreAndId, key=lambda x: x[0][0], reverse=True)
    pickList = [sorted_list[i][1] for i in range(topN)]
    return pickList

while True:
    SelectedType = []
    Season = ["Spring", "Summer", "Fall"]
    Purpose = ["Formal", "Sports", "Casual"]
    for i in range(2):
        if i == 0:
            x = int(input("Pick a Season: 1.Spring, 2.Summer, 3.Fall\n"))
            SelectedType.append(Season[x - 1])
        elif i == 1:
            x = int(input("Pick a Purpose: 1.Formal, 2.Sports, 3.Casual\n"))
            SelectedType.append(Purpose[x - 1])
    print("Finish Selected Type:", SelectedType)

    figures = {}
    SelectedImgId = []

    for part, imgList in data["Men"][SelectedType[0]][SelectedType[1]].items():
        key = "Query-" + part
        imgId = str(imgList[3]) + ".jpg"
        value = load_image(str(imgList[3]) + ".jpg")
        figures[key] = value

        pickIndex = df[df['image'] == imgId].index[0]
        recommend_clothes = getSimilarList(pickIndex, topN=1)
        for ID in recommend_clothes:
            figures["Recommended-" + part] = load_image(df.iloc[ID].image)

    plot_figures(figures, 3, 2, (8, 8))
