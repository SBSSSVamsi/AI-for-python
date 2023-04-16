import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

data_dir = "./data/weeds"
categories = ["CELOSIA ARGENTEA L", "CROWFOOT GRASS", "PURPLE CHLORIS"]

data = []


def make_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = np.array(img, dtype=np.float32)
                data.append([img, label])
            except Exception as e:
                pass
    print(len(data))
    pik = open("data.pkl", "wb")
    pickle.dump(data, pik)
    pik.close()


make_data()


def load_data():
    pik = open("data.pkl", "rb")
    data = pickle.load(pik)
    pik.close()
    np.random.shuffle(data)
    feature = []
    labels = []
    for img, label in data:
        feature.append(img)
        labels.append(label)
    feature = np.array(feature, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    feature = feature / 255.0

    return [feature, labels]
