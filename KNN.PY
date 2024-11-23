import cv2
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Citirea datelor
df_train = pd.read_csv("./Datasets/realistic-image-classification/train.csv")
df_validation = pd.read_csv("./Datasets/realistic-image-classification/validation.csv")

file_names_train = df_train["image_id"]
file_names_validation = df_validation["image_id"]


y_train = df_train["label"]
x_train = np.array([cv2.cvtColor(cv2.imread("./Datasets/realistic-image-classification/train/" + x + '.png'),cv2.COLOR_BGR2RGB) for x in file_names_train])


x_train = x_train / 255

y_validation = df_validation["label"]
x_validation = np.array([cv2.cvtColor(cv2.imread("./Datasets/realistic-image-classification/validation/" + x + '.png'),cv2.COLOR_BGR2RGB) for x in file_names_validation])
x_validation = x_validation / 255


x_train = x_train.reshape(x_train.shape[0],-1)
x_validation = x_validation.reshape(x_validation.shape[0],-1)

# KNN
clf = KNeighborsClassifier(n_neighbors = 17,metric= "manhattan")
print("Fitting")
clf.fit(x_train,y_train)
print("Predicting")
predictions = clf.predict(x_validation)

print(accuracy_score(y_validation,predictions))
