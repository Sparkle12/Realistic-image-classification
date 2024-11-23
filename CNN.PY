import cv2
import numpy as np
import pandas as pd
from skimage import io
import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers
from copy import deepcopy
import time

# Citirea datelor de antrenare si validare
df_train = pd.read_csv("./Datasets/realistic-image-classification/train.csv")
df_validation = pd.read_csv("./Datasets/realistic-image-classification/validation.csv")

file_names_train = df_train["image_id"]
file_names_validation = df_validation["image_id"]


y_train = df_train["label"]
x_train = np.array([cv2.cvtColor(cv2.imread("./Datasets/realistic-image-classification/train/" + x + '.png'),cv2.COLOR_BGR2RGB) for x in file_names_train])


y_validation = df_validation["label"]
x_validation = np.array([cv2.cvtColor(cv2.imread("./Datasets/realistic-image-classification/validation/" + x + '.png'),cv2.COLOR_BGR2RGB) for x in file_names_validation])
x_validation = x_validation / 255

# Data Augmentation
rotated1 = np.array(tf.image.rot90(x_train))
rotated2 = np.array(tf.image.rot90(x_train,k = 2))
rotated3 = np.array(tf.image.rot90(x_train, k = 3))

x_train = np.concatenate((x_train,rotated1),axis = 0)
x_train = np.concatenate((x_train,rotated2),axis = 0)
x_train = np.concatenate((x_train,rotated3),axis = 0)

initial = deepcopy(y_train)

y_train = np.concatenate((y_train,initial),axis = 0)
y_train = np.concatenate((y_train,initial),axis = 0)
y_train = np.concatenate((y_train,initial),axis = 0)

x_train = x_train / 255



lr = 0.001
i = 1
while True:

    # Definirea modelului
    model = keras.models.Sequential()
    model.add(layers.Conv2D(16,(3,3), input_shape = (80,80,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(16,(3,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32,(3,3),strides=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32,(3,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64,(3,3),strides=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64,(3,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128,(3,3),strides=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256,(3,3),strides=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256,(3,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.GlobalAveragePooling2D("channels_last"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(3))

    # Compilarea modelului
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate = lr)


    model.compile(optimizer = optimizer,loss = loss_function, metrics = 'accuracy')

    # Antrenarea modelului
    model.fit(x_train,y_train,epochs= 10,batch_size= 64,verbose=1)

    # Evaluarea modelului
    metrics = model.evaluate(x_validation,y_validation,batch_size= 64, verbose = 1, return_dict = True)

    # Salvarea modelului
    if(metrics["accuracy"] > 0.715):
        model.save(f"CNN.Kaggle{metrics['accuracy']*100}" + str(i))
        i += 1
    
    time.sleep(240)
    





