import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import keras.api._v2.keras as keras
import tensorflow as tf
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import optuna
from copy import deepcopy
import torch

def get_bits(number):
    bits = []
    for i, c in enumerate(bin(number)[:1:-1], 1):
        if c == '1':
            bits.append(i)
    return bits

#Citirea datelor
df_train = pd.read_csv("./Datasets/realistic-image-classification/train.csv")
df_validation = pd.read_csv("./Datasets/realistic-image-classification/validation.csv")
df_test = pd.read_csv("./Datasets/realistic-image-classification/test.csv")

file_names_test = df_test["image_id"]
x_test = np.array([cv2.cvtColor(cv2.imread("./Datasets/realistic-image-classification/test/" + x + '.png'),cv2.COLOR_BGR2RGB) for x in file_names_test])
x_test = x_test / 255
file_names_train = df_train["image_id"]
y_train = df_train["label"]

x_train = np.array([cv2.cvtColor(cv2.imread("./Datasets/realistic-image-classification/train/" + x + '.png'),cv2.COLOR_BGR2RGB) for x in file_names_train])
x_train = x_train / 255


file_names_validation = df_validation["image_id"]
y_validation = df_validation["label"]

x_validation = np.array([cv2.cvtColor(cv2.imread("./Datasets/realistic-image-classification/validation/" + x + '.png'),cv2.COLOR_BGR2RGB) for x in file_names_validation])
x_validation = x_validation / 255

#Incarcarea modelelor
model1 = keras.models.load_model("./Kaggle/CNN.Kaggle72.6/76.6fara_strat_512")
model2 = keras.models.load_model("./Kaggle/CNN.Kaggle72.37/77.49")
model3 = keras.models.load_model("./Kaggle/CNN1.Kaggle71.33_76.86")
model4 = keras.models.load_model("./Kaggle/CNN.Kaggle71.23/76.97farastrat512")
model5 = keras.models.load_model("./Kaggle/CNN1.Kaggle71.00/76.74")
model6 = keras.models.load_model("./Kaggle/CNN1.Kaggle74.9_76.38fara_strat512")
model7 = keras.models.load_model("./Kaggle/CNN1.Kaggle71.8_76.25fara512")
model8 = keras.models.load_model("./Kaggle/CNN.Kaggle71.57")
model9 = keras.models.load_model("./Kaggle/CNN.Kaggle73.5cu_strat_256")
model10 = keras.models.load_model("./Kaggle/CNN.Kaggle72.43cu_strat_256")
model11 = keras.models.load_model("./Kaggle/CNN.Kaggle72.2cu_strat_256")
model12 = torch.load("./Kaggle/CNNShape72.93.pth" , map_location = torch.device('cpu'))
model13 = torch.load("./Kaggle/CNNShape72.73.pth" , map_location = torch.device('cpu'))
model14 = torch.load("./Kaggle/CNNShape72.69.pth" , map_location = torch.device('cpu'))
model15 = torch.load("./Kaggle/CNN.Kaggle74.13.pth" , map_location = torch.device('cpu'))
model16 = torch.load("./Kaggle/CNN.Kaggle73.96.pth" , map_location = torch.device('cpu'))
model17 = torch.load("./Kaggle/CNN.Kaggle73.6.pth" , map_location = torch.device('cpu'))
model18 = torch.load("./Kaggle/CNN.Kaggle73.2.pth" , map_location = torch.device('cpu'))
model19 = torch.load("./Kaggle/CNN.Kaggle72.66.pth", map_location = torch.device('cpu'))

def make_prediction(index,x_test):
    #Creem predictiile pentru fiecare model
    predictions1 = np.array(model1.predict(x_test))
    predictions2 = np.array(model2.predict(x_test))
    predictions3 = np.array(model3.predict(x_test))
    predictions4 = np.array(model4.predict(x_test))
    predictions5 = np.array(model5.predict(x_test))
    predictions6 = np.array(model6.predict(x_test))
    predictions7 = np.array(model7.predict(x_test))
    predictions8 = np.array(model8.predict(x_test))
    predictions9 = np.array(model9.predict(x_test))
    predictions10 = np.array(model10.predict(x_test))
    predictions11 = np.array(model11.predict(x_test))
    with torch.no_grad():
        x_test = torch.tensor(x_test).permute(0,3,1,2).float()
        predictions12 = np.array(model12(x_test))
        predictions13 = np.array(model13(x_test))
        predictions14 = np.array(model14(x_test))
        predictions15 = np.array(model15(x_test))
        predictions16 = np.array(model16(x_test))
        predictions17 = np.array(model17(x_test))
        predictions18 = np.array(model18(x_test))
        predictions19 = np.array(model19(x_test))
    
    
    pred = []
    pred.append(predictions1)
    pred.append(predictions2)
    pred.append(predictions3)
    pred.append(predictions4)
    pred.append(predictions5)
    pred.append(predictions6)
    pred.append(predictions7)
    pred.append(predictions8)
    pred.append(predictions9)
    pred.append(predictions10)
    pred.append(predictions11)
    pred.append(predictions12)
    pred.append(predictions13)
    pred.append(predictions14)
    pred.append(predictions15)
    pred.append(predictions16)
    pred.append(predictions17)
    pred.append(predictions18)
    pred.append(predictions19)
    
    pred = np.array(pred)
    bits = np.array(get_bits(index)) - 1
    probs = np.array([0.726,0.7237,0.7133,0.7123,0.71,0.749,0.718,0.7157,0.735,0.7243,0.722,0.7293,0.7273,0.7269,0.7413,0.7396,0.736,0.732,0.7266])
    sum = np.sum(probs[bits])

    #Calculam predictiile
    predictions = np.sum(((probs[bits]/sum)[:,np.newaxis,np.newaxis]) * pred[bits] ,axis = 0)

    predictions = np.argmax(predictions, axis = 1)
    return predictions

def getBestModel(x_validation,y_validation):

    #Calculam predictiile pentru fiecare model
    predictions1 = np.array(model1.predict(x_validation))
    predictions2 = np.array(model2.predict(x_validation))
    predictions3 = np.array(model3.predict(x_validation))
    predictions4 = np.array(model4.predict(x_validation)) 
    predictions5 = np.array(model5.predict(x_validation))
    predictions6 = np.array(model6.predict(x_validation))
    predictions7 = np.array(model7.predict(x_validation))
    predictions8 = np.array(model8.predict(x_validation))
    predictions9 = np.array(model9.predict(x_validation))
    predictions10 = np.array(model10.predict(x_validation))
    predictions11 = np.array(model11.predict(x_validation))
    with torch.no_grad():
        x_validation = torch.tensor(x_validation).permute(0,3,1,2).float()
        predictions12 = np.array(model12(x_validation))
        predictions13 = np.array(model13(x_validation))
        predictions14 = np.array(model14(x_validation))
        predictions15 = np.array(model15(x_validation))
        predictions16 = np.array(model16(x_validation))
        predictions17 = np.array(model17(x_validation))
        predictions18 = np.array(model18(x_validation))
        predictions19 = np.array(model19(x_validation))

    pred = []
    pred.append(predictions1)
    pred.append(predictions2)
    pred.append(predictions3)
    pred.append(predictions4)
    pred.append(predictions5)
    pred.append(predictions6)
    pred.append(predictions7)
    pred.append(predictions8)
    pred.append(predictions9)
    pred.append(predictions10)
    pred.append(predictions11)
    pred.append(predictions12)
    pred.append(predictions13)
    pred.append(predictions14)
    pred.append(predictions15)
    pred.append(predictions16)
    pred.append(predictions17)
    pred.append(predictions18)
    pred.append(predictions19)
    pred = np.array(pred)
    name = ["pred1" , "pred2", "pred3","pred4","pred5","pred6","pred7","pred8","pred9","pred10","pred11","pred12","pred13","pred14","pred15","pred16","pred17","pred18","pred19"]
    probs = np.array([0.726,0.7237,0.7133,0.7123,0.71,0.749,0.718,0.7157,0.735,0.7243,0.722,0.7293,0.7273,0.7269,0.7413,0.7396,0.736,0.732,0.7266])

    # (probs[bits]/np.sum(probs[bits]))[:,np.newaxis,np.newaxis] pentru a pondera 

    #Calculam acuratetea pentru fiecare combinatie de modele
    acc = []
    for i in range(1,2**len(name)):
        print(i)
        bits = get_bits(i)
        bits = np.array(bits) - 1
        predictions = np.sum(((probs[bits]/np.sum(probs[bits]))[:,np.newaxis,np.newaxis]) * pred[bits] ,axis = 0)
        predictions = np.argmax(predictions, axis = 1)
        acc.append((accuracy_score(y_validation, predictions),i))

    #Sortam combinatiile de modele in functie de acuratete
    acc.sort(reverse = True)
    print(acc[:10])


#Salvam predictiile
predictions = make_prediction(314291,x_test)
df_test["label"] = predictions
df_test.to_csv("submission11.csv",index = False)



