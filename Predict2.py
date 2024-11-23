import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import keras.api._v2.keras as keras
import tensorflow as tf
import cv2
from copy import deepcopy
import torch
import torch.nn as nn
from tqdm import tqdm

class ResBlock(nn.Module):
    def init(self,in_chan,out_chan,proj = False,pool = False):

        super(ResBlock,self).init()

        self.proj = None

        if proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chan,out_chan,(1,1)),
                nn.BatchNorm2d(out_chan)
            )

        if pool:
            self.block = nn.Sequential(
                nn.Conv2d(in_chan,out_chan,(3,3),stride= 2,padding= 1),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chan,out_chan,(3,3),padding= 1),
                nn.BatchNorm2d(out_chan)
            )
            self.proj = nn.Sequential(
                nn.Conv2d(in_chan,out_chan,(1,1),stride = 2),
                nn.BatchNorm2d(out_chan)
            )

        else:
            self.block =  nn.Sequential(
                nn.Conv2d(in_chan,out_chan,(3,3),padding= 1),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chan,out_chan,(3,3),padding= 1),
                nn.BatchNorm2d(out_chan)
            )

    def forward(self,x):
        id = x
        out = self.block(x)

        if self.proj is not None:
            id = self.proj(x)

        out += id
        out = nn.functional.relu(out)
        return out

def get_bits(number):
    bits = []
    for i, c in enumerate(bin(number)[:1:-1], 1):
        if c == '1':
            bits.append(i)
    return bits


# Citirea datelor
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

# Incarcarea modelelor
model1 = keras.models.load_model("./Kaggle/CNN1.Kaggle74.9_76.38fara_strat512")
model2 = torch.load("./Kaggle/CNN.Kaggle74.13.pth", map_location=torch.device('cpu'))
model3 = torch.load("./Kaggle/CNN11.Kaggle74.03.pth", map_location=torch.device('cpu'))
model4 = torch.load("./Kaggle/CNN11.Kaggle74.26.pth", map_location=torch.device('cpu'))
model5 = torch.load("./Kaggle/CNN11.Kaggle74.76.pth", map_location=torch.device('cpu'))
model6 = torch.load("./Kaggle/CNN11.Kaggle74.93.pth", map_location=torch.device('cpu'))
model7 = torch.load("./Kaggle/CNN.Kaggle73.96.pth", map_location=torch.device('cpu'))
model8 = torch.load("./Kaggle/CNN12.Kaggle74.73.pth", map_location=torch.device('cpu'))
model9 = torch.load("./Kaggle/CNN11.Kaggle73.69.pth", map_location=torch.device('cpu'))
model10 = torch.load("./Kaggle/CNN13.Kaggle74.5.pth", map_location=torch.device('cpu'))
model11 = torch.load("./Kaggle/CNN13.Kaggle74.06.pth", map_location=torch.device('cpu'))
model12 = torch.load("./Kaggle/CNN13.Kaggle74.83.pth", map_location=torch.device('cpu'))
model13 = torch.load("./Kaggle/CNN13.Kaggle75.63.pth", map_location=torch.device('cpu'))
model14 = torch.load("./Kaggle/CNN13.Kaggle74.90.pth", map_location=torch.device('cpu'))
model15 = torch.load("./Kaggle/CNN14.Kaggle74.66.pth", map_location=torch.device('cpu'))
model16 = torch.load("./Kaggle/ResNetDIY.Kaggle76.63.pth", map_location=torch.device('cpu'))

def make_prediction(index,x_test):
    # Creez predictiile pentru fiecare model
    predictions1 = np.array(model1.predict(x_test))
    with torch.no_grad():
        x_test = torch.tensor(x_test).permute(0,3,1,2).float()
        predictions2 = np.array(model2(x_test))
        predictions3 = np.array(model3(x_test))
        predictions4 = np.array(model4(x_test))
        predictions5 = np.array(model5(x_test))
        predictions6 = np.array(model6(x_test))
        predictions7 = np.array(model7(x_test))
        predictions8 = np.array(model8(x_test))
        predictions9 = np.array(model9(x_test))
        predictions10 = np.array(model10(x_test))
        predictions11 = np.array(model11(x_test))
        predictions12 = np.array(model12(x_test))
        predictions13 = np.array(model13(x_test))
        predictions14 = np.array(model14(x_test))
        predictions15 = np.array(model15(x_test))
        predictions16 = np.array(model16(x_test))

    
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

    pred = np.array(pred)
    bits = np.array(get_bits(index)) - 1
    probs = np.array([0.749,0.7413,0.7403,0.7426,0.7476,0.7493,0.7396,0.7473,0.7369,0.745,0.7406,0.7483,0.7563,0.749,0.7466,0.7663])
    sum = np.sum(probs[bits])
    
    # Creez predictiile finale
    predictions = np.sum(((probs[bits]/sum)[:,np.newaxis,np.newaxis]) * pred[bits] ,axis = 0)

    predictions = np.argmax(predictions, axis = 1)
    return predictions


def get_highest(x_validation,y_validation):
    # Creez predictiile pentru fiecare model
    y_pred1 = np.array(model1.predict(x_validation))
    with torch.no_grad():
        x_validation = torch.tensor(x_validation).permute(0,3,1,2).float()
        y_pred2 = np.array(model2(x_validation))
        y_pred3 = np.array(model3(x_validation))
        y_pred4 = np.array(model4(x_validation))
        y_pred5 = np.array(model5(x_validation))
        y_pred6 = np.array(model6(x_validation))
        y_pred7 = np.array(model7(x_validation))
        y_pred8 = np.array(model8(x_validation))
        y_pred9 = np.array(model9(x_validation))
        y_pred10 = np.array(model10(x_validation))
        y_pred11 = np.array(model11(x_validation))
        y_pred12 = np.array(model12(x_validation))
        y_pred13 = np.array(model13(x_validation))
        y_pred14 = np.array(model14(x_validation))
        y_pred15 = np.array(model15(x_validation))
        y_pred16 = np.array(model16(x_validation))

    pred = []
    pred.append(y_pred1)
    pred.append(y_pred2)
    pred.append(y_pred3)
    pred.append(y_pred4)
    pred.append(y_pred5)
    pred.append(y_pred6)
    pred.append(y_pred7)
    pred.append(y_pred8)
    pred.append(y_pred9)
    pred.append(y_pred10)
    pred.append(y_pred11)
    pred.append(y_pred12)
    pred.append(y_pred13)
    pred.append(y_pred14)
    pred.append(y_pred15)
    pred.append(y_pred16)
    pred = np.array(pred)

    probs = np.array([0.749,0.7413,0.7403,0.7426,0.7476,0.7493,0.7396,0.7473,0.7369,0.745,0.7406,0.7483,0.7563,0.749,0.7466,0.7663])
    
    # Calculez acuratetea pentru fiecare combinatie de modele
    acc = []
    for i in tqdm(range(1,2**len(probs))):
        bits = get_bits(i)
        bits = np.array(bits) - 1
        predictions = np.sum(((probs[bits]/np.sum(probs[bits]))[:,np.newaxis,np.newaxis]) * pred[bits] ,axis = 0)
        predictions = np.argmax(predictions, axis = 1)
        acc.append((accuracy_score(y_validation, predictions),i))

    # Sortez acuratetea si afisez primele 10
    acc.sort(reverse = True)
    print(acc[:10])

def get_highest_equal_weight(x_validation,y_validation):
    # Creez predictiile pentru fiecare model
    y_pred1 = np.array(model1.predict(x_validation))
    with torch.no_grad():
        x_validation = torch.tensor(x_validation).permute(0,3,1,2).float()
        y_pred2 = np.array(model2(x_validation))
        y_pred3 = np.array(model3(x_validation))
        y_pred4 = np.array(model4(x_validation))
        y_pred5 = np.array(model5(x_validation))
        y_pred6 = np.array(model6(x_validation))
        y_pred7 = np.array(model7(x_validation))
        y_pred8 = np.array(model8(x_validation))
        y_pred9 = np.array(model9(x_validation))
        y_pred10 = np.array(model10(x_validation))
        y_pred11 = np.array(model11(x_validation))
        y_pred12 = np.array(model12(x_validation))
        y_pred13 = np.array(model13(x_validation))
        y_pred14 = np.array(model14(x_validation))
        y_pred15 = np.array(model15(x_validation))
        y_pred16 = np.array(model16(x_validation))

    pred = []
    pred.append(y_pred1)
    pred.append(y_pred2)
    pred.append(y_pred3)
    pred.append(y_pred4)
    pred.append(y_pred5)
    pred.append(y_pred6)
    pred.append(y_pred7)
    pred.append(y_pred8)
    pred.append(y_pred9)
    pred.append(y_pred10)
    pred.append(y_pred11)
    pred.append(y_pred12)
    pred.append(y_pred13)
    pred.append(y_pred14)
    pred.append(y_pred15)
    pred.append(y_pred16)
    pred = np.array(pred)

    # Calculez acuratetea pentru fiecare combinatie de modele
    acc = []
    for i in tqdm(range(1,2**len(pred))):
        bits = get_bits(i)
        bits = np.array(bits) - 1
        predictions = np.sum(pred[bits] ,axis = 0)
        predictions = np.argmax(predictions, axis = 1)
        acc.append((accuracy_score(y_validation, predictions),i))

    # Sortez acuratetea si afisez primele 10
    acc.sort(reverse = True)
    print(acc[:10])

#get_highest_equal_weight(x_validation,y_validation)

#get_highest(x_validation,y_validation)

# Salvez predictiile in fisier
predictions = make_prediction(55935,x_test)
df_test["label"] = predictions
df_test.to_csv("submission22.csv",index = False)




