import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader , TensorDataset
from copy import deepcopy
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

# Definirea Blocului Residual
class ResBlock(nn.Module):
    def __init__(self,in_chan,out_chan,proj = False,pool = False):

        super(ResBlock,self).__init__()

        self.proj = None

        if proj:
            # Proiectia spatiala a identitatii pentru a se potrivi cu dimensiunile blocului
            self.proj = nn.Sequential(
                nn.Conv2d(in_chan,out_chan,(1,1)),
                nn.BatchNorm2d(out_chan)
            )

        if pool:
            # Pooling-ul spatial pentru a reduce dimensiunile blocului
            self.block = nn.Sequential(
                nn.Conv2d(in_chan,out_chan,(3,3),stride= 2,padding= 1),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chan,out_chan,(3,3),padding= 1),
                nn.BatchNorm2d(out_chan)
            )
            # Proiectia spatiala a identitatii pentru a se potrivi cu dimensiunile blocului
            self.proj = nn.Sequential(
                nn.Conv2d(in_chan,out_chan,(1,1),stride = 2),
                nn.BatchNorm2d(out_chan)
            )

        else:
            # Blocul fara pooling spatial
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

        # Adaugarea identitatii la iesirea blocului
        if self.proj is not None:
            id = self.proj(x)

        out += id
        out = nn.functional.relu(out)
        return out

# Citirea datelor
df_train = pd.read_csv("./Datasets/realistic-image-classification/train.csv")
df_validation = pd.read_csv("./Datasets/realistic-image-classification/validation.csv")

file_names_train = df_train["image_id"]
file_names_validation = df_validation["image_id"]

y_train = torch.tensor(df_train["label"].values)
x_train = torch.stack([transforms.ToTensor()(cv2.cvtColor(cv2.imread(f"./Datasets/realistic-image-classification/train/{x}.png"), cv2.COLOR_BGR2RGB)) for x in file_names_train])

y_validation = torch.tensor(df_validation["label"].values)
x_validation = torch.stack([transforms.ToTensor()(cv2.cvtColor(cv2.imread(f"./Datasets/realistic-image-classification/validation/{x}.png"), cv2.COLOR_BGR2RGB)) for x in file_names_validation])

# Augmentarea datelor
rotated1 = torch.tensor(torch.rot90(x_train.permute(0, 2, 3, 1), k=1, dims=(1, 2)).permute(0, 3, 1, 2))
rotated2 = torch.tensor(torch.rot90(x_train.permute(0, 2, 3, 1), k=2, dims=(1, 2)).permute(0, 3, 1, 2))
rotated3 = torch.tensor(torch.rot90(x_train.permute(0, 2, 3, 1), k=3, dims=(1, 2)).permute(0, 3, 1, 2))

x_train = torch.cat((x_train, rotated1, rotated2, rotated3))

initial = deepcopy(y_train)

y_train = torch.cat((y_train, initial, initial, initial))


# Definirea datelor de antrenare si de validare
train_dataset = TensorDataset(x_train,y_train)
train_data_loader = DataLoader(train_dataset,batch_size= 64 , shuffle= True)

validation_dataset = TensorDataset(x_validation,y_validation)
validation_data_loader = DataLoader(validation_dataset,batch_size=64,shuffle=True)

lr = 0.001
index = 1
while True:
    # Definirea modelului
    model = nn.Sequential(ResBlock(3,32,True),
                        ResBlock(32,32),
                        ResBlock(32,32),
                        ResBlock(32,64,True,True),
                        ResBlock(64,64),
                        ResBlock(64,64),
                        ResBlock(64,128,True,True),
                        ResBlock(128,128),
                        ResBlock(128,128),
                        ResBlock(128,256,True,True),
                        ResBlock(256,256),
                        ResBlock(256,256),
                        nn.AdaptiveAvgPool2d((1, 1)), 
                        nn.Flatten(),
                        nn.Linear(256, 3),
                        )
    # Definirea functiei de loss si a optimizatorului
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda")
    model.to(device)
    # Antrenarea modelului
    for epoch in tqdm(range(50)):
        start = time.time()
        for x_batch, y_batch in train_data_loader:
            x_batch, y_batch = x_batch.cuda() , y_batch.cuda()
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
       # end = time.time()
        #print(end-start)

    model.eval()

    # Validarea modelului
    with torch.no_grad():
        x_validation = x_validation.cuda()
        y_pred = model(x_validation)
        x_validation = x_validation.cpu()
        y_validation = y_validation.cuda()
        loss = loss_function(y_pred, y_validation)
        print(loss)
        accuracy = torch.sum(torch.argmax(y_pred,1) == y_validation) / len(x_validation)
        if accuracy.item() >= 0.745:
            torch.save(model, f"{index}ResNetDIY.Kaggle{accuracy.item()*100}.pth")
            index += 1
        print(accuracy)
        model.cpu()
        y_validation = y_validation.cpu()
        del model
        torch.cuda.empty_cache()