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
import time

# Citirea datelor de antrenare si validare
df_train = pd.read_csv("./Datasets/realistic-image-classification/train.csv")
df_validation = pd.read_csv("./Datasets/realistic-image-classification/validation.csv")

file_names_train = df_train["image_id"]
file_names_validation = df_validation["image_id"]

y_train = torch.tensor(df_train["label"].values)
x_train = torch.stack([transforms.ToTensor()(cv2.cvtColor(cv2.imread(f"./Datasets/realistic-image-classification/train/{x}.png"), cv2.COLOR_BGR2RGB)) for x in file_names_train])

y_validation = torch.tensor(df_validation["label"].values)
x_validation = torch.stack([transforms.ToTensor()(cv2.cvtColor(cv2.imread(f"./Datasets/realistic-image-classification/validation/{x}.png"), cv2.COLOR_BGR2RGB)) for x in file_names_validation])

# Data Augmentation
rotated1 = torch.tensor(torch.rot90(x_train.permute(0, 2, 3, 1), k=1, dims=(1, 2)).permute(0, 3, 1, 2))
rotated2 = torch.tensor(torch.rot90(x_train.permute(0, 2, 3, 1), k=2, dims=(1, 2)).permute(0, 3, 1, 2))
rotated3 = torch.tensor(torch.rot90(x_train.permute(0, 2, 3, 1), k=3, dims=(1, 2)).permute(0, 3, 1, 2))

x_train = torch.cat((x_train, rotated1, rotated2, rotated3))


initial = deepcopy(y_train)

y_train = torch.cat((y_train, initial, initial, initial))



# Definirea DataLoader-elor
train_dataset = TensorDataset(x_train,y_train)
train_data_loader = DataLoader(train_dataset,batch_size= 64 , shuffle= True)

validation_dataset = TensorDataset(x_validation,y_validation)
validation_data_loader = DataLoader(validation_dataset,batch_size=64,shuffle=True)

lr = 0.001
index = 1
while True:
    # Definirea modelului
    model = nn.Sequential(
        #80x80x3
        nn.Conv2d(3, 32, (3, 3)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        #78x78x32
        nn.Conv2d(32, 32, (3, 3)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        #76x76x32
        nn.Conv2d(32, 32, (3, 3)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        #74x74x32
        nn.Conv2d(32, 64, (3, 3), stride=(2, 2)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        #36x36x64
        nn.Conv2d(64, 64, (3, 3)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        #34x34x64
        nn.Conv2d(64, 64, (3, 3)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        #32x32x64
        nn.Conv2d(64, 64, (3, 3)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        #30x30x64
        nn.Conv2d(64, 128, (3, 3), stride=(2, 2)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        #14x14x128
        nn.Conv2d(128, 128, (3, 3)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        #12x12x128
        nn.Conv2d(128, 256, (3, 3), stride=(2, 2)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        #5x5x256
        nn.Conv2d(256, 256, (3, 3) ),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        #3x3x256
        nn.Conv2d(256, 256, (3, 3) ),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        #1x1x256
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Dropout(0.5),
        nn.Flatten(),
        nn.Linear(256, 3),
    )

    # Compilarea modelului
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda")
    model.to(device)

    x_train = x_train.permute(0, 3, 1, 2)
    x_validation = x_validation.permute(0, 3, 1, 2)
    x_train = torch.swapaxes(x_train,1,2)
    x_validation = torch.swapaxes(x_validation,1,2)


    
    # Antrenarea modelului
    for epoch in range(25):
        print(epoch+1)
        start = time.time()
        for x_batch, y_batch in train_data_loader:
            x_batch, y_batch = x_batch.cuda() , y_batch.cuda()
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
        end = time.time()
        print(end-start)

    # Evaluarea modelului
    model.eval()

    with torch.no_grad():
        x_validation = x_validation.cuda()
        y_pred = model(x_validation)
        x_validation = x_validation.cpu()
        y_validation = y_validation.cuda()
        loss = loss_function(y_pred, y_validation)
        print(loss)
        accuracy = torch.sum(torch.argmax(y_pred,1) == y_validation) / len(x_validation)
        if accuracy.item() >= 0.745:
            torch.save(model, f"{index}aCNN13.Kaggle{accuracy.item()*100}.pth")
            index += 1
        print(accuracy)
        model.cpu()
        y_validation = y_validation.cpu()
        del model
        torch.cuda.empty_cache()

    



    