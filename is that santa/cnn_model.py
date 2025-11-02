import os
from time import time
# from tqdm import tqdm
# import numpy as np
import torch

from toch.nn import Linear,CrossEntropy
from torch.nn import Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import transforms
## device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(degress=10),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

## Create Dataset
TRAIN_ROOT="is that santa/train"
TEST_ROOT=" is that santa/test"
train_ds=ImageFolder(TRAIN_ROOT,transform=tfm)
test_ds=ImageFolder(TEST_ROOT,transform=tfm)

## Length of Train and Test Daataset

len_train=len(train_ds)
len_test=len(test_ds)
len_train,len_test
## step 3: Build Model, Optimizer,and Loss FUnction

## Model
model=resnet18(pretrained=True)
## Replace output of Fully Connected Layer
model.fc=Linear(in_features=512,out_featuers=2)
model=model.to(device)
## optimizer
optimizers=Adam(model.parameters(),lr=3e-14)
## lOSS Function
loss_fn=CrossEntropyLoss()


## step4: Train and Evaluate the Model
for epoch in range(3):
    tr_acc=0
    test_acc=0
    model.train()
    with tqdm(train_loader,unit="batch") as tepoch:
        for xtrain,ytrain in tepoch:
            optimiser.zero_grad()
            x_train=xtrain.to(device)
            train_prob=model(xtrain)
            train_prob=train_prob.cpu()
            loss_fn=loss_fn(train_prob,ytrain)
    model.eval()

for x,y in train_loader:
    x,y
    break
x.shape,y.shape

x=x.to(device)
yprob=model(x.float())
yprob
torch.max(yprob,1).indices
yprob=yprob.cpu()
ypred=torch.max(yprob,1).indices
y==ypred
torch.sum(y==ypred)
## Model performance on Samples
sample_1='is that santa/test/santa/283.santa.jpg'
sample_2='is that santa/test/santa/474.santa.jpg'
sample_3='is that santa/test/not-a-santa/58.not-a-santa.jpg'
sample_4='is that santa/test/not-a-santa/340.not-a-santa.jpg'
sample_list=[sample_1,sample_2,sample_3,sample_4]
from PIL import Image
img=Image.open(r'C:\Users\prativadis\OneDrive - TESSCO Technologies\Desktop\Implement a PreTrained (ResNet18) CNN Model using PyTorch from Scratch on a Kaggle Image Dataset\santa.jpeg')
display(img.resize(224,224))
tfm_2=tfm=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(degress=10),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

img=Image.open(sample_1)
display(img.resize(224,224))
img_array=np.array(img)
img_tensor=tfm_2(img_array)
img_tensor.shape ## to get the size
img_tensor=img_tensor[np.newaxis,:]
img_tensor.to(device)
pred_prob=model(img_tensor)
pred=torch.max(pred_prob,1).indices
pred=pred.item()
if pred==1:
    print(f"Model Prediction {pred}, hence a Santa")
else:
    print("Model Prediction {pred}, hence not a Santa")
print(pred)