

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer,AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
device='cuda' if torch.cuda.is_available() else 'cpu'
print('Available device:',device)

df=pd.read_json("/content/Sarcasm.json",lines=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.head()

x_train,x_test,y_train,y_test=train_test_split(df['headline'],df['is_sarcastic'],test_size=0.2,stratify=df['is_sarcastic'])
x_val,x_test,y_val,y_test=train_test_split(x_test,y_test,test_size=0.5,stratify=y_test)
print('Training Shape',x_train.shape[0])

# model_name = "distilbert-base-uncased"  # ~2x faster than BERT
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

tokenizer=AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
bert_model=AutoModel.from_pretrained('google-bert/bert-base-cased')

class dataset(Dataset):
    def __init__(self, X, Y):
        # 🔹 Ensure all texts are strings
        X = [str(x) for x in X]

        # 🔹 Tokenize the batch
        encodings = tokenizer(
            X,
            max_length=2,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        # 🔹 Store tokenized inputs
        self.input_ids = encodings['input_ids'].to(device)
        self.attention_mask = encodings['attention_mask'].to(device)

        # 🔹 Convert labels to NumPy array, then to LongTensor
        self.y = torch.tensor(np.array(Y), dtype=torch.long).to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.y[idx]
        }

training_data = dataset(x_train, y_train)
val_data = dataset(x_val, y_val)
test_data = dataset(x_test, y_test)

BATCH_SIZE=32
EPOCHS=10
LEARNING_RATE=0.001

from torch.utils.data import DataLoader

# 🔹 Training DataLoader
train_dataloader = DataLoader(
    training_data,
    batch_size=8,              # Smaller batch size reduces memory load
    shuffle=True,              # Always shuffle for training
    num_workers=2,             # Parallel loading without overwhelming CPU
    pin_memory=False           # Only needed for GPU
)

# 🔹 Validation DataLoader
val_dataloader = DataLoader(
    val_data,
    batch_size=8,              # Match training batch size
    shuffle=False,             # No need to shuffle for validation
    num_workers=2,
    pin_memory=False
)

# 🔹 Test DataLoader
test_dataloader = DataLoader(
    test_data,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=False
)

class MyModel(nn.Module):
    def __init__(self, bert):
        super(MyModel, self).__init__()
        self.bert = bert
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)  # no comma
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # Unpack tuple output
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.drop(pooled_output)
        x = self.out(x)
        x = self.sigmoid(x)
        return x

for param in bert_model.parameters():
  param.requires_grad=False
model=MyModel(bert_model)
model=model.to(device)

model

criterion = nn.CrossEntropyLoss()
optimizer=Adam(model.parameters(),lr=LEARNING_RATE)

total_loss_train_plot=[]
total_acc_validation_plot=[]
total_acc_train_plot=[]
total_loss_validation_plot=[]

model.train()  # Ensure model is in training mode

for batch in train_dataloader:
    # Move inputs and labels to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device).long()  # CrossEntropyLoss expects LongTensor

    # Forward pass
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # logits: [batch_size, num_classes]

    # Compute loss
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

# criterion = nn.CrossEntropyLoss()  # Correct loss for multi-class logits

# # for batch in train_dataloader:
# #     input_ids = batch['input_ids'].to(device)
# #     attention_mask = batch['attention_mask'].to(device)
# #     labels = batch['labels'].to(device).long()  # <-- convert to long here

# #     optimizer.zero_grad()
# #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # logits [batch_size, 2]

# #     loss = criterion(outputs, labels)  # now no error

# #     loss.backward()
# #     optimizer.step()

with torch.no_grad():
    total_acc_val = 0
    total_loss_val = 0

    for idx, data in enumerate(val_dataloader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device).long()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        total_loss_val += loss.item()
        preds = outputs.argmax(dim=1)
        total_acc_val += (preds == labels).sum().item()

# Normalize and append to plot lists
total_loss_train_plot.append(round(total_loss_train / len(train_dataloader), 3))
total_loss_val_plot.append(round(total_loss_val / len(val_dataloader), 3))
total_acc_train_plot.append(round(total_acc_train / len(train_dataloader.dataset), 3))
total_acc_val_plot.append(round(total_acc_val / len(val_dataloader.dataset), 3))

with torch.no_grad():
    total_acc_test = 0
    total_loss_test = 0
    for idx,data in enumerate(testing_dataloader):
      inputs,labels=data
      inputs.to(device)
      labels.to(device)
      prediction=model(inputs["input_ids"].squeeze(1),inputs["attention_mask"].squeeze(1)).squeeze(1)
      batch_loss=criterion(prediction,labels)
      total_loss_test+=batch_loss.item()
      acc=(prediction.round()==labels).sum().item()
      total_acc_test+=acc
print(f"Accuracy Score on testing data:{round((total_acc_test/testing_data.__len__())*100,4)}")

fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].plot(total_loss_train_plot,label='Training class')
axs[0].plot(total_loss_validation_plot,label='Validation class')
axs[0].set_title('Training and validation loss over epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[1].plot(total_acc_train_plot,label='Training class')
axs[1].plot(total_acc_validation_plot,label='Validation class')
axs[1].set_title('Training and validation accuracy over epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()