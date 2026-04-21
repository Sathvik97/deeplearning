import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import pandas as pd
df = pd.read_csv('fmnist_small.csv')
df.head()
df.isnull().sum()
X = df.iloc[:,1:]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#scaling
X_train = X_train/255.0
X_test = X_test/255.0

#dataset
def CustomDataset(Dataset):
  def __init__(self,features,labels):
    self.features = torch.tensor(features,dtype=torch.float32)
    self.labels = torch.tensor(labels,dtype=torch.long)
  
  def __len__(self):
    return len(self.features)

  def __getitem__(self,index):
    return self.features[index],self.labels[index]

train_dataset = CustomDataset(X_train,y_train)
test_dataset = CustomDataset(X_test,y_test)
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)
n_features = X.shape[1]

#Neural Network
class NN(nn.Module):
  def __init__(self,n_features):
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(n_features,128),
      nn.ReLU(),
      nn.Linear(128,64),
      nn.ReLU()
      nn.Linear(64,10)
    )

  def forward(self,x):
    return self.model(x)

#training loop
lr = 0.01
epochs = 100

for epoch in range(epochs):
  epoch_loss = 0
  for X_batch,y_batch in train_loader:
    y_hat = model(X_batch)
    loss = criterion(y_hat,y_batch)
    epoch_loss+=loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(f"Epoch: {epoch+1},Loss: {epoch_loss}")


#accuracy checking
correct = 0
total = 0
with torch.no_grad():
    for X, y in test_loader:
        preds = model(X).argmax(dim=1)#dimension along the row each row has one max probability prediction of the labels for each image ie multiclassification
        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")

-------------------------------------------------------IMPORTANT NOTES---------------------------------------------------------------------------------------------
 #In PyTorch, torch.argmax() is used to find the index of the maximum value in a tensor, optionally along a specified dimension.

    





