import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
df = iris['frame']
df.head()
X = df.iloc[:,1:4]
y = df['target']
import seaborn as sns
sns.scatterplot(x='petal length (cm)',y='petal width (cm)',hue='target',data=df)
sns.scatterplot(x='sepal length (cm)',y='sepal width (cm)',hue='target',data=df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
X_test_tensor = torch.tensor(X_test,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values , dtype=torch.float32).view(-1)
y_train_tensor = torch.tensor(y_train.values,dtype=torch.float32).view(-1)

------------------------------------------**model**------------------------------------------------------------------------------------------------
class Softmax(nn.Module):
  def __init__(self,in_features,out_features):
    super().__init__()
    self.linear_1 = nn.Linear(in_features,out_features)
  def forward(self,x):
    return self.linear_1(x)

----------------------------------------------------------------------------------------------------------------------------------------------------
in_features = X.shape[1]
out_features = len(set(y))

model = Softmax(in_features,out_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)
epochs = 200

for epoch in range(epochs):
  
  outputs = model(X_train_tensor)
  loss = criterion(outputs,y_train_tensor.long())

  optimizer.zero_grad()
  loss.backward()
  optimizer.step() 

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test_tensor,torch.argmax(model(X_test_tensor),dim=1))
print(accuracy_score)  //97%//








