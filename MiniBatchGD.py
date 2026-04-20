import numpy as np
import torch
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import pandas as pd

#DATA
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target
print(df.head())
X = df.iloc[:,0:8]
y = df['Prices']

#FEATURE SCALING AND VISUALIZATIOB
corr_matrix = df.corr(method='pearson')#correlation matrix
sns.heatmap(
  corr_matrix,
  annot = True,
  cmap = 'coolwarm'
)
df.isnull().sum() #check for null values

# SKLEARN BASIC LINEAR REGRESSION MODEL
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, shuffle=True
)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
predicts = lr.predict(X_test)
pre = lr.predict(X_train)
from sklearn.metrics import r2_score
r2_test = r2_score(y_test,predicts)
r2_train = r2_score(y_train,pre)

# NEURAL NETWORK
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class Linear_regression(nn.Module):
  def __init__(self,n_input_features):
    super(Linear_regression,self).__init__()
    self.linear = nn.Linear(n_input_features,1)

  def forward(self,x):
    o = self.linear(x)
    return o

X_train_tensor = torch.from_numpy(X_train.values).type(torch.float32)
y_train_tensor = torch.from_numpy(y_train.values).type(torch.float32).view(-1,1)
X_test_tensor = torch.from_numpy(X_test.values).type(torch.float32)
y_test_tensor = torch.from_numpy(y_test.values).type(torch.float32).view(-1,1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_tensor = scaler.fit_transform(X_train_tensor)
X_test_tensor = scaler.transform(X_test_tensor)
from torch.utils.data import TensorDataset, DataLoader
import torch 
X_train_tensor = torch.from_numpy(X_train_tensor).type(torch.float32)

dataset = TensorDataset(X_train_tensor, y_train_tensor)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = Linear_regression(X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

num_epochs = 50
for epoch in range(num_epochs):
  model.train()
  epoch_loss = 0
  for X_batch, y_batch in loader:
    outputs = model(X_batch)
    loss = criterion(outputs,y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss+= loss.item() #loss.item() makes it a normal float value from tensor

y_hat = model(torch.from_numpy(X_test_tensor).type(torch.float32))
print(r2_score(y_hat.detach().numpy(),y_test_tensor.numpy())

