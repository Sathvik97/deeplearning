import numpy as np
import pandas as pd
import torch
from torch.nn import Linear

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
print(X.shape)
print(y.shape)

n_input_features = X.shape[1]
#SCALING THE DATA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_tensor = torch.from_numpy(X_train).type(torch.float32)
y_train_tensor = torch.from_numpy(y_train).type(torch.float32)
X_test_tensor = torch.from_numpy(X_test).type(torch.float32)
y_test_tensor = torch.from_numpy(y_test).type(torch.float32)
criterion = nn.BCEWithLogitsLoss()

class Logistic_regression(nn.Module):
  def __init__(self,n_input_features):
    super(Logistic_regression,self).__init__()
    self.linear = nn.Linear(n_input_features,1)

  def forward(self,x):
    o = torch.sigmoid(self.linear(x))
    return o

model = Logistic_regression(30)
lr = 0.001
epochs = 200

#training loop
for epoch in range(epochs):
  y_pred = model(X_train_tensor)
  loss = criterion(y_pred, y_train_tensor.unsqueeze(1))
  loss.backward()
  with torch.no_grad():
    model.linear.weight -= lr*model.linear.weight.grad
    model.linear.bias -= lr*model.linear.bias.grad

  model.linear.weight.grad.zero_()
  model.linear.bias.grad.zero_()

#accuracy and pred
y_hat = model(X_test_tensor)
accuracy = y_hat == y_test_tensor
print(accuracy.float().mean())









