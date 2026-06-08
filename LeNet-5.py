 #LeNet - 5, the most well-known version of LeNet, consists of two convolutional layers followed by average pooling layers, and then three fully-connected layers.
import torch
from torch import nn

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.features = nn.Sequential(
        #1
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1,padding=2),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2,stride=2), 

        #2
        nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=2),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2,stride=2)
        )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=16*5*5,out_features=120),
        nn.Tanh(),
        nn.Linear(in_features=120,out_features=84),
        nn.Tanh(),
        nn.Linear(in_features=84,out_features=10)
    )

  def forward(self,x):
    return self.classifier(self.features(x))
