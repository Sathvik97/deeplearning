import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import pandas as pd

df = pd.read_csv('fmnist_small.csv')
df.head()
