from torch.utils.data import Dataset,Dataloader
import torch
from sklearn.datasets import make_classification
X,y = make_classification(
    n_samples = 10,
    n_features = 2,
    n_informative = 2,
    n_redundant = 0,
    n_classes = 2,
    random_state = 42
)
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

#custom dataset
class CustomDataset(Dataset):
  def __init__(self,features,labels):
    self.features = features
    self.labels = labels
  def __len__(self):
    return len(self.features.shape[1])
  def __getitem__(self,index):
    return self.labels[index],self.features[index]

dataset = CustomDataset(X,y)

#dataloader
dataloader = Dataloader(dataset,batch_size=2,shuffle=True)

for batch_label,batch_features in dataloader:
  print(batch_labels,batch_features)
