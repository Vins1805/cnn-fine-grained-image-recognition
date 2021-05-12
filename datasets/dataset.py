import os
import cv2
import torch
import torch.optim
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

test_size = 1/6
batch_size = 10
random_state = 0
shuffle = True

filepath = "samples/cub_data.hdf5"

class CnnDataset(Dataset):
    def __init__(self, x, y):
        super(CnnDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        img = torch.tensor(cv2.resize(self.x[index], (448, 448))).float()
        img = img.permute(2,0,1)
        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.x)
    

x, y = [], []

# Load images and append matrices and labels to lists, respectively.
with h5py.File(filepath, 'r') as f:
    for index, name in enumerate(f):
        for file in f[name]:
            x.append(f[name][file].value)
            y.append(index)

            
x, tx, y, ty = train_test_split(x, y, test_size=test_size, random_state=random_state)

# Create Dataset and DataLoder instances of both train and test data.
train_set = CnnDataset(x, y)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
test_set = CnnDataset(tx, ty)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle)