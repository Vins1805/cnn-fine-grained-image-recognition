import os
import cv2
import torch
import torch.optim
import h5py
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

#batch_size = 10
#shuffle = True


class CUBDataset(Dataset):
    def __init__(self, is_test: bool = False, test_size: int = 1/6, random_state: int = 0):
        super(CUBDataset, self).__init__()
        self.is_test = is_test
        self.data = h5py.File("samples/cub_data.hdf5", 'r')
        
        keys = list(self.data.keys())
        self.itrain, self.itest, _, _ = train_test_split(keys, keys, test_size=test_size, random_state=random_state)
        

    def __getitem__(self, index):
        if self.is_test:
            key = self.itest[index]
        else:
            key = self.itrain[index]
        img = torch.tensor(cv2.resize(self.data[key]["x"][()], (448, 448))).float()
        img = img.permute(2,0,1)
        label = self.data[key]["y"][()]
        return img, label

    def __len__(self):
        if self.is_test:
            return len(self.itest)
        else:
            return len(self.itrain)
    
    
    
    
    
"""
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
            x.append(f[name][file][()])
            y.append(index)

            
x, tx, y, ty = train_test_split(x, y, test_size=test_size, random_state=random_state)

# Create Dataset and DataLoder instances of both train and test data.
train_set = CnnDataset(x, y)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
test_set = CnnDataset(tx, ty)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle)
"""