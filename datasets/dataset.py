import os
import cv2
import h5py
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CUBDataset(Dataset):
    def __init__(self, is_test: bool = False, test_size: int = 1/6, random_state: int = 0):
        """
            Every instance of this class has the same train and test data due to fixed random state.        
        """
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
        img = self.data[key]["x"][()]
        label = self.data[key]["y"][()]
        return img, label
        

    def __len__(self):
        if self.is_test:
            return len(self.itest)
        else:
            return len(self.itrain)
    