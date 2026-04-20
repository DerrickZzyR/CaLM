import os
import torch
import numpy as np
from torch.utils.data import Dataset

class UnifiedDataset(Dataset):
    def __init__(self, root_path, flag='train'):
        if flag == 'train':
            self.X = np.load(os.path.join(root_path, 'X_train.npy'))
            self.y = np.load(os.path.join(root_path, 'y_train.npy'))
            self.ids = np.load(os.path.join(root_path, 'id_train.npy'))

        elif flag == 'test':
            self.X = np.load(os.path.join(root_path, 'X_test.npy'))
            self.y = np.load(os.path.join(root_path, 'y_test.npy'))
            self.ids = np.load(os.path.join(root_path, 'id_test.npy'))

        elif flag == 'all':
            X_train = np.load(os.path.join(root_path, 'X_train.npy'))
            y_train = np.load(os.path.join(root_path, 'y_train.npy'))
            id_train = np.load(os.path.join(root_path, 'id_train.npy'))

            X_test = np.load(os.path.join(root_path, 'X_test.npy'))
            y_test = np.load(os.path.join(root_path, 'y_test.npy'))
            id_test = np.load(os.path.join(root_path, 'id_test.npy'))

            self.X = np.concatenate([X_train, X_test])
            self.y = np.concatenate([y_train, y_test])
            self.ids = np.concatenate([id_train, id_test])

        else:
            raise ValueError(flag)

        print(f"Loaded {flag} set: {self.X.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
            int(self.ids[idx])
        )