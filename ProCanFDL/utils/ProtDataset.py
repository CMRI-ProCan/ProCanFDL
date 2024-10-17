import torch
from torch.utils.data import Dataset
import numpy as np


class ProtDataset(Dataset):
    def __init__(self, data, labels, prot_id_to_name, prot_name_to_id, label_map):
        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.prot_id_to_name = prot_id_to_name
        self.prot_name_to_id = prot_name_to_id
        self.label_map = label_map
        self.feature_dim = data.shape[1]
        self.num_classes = np.max(labels) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]
