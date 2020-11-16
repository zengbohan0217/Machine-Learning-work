import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class DNN_data_set(Dataset):
    def __init__(self, json_path):
        self.character_dict = ["1", "2", "3", "4", "A", "B", "C", "D"]

