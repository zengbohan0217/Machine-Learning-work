import torch
import torchvision
import torch.nn as nn
import random
import json
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

class LSTM(nn.Module):
    def __init__(self, batch_size):
        super.__init__()
        self.lstm = nn.LSTM(2, 8)
        self.batch_size = batch_size

    def forword(self, x):
        # x的size是[3, batch_size, 2]，其中2分别代表speed angle
        hidden = (torch.randn(1, self.batch_size, 8),
                  torch.randn(1, self.batch_size, 8))
        out, h0, c0 = self.lstm(x, hidden)
        return h0

def get_dataset():
    dataset_list = []              # size:n*3*2
    character_dict = ["1", "2", "3", "4", "A", "B", "C", "D"]
    for k in range(8):
        with open('./data_set/{}.json'.format(character_dict[k]), 'r', encoding="UTF-8") as fp:
            json_data = json.load(fp)
            speed_list = []
            angel_list = []
            for need_key in json_data.keys():
                speed_list.extend(json_data[need_key]["speed_accel"])
                angel_list.extend(json_data[need_key]["angle_accel"])
            for i in range(len(speed_list)):
                data_part = []
                label_part = [0]*8
                for j in range(3):
                    data_part.append([speed_list[i][j], angel_list[i][j]])
                label_part[k] = 1
                dataset_list.append((data_part, label_part))
    random.shuffle(dataset_list)
    return dataset_list

def train(model, train_data_set):

    model.train()

def test(model, test_data_set):
    model.test()

