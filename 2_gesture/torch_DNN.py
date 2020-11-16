import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random
import json

BATCH_SIZE = 64
EPOCH = 20

def get_dataset():
    dataset_list = []  # size:n*6
    character_dict = ["1", "2", "3", "4", "A", "B", "C", "D"]
    for k in range(8):
        with open('./data_set/{}.json'.format(character_dict[k]), 'r', encoding="UTF-8") as fp:
            json_data = json.load(fp)
            input_list = []
            speed_list = []
            angel_list = []
            for need_key in json_data.keys():
                speed_list.extend(json_data[need_key]["speed_accel"])
                angel_list.extend(json_data[need_key]["angle_accel"])
            for i in range(len(speed_list)):
                data_part = []
                label_part = [0] * 8
                for j in range(3):
                    data_part.append(speed_list[i][j])
                    data_part.append(speed_list[i][j])
                label_part[k] = 1
                dataset_list.append((data_part, label_part))
    random.shuffle(dataset_list)
    return dataset_list

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.link1 = nn.Linear(6, 200)
        self.link2 = nn.Linear(200, 100)
        self.link3 = nn.Linear(100, 50)
        self.link4 = nn.Linear(50, 8)

    def forward(self, x):
        out = self.link1(x)
        out = F.relu(out)
        out = self.link2(out)
        out = F.relu(out)
        out = self.link3(out)
        out = F.relu(out)
        out = self.link4(out)
        out = F.relu(out)
        return out

def train(model, optimizer, train_set, epoch):
    model.train()
    point = 0
    while point + BATCH_SIZE <= len(train_set):
        part_data = train_set[point:point + BATCH_SIZE]
        input_data = []
        input_label = []
        for data, label in part_data:
            input_data.append(data)
            input_label.append(label)
        input_data = torch.Tensor(input_data)
        input_label = torch.Tensor(input_label)
        optimizer.zero_grad()
        out_put = model(input_data)
        labels_ = torch.max(input_label, 1)[1]
        criteria = nn.CrossEntropyLoss()
        loss = criteria(out_put, labels_)
        loss.backward()
        optimizer.step()
        point = point + BATCH_SIZE
    print("epoch {} finish loss is {}".format(epoch, loss.item()))

def test(model, test_data, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        point = 0
        while point + BATCH_SIZE <= len(test_data):
            part_data = test_data[point:point + BATCH_SIZE]
            input_data = []
            input_label = []
            for data, label in part_data:
                input_data.append(data)
                input_label.append(label)
            input_data = torch.Tensor(input_data)
            input_label = torch.Tensor(input_label)
            out_put = model(input_data)
            labels_ = torch.max(input_label, 1)[1]
            pred = out_put.max(1, keepdim=True)[1]
            correct += pred.eq(labels_.view_as(pred)).sum().item()
            point += BATCH_SIZE
    print("Test set Accuracy: {:.2f}%\n".format(100. * correct / len(test_data)))

model = DNN()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
train_data_set = get_dataset()
for epoch in range(0, EPOCH):
    train(model, optimizer, train_data_set, epoch)
    test(model, train_data_set, epoch)

