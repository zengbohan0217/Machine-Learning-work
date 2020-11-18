import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import random
import json

BATCH_SIZE = 64
EPOCH = 100

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
        self.link1 = nn.Linear(1200, 400)
        self.link2 = nn.Linear(400, 200)
        self.link3 = nn.Linear(200, 50)
        self.link4 = nn.Linear(50, 8)

    def forward(self, x):
        out = self.link1(x)
        out = F.relu(out)
        out = self.link2(out)
        out = F.relu(out)
        out = self.link3(out)
        out = F.relu(out)
        out = self.link4(out)
        out = F.softmax(out, dim=-1)
        return out

class dataset(Dataset):
    def __init__(self, json_path, start, length):
        self.json_path = json_path
        self.start = start
        self.length = length
        self.data_load = self.load_data(json_path)

    def __getitem__(self, i):
        key = self.start + i
        if key == 67:
            key = 66
        elif key == 627:
            key = 626
        key = str(key)
        data = self.data_load[key]["data"]
        data = np.array(data)
        label = self.data_load[key]["label"]
        data = torch.tensor(data, dtype=torch.float32)
        return data, label

    def __len__(self):
        return self.length

    def load_data(self, json_path):
        with open(json_path, 'r', encoding="UTF-8") as fp:
            loader = json.load(fp)
        return loader

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

def train_with_dataloader(model, optimizer, trainloader, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target)
        loss.backward()
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {:.0f}%'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item(), 100. * correct / len(trainloader.dataset)))

def test_with_dataloader(model, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)
    ))


model = DNN()
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
#train_data_set = get_dataset()
json_path = "./data_set/deal.json"       # 1120条
train_data = dataset(json_path=json_path, start=0, length=700)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_data = dataset(json_path=json_path, start=700, length=300)
test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=True)
for epoch in range(0, EPOCH):
    #train(model, optimizer, train_data_set, epoch)
    #test(model, train_data_set, epoch)
    train_with_dataloader(model, optimizer, train_loader, epoch)
    test_with_dataloader(model, test_loader)

