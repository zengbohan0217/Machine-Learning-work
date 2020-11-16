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
train_num = 15

class LSTM(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.lstm = nn.LSTM(2, 20)
        self.link1 = nn.Linear(20, 10)
        self.link2 = nn.Linear(10, 8)
        self.batch_size = batch_size

    def forward(self, x):
        # x的size是[3, batch_size, 2]，其中2分别代表speed angle
        hidden = (torch.randn(1, self.batch_size, 20),
                  torch.randn(1, self.batch_size, 20))
        out, h0 = self.lstm(x, hidden)
        out_put = self.link1(h0[0])
        out_put = F.relu(out_put)
        out_put = self.link2(out_put)
        out_put = F.relu(out_put)
        return out_put

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

def train(model, optimizer, train_data_set):
    model.train()
    for epoch in range(0, train_num):
        point = 0
        loss_sum = 0
        while point + BATCH_SIZE <= len(train_data_set):
            part_data = train_data_set[point:point+BATCH_SIZE]
            input_data_1 = []
            input_data_2 = []
            input_data_3 = []
            input_label = []
            for data, label in part_data:
                input_label.append(label)
                input_data_1.append(data[0])
                input_data_2.append(data[1])
                input_data_3.append(data[2])
            input_data = [input_data_1, input_data_2, input_data_3]
            input_data = torch.Tensor(input_data)
            input_label = torch.Tensor(input_label)
            optimizer.zero_grad()
            out_put = model(input_data)
            #print(out_put[0].view(512))
            #print(input_label)
            labels_ = torch.max(input_label, 1)[1]
            #criteria = nn.CrossEntropyLoss()
            #loss = criteria(out_put[0], labels_)
            loss = F.nll_loss(out_put[0], labels_)
            #loss = F.nll_loss(out_put, labels_)
            loss_sum = loss.item()
            loss.backward()
            optimizer.step()
            point = point + BATCH_SIZE
        print("epoch {} finish loss is {}".format(epoch, loss_sum))

def test(model, test_data_set):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        point = 0
        len_with_batch = 0
        while point + BATCH_SIZE <= len(test_data_set):
            part_data = test_data_set[point:point+BATCH_SIZE]
            input_data_1 = []
            input_data_2 = []
            input_data_3 = []
            input_label = []
            for data, label in part_data:
                input_label.append(label)
                input_data_1.append(data[0])
                input_data_2.append(data[1])
                input_data_3.append(data[2])
            input_data = [input_data_1, input_data_2, input_data_3]
            input_data = torch.Tensor(input_data)
            input_label = torch.Tensor(input_label)
            out_put = model(input_data)
            labels_ = torch.max(input_label, 1)[1]
            #test_loss += F.nll_loss(out_put[0], target, reduction='sum')
            pred = out_put[0].max(1, keepdim=True)[1]
            #pred = out_put.max(1, keepdim=True)[1]
            correct += pred.eq(labels_.view_as(pred)).sum().item()
            point += BATCH_SIZE
            len_with_batch += 1

    #test_loss /= len(test_data_set)
    print("\nTest set Accuracy: {:.2f}%".format(100.*correct/len(test_data_set)))

model = LSTM(BATCH_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.005)
train_data = get_dataset()
train(model, optimizer, train_data)
test(model, test_data_set=train_data)
