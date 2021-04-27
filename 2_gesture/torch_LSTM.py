import torch
import torchvision
import torch.nn as nn
import random
import json
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
train_num = 50

class LSTM(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.lstm = nn.LSTM(30, 8)                     # [input_size, output_size, num_layers]
        # bidirectional – If True, becomes a bidirectional LSTM. Default: False
        self.link1 = nn.Linear(100, 50)
        self.link2 = nn.Linear(50, 10)
        self.link3 = nn.Linear(10, 8)
        self.batch_size = batch_size

    def forward(self, x):
        # The size of x is [sequence_len, 1, group_size * 6], where 6 includes speed and Angle information
        hidden = (torch.randn(1, self.batch_size, 8),
                  torch.randn(1, self.batch_size, 8))    # size of h0 and c0 is [num_layers * num_directions, batch, hidden_size]
        out, h0 = self.lstm(x, hidden)
        #out_put = self.link1(h0[0])
        #out_put = F.relu(out_put)
        #out_put = self.link2(out_put)
        #out_put = F.relu(out_put)
        #out_put = self.link3(out_put)

        # out_put = F.softmax(h0[0], dim=-1)
        out_put = F.softmax(h0[1], dim=-1)
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
            criteria = nn.CrossEntropyLoss()
            loss = criteria(out_put[0], labels_)
            #loss = F.nll_loss(out_put[0], labels_)
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

def new_get_dataset(start, length):
    with open("./data_set/data_for_LSTM.json", 'r', encoding="UTF-8") as fp:
        loader = json.load(fp)
    data_list = []
    for i in range(length):
        key = i + start
        key = str(key)
        data = loader[key]["data"]
        label = loader[key]["label"]
        data_list.append((data, label))
    return data_list

def new_train(model, optimizer, epoch, train_data):
    model.train()
    for data, label in train_data:
        input_data = torch.Tensor(data)
        label = [label]
        input_label = torch.LongTensor(label)
        optimizer.zero_grad()
        output = model(input_data)
        #criteria = nn.MSELoss()
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output[0], input_label)
        #loss = F.nll_loss(output[0], input_label)
        loss.backward()
        optimizer.step()
        #print("finish a train")
    print("epoch {} finish loss is {}".format(epoch, loss.item()))

def new_test(model, epoch, test_data):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in test_data:
            input_data = torch.Tensor(data)
            label = [label]
            input_label = torch.LongTensor(label)
            out_put = model(input_data)
            pred = out_put[0].max(1, keepdim=True)[1]
            correct += pred.eq(input_label.view_as(pred)).sum().item()
    print("Test set Accuracy: {:.2f}%\n".format(100. * correct / len(test_data)))

model = LSTM(BATCH_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
train_data = new_get_dataset(0, 700)
test_data = new_get_dataset(700, 300)
for epoch in range(0, train_num):
    new_train(model, optimizer, epoch, train_data)
    new_test(model, epoch, test_data)
