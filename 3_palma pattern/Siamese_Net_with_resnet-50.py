import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import Siamese_Net_dataset as SN_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
from PIL import Image
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

size = (256, 256)
transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
# pic = cv2.imread('./BMP600/100_1.bmp')
# pic = Image.fromarray(pic)
# img = transform(pic)
# img = img.view(1, 3, 256, 256)
# img.to(DEVICE)
#
# model = torchvision.models.resnet50(pretrained=True)
# output = model(img)
# print(output.size())

class Siamese_Net_R(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.fc0 = nn.Sequential(nn.Linear(1000, 500), nn.Dropout(p=0.5))
        self.fc1 = nn.Sequential(nn.Linear(500, 50), nn.Dropout(p=0.5))

    def forward_once(self, x):
        x = x.view(-1, 3, 256, 256)
        out = self.resnet(x)
        # out = self.fc0(out)
        # out = F.relu(out)
        # out = self.fc1(out)
        out = F.log_softmax(out, dim=1)
        return out

    def forward(self, input_1, input_2):
        output_1 = self.forward_once(input_1)
        output_2 = self.forward_once(input_2)
        return output_1, output_2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        loss_contrastive = torch.mean(1.46 * label * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def test(model, device, test_loader):
    model.eval()
    correct = 0
    eval_judge = 0
    with torch.no_grad():

        same_norm_dis = 0
        same_num = 0
        same_correct = 0
        dif_norm_dis = 0
        dif_num = 0
        dif_correct = 0

        for i, data in tqdm(enumerate(test_loader, 0)):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            output1, output2 = model(img0, img1)
            output1, output2 = output1.to(device), output2.to(device)
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            euclidean_distance = torch.mean(euclidean_distance).item()
            # euclidean_distance = euclidean_distance.to(device)
            if abs(euclidean_distance-0) < abs(euclidean_distance-2):
                eval_judge = torch.tensor([1])
            elif abs(euclidean_distance-0) >= abs(euclidean_distance-2):
                eval_judge = torch.tensor([0])
            eval_judge = eval_judge.to(device)
            correct += eval_judge.eq(label.view_as(torch.tensor(eval_judge))).sum().item()
            check_label = label.sum().item()

            if check_label == 1:
                same_norm_dis += euclidean_distance
                same_correct += eval_judge.eq(label.view_as(torch.tensor(eval_judge))).sum().item()
                same_num += 1
            elif check_label == 0:
                dif_norm_dis += euclidean_distance
                dif_correct += eval_judge.eq(label.view_as(torch.tensor(eval_judge))).sum().item()
                dif_num += 1
        print(f"size of same is {same_num / (same_num + dif_num)}")
        print(f"accuracy of same is {same_correct / same_num}")
        print(f"accuracy of dif is {dif_correct / dif_num}")

    return correct/len(test_loader)

net = Siamese_Net_R().to(DEVICE)
criterion = ContrastiveLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.0005)
optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

counter = []
loss_history = []
iteration_number = 0

train_number_epochs = 30
train_image_dir = '.\BMP600'
train_data = SN_data.SN_dataset(image_dir=train_image_dir, repeat=1, length=1600)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_data = SN_data.SN_dataset(image_dir=train_image_dir, repeat=1, length=800)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

#开始训练
for epoch in range(0, train_number_epochs):
    pbar = tqdm(len(train_loader))
    for i, data in enumerate(train_loader, 0):
        img0, img1, label = data
        # The img0 dimension is torch.size([32, 1, 100, 100]), 32 is batch, label is torch.Size([32, 1])
        img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0 :
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
            # print(f"epoch{epoch} turn {i} finished")
        pbar.update(1)
    pbar.close()
    print("Epoch number: {} , Current loss: {:.4f}".format(epoch, loss_contrastive.item()))
    accuracy = test(net, DEVICE, test_loader)
    print("Epoch number: {} , test accuracy: {:.4f}".format(epoch, accuracy))
