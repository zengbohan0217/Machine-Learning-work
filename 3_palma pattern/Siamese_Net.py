import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import Siamese_Net_dataset as SN_data
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*1*28*28
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc0 = nn.Linear(3380, 500)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 5)

    def forward_once(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 1* 10 * 12 * 12
        out = F.max_pool2d(out, 2, 2)
        out = F.max_pool2d(out, 2, 2)
        out = F.max_pool2d(out, 2, 2)
        out = self.conv2(out)  # 1* 20 * 10 * 10
        out = F.relu(out)
        out = out.view(in_size, -1)  # 1 * 2000
        out = self.fc0(out)
        out = F.relu(out)
        #out = self.fc1(out)  # 1 * 500
        #out = F.relu(out)
        out = self.fc2(out)  # 1 * 10
        out = F.log_softmax(out, dim=1)
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


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
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

#net = SiameseNetwork().cuda() #定义模型且移至GPU
net = SiameseNetwork().to(DEVICE)
criterion = ContrastiveLoss() #定义损失函数
optimizer = optim.Adam(net.parameters(), lr = 0.0005) #定义优化器

counter = []
loss_history = []
iteration_number = 0

train_number_epochs = 20
train_image_dir = '.\BMP600'
train_data = SN_data.SN_dataset(image_dir=train_image_dir, repeat=1)
train_loader = DataLoader(dataset=train_data, batch_size=40, shuffle=True)

#开始训练
for epoch in range(0, train_number_epochs):
    for i, data in enumerate(train_loader, 0):
        img0, img1 , label = data
        #img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
        img0, img1 , label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE) #数据移至GPU
        optimizer.zero_grad()
        output1,output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0 :
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
    print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, loss_contrastive.item()))
