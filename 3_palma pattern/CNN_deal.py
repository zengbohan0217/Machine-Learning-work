import torch
import torch.nn as nn
import torch.nn.functional as F
import CNN_dataset as my_get
from torch.utils.data import DataLoader
import torch.optim as optim

BATCH_SIZE = 512
EPOCHS = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*1*28*28
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc0 = nn.Linear(3380, 500)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 100)

    def forward(self, x):
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

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = data.to(device)
        #target_list = [0]*100
        #target_list[int(target)] = 1
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        pred = output.max(1, keepdim=True)[1]  # Find the index with the highest probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {:.0f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 100. * correct / len(train_loader.dataset)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')
            pred = output.max(1, keepdim=True)[1]  # Find the index with the highest probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

train_image_dir = '.\BMP600'
train_data = my_get.new_train_data(image_dir=train_image_dir, repeat=1)
train_loader = DataLoader(dataset=train_data, batch_size=12, shuffle=True)

test_image_dir = r'.\test_data'
test_data = my_get.Test_Dataset(image_dir=test_image_dir, repeat=1)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

# Finally start training and testing
for epoch in range(1, EPOCHS + 1):
    train(model,  DEVICE, train_loader, optimizer, epoch)
    #train(model, DEVICE, test_loader, optimizer, epoch)

test(model, DEVICE, test_loader)