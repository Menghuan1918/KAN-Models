import torch
import torchvision
import torchvision.transforms as transforms

num = 100
seed = 114
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 512

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

import torch.nn as nn
import torch.nn.functional as F
from KANLinear import KANLinear

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.KAN1 = KANLinear(6 * 14 * 14, 28 * 28)
        self.KAN2 = KANLinear(28 * 28, 28 * 28)
        self.KAN3 = KANLinear(28 * 28, 28 * 28)
        self.KAN4 = KANLinear(28 * 28, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.KAN1(x)
        x = self.KAN2(x)
        x = self.KAN3(x)
        x = self.KAN4(x)
        x = F.log_softmax(x,dim=1)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

net = Net().to(device)

import torch.optim as optim
from torch.optim import lr_scheduler

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
scheduler=lr_scheduler.CosineAnnealingLR(optimizer,T_max=num,eta_min=0)


best_acc = 0
avg_loss_list = []
acc_list = []
for epoch in range(num):
    net.train()
    avg_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    avg_loss /= len(trainloader)
    avg_loss_list.append(avg_loss)
    scheduler.step()
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
    acc_list.append(acc)
    print(f"Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {acc}")
print("Best Accuracy: ", best_acc)