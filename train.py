import torch
import model
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

cifar10_train = datasets.CIFAR10(
    root="/data/cifar10", 
    train=True, 
    transform=transform,
    download=False
)

cifar10_train_loader = DataLoader(cifar10_train, batch_size=32, shuffle=True)
# print(next(enumerate(cifar10_train_loader)))

# convModel = model.ConvModel()
convModel = model.resModel()

DEVICE = torch.device('cuda')
convModel.to(DEVICE)

# 交叉熵损失
criterion = nn.CrossEntropyLoss()
criterion.to(DEVICE)

losses = []

# lr = 0.01, losses(last) = 1.1925249099731445
# lr = 0.005, loss = 
lr = 0.001

optimizer = torch.optim.Adam(params=convModel.parameters(), lr=lr)

total_epoch = 5

for epoch in range(total_epoch):
    for i, data_batch in enumerate(cifar10_train_loader):
        feature_map = data_batch[0].to(DEVICE)
        labels = data_batch[1].to(DEVICE)

        optimizer.zero_grad()
        output = convModel(feature_map)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().data.item())

        if (i + 1) % 100 == 0:
            print("epoch: {} / {}, Iter: {} / {}, loss: {}".format(epoch+1, total_epoch, i+1, len(cifar10_train)//32, loss.data.item()))
        
# print("loss in cuda: {}".format(loss))

plt.xlabel('epoch #')
plt.ylabel('loss #')
plt.plot(losses)
plt.show()

torch.save(convModel.state_dict(), "res_model_params.pth")
