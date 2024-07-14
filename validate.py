import torch
import model
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

model = model.resModel()
model.load_state_dict(torch.load("res_model_params.pth"))
model.to(torch.device('cuda'))
model.eval()

# print(convModel)
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

val_cifar10 = datasets.CIFAR10(
    root="/data/cifar10",
    train=False,
    transform=transform
)

correct = 0
total = len(val_cifar10)

# print(val_cifar10)
val_cifar10_loader = DataLoader(val_cifar10, batch_size=32)
for  feature_map, labels in val_cifar10_loader:
    feature_map = feature_map.to(torch.device('cuda'))
    predict = model(feature_map)
    predict = predict.cpu()
    # print(predict)
    _, tag = torch.max(predict, 1)
    # print(tag)
    # print(labels)
    correct += (tag == labels).sum()
    
# initial conv
# lr = 1e-3, acc=68.38%
# lr = 5e-3, acc=69.77%

# residual net
# lr = 1e-3, acc=77.05%

print('correct / total : {} / {}'.format(correct, total))
print('acc: {}%'.format(correct/total*100))



