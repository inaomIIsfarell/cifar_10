import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # (C, H, W) : (3, 32, 32) => (16, 28, 28) 
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # downSample
        # (C, H, W) = (16, 28, 28) => (16, 14, 14)
        self.pool1 = nn.MaxPool2d(2)
        # (C, H, W): (16, 14, 14) => (16, 12, 12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # (C, H, W): (32, 12, 12) => (64, 10, 10)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # dowmSample
        # (C, H, W): (64, 10, 10) => (64, 5, 5)
        self.pool2 = nn.MaxPool2d(2)
        # num_label: 10
        self.fc = nn.Linear(5*5*64, 10)

    def forward(self, input):
        output = self.conv1(input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.pool2(output)
        # print("output.shape: {}".format(output.shape))
        output = output.view(output.size(0), -1)
        # print("output.shape: {}".format(output.shape))
        output = self.fc(output)
        return output
    
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1
    )

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
    )

class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downSample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downSample = downSample
        self.stride = stride

    def forward(self, x):
        iden = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downSample is not None:
            # print("out.shape: {}".format(out.shape))
            iden = self.downSample(x)
        
        # print("iden.shape: {}".format(iden.shape))
        out += iden

        out = self.relu1(out)

        return out
        
class resModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # (3, 32, 32) => (16, 28, 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # (16, 28, 28) => (64, 28, 28)
        self.ds1 = nn.Sequential(
            conv1x1(in_channels=16, out_channels=64),
            nn.BatchNorm2d(64)
        )
        self.res1 = resBlock(in_channels=16, out_channels=64, downSample=self.ds1)
        # (64, 28, 28) = (64, 14, 14)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # (64, 14, 14) => (256, 14, 14)
        self.ds2 = nn.Sequential(
            conv1x1(in_channels=64, out_channels=256),
            nn.BatchNorm2d(256)
        )
        self.res2 = resBlock(in_channels=64, out_channels=256, downSample=self.ds2)
        # (256, 14, 14) => (256, 7, 7)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # (256, 7, 7) => (256, 1, 1)
        self.avg1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)
        

    
    def forward(self, x):
        y = self.conv1(x)
        y = self.res1(y)
        # print("y.shape: {}".format(y.shape))
        # exit()
        y = self.pool1(y)
        y = self.res2(y)
        y = self.pool2(y)
        y = self.avg1(y)
        # print("before flatten y.shape: {}".format(y.shape))
        y = y.view(y.size(0), -1)
        # print("after flatten y.shape: {}".format(y.shape))
        # exit()
        y = self.fc(y)
        return y