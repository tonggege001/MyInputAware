import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Classifier(nn.Module):
    def __init__(self, ch_in, classes_num):
        super(Classifier, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
            nn.Conv2d(ch_in, ndf, kernel_size=3, stride=1, padding=1, bias=False),          # 3 x 32 x 32
            nn.BatchNorm2d(ndf),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                          # 3 x 16 x 16

            nn.Conv2d(ndf, 2 * ndf, kernel_size=3, stride=1, padding=1, bias=False),        # 3 x 16 x 16
            nn.BatchNorm2d(2 * ndf),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                          # 3 x 8 x 8
            nn.Dropout2d(0.2),

            nn.Conv2d(2 * ndf, 4 * ndf, kernel_size=3, stride=1, padding=1, bias=False),    # 3 x 8 x 8
            nn.BatchNorm2d(4 * ndf),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                          # 3 x 4 x 4
            nn.Dropout2d(0.2),

            nn.Conv2d(4 * ndf, 8 * ndf, kernel_size=3, stride=1, padding=1, bias=False),    # 3 x 4 x 4
            nn.BatchNorm2d(8 * ndf),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                          # 3 x 2 x 2
            nn.Dropout2d(0.2),

            nn.Conv2d(8 * ndf, classes_num, kernel_size=2, stride=1)
        )

    def forward(self, x_train):
        return self.main(x_train).view(x_train.shape[0], -1)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)      # same
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )


    def forward(self, x_train):
        out = F.relu(self.bn1(self.conv1(x_train)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x_train)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_ch, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_ch, out_ch, stride))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x_train):
        out = F.relu(self.bn1(self.conv1(x_train)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])



# ---------------------------    Generator    ---------------------------

class Generator(nn.Module):
    def __init__(self, ngf=32, ch_in=3, ch_out=3):
        super(Generator, self).__init__()
        self.list = []
        self.main = nn.Sequential()

        conv_in, conv_out = ch_in, ngf
        for i in range(3):
            self._add_conv_block(conv_in, conv_out)
            self._add_conv_block(conv_out, conv_out),
            self.list = self.list + [nn.MaxPool2d(kernel_size=2, stride=2)]

            conv_in = conv_out
            conv_out = 2 * conv_out

        conv_out = conv_out // 2
        for i in range(3):
            self._add_conv_block(conv_in, conv_out)          # 后面的
            self.list.append(nn.Upsample(scale_factor=(2,2), mode="bilinear"))
            self._add_conv_block(conv_out, conv_out)

            conv_in = conv_out
            conv_out = conv_out // 2

        self.list.append(nn.Conv2d(conv_in, ch_out, kernel_size=3, stride=1, padding=1))
        # self.list.append(nn.Sigmoid())
        self.main = nn.Sequential(*self.list)


    def forward(self, x_train):
        x_train = self.main(x_train)
        return F.tanh(x_train) / 2 + 0.5


    def _add_conv_block(self, ch_in, ch_out):
        self.list = self.list + [
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(True)
        ]






if __name__ == "__main__":
    model = Generator().cuda()
    x = torch.rand((2, 3, 32, 32)).cuda()
    res = model(x)
    print("Res: ")
    print(res.size())

