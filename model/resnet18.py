import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, class_num, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # x -> [64, 32, 32]
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # x -> [128, 16, 16]
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # x -> [256, 8, 8]
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # x -> [512, 4, 4]
        self.fc = nn.Linear(512 * block.expansion, class_num)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        x = F.relu(self.bn1(self.conv1(x)))

        for layer in self.layers:
            feat = x
            x = layer(x)
        pool_layers = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()])
        x = pool_layers(x)
        x = self.fc(x)

        if return_feat:
            return x, feat
        return x


def resnet18_cifar100(args):
    return ResNet(class_num=args.class_num, block=BasicBlock, num_blocks=[2, 2, 2, 2])