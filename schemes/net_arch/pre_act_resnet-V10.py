import torch
import torch.nn as nn
import math

class BasicBlock3D(nn.Module):
    expansion=1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.bn3 = nn.BatchNorm3d(inplanes)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)

        self.bn3 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)

        #self.bn4 = nn.BatchNorm3d(planes)
        self.downsample = downsample



    def forward(self, x):
        residual = x

        #  --- left
        out1 = self.bn1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        out1 = self.bn2(out1)
        out1 = self.relu(out1)
        out1 = self.conv2(out1)

        out1 = self.bn3(out1)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)


        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out = residual + out1

        return out

class LayerRegion(nn.Module):
    def __init__(self, block, inplanes, planes, block_num, stride=1):
        super(LayerRegion, self).__init__()
        self.inlayer = self._block_stack(block, planes, block_num)

        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None

    def _block_stack(self, block, planes, block_num):
        layers = []
        for _ in range(block_num):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        residual = x
        x = self.inlayer(x)
        out = x - residual

        return out



class MixedNet_V10(nn.Module):
    def __init__(self, block, layers,
                 in_channel=3, num_classes=10, sample_duration=8, sample_size=128,
                 ):
        super(MixedNet_V10, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv3d(in_channel, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.maxpool = maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)

        # self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        # self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer1 = LayerRegion(block, 64, 64, layers[0], stride=1)
        self.layer2 = LayerRegion(block, 64, 128, layers[1], stride=2)
        self.layer3 = LayerRegion(block, 128, 256, layers[2], stride=2)
        self.layer4 = LayerRegion(block, 256, 512, layers[3], stride=2)

        self.bn1 = nn.BatchNorm3d(512 * block.expansion)
        self.relu1 = nn.ReLU(inplace=True)

        last_duration = int(math.ceil(sample_duration / 8))
        last_size = int(math.ceil(sample_size / 32))

        # self.avgpool = nn.AvgPool3d(kernel_size=(last_duration, 7, 7), stride=(1, 7, 7))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)

        self.fc = nn.Linear(512, num_classes)

        # initialize conv and bn parameters
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        #
        # x = self.fc(x)

        return x


def mixed_net34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = MixedNet_V10(BasicBlock3D, [3, 4, 6, 3], **kwargs)
    return model


def mixed_net50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = MixedNet_V10(BasicBlock3D, [3, 4, 6, 3], **kwargs)

    return model


def mixed_net101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = MixedNet_V10(BasicBlock3D, [3, 4, 23, 3], **kwargs)
    return model


def mixed_net152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = MixedNet_V10(BasicBlock3D, [3, 8, 36, 3], **kwargs)
    return model


def mixed_net200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = MixedNet_V10(BasicBlock3D, [3, 24, 36, 3], **kwargs)
    return model

if __name__ == "__main__":

    model = mixed_net101(in_channel=3, num_classes=16).cuda()
    x = torch.randn(1, 3, 8, 128, 128).cuda()

   # model.forward(x)
    print(model)
    out = model(x)
    print(out.shape)
    """
    a = torch.randn(32, 256, 2, 28, 28)
    pool = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=0)

    b = pool(a)

    print(b.shape)
    """
    # loss_fn = nn.CrossEntropyLoss()
    # output = model(x)
    # output = output.double()
    # target = torch.empty(32, dtype=torch.long).random_(16).cuda()
    # res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    # print(res)





