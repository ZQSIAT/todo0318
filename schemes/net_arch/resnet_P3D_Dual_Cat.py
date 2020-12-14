import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNetCatP3', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1, 1), downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 3, 3),
                               stride=(1, stride[1], stride[2]), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1),
                               stride=(stride[0], 1, 1), padding=(1, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
                               stride=1, padding=(0, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(planes)

        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1),
                               stride=1, padding=(1, 0, 0), bias=False)
        self.bn4 = nn.BatchNorm3d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1, 1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
                               stride=(1, stride[1], stride[2]), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1),
                               stride=(stride[0], 1, 1), padding=(1, 0, 0), bias=False)
        self.bn3 = nn.BatchNorm3d(planes)

        self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm3d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetCatP3(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 in_channel,
                 sample_size=224,
                 sample_duration=8,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNetCatP3, self).__init__()
        self.in_channel = in_channel

        assert in_channel == 4, "Invalid input dim"
        self.conv1_D = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1_D = nn.BatchNorm3d(64)

        self.conv1_G = nn.Conv3d(in_channel-1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1_G = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)

        self.inplanes = self.inplanes * 2

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 8))

        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x_d = self.conv1_D(x[:, 3].unsqueeze(1))  # Depth
        x_d = self.bn1_D(x_d)
        x_d = self.relu(x_d)
        x_d = self.maxpool(x_d)

        x_g = self.conv1_G(x[:, 0:3])  # Gradient
        x_g = self.bn1_G(x_g)
        x_g = self.relu(x_g)
        x_g = self.maxpool(x_g)
        # fusing by adding
        x = torch.cat((x_g, x_d), dim=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNetCatP3(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNetCatP3(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNetCatP3(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNetCatP3(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrain = torch.load("/home/xp_ji/workspace/pt-work/action-net/pre_trained/resnet-50-kinetics.pth")
        #state_dict = model_checkpoint["state_dict"]

        pretrained_dict = pretrain["state_dict"]

        model_dict = model.state_dict()
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if model_dict in }

        #print(pretrained_dict.keys())

        # print(pretrain['state_dict'].keys())
        # #print(list(model.named_parameters()))
        # print(pretrain['arch'])
        # print(model.state_dict().keys())
        model.load_state_dict(pretrained_dict, strict=False)

        print("=> using pretrain model")

    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNetCatP3(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNetCatP3(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNetCatP3(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model




if __name__ == "__main__":
    x = torch.randn((1, 4, 8, 224, 224))
    model = resnet34(in_channel=4)
    out = model(x)
    print(out.shape)