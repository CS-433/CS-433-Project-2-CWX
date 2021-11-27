import torch
import torch.nn as nn
import torchvision.models
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms.functional import resize
from torchsummary import summary


def double_conv(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
                               padding=(kernel_size - 1) // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
                               padding=(kernel_size - 1) // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = F.relu(y)
        return y


class Unet(nn.Module):
    def __init__(self, in_channel, out_channel, filters=None, nb_res_block=0):
        super(Unet, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.downs.append(double_conv(in_channel, filters[0]))
        for i in range(1, len(filters)):
            self.downs.append(double_conv(filters[i - 1], filters[i]))

        self.bottleneck = double_conv(filters[-1], filters[-1] * 2)

        self.resblocks = nn.Sequential(
            *(ResBlock(filters[-1] * 2, kernel_size=3) for _ in range(nb_res_block))
        )

        self.ups = nn.ModuleList()
        filters = filters[::-1]
        self.ups.append(
            nn.ConvTranspose2d(filters[0] * 2, filters[0], kernel_size=2, stride=2)
        )
        self.ups.append(double_conv(filters[0] * 2, filters[0]))
        for i in range(1, len(filters)):
            self.ups.append(
                nn.ConvTranspose2d(filters[i - 1], filters[i], kernel_size=2, stride=2)
            )
            self.ups.append(double_conv(2 * filters[i], filters[i]))

        self.out = nn.Conv2d(filters[-1], out_channel, kernel_size=1)

    def forward(self, x):
        xs = []
        for down in self.downs:
            x = down(x)
            xs.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bottleneck(x)
        x = self.resblocks(x)
        xs = xs[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            if x.shape[2:] != xs[i // 2].shape[2:]:
                x = resize(x, xs[i // 2].shape[2:])
            x = torch.cat((xs[i // 2], x), dim=1)
            x = self.ups[i + 1](x)

        return self.out(x)


class LinkNetDecoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LinkNetDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel // 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channel // 4)

        self.up = nn.ConvTranspose2d(in_channel // 4, in_channel // 4, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        self.bn2 = nn.BatchNorm2d(in_channel // 4)
        self.conv2 = nn.Conv2d(in_channel // 4, out_channel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.up(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = F.relu(x)
        return x


class LinkNet(nn.Module):
    def __init__(self, in_channel, out_channel, filters=None, resnet=None, pretrained=False):
        super(LinkNet, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        if resnet is None:
            resnet = models.resnet34(pretrained=pretrained)
        assert len(filters) == 4
        self.conv1 = resnet.conv1 if in_channel == 3 else nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = resnet.bn1
        self.maxpool1 = resnet.maxpool
        self.encoders = nn.ModuleList()
        self.encoders.append(resnet.layer1)
        self.encoders.append(resnet.layer2)
        self.encoders.append(resnet.layer3)
        self.encoders.append(resnet.layer4)

        self.decoders = nn.ModuleList()
        filters = filters[::-1]
        for i in range(len(filters) - 1):
            self.decoders.append(LinkNetDecoder(filters[i], filters[i + 1]))
        self.decoders.append(LinkNetDecoder(filters[3], filters[3]))
        self.up = nn.ConvTranspose2d(filters[3], 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, out_channel, kernel_size=1)

    def forward(self, x):
        xs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        for enc in self.encoders:
            x = enc(x)
            xs.append(x)
        xs = xs[::-1]
        for i in range(3):
            x = self.decoders[i](x)
            if x.shape[2:] != xs[i + 1].shape[2:]:
                x = resize(x, xs[i + 1].shape[2:])
            x = x + xs[i + 1]
        x = self.decoders[3](x)

        x = self.up(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


class DBlock(nn.Module):
    def __init__(self, channel):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1, bias=False)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, bias=False)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4, bias=False)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8, bias=False)

    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x))
        dilate2_out = F.relu(self.dilate2(dilate1_out))
        dilate3_out = F.relu(self.dilate3(dilate2_out))
        dilate4_out = F.relu(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DinkNet(nn.Module):
    def __init__(self, in_channel, out_channel, filters=None, resnet=None, pretrained=False):
        super(DinkNet, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        if resnet is None:
            resnet = models.resnet34(pretrained=pretrained)
        assert len(filters) == 4
        self.conv1 = resnet.conv1 if in_channel == 3 else nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = resnet.bn1
        self.maxpool1 = resnet.maxpool
        self.encoders = nn.ModuleList()
        self.encoders.append(resnet.layer1)
        self.encoders.append(resnet.layer2)
        self.encoders.append(resnet.layer3)
        self.encoders.append(resnet.layer4)

        self.dblock = DBlock(filters[3])

        self.decoders = nn.ModuleList()
        filters = filters[::-1]
        for i in range(len(filters) - 1):
            self.decoders.append(LinkNetDecoder(filters[i], filters[i + 1]))
        self.decoders.append(LinkNetDecoder(filters[3], filters[3]))
        self.up = nn.ConvTranspose2d(filters[3], 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, out_channel, kernel_size=1)

    def forward(self, x):
        xs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        for enc in self.encoders:
            x = enc(x)
            xs.append(x)

        x = self.dblock(x)

        xs = xs[::-1]
        for i in range(3):
            x = self.decoders[i](x)
            if x.shape[2:] != xs[i + 1].shape[2:]:
                x = resize(x, xs[i + 1].shape[2:])
            x = x + xs[i + 1]
        x = self.decoders[3](x)

        x = self.up(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


if __name__ == '__main__':
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_pretrained('efficientnet-b0').cuda()
    img=torch.zeros((1,3,400,400)).cuda()
    features = model.extract_features(img)
    print(features.shape)
