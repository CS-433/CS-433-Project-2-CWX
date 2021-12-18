import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torchvision.transforms.functional import resize
from torchsummary import summary
import torchvision.models as model


def double_conv(in_channel, out_channel):
    """It returns a double convolutional layer"""
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    )


class ResBlock(nn.Module):
    """ResBlock class"""

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
    """Unet class"""

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
    """LinkNet decoder"""

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
    """LinkNet"""

    def __init__(self, in_channel, out_channel, filters=None, resnet=None, pretrained=False):
        super(LinkNet, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        if resnet is None:
            resnet = model.resnet34(pretrained=pretrained)
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


class LinkNet1(nn.Module):
    """LinkNet with + in cross connection replaced by concatenation and convolutional layer"""

    def __init__(self, in_channel, out_channel, filters=None, resnet=None, pretrained=False):
        super(LinkNet1, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        if resnet is None:
            resnet = model.resnet34(pretrained=pretrained)
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
        self.cs = nn.ModuleList()
        filters = filters[::-1]
        for i in range(len(filters) - 1):
            self.decoders.append(LinkNetDecoder(filters[i], filters[i + 1]))
            self.cs.append(nn.Conv2d(2 * filters[i + 1], filters[i + 1], kernel_size=1))
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
            x = torch.cat((xs[i + 1], x), dim=1)
            x = self.cs[i](x)
        x = self.decoders[3](x)

        x = self.up(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


class DBlock(nn.Module):
    """DinkNet block, to dilate the field of view"""

    def __init__(self, channel):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1, bias=False)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, bias=False)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4, bias=False)

    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x))
        dilate2_out = F.relu(self.dilate2(dilate1_out))
        dilate3_out = F.relu(self.dilate3(dilate2_out))
        out = x + dilate1_out + dilate2_out + dilate3_out
        return out


class DinkNet(nn.Module):
    """DinkNet"""

    def __init__(self, in_channel, out_channel, filters=None, resnet=None, pretrained=False):
        super(DinkNet, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        if resnet is None:
            resnet = model.resnet34(pretrained=pretrained)
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


class UnetPP(nn.Module):
    """Unet++"""

    def __init__(self, in_channel, out_channel, filters=None):
        super(UnetPP, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512, 1024]

        self.conv1 = nn.Conv2d(in_channel, filters[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.downs = nn.ModuleList()
        for i in range(len(filters) - 1):
            self.downs.append(double_conv(filters[i], filters[i + 1]))

        self.ups = nn.ModuleDict()

        for i in range(len(filters)):
            self.ups[str(i)] = nn.ModuleList()

        for i in range(len(filters)):
            for j in range(i):
                if i - j == len(filters) - 1:
                    self.ups[str(i)].append(nn.ConvTranspose2d(filters[-1], filters[- 2], kernel_size=2, stride=2))
                    self.ups[str(i)].append(double_conv((j + 2) * filters[-2], filters[- 2]))
                    continue
                self.ups[str(i)].append(nn.ConvTranspose2d(filters[i - j], filters[i - j - 1], kernel_size=2, stride=2))
                self.ups[str(i)].append(double_conv((j + 2) * filters[i - j - 1], filters[i - j - 1]))

        self.convTransposes = nn.ModuleList()
        self.outputs = nn.ModuleList()
        for i in range(0, len(filters)):
            self.convTransposes.append(nn.ConvTranspose2d(filters[0], 32, kernel_size=2, stride=2))
            self.outputs.append(nn.Conv2d(32, out_channel, kernel_size=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        xs = dict()
        for i, down in enumerate(self.downs):
            xs[str(i)] = []
            xs[str(i)].append(x)
            x = down(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        xs[str(len(self.downs))] = []
        xs[str(len(self.downs))].append(x)

        for k in self.ups.keys():
            for i in range(0, 2 * int(k), 2):
                x = self.ups[k][i](xs[k][i // 2])
                for j in range(1, i // 2 + 2, 1):
                    if x.shape[2:] != xs[str(int(k) - j)][i // 2 + 1 - j].shape[2:]:
                        x = resize(x, xs[str(int(k) - j)][i // 2 + 1 - j].shape[2:])
                    x = torch.cat((xs[str(int(k) - j)][i // 2 + 1 - j], x), dim=1)
                x = self.ups[k][i + 1](x)
                xs[k].append(x)

        outs = []
        for k in xs.keys():
            index = int(k)
            if index == 0:
                continue
            x = self.convTransposes[index - 1](xs[k][index])
            x = self.outputs[index - 1](x)
            outs.append(x)
        x = outs[0]
        for i in range(1, len(outs), 1):
            x = torch.cat((x, outs[i]), dim=1)

        return x


class LinkNetPP(nn.Module):
    """LinkNet++"""

    def __init__(self, in_channel, out_channel, filters=None, resnet=None, pretrained=False):
        super(LinkNetPP, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        if resnet is None:
            resnet = model.resnet34(pretrained=pretrained)

        self.conv1 = resnet.conv1 if in_channel == 3 else nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = resnet.bn1
        self.maxpool1 = resnet.maxpool
        self.downs = nn.ModuleList()
        self.downs.append(resnet.layer1)
        self.downs.append(resnet.layer2)
        self.downs.append(resnet.layer3)
        self.downs.append(resnet.layer4)

        self.ups = nn.ModuleDict()

        for i in range(len(filters)):
            self.ups[str(i)] = nn.ModuleList()

        for i in range(len(filters)):
            for j in range(i):
                if i - j == len(filters) - 1:
                    self.ups[str(i)].append(LinkNetDecoder(filters[-1], filters[- 2]))
                    self.ups[str(i)].append(nn.Conv2d((j + 2) * filters[-2], filters[- 2], kernel_size=1))
                    continue
                self.ups[str(i)].append(LinkNetDecoder(filters[i - j], filters[i - j - 1]))
                self.ups[str(i)].append(nn.Conv2d((j + 2) * filters[i - j - 1], filters[i - j - 1], kernel_size=1))

        self.convTransposes = nn.ModuleList()
        self.outputs = nn.ModuleList()
        for i in range(0, len(filters)):
            self.outputs.append(nn.Sequential(
                LinkNetDecoder(filters[0], filters[0]),
                nn.ConvTranspose2d(filters[0], 32, kernel_size=3, stride=2),
                nn.Conv2d(32, 32, kernel_size=2),
                nn.Conv2d(32, out_channel, kernel_size=1)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        xs = dict()
        for i, down in enumerate(self.downs):
            x = down(x)
            xs[str(i)] = []
            xs[str(i)].append(x)

        for k in self.ups.keys():
            for i in range(0, 2 * int(k), 2):
                x = self.ups[k][i](xs[k][i // 2])
                for j in range(1, i // 2 + 2, 1):
                    if x.shape[2:] != xs[str(int(k) - j)][i // 2 + 1 - j].shape[2:]:
                        x = resize(x, xs[str(int(k) - j)][i // 2 + 1 - j].shape[2:])
                    x = torch.cat((xs[str(int(k) - j)][i // 2 + 1 - j], x), dim=1)
                x = self.ups[k][i + 1](x)
                xs[k].append(x)

        outs = []
        for k in xs.keys():
            index = int(k)
            if index == 0:
                continue
            x = self.outputs[index - 1](xs[k][index])
            outs.append(x)
        x = outs[0]
        for i in range(1, len(outs), 1):
            x = torch.cat((x, outs[i]), dim=1)
        return x


class DoubleUnet(nn.Module):
    """DoubleUnet"""

    def __init__(self, in_channel, out_channel, filters1=None, filters2=None):
        super(DoubleUnet, self).__init__()
        if filters1 is None:
            filters1 = [64, 128, 256, 512]
        if filters2 is None:
            filters2 = [32, 64, 128, 256]
        self.downs1 = nn.ModuleList()
        self.downs1.append(double_conv(in_channel, filters1[0]))
        for i in range(1, len(filters1)):
            self.downs1.append(double_conv(filters1[i - 1], filters1[i]))

        self.bottleneck1 = double_conv(filters1[-1], filters1[-1] * 2)

        self.ups1 = nn.ModuleList()
        filters1 = filters1[::-1]
        self.ups1.append(
            nn.ConvTranspose2d(filters1[0] * 2, filters1[0], kernel_size=2, stride=2)
        )
        self.ups1.append(double_conv(filters1[0] * 2, filters1[0]))
        for i in range(1, len(filters1)):
            self.ups1.append(
                nn.ConvTranspose2d(filters1[i - 1], filters1[i], kernel_size=2, stride=2)
            )
            self.ups1.append(double_conv(2 * filters1[i], filters1[i]))

        self.out1 = nn.Conv2d(filters1[-1], out_channel, kernel_size=1)

        self.downs2 = nn.ModuleList()
        self.downs2.append(double_conv(filters1[-1], filters2[0]))
        for i in range(1, len(filters2)):
            self.downs2.append(double_conv(filters2[i - 1], filters2[i]))

        self.bottleneck2 = double_conv(filters2[-1], filters2[-1] * 2)

        self.ups2 = nn.ModuleList()
        filters2 = filters2[::-1]
        self.ups2.append(
            nn.ConvTranspose2d(filters2[0] * 2, filters2[0], kernel_size=2, stride=2)
        )
        self.ups2.append(double_conv(filters2[0] * 2 + filters1[0], filters2[0]))
        for i in range(1, len(filters2)):
            self.ups2.append(
                nn.ConvTranspose2d(filters2[i - 1], filters2[i], kernel_size=2, stride=2)
            )
            self.ups2.append(double_conv(2 * filters2[i] + filters1[i], filters2[i]))

        self.out2 = nn.Conv2d(filters2[-1], out_channel, kernel_size=1)

    def forward(self, x):

        xs1 = []
        for down in self.downs1:
            x = down(x)
            xs1.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bottleneck1(x)
        xs1 = xs1[::-1]

        for i in range(0, len(self.ups1), 2):
            x = self.ups1[i](x)
            if x.shape[2:] != xs1[i // 2].shape[2:]:
                x = resize(x, xs1[i // 2].shape[2:])
            x = torch.cat((xs1[i // 2], x), dim=1)
            x = self.ups1[i + 1](x)
        outputs = self.out1(x)
        xs2 = []
        for down in self.downs2:
            x = down(x)
            xs2.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bottleneck2(x)
        xs2 = xs2[::-1]

        for i in range(0, len(self.ups2), 2):
            x = self.ups2[i](x)
            if x.shape[2:] != xs2[i // 2].shape[2:]:
                x = resize(x, xs2[i // 2].shape[2:])
            x = torch.cat((xs1[i // 2], xs2[i // 2], x), dim=1)
            x = self.ups2[i + 1](x)
        outputs = torch.cat((outputs, self.out2(x)), dim=1)
        return outputs


if __name__ == '__main__':
    model = LinkNet(in_channel=3, out_channel=1, resnet=torchvision.models.resnet152(pretrained=False),
                    filters=[256, 512, 1024, 2048])
    summary(model.cuda(), input_size=(3, 400, 400))
