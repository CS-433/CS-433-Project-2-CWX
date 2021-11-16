import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize


def double_conv(in_channel, out_channel, p=0.3):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.Dropout2d(p=p),
        nn.ReLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size, p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
                               padding=(kernel_size - 1) // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
                               padding=(kernel_size - 1) // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(nb_channels)
        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = self.dropout(y)
        y = F.relu(y)
        return y


class Unet(nn.Module):
    def __init__(self, in_channel, out_channel, nb_res_block=0, p=0.3):
        super(Unet, self).__init__()
        self.downs = nn.ModuleList()
        self.downs.append(double_conv(in_channel, 64))
        for i in range(3):
            self.downs.append(double_conv(64 * (2 ** i), 64 * (2 ** (i + 1)), p=p))

        self.resblocks = nn.Sequential(
            *(ResBlock(1024, kernel_size=3, p=p) for _ in range(nb_res_block))
        )

        self.ups = nn.ModuleList()
        self.ups.append(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )
        self.ups.append(double_conv(1024, 512, p=p))

        for i in range(3):
            self.ups.append(
                nn.ConvTranspose2d(512 // (2 ** i), 512 // (2 ** (i + 1)), kernel_size=2, stride=2)
            )
            self.ups.append(double_conv(512 // (2 ** i), 512 // (2 ** (i + 1)), p=p))

        self.bottleneck = double_conv(512, 1024, p=p)
        self.out = nn.Conv2d(64, out_channel, kernel_size=1)

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
            if x.size() != xs[i // 2].size():
                x = resize(x, xs[i // 2].shape[2:])
            x = torch.cat((xs[i // 2], x), dim=1)
            x = self.ups[i + 1](x)

        return self.out(x)


if __name__ == '__main__':
    a = torch.zeros((1, 3, 161, 161))
    model = Unet(3, 1)
    print(model(a).shape)
