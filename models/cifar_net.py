from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, p=0.2):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(p),
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)

        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = out + x
        out = self.relu(out)
        out = self.bn2(out)

        return out


class CIFAR10Net(nn.Module):
    def __init__(self, num_repeats=(5, 5, 5)):
        super(CIFAR10Net, self).__init__()

        self.conv = nn.Sequential(
            *self.make_layers(3, 32, num_repeats[0], 1),
            *self.make_layers(32, 64, num_repeats[1], 2),
            *self.make_layers(64, 128, num_repeats[2], 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, 10)
        )

    def make_layers(self, in_ch, out_ch, num_repeat_layers, stride):
        layers = []
        layers.append(ConvBlock(in_ch, out_ch, stride))
        for _ in range(num_repeat_layers):
            layers.append(ResidualBlock(out_ch))
        return layers

    def forward(self, x):
        out = self.conv(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out
