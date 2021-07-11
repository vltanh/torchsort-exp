from torch import nn
from torch.nn import functional as F

__all__ = ['CNN']


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    def __init__(self, hidden, nclasses) -> None:
        super().__init__()
        self.conv1 = ConvBlock(3, hidden)
        self.conv2 = ConvBlock(hidden, hidden)
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * hidden, 512),
            nn.ReLU(),
            nn.Linear(512, nclasses),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
