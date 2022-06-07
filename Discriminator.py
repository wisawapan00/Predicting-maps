import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, 
                      padding=padding, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels=64, kernel_size=4, stride=2,
                padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        features=[64, 128, 256, 512]
        for feature in features[:-1]:
            layers.append(
                CNNBlock(feature, feature*2)
            )

        layers.append(
            nn.Conv2d(
                features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    Dis = Discriminator()
    Dis_preds = Dis(x, y)
    print(Dis_preds.shape)