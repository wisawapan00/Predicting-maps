import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, 
                 down_sampling=True, use_dropout=False):
        super().__init__()
        self.down_sampling = down_sampling
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, 
                      padding=padding, bias=False, padding_mode="reflect")
            if self.down_sampling else nn.ConvTranspose2d(in_channels, out_channels, 
                                                     kernel_size=kernel_size, stride=stride, 
                                                     padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if not self.down_sampling else nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, 
                      padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        features=[64, 128, 256, 512, 512, 512, 512]

        self.down1 = Block(features[0], features[1], down_sampling=True, use_dropout=False) # 64 > 128
        self.down2 = Block(features[1], features[2], down_sampling=True, use_dropout=False) # 128 > 256
        self.down3 = Block(features[2], features[3], down_sampling=True, use_dropout=False) # 256 > 512
        self.down4 = Block(features[3], features[4], down_sampling=True, use_dropout=False) # 512 > 512
        self.down5 = Block(features[4], features[5], down_sampling=True, use_dropout=False) # 512 > 512
        self.down6 = Block(features[5], features[6], down_sampling=True, use_dropout=False) # 512 > 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], kernel_size=4, stride=2, padding=1), 
            nn.ReLU()
        )

        features = features[::-1]
        self.initial_up = Block(features[0], features[0], down_sampling=False, use_dropout=True) # 512 > 512
        self.up2 = Block(features[0]*2, features[1], down_sampling=False, use_dropout=True) # 512 > 512
        self.up3 = Block(features[1]*2, features[2], down_sampling=False, use_dropout=True) # 512 > 512
        self.up4 = Block(features[2]*2, features[3], down_sampling=False, use_dropout=False) # 512 > 512
        self.up5 = Block(features[3]*2, features[4], down_sampling=False, use_dropout=False) # 512 > 256
        self.up6 = Block(features[4]*2, features[5], down_sampling=False, use_dropout=False) # 256 > 128
        self.up7 = Block(features[5]*2, features[6], down_sampling=False, use_dropout=False) # 128 > 64

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features[-1]*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        down1 = self.initial_down(x)
        down2 = self.down1(down1)
        down3 = self.down2(down2)
        down4 = self.down3(down3)
        down5 = self.down4(down4)
        down6 = self.down5(down5)
        down7 = self.down6(down6)
        bottleneck = self.bottleneck(down7)
        up1 = self.initial_up(bottleneck)
        up2 = self.up2(torch.cat([up1, down7], 1))
        up3 = self.up3(torch.cat([up2, down6], 1))
        up4 = self.up4(torch.cat([up3, down5], 1))
        up5 = self.up5(torch.cat([up4, down4], 1))
        up6 = self.up6(torch.cat([up5, down3], 1))
        up7 = self.up7(torch.cat([up6, down2], 1))
        return self.final(torch.cat([up7, down1], 1))


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    Gen = Generator()
    Gen_preds = Gen(x)
    print(Gen_preds.shape)