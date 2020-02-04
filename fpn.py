import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class FPN(nn.Module):

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(1, len(in_channels)):
            self.lateral_convs.append(nn.Conv2d(in_channels[i], out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        self.fpn_convs.append(nn.Conv2d(in_channels[-1], out_channels, 3, stride=2, padding=1))
        self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m)

    def forward(self, inputs):
        layer1, layer2, layer3, layer4 = inputs

        c1 = self.lateral_convs[0](layer2)
        c2 = self.lateral_convs[1](layer3)
        c3 = self.lateral_convs[2](layer4)

        c2 += F.interpolate(c3, scale_factor=2, mode='nearest')
        c1 += F.interpolate(c2, scale_factor=2, mode='nearest')

        p1 = self.fpn_convs[0](c1)
        p2 = self.fpn_convs[1](c2)
        p3 = self.fpn_convs[2](c3)

        # part 2: add extra levels
        p4 = self.fpn_convs[3](layer4)
        p5 = self.fpn_convs[4](p4)

        return p1, p2, p3, p4, p5


if __name__ == '__main__':
    from resnet import ResNet

    net = ResNet()
    fpn = FPN()
    import torch

    x = torch.randn((1, 3, 800, 800))
    feats = net(x)
    feats = fpn(feats)
    print([feat.shape for feat in feats])
