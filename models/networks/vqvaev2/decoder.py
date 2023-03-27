from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out

class VQVAEDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_resblocks, n_res_channels, stride) :
        super().__init__()
        blocks = [nn.Conv2d(in_channels, hidden_channels, 3, padding=1)]
        blocks.extend([ResBlock(hidden_channels, n_res_channels)] * n_resblocks)
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend(
            [
                nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    hidden_channels // 2, out_channels, 4, stride=2, padding=1
                ),
            ])

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(hidden_channels, out_channels, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)