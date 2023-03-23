import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_channels:int, out_channels:int, kernel_size:int=3):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())
def double_conv_block(in_channels:int, out_channels:int, kernel_size:int=3):
    return nn.Sequential(conv_bn_relu(in_channels, out_channels, kernel_size), conv_bn_relu(out_channels, out_channels, kernel_size))

class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        encoder_channels = encoder_channels[1:] # (64, 256, 512, 1024, 2048)
        encoder_channels = encoder_channels[::-1] # (2048, 1024, 512, 256, 64)
       
        n_blocks = len(decoder_channels)
        blocks = []
        prev_channels = 0
        for i in range(n_blocks):
            blocks.append(double_conv_block(
                in_channels=encoder_channels[i]+prev_channels,
                out_channels=decoder_channels[i]))
            prev_channels = decoder_channels[i]
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, *features):
        features = features[1:] # without input 
        features = features[::-1] # reverse deep -> shallow
        cat_x = features[0]
        for i in range(len(self.blocks)-1):
            output = self.blocks[i](cat_x)
            cat_x = torch.cat((F.interpolate(output, features[i+1].shape[-2:], mode='bilinear'), 
                              features[i+1]),
                              dim=1)
        output = self.blocks[-1](cat_x)
        return output
            
if __name__ == '__main__':
    x = double_conv_block(3, 32, 3)
    a=1
        