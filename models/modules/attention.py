from functools import reduce
from typing import List
import torch
from torch.nn import Module, Conv2d, Softmax, Parameter, ModuleList, Identity
from torch import nn
def make_attentions(attention:Module, encoder_channels, flag):
    attentions = [attention(ch) if f else Identity() for ch, f in zip(encoder_channels, flag)]
    attentions = ModuleList(attentions)
    return attentions
    
class DualAttention(Module):
    def __init__(self, in_dim:int):
        super().__init__()
        self.pam = PAM_Module(in_dim)
        self.cam = CAM_Module(in_dim)
    def forward(self, x):
        pam_out = self.pam(x)
        cam_out = self.cam(pam_out)
        return cam_out
        
class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
##### ------ DRSAM
class DRSAM(nn.Module):
    def __init__(self, in_channels, kernel_size_list:List=[3, 7]) :
        super().__init__()
        self.dwconv_blocks = nn.ModuleList([DWConvBlock(in_channels, in_channels, kernel_size=i) for i in kernel_size_list])
        self.fc_list = nn.ModuleList([FC(in_channels) for _ in range(len(kernel_size_list))])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.last_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        conv_outs = [dwconv(x) for dwconv in self.dwconv_blocks] 
        gap_outs = [torch.squeeze(self.gap(conv_out)) for conv_out in conv_outs] # (N, C) x len(kernel_size_list)
        fc_outs = [fc(gap_out) for gap_out, fc in zip(gap_outs, self.fc_list)] # (N, C) x len(kernel_size_list)
        weights = torch.stack(fc_outs, dim=1) # (N, len(kernel_size_list), C)
        weights = torch.softmax(weights, dim=1) # (N, len(kernel_size_list), C)
        weighted = [weights[:, i, :].unsqueeze(-1).unsqueeze(-1) * conv_out for i, conv_out in enumerate(conv_outs)]
        output = reduce(lambda x,y:x+y, weighted)
        output = self.last_conv(output)
        return output
    
class DWConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, padding_mode='reflect'):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=bias, groups=in_channels, padding_mode=padding_mode),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU()]
        super().__init__(*layers)
class FC(nn.Sequential):
    def __init__(self, in_channels):
        layers = [nn.Linear(in_channels, in_channels//2),
                nn.Linear(in_channels//2, in_channels)]
        super().__init__(*layers)
        
    