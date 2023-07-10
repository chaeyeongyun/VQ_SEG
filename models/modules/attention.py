from functools import reduce
from typing import List
import torch
from torch.nn import Module, Conv2d, Softmax, Parameter, ModuleList, Identity
from torch import nn
from functools import reduce
def make_attentions(attention:Module, encoder_channels, flag):
    attentions = [attention(ch) if f else Identity() for ch, f in zip(encoder_channels, flag)]
    attentions = ModuleList(attentions)
    return attentions
class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 bn_eps=1e-5,
                 activation=nn.ReLU):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=bn_eps)
        if self.activate:
            self.activ = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x
    
class SKA(nn.Module):
    """
        SKNet specific convolution block.

        Parameters:
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        stride : int or tuple/list of 2 int
            Strides of the convolution.
        groups : int, default 32
            Number of groups in branches.
        num_branches : int, default 2
            Number of branches (`M` parameter in the paper).
        reduction : int, default 16
            Reduction value for intermediate channels (`r` parameter in the paper).
        min_channels : int, default 32
            Minimal number of intermediate channels (`L` parameter in the paper).
    """
    def __init__(self,
                    in_channels,
                    out_channels,
                    stride=1,
                    num_branches=2,
                    reduction=16,
                    min_channels=32):
        super(SKA, self).__init__()
        self.num_branches = num_branches
        self.out_channels = out_channels
        mid_channels = max(in_channels // reduction, min_channels)

        branches =[]
        for i in range(num_branches):
            branches.append(
                ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(2*(i+1)+1),
                stride=stride,
                padding=1,
                groups=in_channels)
                ) # depthwise
        self.branches = nn.ModuleList(branches)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = ConvBlock(
            in_channels=out_channels,
            out_channels=mid_channels,
            kernel_size=1)
        self.fc2 = Conv2d(
            in_channels=mid_channels,
            out_channels=(out_channels * num_branches), kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        outs = []
        for branch in self.branches:
            outs.append(branch(x))
    
        u = reduce(lambda x,y: x+y, outs) # (N, C, H, W)
        s = self.pool(u) # (N, C, 1, 1)
        z = self.fc1(s) # (N, C/r, 1, 1)
        w = self.fc2(z) # (N, C*2, 1, 1)

        batch = w.size(0)
        w = w.view(batch, self.num_branches, self.out_channels) # (N, 2, C)
        w = self.softmax(w) # (N, 2, C)
        w = w.unsqueeze(-1).unsqueeze(-1) # (N, 2, C, 1, 1)
        for i in range(len(outs)):
            outs[i] = outs[i] * w[:, i, :, :, :]  #(N, C, 1, 1)
        y = reduce(lambda x, y:x+y, outs)
        return y
        



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
### CCA ### contrast-aware channel attention module
def nanstd(x): 
    return torch.sqrt(torch.mean(torch.pow(x-torch.nanmean(x,dim=1).unsqueeze(-1),2)))
class CCA(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) :
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(in_channels, in_channels//16, 1, bias=True),
                                 nn.ReLU(),
                                nn.Conv2d(in_channels//16, in_channels, 1, bias=True),
                                )
        
        # self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False) if out_channels != in_channels else nn.Identity()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False, groups=in_channels),
                                  nn.BatchNorm2d(in_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(),
                                  )
        # self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
       
    def forward(self, x):
        mean = torch.nanmean(x, dim=(2, 3), keepdim=True) # (B, C, 1, 1)
        # std = torch.std(x, dim=(2, 3), keepdim=True) # (B, C, 1, 1)
        std = torch.sqrt(torch.nanmean((x-mean).pow(2), dim=(2, 3), keepdim=True))
        weight = mean + std
        weight = self.mlp(weight)
        weight = torch.sigmoid(weight)
        output = x * weight
        output = self.conv(output)
        return output 
        # return output
        
### IMDB ###
class CL(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', activation=nn.ReLU):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode),
                        activation()]
        super().__init__(*layers)
        
class ContrastAttention(nn.Module):
    def __init__(self, in_channels:int) :
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(in_channels, in_channels//16, 1, bias=True),
                                 nn.ReLU(),
                                nn.Conv2d(in_channels//16, in_channels, 1, bias=True),
                                )
        
        
    def forward(self, x):
        mean = torch.nanmean(x, dim=(2, 3), keepdim=True) # (B, C, 1, 1)
        std = torch.sqrt(torch.nanmean((x-mean).pow(2), dim=(2, 3), keepdim=True))
        weight = mean + std
        weight = self.mlp(weight)
        weight = torch.sigmoid(weight)
        output = x * weight
        return output  
    
class IMDB(nn.Module):
    def __init__(self, in_channels,  split=3, activation=nn.GELU) :
        super().__init__()
        self.split = split
        self.in_channels = in_channels
        self.refine_channels = in_channels // (split+1)
        self.first_conv = CL(in_channels, in_channels, 3, padding=1,  activation=activation)
        self.split_conv = nn.ModuleList([CL((in_channels-self.refine_channels), in_channels, 3, padding=1, activation=activation) for _ in range(split-1)] \
                                                                + [CL((in_channels-self.refine_channels), self.refine_channels, 3, padding=1, activation=activation)])
        self.cca = ContrastAttention(in_channels)
        self.last_conv = nn.Conv2d(self.refine_channels*(split+1), in_channels, 1, bias=False)
        
    def forward(self, x):
        first_conv_out = self.first_conv(x)
        refine_list = []
        
        course = first_conv_out
        for i in range(self.split):
            refine, course = torch.split(course, [self.refine_channels, self.in_channels-self.refine_channels], dim=1)[:]
            refine_list.append(refine)
            course = self.split_conv[i](course)
        cat_feat = torch.cat(refine_list+[course], dim=1)
        cca_out = self.cca(cat_feat)
        output = self.last_conv(cca_out)
        return x + output    
        
