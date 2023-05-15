from typing import Optional

import torch
import torch.nn as nn
from models.networks.unet.decoder import UnetDecoder
from models.encoders import make_encoder
from models.modules.segmentation_head import SegmentationHead, AngularSegmentationHead, AngularSegmentationHeadv2
from models.modules.prototype import *
from models.modules.attention import DualAttention, make_attentions
from vector_quantizer.vq_img import VectorQuantizer
from vector_quantizer import make_vq_module
from loss import SupConLoss

 
class VQUnet_v1(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth)    
        encoder_channels = self.encoder.out_channels()
        self.codebook = VectorQuantizer(**vq_cfg, dim=encoder_channels[-1])
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
    def forward(self, x, code_usage_loss=False):
        features = self.encoder(x)[1:]
        quantize, _embed_index, commitment_loss, code_usage = self.codebook(features[-1]) 
        features[-1] = quantize
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        if code_usage_loss: 
            usage_loss = code_usage
            return output, commitment_loss, code_usage.detach().cpu(), usage_loss
        return output, commitment_loss, code_usage.detach().cpu()

    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.codebook.parameters():
            param.requires_grad = False
        
        
class VQUnet_v2(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights)    
        encoder_channels = self.encoder.out_channels()
        
        # if isinstance(vq_cfg.num_embeddings, (int)):
        #     self.codebook = nn.ModuleList([VectorQuantizer(**vq_cfg, dim=encoder_channels[i+1])for i in range(depth)])
        # elif isinstance(vq_cfg.num_embeddings, list):
        #     l = []
        #     for i in range(depth):
        #         num_embeddings = copy.deepcopy(vq_cfg.num_embeddings)
        #         vq_cfg.num_embeddings = num_embeddings[i]
        #         l.append(VectorQuantizer(**vq_cfg, dim=encoder_channels[i+1]))
        #     self.codebook = nn.ModuleList(l)
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
    def forward(self, x, code_usage_loss=False):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst)
    
    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.codebook.parameters():
            param.requires_grad = False

class VQPTUnet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        margin=1.5,
        scale=1.,
        use_feature=False,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights, padding_mode='reflect')    
        encoder_channels = self.encoder.out_channels()
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
        self.prototype_loss = PrototypeLoss(num_classes, decoder_channels[-1], margin=margin, scale=scale, use_feature=use_feature)
        self.device = None
        
    def forward(self, x, gt=None, code_usage_loss=False):
        if self.device is None:
            self.device = x.device
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        prototype_loss = self.prototype_loss(decoder_out, gt) if self.training else None
        output = self.segmentation_head(decoder_out)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
    
    @torch.no_grad()
    def pseudo_label(self, x):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        for i in range(len(features)):
            quantize, _, _, _ = self.codebook[i](features[i])
            features[i] = quantize
        
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        return torch.argmax(output, dim=1).long()
    
    
    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)
   
    

class VQEuPTUnet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        learnable_alpha=True,
        use_feature=False,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights, padding_mode='reflect')    
        encoder_channels = self.encoder.out_channels()
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
        self.prototype_loss = EuclideanPrototypeLoss(num_classes, decoder_channels[-1], use_feature=use_feature) if not learnable_alpha \
            else LearnableEuclideanPrototypeLoss(num_classes, decoder_channels[-1], use_feature=use_feature)
        
    def forward(self, x, gt=None, code_usage_loss=False):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        prototype_loss = self.prototype_loss(decoder_out, gt) if self.training else None
        output = self.segmentation_head(decoder_out)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
    
    @torch.no_grad()
    def pseudo_label(self, x):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        for i in range(len(features)):
            quantize, _, _, _ = self.codebook[i](features[i])
            features[i] = quantize
        
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        return torch.argmax(output, dim=1).long()
    
    
    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)


class VQASHUnet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        margin=1.5,
        scale=1.,
        use_feature=False,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights, padding_mode='reflect')    
        encoder_channels = self.encoder.out_channels()
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = AngularSegmentationHead(decoder_channels[-1], decoder_channels[-1], num_classes, decoder_channels[-1],
                                                         scale=scale, margin=margin, upsampling=upsampling, activation=activation)
        
    def forward(self, x, gt=None, code_usage_loss=False):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        
        output, angular_loss = self.segmentation_head(decoder_out, gt)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), angular_loss
    
    @torch.no_grad()
    def pseudo_label(self, x):
        self.eval()
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        for i in range(len(features)):
            quantize, _, _, _ = self.codebook[i](features[i])
            features[i] = quantize
        
        decoder_out = self.decoder(*features)
        output, _ = self.segmentation_head(decoder_out, None)
        self.train()
        return torch.argmax(output, dim=1).long()
    
    
    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)

class VQASHUnetv2(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        margin=1.5,
        scale=1.,
        use_feature=False,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights)    
        encoder_channels = self.encoder.out_channels()
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = AngularSegmentationHeadv2(decoder_channels[-1], decoder_channels[-1], num_classes, decoder_channels[-1],
                                                         scale=scale, margin=margin, upsampling=upsampling, activation=activation)
        
    def forward(self, x, gt=None, code_usage_loss=False):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        
        output, angular_loss, seghead_commit_loss = self.segmentation_head(decoder_out, gt)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), angular_loss, seghead_commit_loss
    
    @torch.no_grad()
    def pseudo_label(self, x):
        self.eval()
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        for i in range(len(features)):
            quantize, _, _, _ = self.codebook[i](features[i])
            features[i] = quantize
        
        decoder_out = self.decoder(*features)
        output, _, _ = self.segmentation_head(decoder_out, None)
        self.train()
        return torch.argmax(output, dim=1).long()
    
    
    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)


class VQUnetwithSalientloss(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth)    
        encoder_channels = self.encoder.out_channels()
        
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
        self.auxiliary_decoder = nn.Sequential(nn.Conv2d(encoder_channels[-1], 512, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(512),
                                               nn.ReLU(),
                                               nn.Upsample(scale_factor=2, mode='bilinear'),
                                               nn.Conv2d(512, 256, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(256),
                                               nn.ReLU(),
                                               nn.Upsample(scale_factor=2, mode='bilinear'),
                                               nn.Conv2d(256, 64, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(64),
                                               nn.ReLU(),
                                               nn.Upsample(scale_factor=2, mode='bilinear'),
                                               nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(32),
                                               nn.ReLU(),
                                               nn.Upsample(scale_factor=2, mode='bilinear'),
                                               nn.Conv2d(32, 1, 3, padding=1, bias=False),
                                               nn.Sigmoid()
                                               )
    def forward(self, x, code_usage_loss=False):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        
        if self.training:
            auxiliary_output = self.auxiliary_decoder(features[-1])
        else:
            return output, loss, torch.tensor(code_usage_lst)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss, auxiliary_output
        return output, loss, torch.tensor(code_usage_lst), auxiliary_output
     
class VQATUnet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
        upsampling=2,
        attention=DualAttention
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights)    
        encoder_channels = self.encoder.out_channels()
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
        flag = list(map(lambda x: x!=0, vq_cfg.num_embeddings))
        self.attention = make_attentions(attention, encoder_channels[1:], flag=flag)
        
    def forward(self, x, code_usage_loss=False):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            features[i] = self.attention[i](features[i])
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst)
    
    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.codebook.parameters():
            param.requires_grad = False
    def get_all_modules(self):
        return {"encoder":self.encoder, 
                "decoder":self.decoder, 
                "seg_head":self.segmentation_head, 
                "codebook":self.codebook, 
                "attention":self.attention}
        
class VQNEDPTUnet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        use_feature=False,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights)    
        encoder_channels = self.encoder.out_channels()
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
        self.prototype_loss = NEDPrototypeLoss(num_classes=num_classes, 
                                               embedding_dim=decoder_channels[-1], 
                                               use_feature=use_feature)
        
    def forward(self, x, gt=None, code_usage_loss=False):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        prototype_loss = self.prototype_loss(decoder_out, gt) if self.training else None
        output = self.segmentation_head(decoder_out)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), prototype_loss
    
    @torch.no_grad()
    def pseudo_label(self, x):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        for i in range(len(features)):
            quantize, _, _, _ = self.codebook[i](features[i])
            features[i] = quantize
        
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        return torch.argmax(output, dim=1).long()
    
    
    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)

class SupConVQUnet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        vq_cfg:dict,
        use_feature=False,
        encoder_weights=None,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Softmax2d,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth, weights=encoder_weights)    
        encoder_channels = self.encoder.out_channels()
        self.codebook = make_vq_module(vq_cfg, encoder_channels, depth)
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
        self.auxiliary_loss = SupConLoss()
        
    def forward(self, x, gt=None, split=None, code_usage_loss=False):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        code_usage_lst = []
        if code_usage_loss : usage_loss = torch.tensor([0.], device=x.device, requires_grad=self.training)
        for i in range(len(features)):
            quantize, _embed_index, commitment_loss, code_usage = self.codebook[i](features[i])
            features[i] = quantize
            # sum
            if commitment_loss is not None: loss = loss + commitment_loss
            
            if code_usage is not None: 
                code_usage_lst.append(code_usage.detach().cpu())
                if code_usage_loss: 
                    usage_loss = usage_loss + code_usage
        # mean
        loss = loss / len(features)
        decoder_out = self.decoder(*features)
        auxiliary_loss = self.auxiliary_loss(decoder_out, gt) if self.training and split=='label' else None
        output = self.segmentation_head(decoder_out)
        if code_usage_loss : 
            usage_loss = usage_loss / len(features)
            return output, loss, torch.tensor(code_usage_lst), usage_loss
        return output, loss, torch.tensor(code_usage_lst), auxiliary_loss
    
    @torch.no_grad()
    def pseudo_label(self, x):
        features = self.encoder(x)[1:]
        if len(features) != len(self.codebook) : raise NotImplementedError
        
        for i in range(len(features)):
            quantize, _, _, _ = self.codebook[i](features[i])
            features[i] = quantize
        
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        return torch.argmax(output, dim=1).long()
    
    
    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)
        
class Unet(nn.Module):
    def __init__(
        self, 
        encoder_name:str,
        num_classes:int,
        in_channels:int=3,
        decoder_channels=None,
        depth:int=5,
        activation=nn.Identity,
        upsampling=2,
        ):
        super().__init__()
        
        self.encoder = make_encoder(encoder_name, in_channels, depth)    
        encoder_channels = self.encoder.out_channels()
        
        if decoder_channels == None:
            decoder_channels = [i//2 for i in encoder_channels[1:]] 
            decoder_channels = decoder_channels[::-1]
        self.decoder = UnetDecoder(encoder_channels,
                                   decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1],
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=3)
    def forward(self, x):
        features = self.encoder(x)[1:]
        decoder_out = self.decoder(*features)
        output = self.segmentation_head(decoder_out)
        
        return output

    def load_pretrained(self, encoder_weights_path, codebook_weights_path):
        print(f"... load pretrained weights ... \nencoder:{encoder_weights_path}, codebook:{codebook_weights_path}")
        encoder_weights = torch.load(encoder_weights_path)
        codebook_weights = torch.load(codebook_weights_path)
        self.encoder.load_state_dict(encoder_weights)
        self.codebook.load_state_dict(codebook_weights)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

