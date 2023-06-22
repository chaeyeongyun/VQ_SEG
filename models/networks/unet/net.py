from typing import Optional

import torch
import torch.nn as nn
from models.networks.unet.decoder import UnetDecoder
from models.encoders import make_encoder
from models.modules.segmentation_head import SegmentationHead, AngularSegmentationHead, AngularSegmentationHeadv2
from models.modules.prototype import *
from models.modules.attention import DualAttention, make_attentions, CCA
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
        activation=nn.Identity,
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
    def get_all_modules(self):
        return {"encoder":self.encoder, 
                "decoder":self.decoder, 
                "seg_head":self.segmentation_head, 
                "codebook":self.codebook, 
                "prototype_loss":self.prototype_loss}
    

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
        encoder_weights=None
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

class DBConv(nn.Sequential):
    '''
    double 3x3 conv layers with Batch normalization and ReLU
    '''
    def __init__(self, in_channels, out_channels):
        conv_layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(DBConv, self).__init__(*conv_layers)


class ContractingPath(nn.Module):
    def __init__(self, in_channels, first_outchannels):
        super(ContractingPath, self).__init__()
        self.conv1 = DBConv(in_channels, first_outchannels)
        self.conv2 = DBConv(first_outchannels, first_outchannels*2)
        self.conv3 = DBConv(first_outchannels*2, first_outchannels*4)
        self.conv4 = DBConv(first_outchannels*4, first_outchannels*8)

        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        output1 = self.conv1(x) # (N, 64, 568, 568)
        output = self.maxpool(output1) # (N, 64, 284, 284)
        output2 = self.conv2(output) # (N, 128, 280, 280)
        output = self.maxpool(output2) # (N, 128, 140, 140)
        output3 = self.conv3(output) # (N, 256, 136, 136)
        output = self.maxpool(output3) # (N, 256, 68, 68)
        output4 = self.conv4(output) # (N, 512, 64, 64)
        output = self.maxpool(output4) # (N, 512, 32, 32)
        return output1, output2, output3, output4, output


class ExpansivePath(nn.Module):
    '''
    pass1, pass2, pass3, pass4 are the featuremaps passed from Contracting path
    '''
    def __init__(self, in_channels):
        super(ExpansivePath, self).__init__()
        # input (N, 1024, 28, 28)
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2) # (N, 512, 56, 56)
        self.conv1 = DBConv(in_channels, in_channels//2) # (N, 512, 52, 52)
        
        self.upconv2 = nn.ConvTranspose2d(in_channels//2, in_channels//4, 2, 2) # (N, 256, 104, 104)
        self.conv2 = DBConv(in_channels//2, in_channels//4) # (N, 256, 100, 100)
        
        self.upconv3 = nn.ConvTranspose2d(in_channels//4, in_channels//8, 2, 2) # (N, 128, 200, 200)
        self.conv3 = DBConv(in_channels//4, in_channels//8) # (N, 128, 196, 196)

        self.upconv4 = nn.ConvTranspose2d(in_channels//8, in_channels//16, 2, 2) # (N, 64, 392, 392)
        self.conv4 = DBConv(in_channels//8, in_channels//16) # (N, 64, 388, 388)
        
        # for match output shape with 

    def forward(self, x, pass1, pass2, pass3, pass4):
        # input (N, 1024, 28, 28)
        output = self.upconv1(x)# (N, 512, 56, 56)
        diffY = pass4.size()[2] - output.size()[2]
        diffX = pass4.size()[3] - output.size()[3]
        output = F.pad(output, (diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2)) # (N, 512, 64, 64)
        output = torch.cat((output, pass4), 1) # (N, 1024, 64, 64)
        output = self.conv1(output) # (N, 512, 60, 60)
        
        output = self.upconv2(output) # (N, 256, 120, 120)
        diffY = pass3.size()[2] - output.size()[2]
        diffX = pass3.size()[3] - output.size()[3]
        output = F.pad(output, (diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2)) # (N, 256, 136, 136)
        output = torch.cat((output, pass3), 1) # (N, 512, 136, 136)
        output = self.conv2(output) # (N, 256, 132, 132)
        
        output = self.upconv3(output) # (N, 128, 264, 264)
        diffY = pass2.size()[2] - output.size()[2]
        diffX = pass2.size()[3] - output.size()[3]
        output = F.pad(output, (diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2)) # (N, 128, 280, 280)
        output = torch.cat((output, pass2), 1) # (N, 256, 280, 280)
        output = self.conv3(output) # (N, 128, 276, 276)
        
        output = self.upconv4(output) # (N, 64, 552, 552)
        diffY = pass1.size()[2] - output.size()[2]
        diffX = pass1.size()[3] - output.size()[3]
        output = F.pad(output, (diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2)) # (N, 64, 568, 568)
        output = torch.cat((output, pass1), 1) # (N, 128, 568, 568)
        output = self.conv4(output) # (N, 64, 564, 564)
        
        return output

class UnetOriginal(nn.Module):
    def __init__(self, in_channels=3, first_outchannels=64, num_classes=3, init_weights=True, upsampling=1, activation=nn.Identity):
        super(UnetOriginal, self).__init__()
        self.encoder = nn.ModuleList([ContractingPath(in_channels=in_channels, first_outchannels=first_outchannels), DBConv(first_outchannels*8, first_outchannels*16)])
        self.decoder = ExpansivePath(in_channels=first_outchannels*16)
        self.segmentation_head = SegmentationHead(in_channels=first_outchannels,
                                                  out_channels=num_classes,
                                                  upsampling=upsampling,
                                                  activation=activation,
                                                  kernel_size=1)
        
        if init_weights:
            print('initialize weights...')
            self._initialize_weights()
    
    def forward(self, x):
        factor=4
        orgh, orgw = x.shape[-2], x.shape[-1]
        H, W = ((orgh+factor)//factor)*factor, ((orgw+factor)//factor)*factor
        padh = H-orgh if orgh%factor!=0 else 0
        padw = W-orgw if orgh%factor!=0 else 0
        x = F.pad(x, (4, padw+4, 4, padh+4), mode='reflect')
        # image = F.pad(x, (4, 4, 4, 4))
        pass1, pass2, pass3, pass4, output = self.encoder[0](x)
        output = self.encoder[1](output)
        output = self.decoder(output, pass1, pass2, pass3, pass4)
        output = self.segmentation_head(output)
        output = output[:, :, :orgh, :orgw]
        return output
   
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # He initialization
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
