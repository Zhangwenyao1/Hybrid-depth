import torch
import torch.nn as nn
import torch.nn.functional as F
class SPM(nn.Module):
    """ Structure Perception Module """
    def __init__(self, num_ch_enc):
        super(SPM, self).__init__()
        # self.chanel_in = in_dim
        self.num_ch_enc = num_ch_enc
        self.softmax = nn.Softmax(dim=-1)
        self.ocr_gather_head = SpatialGather_Module()
        self.ocr_distri_head = SpatialOCR_Module(in_channels=32,
                                                 key_channels=16,
                                                 out_channels=16,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        # self.ocr_distri_head = SpatialOCR_Module(in_channels=2048,
        #                                          key_channels=16,
        #                                          out_channels=16,
        #                                          scale=1,
        #                                          dropout=0.05,
        #                                          )
        # self.ocr_distri_head_1 = SpatialOCR_Module(in_channels=16,
        #                                     key_channels=self.num_ch_enc[1],
        #                                     out_channels=16,
        #                                     scale=1,
        #                                     dropout=0.05,
        #                                     )
        # self.ocr_distri_head_2 = SpatialOCR_Module(in_channels=16,
        #                                     key_channels=self.num_ch_enc[2],
        #                                     out_channels=16,
        #                                     scale=1,
        #                                     dropout=0.05,
        #                                     )
        # for i, num_ch in enumerate(self.num_ch_enc):
        #     setattr(self, f'ocr_distri_head_{i}', SpatialOCR_Module(
        #         in_channels=num_ch,
        #         key_channels=16,
        #         out_channels=16,
        #         scale=1,
        #         dropout=0.05))
            
        
    # def forward(self,x):
    #     """
    #         inputs :
    #             x : input feature maps(B X C X H X W)
    #         returns :
    #             out : attention value + input feature
    #             attention: B X C X C
    #     """
    #     m_batchsize, C, height, width = x.size()
    #     proj_query = x.view(m_batchsize, C, -1)
    #     proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
    #     energy = torch.bmm(proj_query, proj_key)
    #     energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
    #     attention = self.softmax(energy_new)
    #     proj_value = x.view(m_batchsize, C, -1)
    #     out = torch.bmm(attention, proj_value)
    #     out = out.view(m_batchsize, C, height, width)
    #     out = out + x

        # return out
        
    def forward(self, feats, text_alignment,index=0):       
         
        # ocr
        # out_aux = self.aux_head(feats)
        # # compute contrast feature
        # feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, text_alignment)
        # feats = getattr(self, f'ocr_distri_head_{index}')(feats, context)
        feats = self.ocr_distri_head(feats, context)

        # out = self.cls_head(feats)      
        return feats
        
        


# self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
#                                                  key_channels=ocr_key_channels,
#                                                  out_channels=ocr_mid_channels,
#                                                  scale=1,
#                                                  dropout=0.05,
#                                                  )
# feats = self.ocr_distri_head(feats, context) 
#  SpatialOCR_Module



# import torch
# import functools

# if torch.__version__.startswith('0'):
#     from .sync_bn.inplace_abn.bn import InPlaceABNSync
#     BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
#     BatchNorm2d_class = InPlaceABNSync
#     relu_inplace = False
# else:
#     BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
#     relu_inplace = True


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            torch.nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return torch.nn.BatchNorm2d


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = _ObjectAttentionBlock(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output



class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context




class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context    
    
    
