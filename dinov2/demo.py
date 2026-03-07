import torch
import torch.nn as nn
dinov2:nn.Module = torch.hub.load('/code/dinov2', 'dinov2_vitb14', source='local', verbose=True, pretrained=False)
dinov2.load_state_dict(torch.load('checkpoints/dinov2_vitb14_pretrain.pth'))
dinov2.cuda()
dummy_input=torch.zeros((1, 3, 224, 224)).cuda()
ret = dinov2.forward_features(dummy_input)
print(ret['x_norm_patchtokens'].shape)