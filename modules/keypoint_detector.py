from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d, Ct_encoder, EmotionNet, AF2F, AF2F_s, draw_heatmap


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        
        
        
        
    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1) #[4,10,58,58,1]
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0) #[1,1,58,58,2]
        value = (heatmap * grid).sum(dim=(2, 3)) #[4,10,2]
        kp = {'value': value}

        return kp
    
    def audio_feature(self, x, heatmap):
        
      #  prediction = self.kp(x) #[4,10,H/4-6, W/4-6]

      #  final_shape = prediction.shape
      #  heatmap = prediction.view(final_shape[0], final_shape[1], -1) #[4, 10, 58*58]
     #   heatmap = F.softmax(heatmap / self.temperature, dim=2)
     #   heatmap = heatmap.view(*final_shape) #[4,10,58,58]

     #   out = self.gaussian2kp(heatmap)
        final_shape = heatmap.squeeze(2).shape   
     
        if self.jacobian is not None:
            jacobian_map = self.jacobian(x) ##[4,40,H/4-6, W/4-6]
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map #[4,10,4,H/4-6, W/4-6]
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1) #[4,10,4]
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2) #[4,10,2,2]
            
        return jacobian
    
    def forward(self, x): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]
        prediction = self.kp(feature_map) #[4,10,H/4-6, W/4-6]

        final_shape = prediction.shape
        
        heatmap = prediction.view(final_shape[0], final_shape[1], -1) #[4, 10, 58*58]
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape) #[4,10,58,58]
        
        out = self.gaussian2kp(heatmap)
        out['heatmap'] = heatmap
        
        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map) ##[4,40,H/4-6, W/4-6]
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map #[4,10,4,H/4-6, W/4-6]
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1) #[4,10,4]
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2) #[4,10,2,2]
            out['jacobian'] = jacobian

        return out
    
    


class KPDetector_a(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels,num_channels_a, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector_a, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels_a,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        
        
        
        
    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1) #[4,10,58,58,1]
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0) #[1,1,58,58,2]
        value = (heatmap * grid).sum(dim=(2, 3)) #[4,10,2]
        kp = {'value': value}

        return kp
    
    def audio_feature(self, x, heatmap):
        
      #  prediction = self.kp(x) #[4,10,H/4-6, W/4-6]

      #  final_shape = prediction.shape
      #  heatmap = prediction.view(final_shape[0], final_shape[1], -1) #[4, 10, 58*58]
     #   heatmap = F.softmax(heatmap / self.temperature, dim=2)
     #   heatmap = heatmap.view(*final_shape) #[4,10,58,58]

     #   out = self.gaussian2kp(heatmap)
        final_shape = heatmap.squeeze(2).shape   
     
        if self.jacobian is not None:
            jacobian_map = self.jacobian(x) ##[4,40,H/4-6, W/4-6]
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map #[4,10,4,H/4-6, W/4-6]
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1) #[4,10,4]
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2) #[4,10,2,2]
            
        return jacobian
    
    def forward(self,  feature_map): #torch.Size([4, 3, H, W])
       
        prediction = self.kp(feature_map) #[4,10,H/4-6, W/4-6]

        final_shape = prediction.shape
        
        heatmap = prediction.view(final_shape[0], final_shape[1], -1) #[4, 10, 58*58]
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape) #[4,10,58,58]
        
        out = self.gaussian2kp(heatmap)
        out['heatmap'] = heatmap
        
        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map) ##[4,40,H/4-6, W/4-6]
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map #[4,10,4,H/4-6, W/4-6]
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1) #[4,10,4]
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2) #[4,10,2,2]
            out['jacobian'] = jacobian

        return out
  
    
class Audio_Feature(nn.Module):
    def __init__(self):
        super(Audio_Feature, self).__init__()
        
        self.con_encoder = Ct_encoder()
        self.emo_encoder = EmotionNet()
        self.decoder = AF2F_s()

    
    
    def forward(self, x):
        x = x.unsqueeze(1)
      
        c = self.con_encoder(x)
        e = self.emo_encoder(x)
        
     #   d = torch.cat([c, e], dim=1)
        d = self.decoder(c)
        
        
        return d
'''
def forward(self, x, cube, audio): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]
        
        cube = cube.unsqueeze(1)
        feature = torch.cat([x,cube,audio],dim=1)
        feature_map = self.predictor(feature) #[4,3+32,H/4, W/4]
        prediction = self.kp(feature_map) #[4,10,H/4-6, W/4-6]

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1) #[4, 10, 58*58]
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape) #[4,10,58,58]

        out = self.gaussian2kp(heatmap)
        out['heatmap'] = heatmap
        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map) ##[4,40,H/4-6, W/4-6]
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map #[4,10,4,H/4-6, W/4-6]
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1) #[4,10,4]
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2) #[4,10,2,2]
            out['jacobian'] = jacobian

        return out
'''
