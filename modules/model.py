from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0) #[1,256,256,2]
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")
    
    def inverse_transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0) #[1,256,256,2]
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.inverse_warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")
    
    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def inverse_warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        a = torch.FloatTensor([[[[0,0,1]]]]).repeat([self.bs,1,1,1]).cuda()
        c = torch.cat((theta,a),2)
        d = c.inverse()[:,:,:2,:]
        d = d.type(coordinates.type())
        transformed = torch.matmul(d[:, :, :, :2], coordinates.unsqueeze(-1)) + d[:, :, :, 2:]
        transformed = transformed.squeeze(-1)
        
        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        
        
        return transformed

    def jacobian(self, coordinates):
        coordinates.requires_grad=True
        new_coordinates = self.warp_coordinates(coordinates)#[4,10,2]
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}

class TrainFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, emo_detector, kp_extractor_a, audio_feature, generator, discriminator, train_params, device_ids):
        super(TrainFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.kp_extractor_a = kp_extractor_a
        self.emo_detector = emo_detector
    #    self.content_encoder = content_encoder
    #    self.emotion_encoder = emotion_encoder
        self.audio_feature = audio_feature
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
        
       # self.pca = torch.FloatTensor(np.load('/mnt/lustre/jixinya/Home/LRW/list/U_106.npy'))[:, :16].to(device_ids[0])
      #  self.mean = torch.FloatTensor(np.load('/mnt/lustre/jixinya/Home/LRW/list/mean_106.npy')).to(device_ids[0])
        self.mse_loss_fn   =  nn.MSELoss().cuda()
        self.CroEn_loss =  nn.CrossEntropyLoss().cuda()
    def forward(self, x):
   #     source_a_f = self.audio_feature(x['source_audio'],x['source_lm'],x[])
      #  source_a_f = self.audio_feature(self.content_encoder(x['source_audio'].unsqueeze(1)), self.emotion_encoder(x['source_audio'].unsqueeze(1)))
        kp_source = self.kp_extractor(x['example_image'])

        kp_driving = []
        emo_features = []
        fakes = []
        for i in range(16):
            kp_driving.append(self.kp_extractor(x['driving'][:,i]))
            
            out, fake = self.emo_detector(x['transformed_driving'][:,i])   
            emo_features.append(out)
            fakes.append(fake)
        
    #    print('KP_driving ', file=open('/mnt/lustre/jixinya/Home/fomm_audio/log/LRW_test.txt', 'a'))
        kp_driving_a = [] #x['example_image'],
        if self.train_params['type'] == 'add':
            deco_out = self.audio_feature(x['example_image'], x['driving_audio'], x['driving_pose'], self.train_params['jaco_net'], emo_features)
        elif self.train_params['type'] == 'adain':
            deco_out = self.audio_feature.adain_forward(x['example_image'], x['driving_audio'], x['driving_pose'], self.train_params['jaco_net'], emo_features)
        elif self.train_params['type'] == 'adain_feature':
            deco_out = self.audio_feature.adain_feature(x['example_image'], x['driving_audio'], x['driving_pose'], self.train_params['jaco_net'], emo_features)
        elif self.train_params['type'] == 'adain_feature2':
            deco_out = self.audio_feature.adain_feature2(x['example_image'], x['driving_audio'], x['driving_pose'], self.train_params['jaco_net'], emo_features)
        loss_values = {}
        
        if self.loss_weights['emo'] != 0:
            
            kp_driving_a = []
            for i in range(16):
                kp_driving_a.append(self.kp_extractor_a(deco_out[:,i]))#
       
    #    print('Kp_audio_driving ', file=open('/mnt/lustre/jixinya/Home/fomm_audio/log/LRW_test.txt', 'a'))
        loss_value = 0
        loss_heatmap = 0
        loss_jacobian = 0
        loss_perceptual = 0
        loss_classify = 0
     #   kp_all = kp_emo
        for i in range(len(kp_driving)):
            loss_jacobian += (torch.abs(kp_driving[i]['jacobian'] - kp_driving_a[i]['jacobian']).mean())*self.loss_weights['emo']
            
         #   loss_jacobian = loss_jacobian*self.loss_weights['audio']
            loss_heatmap += (torch.abs(kp_driving[i]['heatmap'] - kp_driving_a[i]['heatmap'] ).mean())*self.loss_weights['emo']*100
           
            
            loss_value += (torch.abs(kp_driving[i]['value'].detach() - kp_driving_a[i]['value']).mean())*self.loss_weights['emo']
            loss_classify += self.CroEn_loss(fakes[i],x['emotion'])
            
     #       kp_all[i]['jacobian'] = kp_emo[i]['jacobian'] + kp_driving[i]['jacobian']
     #       kp_all[i]['value'] = kp_emo[i]['value'] + kp_driving[i]['value']
            
        loss_values['loss_value'] = loss_value/len(kp_driving)
        loss_values['loss_heatmap'] = loss_heatmap/len(kp_driving)
        loss_values['loss_jacobian'] = loss_jacobian/len(kp_driving)
        loss_values['loss_classify'] = loss_classify/len(kp_driving)
        
   
        if self.train_params['generator'] == 'not':
            loss_values['perceptual'] = self.mse_loss_fn(deco_out,deco_out)
            for i in range(1): #0,len(kp_driving),4
 
                generated = self.generator(x['example_image'], kp_source=kp_source, kp_driving=kp_driving_a[i])
                generated.update({'kp_source': kp_source, 'kp_driving': kp_driving_a})
        elif self.train_params['generator'] == 'visual':
            for i in range(0,len(kp_driving),4): #0,len(kp_driving),4
 
                generated = self.generator(x['example_image'], kp_source=kp_source, kp_driving=kp_driving[i])
                generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
                
                pyramide_real = self.pyramid(x['driving'][:,i])
                pyramide_generated = self.pyramid(generated['prediction'])
        
                if sum(self.loss_weights['perceptual']) != 0:
                    value_total = 0
                    for scale in self.scales:
                        x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                        y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                        for i, weight in enumerate(self.loss_weights['perceptual']):
                            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                            value_total += self.loss_weights['perceptual'][i] * value
                    loss_perceptual += value_total
        
            length = int((len(kp_driving)-1)/4)+1
            loss_values['perceptual'] = loss_perceptual/length
        elif self.train_params['generator'] == 'audio':
            for i in range(0,len(kp_driving),4): #0,len(kp_driving),4
 
                generated = self.generator(x['example_image'], kp_source=kp_source, kp_driving=kp_driving_a[i])
                generated.update({'kp_source': kp_source, 'kp_driving': kp_driving_a})
                
                pyramide_real = self.pyramid(x['driving'][:,i])
                pyramide_generated = self.pyramid(generated['prediction'])
        
                if sum(self.loss_weights['perceptual']) != 0:
                    value_total = 0
                    for scale in self.scales:
                        x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                        y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                        for i, weight in enumerate(self.loss_weights['perceptual']):
                            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                            value_total += self.loss_weights['perceptual'][i] * value
                    loss_perceptual += value_total
        
            length = int((len(kp_driving)-1)/4)+1
            loss_values['perceptual'] = loss_perceptual/length
        else:
            print('wrong train_params: ', self.train_params['generator'])
      
        
      
        return loss_values,generated

class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, kp_extractor_a, audio_feature, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.kp_extractor_a = kp_extractor_a
    #    self.content_encoder = content_encoder
    #    self.emotion_encoder = emotion_encoder
        self.audio_feature = audio_feature
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
        
        self.pca = torch.FloatTensor(np.load('.../LRW/list/U_106.npy'))[:, :16].cuda()
        self.mean = torch.FloatTensor(np.load('.../LRW/list/mean_106.npy')).cuda()
        
    def forward(self, x):
   #     source_a_f = self.audio_feature(x['source_audio'],x['source_lm'],x[])
      #  source_a_f = self.audio_feature(self.content_encoder(x['source_audio'].unsqueeze(1)), self.emotion_encoder(x['source_audio'].unsqueeze(1)))
   #     kp_source = self.kp_extractor(x['source'])
   #     kp_source_a = self.kp_extractor_a(x['source'], x['source_cube'], source_a_f)
      #  driving_a_f = self.audio_feature(self.content_encoder(x['driving_audio'].unsqueeze(1)), self.emotion_encoder(x['driving_audio'].unsqueeze(1)))
      #  driving_a_f = self.audio_feature(x['driving_audio'])
      #  kp_driving = self.kp_extractor(x['driving'])
   #     kp_driving_a = self.kp_extractor_a(x['driving'], x['driving_cube'], driving_a_f)
       
        kp_driving = []
        for i in range(16):
            kp_driving.append(self.kp_extractor(x['driving'][:,i],x['driving_landmark'][:,i],self.loss_weights['equivariance_value']))
        
        kp_driving_a = []
        fc_out, deco_out = self.audio_feature(x['example_landmark'], x['driving_audio'], x['driving_pose'])
        fake_lmark=fc_out + x['example_landmark'].expand_as(fc_out)
        
      
        fake_lmark = torch.mm( fake_lmark, self.pca.t() )
        fake_lmark = fake_lmark + self.mean.expand_as(fake_lmark)
    

        fake_lmark = fake_lmark.unsqueeze(0) 

    #    for i in range(16):
    #        kp_driving_a.append()
        
   #     generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
   #     generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
        
        if self.loss_weights['audio'] != 0:
            value = torch.abs(kp_source['jacobian'].detach() - kp_source_a['jacobian'].detach()).mean() + torch.abs(kp_driving['jacobian'].detach() - kp_driving_a['jacobian']).mean()
            value = value/2
            loss_values['jacobian'] = value*self.loss_weights['audio']
            value = torch.abs(kp_source['heatmap'].detach() - kp_source_a['heatmap'].detach()).mean() + torch.abs(kp_driving['heatmap'].detach() - kp_driving_a['heatmap']).mean()
            value = value/2
            loss_values['heatmap'] = value*self.loss_weights['audio']
            value = torch.abs(kp_source['value'].detach() - kp_source_a['value'].detach()).mean() + torch.abs(kp_driving['value'].detach() - kp_driving_a['value']).mean()
            value = value/2
            loss_values['value'] = value*self.loss_weights['audio']
            
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_landmark =  transform.inverse_warp_coordinates(x['driving_landmark'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp
            
            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
