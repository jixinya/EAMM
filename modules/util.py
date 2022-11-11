from torch import nn

import torch.nn.functional as F
import torch
import numpy as np
import cv2
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d

from modules.stylegan2 import Generator

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from modules.function import adaptive_instance_normalization as adain

import pdb



# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                      channels * 2,
                      gain=1.0,
                      use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 6

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 6,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def draw_heatmap(landmark, width, height):
    batch = landmark.shape[0]
    number = landmark.shape[1]
    heatmap = np.zeros((batch, number,width, height), dtype=np.float32)
    # draw mouth from mouth landmarks, landmarks: mouth landmark points, format: x1, y1, x2, y2, ..., x20,


    landmark = (landmark+1)*29
    for i in range(batch):
        for pts_idx in range(number):
            if int(landmark[i,pts_idx,0])<0:
                landmark[i,pts_idx,0] = 0
            if int(landmark[i,pts_idx,1])<0:
                landmark[i,pts_idx,1] = 0
            if int(landmark[i,pts_idx,0])>57:
                landmark[i,pts_idx,0] = 57
            if int(landmark[i,pts_idx,1])>57:
                landmark[i,pts_idx,1] = 57
            heatmap[i,pts_idx, int(landmark[i,pts_idx,1]), int(landmark[i,pts_idx,0])]=1
            if heatmap[i,pts_idx].sum()== 1 :

                heatmap[i,pts_idx] = cv2.GaussianBlur(heatmap[i,pts_idx], ksize=(3, 3), sigmaX=1, sigmaY=1)


    heatmap = torch.tensor(heatmap).cuda()
    return heatmap

class NA_net(nn.Module):
    def __init__(self):
        super(NA_net, self).__init__()



        self.decon = nn.Sequential(
                nn.ConvTranspose2d(1, 16, kernel_size=(2,3), stride=2, padding=(2,1), bias=True),#16,16
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1, bias=True),#8,8
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32+3, kernel_size=4, stride=2, padding=1, bias=True)#16,16


                )



    def forward(self, neutral):

        feature = neutral.unsqueeze(1)
        current_feature = self.decon(feature)


        return current_feature

class AT_net(nn.Module):
    def __init__(self):
        super(AT_net, self).__init__()

        down_blocks = []
        for i in range(8):
            down_blocks.append(DownBlock2d(3 if i == 0 else  2 * (2 ** i),
                                            2 * (2 ** (i + 1)),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)


     #   self.lmark_encoder = nn.Sequential(
     #       nn.Linear(16,256),
     #       nn.ReLU(True),
     #       nn.Linear(256,512),
     #       nn.ReLU(True),
     #       )
        self.pose_encoder = nn.Sequential(
            nn.Linear(6,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),

            )
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),

            )
        self.lstm = nn.LSTM(256*4,256,3,batch_first = True)
    #    self.lstm_fc = nn.Sequential(
    #        nn.Linear(256,16),
    #        )
        self.decon = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=6, stride=2, padding=1, bias=True),#4,4
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),#8,8
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True), #16,16
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),#32,32
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 32+3, kernel_size=4, stride=2, padding=1, bias=True),#64,64
            #    nn.ConvTranspose2d(128, 32*4, kernel_size=2, stride=2, padding=3, bias=True),#64,64


                )
        self.generator = Generator(64,256,8)



    def forward(self, example_image, audio, pose, jaco_net):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        outs = example_image
        for down_block in self.down_blocks:
            outs = down_block(outs)
            image_feature = outs
        image_feature = image_feature.view(image_feature.shape[0], -1)
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature)
            pose_f = self.pose_encoder(pose[:,step_t])
            features = torch.cat([image_feature,  current_feature, pose_f], 1)
            lstm_input.append(features)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden)
        fc_out   = []
        deco_out = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
    #        fc_out.append(self.lstm_fc(fc_in))
            if jaco_net == 'cnn':
                fc_feature = torch.unsqueeze(fc_in,2)
                fc_feature = torch.unsqueeze(fc_feature,3)
                deco_out.append(self.decon(fc_feature))
            elif jaco_net == 'gan':
                result,_ = self.generator([fc_in])
                deco_out.append(result)
            else:
                raise Exception("jaco_net type wrong")

        return torch.stack(deco_out,dim=1)

class Classify(nn.Module):
    def __init__(self):
        super(Classify, self).__init__()



        self.last_fc = nn.Linear(512,8)

    def forward(self, feature):
       # mfcc= torch.unsqueeze(mfcc, 1)

        x = self.last_fc(feature)

        return x

class TF_net(nn.Module):
    def __init__(self):
        super(TF_net, self).__init__()

        down_blocks = []
        for i in range(8):
            down_blocks.append(DownBlock2d(3 if i == 0 else  2 * (2 ** i),
                                            2 * (2 ** (i + 1)),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)


     #   self.lmark_encoder = nn.Sequential(
     #       nn.Linear(16,256),
     #       nn.ReLU(True),
     #       nn.Linear(256,512),
     #       nn.ReLU(True),
     #       )
        self.pose_encoder = nn.Sequential(
            nn.Linear(6,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),

            )
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),

            )
        self.lstm = nn.LSTM(256*4,256,3,batch_first = True)
        self.lstm_two = nn.LSTM(256*6,256,3,batch_first = True)
    #    self.lstm_fc = nn.Sequential(
    #        nn.Linear(256,16),
    #        )
        self.decon = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=6, stride=2, padding=1, bias=True),#4,4
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),#8,8
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True), #16,16
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),#32,32
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 32+3, kernel_size=4, stride=2, padding=1, bias=True),#64,64
            #    nn.ConvTranspose2d(128, 32*4, kernel_size=2, stride=2, padding=3, bias=True),#64,64


                )
        self.generator = Generator(64,256,8)
        self.instance_norm = InstanceNorm()
        self.style_mod = ApplyStyle(512, 1024, use_wscale=True)
        self.style_mod1 = ApplyStyle(512, 35, use_wscale=True)


    def adain_forward(self, example_image, audio, pose, jaco_net, emo_features):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        outs = example_image
        for down_block in self.down_blocks:
            outs = down_block(outs)
            image_feature = outs
        image_feature = image_feature.view(image_feature.shape[0], -1)
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature) #256
            pose_f = self.pose_encoder(pose[:,step_t]) #256
            features = torch.cat([image_feature,  current_feature, pose_f], 1)
            features = torch.unsqueeze(torch.unsqueeze(features,-1),-1)
            features = self.instance_norm(features)
            x = self.style_mod(features, emo_features[step_t])
          #  t = adain(torch.unsqueeze(torch.unsqueeze(features,-1),-1), torch.unsqueeze(torch.unsqueeze(emo_features[step_t],1),2))

            lstm_input.append(torch.squeeze(torch.squeeze(x,-1),-1))
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden)
    #    fc_out   = []
        deco_out = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
    #        fc_out.append(self.lstm_fc(fc_in))
            if jaco_net == 'cnn':
                fc_feature = torch.unsqueeze(fc_in,2)
                fc_feature = torch.unsqueeze(fc_feature,3)
                deco_out.append(self.decon(fc_feature))
            elif jaco_net == 'gan':
                result,_ = self.generator([fc_in])
                deco_out.append(result)
            else:
                raise Exception("jaco_net type wrong")

        return torch.stack(deco_out,dim=1)



    def adain_feature2(self, example_image, audio, pose, jaco_net, emo_features):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        outs = example_image
        for down_block in self.down_blocks:
            outs = down_block(outs)
            image_feature = outs
        image_feature = image_feature.view(image_feature.shape[0], -1)
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature) #256
            pose_f = self.pose_encoder(pose[:,step_t]) #256
            features = torch.cat([image_feature,  current_feature, pose_f], 1)

            lstm_input.append(features)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden)
    #    fc_out   = []
        deco_out = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
    #        fc_out.append(self.lstm_fc(fc_in))
            if jaco_net == 'cnn':
                fc_feature = torch.unsqueeze(fc_in,2)
                fc_feature = torch.unsqueeze(fc_feature,3)
                fc_feature = self.decon(fc_feature)
                fc_feature = self.instance_norm(fc_feature)
                t = self.style_mod1(fc_feature, emo_features[step_t])
             #   emo_feature = torch.unsqueeze(torch.unsqueeze(emo_features[step_t],-1),-1)
             #   emo_feature = emo_feature.repeat(1,fc_feature.shape[1],1,1)
             #   t = adain(fc_feature, emo_feature)
                deco_out.append(t)
            elif jaco_net == 'gan':
                result,_ = self.generator([fc_in])
                deco_out.append(result)
            else:
                raise Exception("jaco_net type wrong")

        return torch.stack(deco_out,dim=1)

    def forward(self, example_image, audio, pose, jaco_net, emo_features):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        outs = example_image
        for down_block in self.down_blocks:
            outs = down_block(outs)
            image_feature = outs
        image_feature = image_feature.view(image_feature.shape[0], -1)
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature) #256
            pose_f = self.pose_encoder(pose[:,step_t]) #256
            features = torch.cat([image_feature,  current_feature, pose_f, emo_features[step_t]], 1)
            lstm_input.append(features)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm_two(lstm_input, hidden)
        fc_out   = []
        deco_out = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
    #        fc_out.append(self.lstm_fc(fc_in))
            if jaco_net == 'cnn':
                fc_feature = torch.unsqueeze(fc_in,2)
                fc_feature = torch.unsqueeze(fc_feature,3)
                deco_out.append(self.decon(fc_feature))
            elif jaco_net == 'gan':
                result,_ = self.generator([fc_in])
                deco_out.append(result)
            else:
                raise Exception("jaco_net type wrong")

        return torch.stack(deco_out,dim=1)


class AT_net2(nn.Module):
    def __init__(self):
        super(AT_net2, self).__init__()

        down_blocks = []
        for i in range(8):
            down_blocks.append(DownBlock2d(3 if i == 0 else  2 * (2 ** i),
                                            2 * (2 ** (i + 1)),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)


     #   self.lmark_encoder = nn.Sequential(
     #       nn.Linear(16,256),
     #       nn.ReLU(True),
     #       nn.Linear(256,512),
     #       nn.ReLU(True),
     #       )
        self.pose_encoder = nn.Sequential(
            nn.Linear(6,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),

            )
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),

            )
        self.lstm = nn.LSTM(256*4,256,3,batch_first = True)
    #    self.lstm_fc = nn.Sequential(
    #        nn.Linear(256,16),
    #        )
        self.decon = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=6, stride=2, padding=1, bias=True),#4,4
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),#8,8
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True), #16,16
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),#32,32
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 32+3, kernel_size=4, stride=2, padding=1, bias=True),#64,64
            #    nn.ConvTranspose2d(128, 32*4, kernel_size=2, stride=2, padding=3, bias=True),#64,64


                )
        self.generator = Generator(64,256,8)


    def forward(self, example_image, audio, pose, jaco_net, weight):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        outs = example_image
        for down_block in self.down_blocks:
            outs = down_block(outs)
            image_feature = outs
        image_feature = image_feature.view(image_feature.shape[0], -1)
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature)*weight
            pose_f = self.pose_encoder(pose[:,step_t])
            features = torch.cat([image_feature,  current_feature, pose_f], 1)
            lstm_input.append(features)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden)
        fc_out   = []
        deco_out = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
    #        fc_out.append(self.lstm_fc(fc_in))
            if jaco_net == 'cnn':
                fc_feature = torch.unsqueeze(fc_in,2)
                fc_feature = torch.unsqueeze(fc_feature,3)
                deco_out.append(self.decon(fc_feature))
            elif jaco_net == 'gan':
                result,_ = self.generator([fc_in])
                deco_out.append(result)
            else:
                raise Exception("jaco_net type wrong")

        return torch.stack(deco_out,dim=1)



class Ct_encoder(nn.Module):
    def __init__(self):
        super(Ct_encoder, self).__init__()
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),

            )

    def forward(self, audio):

        feature = self.audio_eocder(audio)
        feature = feature.view(feature.size(0),-1)
        x = self.audio_eocder_fc(feature)

        return x


class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()

        self.emotion_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),

            nn.MaxPool2d((1,3), stride=(1,2)), #[1, 64, 12, 12]
            conv2d(64,128,3,1,1),

            conv2d(128,256,3,1,1),

            nn.MaxPool2d((12,1), stride=(12,1)), #[1, 256, 1, 12]

            conv2d(256,512,3,1,1),

            nn.MaxPool2d((1,2), stride=(1,2)) #[1, 512, 1, 6]

            )
        self.emotion_eocder_fc = nn.Sequential(
            nn.Linear(512 *6,2048),
            nn.ReLU(True),
            nn.Linear(2048,128),
            nn.ReLU(True),

            )

        self.last_fc = nn.Linear(128,8)

        self.re_id = nn.Sequential(
            conv2d(512,1024,3,1,1),

            nn.MaxPool2d((1,2), stride=(1,2)), #[1, 1024, 1, 3]
            conv2d(1024,1024,3,1,1),

            conv2d(1024,2048,3,1,1),

            nn.MaxPool2d((1,2), stride=(1,2)) #[1, 2048, 1, 1]


            )
        self.re_id_fc = nn.Sequential(

            nn.Linear(2048,512),
            nn.ReLU(True),
            nn.Linear(512,128),
            nn.ReLU(True),
            )


    def forward(self, mfcc):
       # mfcc= torch.unsqueeze(mfcc, 1)
        mfcc=torch.transpose(mfcc,2,3)
        feature = self.emotion_eocder(mfcc)

   #     id_feature = feature.detach()

        feature = feature.view(feature.size(0),-1)
        x = self.emotion_eocder_fc(feature)


  #      remove_feature = self.re_id(id_feature)
  #      remove_feature = remove_feature.view(remove_feature.size(0),-1)
  #      y = self.re_id_fc(remove_feature)

        return x


class AF2F(nn.Module):
    def __init__(self):
        super(AF2F, self).__init__()
        self.decon = nn.Sequential(
                nn.ConvTranspose2d(384, 256, kernel_size=6, stride=2, padding=1, bias=True),#4,4
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),#8,8
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True), #16,16
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),#32,32
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32+3, kernel_size=4, stride=2, padding=1, bias=True),#64,64


                )

    def forward(self, content,emotion):
        features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
        features = torch.unsqueeze(features,2)
        features = torch.unsqueeze(features,3)
        x = self.decon(features)


        return x

class AF2F_s(nn.Module):
    def __init__(self):
        super(AF2F_s, self).__init__()
        self.decon = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=6, stride=2, padding=1, bias=True),#4,4
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),#8,8
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True), #16,16
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),#32,32
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32+3, kernel_size=4, stride=2, padding=1, bias=True),#64,64

                nn.ReLU(),
                )

    def forward(self, content):
       # features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
        features = torch.unsqueeze(content,2)
        features = torch.unsqueeze(features,3)
        x = self.decon(features)


        return x


class A2I(nn.Module):
    def __init__(self):
        super(A2I, self).__init__()
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d((1,5), stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),

            nn.MaxPool2d((5,5), stride=(2,2))
            )
        self.decon = nn.Sequential(

                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),#8,8
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True), #16,16
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True),#32,32
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1, bias=True),#64,64

                nn.ReLU(),
                )

    def forward(self, mfcc):
        mfcc= torch.unsqueeze(mfcc, 1)
        mfcc=torch.transpose(mfcc,2,3)
        feature = self.audio_eocder(mfcc)

   #     id_feature = feature.detach()

        x = self.decon(feature)

        return x

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value'] #[4,10,2]

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type()) #[h,w,2]
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape #5
    coordinate_grid = coordinate_grid.view(*shape) #[1,1,h,w,2]
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats) #[4,10,h,w,2]

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape) #[4,10,1,1,2]

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
     #   sigma = (1 / scale - 1) / 2
        sigma = 1.5
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 0.7853975 - 1))
    return norm_angle


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class EmDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion,  num_channels, max_features,
                 num_blocks, scale_factor=1,  num_classes=8):
        super(EmDetector, self).__init__()
        self.inplanes = 64
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)




        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        self.conv1 = nn.Conv2d(self.predictor.out_filters, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = [2,2,2,2]
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.classify = Classify()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def adain_feature(self, x): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]

    #    out = self.fc(out)

        return feature_map

    def forward(self, x): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]
        f = self.conv1(feature_map) #[16,64,64,64]
        f = self.bn1(f) #torch.Size([16, 64, 64, 64])
        f = self.relu(f)
        f = self.maxpool(f) #[16, 64, 32, 32]

        f = self.layer1(f) #[16, 64, 32, 32]
        f = self.layer2(f) #[16, 128, 16, 16])
        f = self.layer3(f) #[16, 256, 8, 8]
        f = self.layer4(f) #[16, 512, 4, 4]
        f = self.avgpool(f) #[16, 512, 1, 1]
        out = f.squeeze(3).squeeze(2)
        fake = self.classify(out)
    #    out = self.fc(out)

        return out, fake






class Emotion_k(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion,  num_channels, max_features,
                 num_blocks, scale_factor=1,  num_classes=8):
        super(Emotion_k, self).__init__()
        self.inplanes = 64
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)




        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        self.conv1 = nn.Conv2d(self.predictor.out_filters, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = [2,2,2,2]
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self.embed_fn, self.input_ch = get_embedder(10, 0)

        self.fc_p = nn.Sequential(
            nn.Linear(10 * 126,1024),
            nn.ReLU(True),
            nn.Linear(1024,512),
            nn.ReLU(True),

            )
        self.fc_n = nn.Sequential(
            nn.Linear(10 * 6,128),
            nn.ReLU(True),
            nn.Linear(128,512),
            nn.ReLU(True),

            )

        self.fc_all = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(True),
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,64),
            nn.ReLU(True),
            )

      #  self.fc_single = nn.Sequential(
      #      nn.Linear(512,256),
      #      nn.ReLU(True),
      #      nn.Linear(256,64),
      #      nn.ReLU(True),
      #      )

        self.final = nn.Sequential(
            nn.Conv1d(1,2,4,2,1),
            nn.MaxPool1d(2,stride=2),
            nn.ReLU(True),
            nn.Conv1d(2,4,4,2,1),
            nn.ReLU(True),
            nn.Conv1d(4,4,3),

            )

        self.final_4 = nn.Sequential(
            nn.Conv1d(4,4,3,1,1),
            nn.MaxPool1d(2,stride=2),
            nn.ReLU(True),
            nn.Conv1d(4,4,3,1)

            )

        self.final_10 = nn.Sequential(
            nn.Conv1d(4,8,3,1,1), #[B,8,16]
            nn.MaxPool1d(2,stride=2), #[B,8,8]
            nn.ReLU(True),
            nn.Conv1d(8,10,3,1), #[B,10,6]


            )

        self.classify = Classify()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def linear_10(self, x, value, jacobian): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]
        f = self.conv1(feature_map) #[16,64,64,64]
        f = self.bn1(f) #torch.Size([16, 64, 64, 64])
        f = self.relu(f)
        f = self.maxpool(f) #[16, 64, 32, 32]

        f = self.layer1(f) #[16, 64, 32, 32]
        f = self.layer2(f) #[16, 128, 16, 16])
        f = self.layer3(f) #[16, 256, 8, 8]
        f = self.layer4(f) #[16, 512, 4, 4]
        f = self.avgpool(f) #[16, 512, 1, 1]
        out = f.squeeze(3).squeeze(2)
        fake = self.classify(out)
        jacobian = jacobian.reshape(jacobian.shape[0],jacobian.shape[1],4)
        neu_input = torch.cat((value,jacobian),2)
        posi_input = self.embed_fn(neu_input)
        posi_input =posi_input.reshape(posi_input.shape[0],-1)
        ner_feature = self.fc_p(posi_input)
        all_fc = self.fc_all(torch.cat((out,ner_feature),1)).reshape(-1,4,16)
        result = self.final_10(all_fc)
        e_value = result[:,:,:2]
        e_jacobian = result[:,:,2:].reshape(result.shape[0],10,2,2)
        kp = {'value': e_value,'jacobian': e_jacobian}

        return kp, fake


    def linear_4(self, x, value, jacobian): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]
        f = self.conv1(feature_map) #[16,64,64,64]
        f = self.bn1(f) #torch.Size([16, 64, 64, 64])
        f = self.relu(f)
        f = self.maxpool(f) #[16, 64, 32, 32]

        f = self.layer1(f) #[16, 64, 32, 32]
        f = self.layer2(f) #[16, 128, 16, 16])
        f = self.layer3(f) #[16, 256, 8, 8]
        f = self.layer4(f) #[16, 512, 4, 4]
        f = self.avgpool(f) #[16, 512, 1, 1]
        out = f.squeeze(3).squeeze(2)
        fake = self.classify(out)
    #    jacobian = jacobian.reshape(jacobian.shape[0],jacobian.shape[1],4)
    #    neu_input = torch.cat((value,jacobian),2)
    #    posi_input = self.embed_fn(neu_input)
    #    posi_input =posi_input.reshape(posi_input.shape[0],-1)
    #    ner_feature = self.fc_p(posi_input)
    #    all_fc = self.fc_all(torch.cat((out,ner_feature),1)).reshape(-1,4,16)
        all_fc = torch.unsqueeze(self.fc_single(out),1)
        result = self.final(all_fc)
        e_value = result[:,:,:2]
        e_jacobian = result[:,:,2:].reshape(result.shape[0],4,2,2)
        kp = {'value': e_value,'jacobian': e_jacobian}
    #    out = self.fc(out)

        return kp, fake

    def linear_np_10(self, x, value, jacobian): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]
        f = self.conv1(feature_map) #[16,64,64,64]
        f = self.bn1(f) #torch.Size([16, 64, 64, 64])
        f = self.relu(f)
        f = self.maxpool(f) #[16, 64, 32, 32]

        f = self.layer1(f) #[16, 64, 32, 32]
        f = self.layer2(f) #[16, 128, 16, 16])
        f = self.layer3(f) #[16, 256, 8, 8]
        f = self.layer4(f) #[16, 512, 4, 4]
        f = self.avgpool(f) #[16, 512, 1, 1]
        out = f.squeeze(3).squeeze(2)
        fake = self.classify(out)
        jacobian = jacobian.reshape(jacobian.shape[0],jacobian.shape[1],4)
        neu_input = torch.cat((value,jacobian),2)

        posi_input =neu_input.reshape(neu_input.shape[0],-1)
        ner_feature = self.fc_n(posi_input)
        all_fc = self.fc_all(torch.cat((out,ner_feature),1)).reshape(-1,4,16)
        result = self.final_10(all_fc)
        e_value = result[:,:,:2]
        e_jacobian = result[:,:,2:].reshape(result.shape[0],10,2,2)
        kp = {'value': e_value,'jacobian': e_jacobian}
    #    out = self.fc(out)

        return kp, fake

    def linear_np_4(self, x, value, jacobian): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]
        f = self.conv1(feature_map) #[16,64,64,64]
        f = self.bn1(f) #torch.Size([16, 64, 64, 64])
        f = self.relu(f)
        f = self.maxpool(f) #[16, 64, 32, 32]

        f = self.layer1(f) #[16, 64, 32, 32]
        f = self.layer2(f) #[16, 128, 16, 16])
        f = self.layer3(f) #[16, 256, 8, 8]
        f = self.layer4(f) #[16, 512, 4, 4]
        f = self.avgpool(f) #[16, 512, 1, 1]
        out = f.squeeze(3).squeeze(2)
        fake = self.classify(out)
        jacobian = jacobian.reshape(jacobian.shape[0],jacobian.shape[1],4)
        neu_input = torch.cat((value,jacobian),2)

        posi_input =neu_input.reshape(neu_input.shape[0],-1)
        ner_feature = self.fc_n(posi_input)
        all_fc = torch.unsqueeze(self.fc_all(torch.cat((out,ner_feature),1)),1)
        result = self.final(all_fc)
        e_value = result[:,:,:2]
        e_jacobian = result[:,:,2:].reshape(result.shape[0],4,2,2)
        kp = {'value': e_value,'jacobian': e_jacobian}
    #    out = self.fc(out)

        return kp, fake


    def emotion_feature(self, feature, value, jacobian): #torch.Size([4, 3, H, W])

        out = feature
        fake = self.classify(out)
        jacobian = jacobian.reshape(jacobian.shape[0],jacobian.shape[1],4)
        neu_input = torch.cat((value,jacobian),2)
        posi_input = self.embed_fn(neu_input)
        posi_input =posi_input.reshape(posi_input.shape[0],-1)
        ner_feature = self.fc_p(posi_input)
        all_fc = torch.unsqueeze(self.fc_all(torch.cat((out,ner_feature),1)),1)
        result = self.final(all_fc)
        e_value = result[:,:,:2]
        e_jacobian = result[:,:,2:].reshape(result.shape[0],4,2,2)
        kp = {'value': e_value,'jacobian': e_jacobian}
    #    out = self.fc(out)

        return kp, fake

    def feature(self, x): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]
        f = self.conv1(feature_map) #[16,64,64,64]
        f = self.bn1(f) #torch.Size([16, 64, 64, 64])
        f = self.relu(f)
        f = self.maxpool(f) #[16, 64, 32, 32]

        f = self.layer1(f) #[16, 64, 32, 32]
        f = self.layer2(f) #[16, 128, 16, 16])
        f = self.layer3(f) #[16, 256, 8, 8]
        f = self.layer4(f) #[16, 512, 4, 4]
        f = self.avgpool(f) #[16, 512, 1, 1]
        out = f.squeeze(3).squeeze(2)

    #    out = self.fc(out)

        return out

    def forward(self, x, value, jacobian): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]
        f = self.conv1(feature_map) #[16,64,64,64]
        f = self.bn1(f) #torch.Size([16, 64, 64, 64])
        f = self.relu(f)
        f = self.maxpool(f) #[16, 64, 32, 32]

        f = self.layer1(f) #[16, 64, 32, 32]
        f = self.layer2(f) #[16, 128, 16, 16])
        f = self.layer3(f) #[16, 256, 8, 8]
        f = self.layer4(f) #[16, 512, 4, 4]
        f = self.avgpool(f) #[16, 512, 1, 1]
        out = f.squeeze(3).squeeze(2)
        fake = self.classify(out)
        jacobian = jacobian.reshape(jacobian.shape[0],jacobian.shape[1],4)
        neu_input = torch.cat((value,jacobian),2)
        posi_input = self.embed_fn(neu_input)
        posi_input =posi_input.reshape(posi_input.shape[0],-1)
        ner_feature = self.fc_p(posi_input)
        all_fc = torch.unsqueeze(self.fc_all(torch.cat((out,ner_feature),1)),1)
        result = self.final(all_fc)
        e_value = result[:,:,:2]
        e_jacobian = result[:,:,2:].reshape(result.shape[0],4,2,2)
        kp = {'value': e_value,'jacobian': e_jacobian}
    #    out = self.fc(out)

        return kp, fake

class Emotion_map(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion,  num_channels, max_features,
                 num_blocks, scale_factor=1,  num_classes=8):
        super(Emotion_map, self).__init__()
        self.inplanes = 64
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)




        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        self.conv1 = nn.Conv2d(self.predictor.out_filters, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = [2,2,2,2]
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self.embed_fn, self.input_ch = get_embedder(10, 0)

        self.fc_p = nn.Sequential(
            nn.Linear(10 * 126,1024),
            nn.ReLU(True),
            nn.Linear(1024,512),
            nn.ReLU(True),

            )

        self.fc_all = nn.Sequential(
            nn.Linear(1024,2048),
            nn.ReLU(True)
            )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),#8,8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True), #16,16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),#32,32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32+3, kernel_size=4, stride=2, padding=1, bias=True),#64,64

            )


        self.classify = Classify()
        self.kp = nn.Conv2d(in_channels=35, out_channels=10, kernel_size=(7, 7),
                            padding=0)
        self.jacobian = nn.Conv2d(in_channels=35,
                                      out_channels=4 * 10, kernel_size=(7, 7), padding=0)
        self.jacobian.weight.data.zero_()
        self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * 10, dtype=torch.float))
        self.temperature = 0.1

        self.kp_4 = nn.Conv2d(in_channels=35, out_channels=4, kernel_size=(7, 7),
                            padding=0)
        self.jacobian_4 = nn.Conv2d(in_channels=35,
                                      out_channels=4 * 4, kernel_size=(7, 7), padding=0)
        self.jacobian_4.weight.data.zero_()
        self.jacobian_4.bias.data.copy_(torch.tensor([1, 0, 0, 1] * 4, dtype=torch.float))


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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

    def map_4(self, x, value, jacobian): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]
        f = self.conv1(feature_map) #[16,64,64,64]
        f = self.bn1(f) #torch.Size([16, 64, 64, 64])
        f = self.relu(f)
        f = self.maxpool(f) #[16, 64, 32, 32]

        f = self.layer1(f) #[16, 64, 32, 32]
        f = self.layer2(f) #[16, 128, 16, 16])
        f = self.layer3(f) #[16, 256, 8, 8]
        f = self.layer4(f) #[16, 512, 4, 4]
        f = self.avgpool(f) #[16, 512, 1, 1]
        out = f.squeeze(3).squeeze(2)
        fake = self.classify(out)
        jacobian = jacobian.reshape(jacobian.shape[0],jacobian.shape[1],4)
        neu_input = torch.cat((value,jacobian),2)
        posi_input = self.embed_fn(neu_input)
        posi_input =posi_input.reshape(posi_input.shape[0],-1)
        ner_feature = self.fc_p(posi_input)
        all_fc = self.fc_all(torch.cat((out,ner_feature),1)).reshape(-1,128,4,4)
        feature_map = self.final(all_fc)
        prediction = self.kp_4(feature_map) #[4,10,H/4-6, W/4-6]

        final_shape = prediction.shape

        heatmap = prediction.view(final_shape[0], final_shape[1], -1) #[4, 10, 58*58]
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape) #[4,10,58,58]

        out = self.gaussian2kp(heatmap)
        out['heatmap'] = heatmap

        if self.jacobian is not None:
            jacobian_map = self.jacobian_4(feature_map) ##[4,40,H/4-6, W/4-6]
            jacobian_map = jacobian_map.reshape(final_shape[0], 4, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map #[4,10,4,H/4-6, W/4-6]
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1) #[4,10,4]
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2) #[4,10,2,2]
            out['jacobian'] = jacobian



        return out, fake

    def forward(self, x, value, jacobian): #torch.Size([4, 3, H, W])
        if self.scale_factor != 1:
            x = self.down(x) # 0.25 [4, 3, H/4, W/4]

        feature_map = self.predictor(x) #[4,3+32,H/4, W/4]
        f = self.conv1(feature_map) #[16,64,64,64]
        f = self.bn1(f) #torch.Size([16, 64, 64, 64])
        f = self.relu(f)
        f = self.maxpool(f) #[16, 64, 32, 32]

        f = self.layer1(f) #[16, 64, 32, 32]
        f = self.layer2(f) #[16, 128, 16, 16])
        f = self.layer3(f) #[16, 256, 8, 8]
        f = self.layer4(f) #[16, 512, 4, 4]
        f = self.avgpool(f) #[16, 512, 1, 1]
        out = f.squeeze(3).squeeze(2)
        fake = self.classify(out)
        jacobian = jacobian.reshape(jacobian.shape[0],jacobian.shape[1],4)
        neu_input = torch.cat((value,jacobian),2)
        posi_input = self.embed_fn(neu_input)
        posi_input =posi_input.reshape(posi_input.shape[0],-1)
        ner_feature = self.fc_p(posi_input)
        all_fc = self.fc_all(torch.cat((out,ner_feature),1)).reshape(-1,128,4,4)
        feature_map = self.final(all_fc)

        prediction = self.kp(feature_map) #[4,10,H/4-6, W/4-6]

        final_shape = prediction.shape

        heatmap = prediction.view(final_shape[0], final_shape[1], -1) #[4, 10, 58*58]
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape) #[4,10,58,58]

        out = self.gaussian2kp(heatmap)
        out['heatmap'] = heatmap

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map) ##[4,40,H/4-6, W/4-6]
            jacobian_map = jacobian_map.reshape(final_shape[0], 10, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map #[4,10,4,H/4-6, W/4-6]
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1) #[4,10,4]
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2) #[4,10,2,2]
            out['jacobian'] = jacobian



        return out, fake


def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Conv2d(channel_in, channel_out,
                     ksize, stride, padding,
                     bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    # init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)

def _apply(layer, activation, normalizer, channel_out=None):
    if normalizer:
        layer.append(normalizer(channel_out))
    if activation:
        layer.append(activation())
    return layer