#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 20:57:27 2021
@author: thea
"""

import matplotlib
matplotlib.use('Agg')
import os,sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from skimage import io, img_as_float32
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from filter1 import OneEuroFilter
import torch.utils

from torch.autograd import Variable
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector, KPDetector_a
from modules.util import AT_net, Emotion_k, Emotion_map, AT_net2
from augmentation import AllAugmentationTransform

from scipy.spatial import ConvexHull

import python_speech_features
from pathlib import Path
import dlib
import cv2
import librosa
from skimage import transform as tf
#from audiolm.models import AT_emoiton
#from audiolm.utils import plot_flmarks
if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.6")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')




def load_checkpoints(opt, checkpoint_path, audio_checkpoint_path, emo_checkpoint_path, cpu=False):

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    kp_detector_a = KPDetector_a(**config['model_params']['kp_detector_params'],
                             **config['model_params']['audio_params'])

    audio_feature = AT_net2()
    if opt.type.startswith('linear'):
        emo_detector = Emotion_k(block_expansion=32, num_channels=3, max_features=1024,
                 num_blocks=5, scale_factor=0.25, num_classes=8)
    elif opt.type.startswith('map'):
        emo_detector = Emotion_map(block_expansion=32, num_channels=3, max_features=1024,
                 num_blocks=5, scale_factor=0.25, num_classes=8)
    if not cpu:
        kp_detector_a.cuda()
        audio_feature.cuda()
        emo_detector.cuda()




    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        audio_checkpoint = torch.load(audio_checkpoint_path, map_location=torch.device('cpu'))
        emo_checkpoint = torch.load(emo_checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
        audio_checkpoint = torch.load(audio_checkpoint_path)
        emo_checkpoint = torch.load(emo_checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    audio_feature.load_state_dict(audio_checkpoint['audio_feature'])
    kp_detector_a.load_state_dict(audio_checkpoint['kp_detector_a'])
    emo_detector.load_state_dict(emo_checkpoint['emo_detector'])
    

    if not cpu:
        generator = generator.cuda()
        kp_detector = kp_detector.cuda()
        audio_feature = audio_feature.cuda()
        kp_detector_a = kp_detector_a.cuda()
        emo_detector = emo_detector.cuda()

    generator.eval()
    kp_detector.eval()
    audio_feature.eval()
    kp_detector_a.eval()
    emo_detector.eval()
    return generator, kp_detector, kp_detector_a, audio_feature, emo_detector

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def get_aligned_image(driving_video, opt):
    aligned_array = []

    video_array = np.array(driving_video)
    source_image=video_array[0]
   # aligned_array.append(source_image)
    source_image = np.array(source_image * 255, dtype=np.uint8)
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)  #detect human face
    for (i, rect) in enumerate(rects):
        template = predictor(gray, rect) #detect 68 points
        template = shape_to_np(template)

    if opt.emotion == 'surprised' or opt.emotion == 'fear':
        template = template-[0,10]
    for i in range(len(video_array)):
        image=np.array(video_array[i] * 255, dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)  #detect human face
        for (j, rect) in enumerate(rects):
            shape = predictor(gray, rect) #detect 68 points
            shape = shape_to_np(shape)

        pts2 = np.float32(template[:35,:])
        pts1 = np.float32(shape[:35,:]) #eye and nose

    #    pts2 = np.float32(np.concatenate((template[:16,:],template[27:36,:]),axis = 0))
    #    pts1 = np.float32(np.concatenate((shape[:16,:],shape[27:36,:]),axis = 0)) #eye and nose
        # pts1 = np.float32(landmark[17:35,:])
        tform = tf.SimilarityTransform()
        tform.estimate( pts2, pts1) #Set the transformation matrix with the explicit parameters.
        dst = tf.warp(image, tform, output_shape=(256, 256))

        dst = np.array(dst, dtype=np.float32)
        aligned_array.append(dst)

    return aligned_array

def get_transformed_image(driving_video, opt):
    video_array = np.array(driving_video)
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    transformations = AllAugmentationTransform(**config['dataset_params']['augmentation_params'])
    transformed_array = transformations(video_array)
    return transformed_array



def make_animation_smooth(source_image, driving_video, transformed_video, deco_out, kp_loss, generator, kp_detector, kp_detector_a, emo_detector, opt, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []

        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

        if not cpu:
            source = source.cuda()

        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        transformed_driving = torch.tensor(np.array(transformed_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector_a(deco_out[:,0])

        emo_driving_all = []
        features = []
        kp_driving_all = []
        for frame_idx in tqdm(range(len(deco_out[0]))):

            driving_frame = driving[:, :, frame_idx]
            transformed_frame = transformed_driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
                transformed_frame = transformed_frame.cuda()
            kp_driving = kp_detector_a(deco_out[:,frame_idx])
            kp_driving_all.append(kp_driving)
            if opt.add_emo:
                value = kp_driving['value']
                jacobian = kp_driving['jacobian']
                if opt.type == 'linear_3':
                    emo_driving,_ = emo_detector(transformed_frame,value,jacobian)
                    features.append(emo_detector.feature(transformed_frame).data.cpu().numpy())
            
                emo_driving_all.append(emo_driving)
        features = np.array(features)
        if opt.add_emo:        
            one_euro_filter_v = OneEuroFilter(mincutoff=1, beta=0.2, dcutoff=1.0, freq=100)#1 0.4
            one_euro_filter_j = OneEuroFilter(mincutoff=1, beta=0.2, dcutoff=1.0, freq=100)#1 0.4

            for j in range(len(emo_driving_all)):
                emo_driving_all[j]['value']=one_euro_filter_v.process(emo_driving_all[j]['value'].cpu()*100)/100
                emo_driving_all[j]['value'] = emo_driving_all[j]['value'].cuda()
                emo_driving_all[j]['jacobian']=one_euro_filter_j.process(emo_driving_all[j]['jacobian'].cpu()*100)/100
                emo_driving_all[j]['jacobian'] = emo_driving_all[j]['jacobian'].cuda()


        one_euro_filter_v = OneEuroFilter(mincutoff=0.05, beta=8, dcutoff=1.0, freq=100)
        one_euro_filter_j = OneEuroFilter(mincutoff=0.05, beta=8, dcutoff=1.0, freq=100)

        for j in range(len(kp_driving_all)):
            kp_driving_all[j]['value']=one_euro_filter_v.process(kp_driving_all[j]['value'].cpu()*10)/10
            kp_driving_all[j]['value'] = kp_driving_all[j]['value'].cuda()
            kp_driving_all[j]['jacobian']=one_euro_filter_j.process(kp_driving_all[j]['jacobian'].cpu()*10)/10
            kp_driving_all[j]['jacobian'] = kp_driving_all[j]['jacobian'].cuda()


        for frame_idx in tqdm(range(len(deco_out[0]))):
            
            if opt.check_add:
                kp_driving = kp_detector_a(deco_out[:,0])
            else:
                kp_driving = kp_driving_all[frame_idx]

       #     kp_driving_real = kp_detector(driving_frame)

       #     kp_driving['value'] = (1-opt.weight)*kp_driving['value'] + opt.weight*kp_driving_real['value']
       #     kp_driving['jacobian'] = (1-opt.weight)*kp_driving['jacobian'] + opt.weight*kp_driving_real['jacobian']

            if opt.add_emo:
                emo_driving = emo_driving_all[frame_idx]
                if opt.type == 'linear_3':
                    kp_driving['value'][:,1] = kp_driving['value'][:,1] + emo_driving['value'][:,0]*0.2
                    kp_driving['jacobian'][:,1] = kp_driving['jacobian'][:,1] + emo_driving['jacobian'][:,0]*0.2
                    kp_driving['value'][:,4] = kp_driving['value'][:,4] + emo_driving['value'][:,1]
                    kp_driving['jacobian'][:,4] = kp_driving['jacobian'][:,4] + emo_driving['jacobian'][:,1]
                    kp_driving['value'][:,6] = kp_driving['value'][:,6] + emo_driving['value'][:,2]
                    kp_driving['jacobian'][:,6] = kp_driving['jacobian'][:,6] + emo_driving['jacobian'][:,2]
                   # kp_driving['value'][:,8] = kp_driving['value'][:,8] + emo_driving['value'][:,3]
                   # kp_driving['jacobian'][:,8] = kp_driving['jacobian'][:,8] + emo_driving['jacobian'][:,3]
               
         
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions, features



def test_auido(example_image, audio_feature, all_pose, opt):
    with open(opt.config) as f:
        para = yaml.load(f, Loader=yaml.FullLoader)

  #  encoder = audio_feature()
    if not opt.cpu:
        audio_feature = audio_feature.cuda()

    audio_feature.eval()
 #   decoder.eval()
    test_file = opt.in_file
    pose = all_pose[:,:6]
    if len(pose) == 1:
        pose = np.repeat(pose,100,0)

    elif opt.smooth_pose:
        one_euro_filter = OneEuroFilter(mincutoff=0.004, beta=0.7, dcutoff=1.0, freq=100)


        for j in range(len(pose)):
            pose[j]=one_euro_filter.process(pose[j])
      #      pose[j]=pose[0]

    example_image = np.array(example_image, dtype='float32').transpose((2, 0, 1))

    


    speech, sr = librosa.load(test_file, sr=16000)
  #  mfcc = python_speech_features.mfcc(speech ,16000,winstep=0.01)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)


    print ('=======================================')
    print ('Start to generate images')

    ind = 3
    with torch.no_grad():
        fake_lmark = []
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc).cuda()
            input_mfcc.append(t_mfcc)
            ind += 1
        input_mfcc = torch.stack(input_mfcc,dim = 0)

        if (len(pose)<len(input_mfcc)):
            gap = len(input_mfcc)-len(pose)
            n = int((gap/len(pose)/2)) +2
            pose = np.concatenate((pose,pose[::-1,:]),axis = 0)
            pose = np.tile(pose, (n,1))
        if(len(pose)>len(input_mfcc)):
            pose = pose[:len(input_mfcc),:]
        
        if not opt.cpu:
            example_image = Variable(torch.FloatTensor(example_image.astype(float)) ).cuda()
            example_image = torch.unsqueeze(example_image,0)
            pose = Variable(torch.FloatTensor(pose.astype(float)) ).cuda()
        
        pose = pose.unsqueeze(0)

        input_mfcc = input_mfcc.unsqueeze(0)

        deco_out = audio_feature(example_image,input_mfcc,pose,para['train_params']['jaco_net'],1.6)

        return deco_out


def save(path, frames, format):

    if format == '.png':
        if not os.path.exists(path):

            os.makedirs(path)
        for j, frame in enumerate(frames):
            imageio.imsave(path+'/'+str(j)+'.png',frame)
    #        imageio.imsave(os.path.join(path, str(j) + '.png'), frames[j])
    else:
        print ("Unknown format %s" % format)
        exit()

class VideoWriter(object):
    def __init__(self, path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.path = path
        self.out = cv2.VideoWriter(self.path, fourcc, fps, (width, height))

    def write_frame(self, frame):
        self.out.write(frame)

    def end(self):
        self.out.release()

def concatenate(number, imgs, save_path):
    width, height = imgs.shape[-3:-1]
    imgs = imgs.reshape(number,-1,width,height,3)
    if number == 2:
        left = imgs[0]
        right = imgs[1]

        im_all = []
        for i in range(len(left)):
            im = np.concatenate((left[i],right[i]),axis = 1)
            im_all.append(im)
    if number == 3:
        left = imgs[0]
        middle = imgs[1]
        right = imgs[2]

        im_all = []
        for i in range(len(left)):
            im = np.concatenate((left[i],middle[i],right[i]),axis = 1)
            im_all.append(im)
    if number == 4:
        left = imgs[0]
        left2 = imgs[1]
        right = imgs[2]
        right2 = imgs[3]

        im_all = []
        for i in range(len(left)):
            im = np.concatenate((left[i],left2[i],right[i],right2[i]),axis = 1)
            im_all.append(im)
    if number == 5:
        left = imgs[0]
        left2 = imgs[1]
        middle = imgs[2]
        right = imgs[3]
        right2 = imgs[4]

        im_all = []
        for i in range(len(left)):
            im = np.concatenate((left[i],left2[i],middle[i],right[i],right2[i]),axis = 1)
            im_all.append(im)


    imageio.mimsave(save_path, [img_as_ubyte(frame) for frame in im_all], fps=25)

def add_audio(video_name=None, audio_dir = None):

    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
    print (command)
    os.system(command)

def crop_image(source_image):
    
    template = np.load('./M003_template.npy')
    image= cv2.imread(source_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)  #detect human face
    if len(rects) != 1:
        return 0
    for (j, rect) in enumerate(rects):
        shape = predictor(gray, rect) #detect 68 points
        shape = shape_to_np(shape)

    pts2 = np.float32(template[:47,:])
    pts1 = np.float32(shape[:47,:]) #eye and nose
    # pts1 = np.float32(landmark[17:35,:])
    tform = tf.SimilarityTransform()
    tform.estimate( pts2, pts1) #Set the transformation matrix with the explicit parameters.
  
    dst = tf.warp(image, tform, output_shape=(256, 256))

    dst = np.array(dst * 255, dtype=np.uint8)
    return dst 

def smooth_pose(pose_file, pose_long):
    start = np.load(pose_file)
    video_pose = np.load(pose_long)
    delta = video_pose - video_pose[0,:]
    print(len(delta))
    
    pose = np.repeat(start,len(delta),axis = 0)
    all_pose =  pose + delta

    return all_pose

def test(opt, name):

    all_pose = np.load(opt.pose_file).reshape(-1,7)
    if opt.pose_long:

        all_pose = smooth_pose(opt.pose_file,opt.pose_given)

    
   # source_image = img_as_float32(io.imread(opt.source_image))
    source_image = img_as_float32(crop_image(opt.source_image))
    source_image = resize(source_image, (256, 256))[..., :3]
  
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

   
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    driving_video = get_aligned_image(driving_video, opt)
    transformed_video = get_transformed_image(driving_video, opt)
    transformed_video = np.array(transformed_video)

    generator, kp_detector,kp_detector_a, audio_feature, emo_detector = load_checkpoints(opt=opt, checkpoint_path=opt.checkpoint, audio_checkpoint_path=opt.audio_checkpoint, emo_checkpoint_path = opt.emo_checkpoint, cpu=opt.cpu)
 
    deco_out = test_auido(source_image, audio_feature, all_pose, opt)
    if len(driving_video) < len(deco_out[0]):
        driving_video = np.resize(driving_video,(len(deco_out[0]),256,256,3))
        transformed_video = np.resize(transformed_video,(len(deco_out[0]),256,256,3))

    else:
        driving_video = driving_video[:len(deco_out[0])]
    opt.add_emo = False
    predictions, _ = make_animation_smooth(source_image, driving_video, transformed_video, deco_out, opt.kp_loss, generator, kp_detector, kp_detector_a, emo_detector, opt, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
  
    imageio.mimsave(os.path.join(opt.result_path,'neutral.mp4'), [img_as_ubyte(frame) for frame in predictions], fps=fps)
    predictions = np.array(predictions)
     
    opt.add_emo = True
  
    predictions1,_ = make_animation_smooth(source_image, driving_video, transformed_video, deco_out, opt.kp_loss, generator, kp_detector, kp_detector_a, emo_detector, opt, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
  
    imageio.mimsave(os.path.join(opt.result_path,'emotion.mp4'), [img_as_ubyte(frame) for frame in predictions1], fps=fps)
    add_audio(os.path.join(opt.result_path,'emotion.mp4'),opt.in_file)
    predictions1 = np.array(predictions1)
    all_imgs = np.concatenate((driving_video,predictions,predictions1),axis = 0)
    save_path = os.path.join(opt.result_path, 'all.mp4')
    concatenate(3, all_imgs, save_path)
    add_audio(save_path,opt.in_file)



if __name__ == "__main__":
   
    
   
    parser = ArgumentParser()
    parser.add_argument("--config", default ='config/MEAD_emo_video_aug_delta_4_crop_random_crop.yaml', help="path to config")#required=True default ='config/vox-256.yaml'
 
    parser.add_argument("--audio_checkpoint", default='log/1-6000.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--checkpoint", default='log/124_52000.pth.tar', help="path to checkpoint to restore")
   # parser.add_argument("--emo_checkpoint", default='ablation/ablation/ten/10-6000.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--emo_checkpoint", default='log/5-3000.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='test/image/21.png', help="path to source image")
 
    parser.add_argument("--driving_video", default='test/video/disgusted.mp4', help="path to driving video")#data/M030/video/M030_angry_
    parser.add_argument('--in_file', type=str, default='test/audio/sample1.mov')
    parser.add_argument('--pose_file', type=str, default='test/pose/21.npy')
    parser.add_argument('--pose_given', type=str, default='test/pose_long/0zn70Ak8lRc_Daniel_Auteuil_0zn70Ak8lRc_0002.npy')

    parser.add_argument("--result_path", default='result/', help="path to output")#'/media/thea/新加卷/fomm/Exp/'+emotion+'.mp4'

    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--kp_loss", default=0, help="keypoint loss.")

    parser.add_argument("--smooth_pose",  default=True, help="cpu mode.")
    parser.add_argument("--pose_long",  default=False, help="use given long poses.")
    parser.add_argument("--weight",  default=0, help="cpu mode.")
    parser.add_argument("--add_emo",  default=False, help="add emotion.")
    parser.add_argument("--check_add",  default=False, help="check emotion displacement.")
    parser.add_argument("--type",  default='linear_3', help="add emotion type.")
    parser.add_argument("--emotion",  default='disgusted', help="emotion category, 'angry', 'contempt','disgusted','fear','happy','neutral','sad','surprised'.")
    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()
 #   opt.cpu = True
   
    test(opt,'test')
         
    