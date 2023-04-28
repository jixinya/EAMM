# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:36:01 2021

@author: Xinya
"""

import os
import glob
import time
import numpy as np
import csv
import cv2
import dlib

from skimage import transform as tf

import librosa
import python_speech_features

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


import imageio



def save(path, frames, format):
    if format == '.mp4':
        imageio.mimsave(path, frames)
    elif format == '.png':
        if not os.path.exists(path):


            os.makedirs(path)
        for j, frame in enumerate(frames):
            cv2.imwrite(path+'/'+str(j)+'.png',frame)
    #        imageio.imsave(os.path.join(path, str(j) + '.png'), frames[j])
    else:
        print ("Unknown format %s" % format)
        exit()

def crop_image(image_path, out_path):
    template = np.load('./M003_template.npy')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)  #detect human face
    if len(rects) != 1:
        return 0
    for (j, rect) in enumerate(rects):
        shape = predictor(gray, rect) #detect 68 points
        shape = shape_to_np(shape)

    pts2 = np.float32(template[:47,:])
    # pts2 = np.float32(template[17:35,:])
    # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
    pts1 = np.float32(shape[:47,:]) #eye and nose
    # pts1 = np.float32(landmark[17:35,:])
    tform = tf.SimilarityTransform()
    tform.estimate( pts2, pts1) #Set the transformation matrix with the explicit parameters.
    
    dst = tf.warp(image, tform, output_shape=(256, 256))

    dst = np.array(dst * 255, dtype=np.uint8)
    
    
    cv2.imwrite(out_path,dst)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords




def crop_image_tem(video_path, out_path):
    image_all = []
    videoCapture = cv2.VideoCapture(video_path)
    success, frame = videoCapture.read()
    n = 0
    while success :
        image_all.append(frame)
        n = n + 1
        success, frame = videoCapture.read()
        
    if len(image_all)!=0 :
        template = np.load('./M003_template.npy')
        image=image_all[0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)  #detect human face
        if len(rects) != 1:
            return 0
        for (j, rect) in enumerate(rects):
            shape = predictor(gray, rect) #detect 68 points
            shape = shape_to_np(shape)

        pts2 = np.float32(template[:47,:])
        # pts2 = np.float32(template[17:35,:])
        # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
        pts1 = np.float32(shape[:47,:]) #eye and nose
        # pts1 = np.float32(landmark[17:35,:])
        tform = tf.SimilarityTransform()
        tform.estimate( pts2, pts1) #Set the transformation matrix with the explicit parameters.
        out = []
        for i in range(len(image_all)):
            image = image_all[i]
            dst = tf.warp(image, tform, output_shape=(256, 256))

            dst = np.array(dst * 255, dtype=np.uint8)
            out.append(dst)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        save(out_path,out,'.png')

def proc_audio(src_mouth_path, dst_audio_path):
    audio_command = 'ffmpeg -i \"{}\" -loglevel error -y -f wav -acodec pcm_s16le ' \
                    '-ar 16000 \"{}\"'.format(src_mouth_path, dst_audio_path)
    os.system(audio_command)


def audio2mfcc(audio_file, save, name):
    speech, sr = librosa.load(audio_file, sr=16000)
  #  mfcc = python_speech_features.mfcc(speech ,16000,winstep=0.01)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)
    if not os.path.exists(save):
        os.makedirs(save)
    time_len = mfcc.shape[0]
    mfcc_all = []
    for input_idx in range(int((time_len-28)/4)+1):
         #   target_idx = input_idx + sample_delay #14

        input_feat = mfcc[4*input_idx:4*input_idx+28,:]
        mfcc_all.append(input_feat)
    np.save(os.path.join(save,name+'.npy'), mfcc_all)

    print(input_idx)

if __name__ == "__main__":
    #video alignment
    video_path = './test/crop/M030_sad_3_001.mp4'
    out_path = './test/crop/M030_sad_3_001'
    crop_image_tem(video_path, out_path)
    
    #image alignment
    image_path = './test/raw_image/brade2.jpg'
    out_path = './test/image/brade2.jpg'
    crop_image(image_path, out_path)
    
    #change_audio_sample_rate
    src_mouth_path = './test/audio/00015.mp3'
    dst_audio_path = './test/audio/00015.mov'
    proc_audio(src_mouth_path, dst_audio_path)

    #audio2mfcc
    #mead
    path = './dataset/MEAD/audio/'
    pathDir = os.listdir(path)
    for i in range(len(pathDir)):#len(pathDir)
        name = pathDir[i]
        filepath = os.path.join(path,name)
        if os.path.exists(filepath):
            Dir = os.listdir(filepath)
            save_path = './dataset/MEAD/MEAD_MFCC/'+name
            os.makedirs(save_path,exist_ok=True)
            for j in range(len(Dir)):

                index = Dir[j].split('.')[0]
                audio_path = os.path.join(filepath,Dir[j])
                audio2mfcc(audio_path, save_path,index)
                print(i,name,j,index)
        else:
            print('not exist ',filepath)
