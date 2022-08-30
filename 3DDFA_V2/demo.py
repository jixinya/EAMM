# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml
import os
import time
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
#from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose, get_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool
import numpy as np
from tqdm import tqdm
import copy

import concurrent.futures
from multiprocessing import Pool

def main(args,img, save_path, pose_path):
 #   begin = time.time()
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
  #  img = cv2.imread(img_path) #args.img_fp

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
      #  sys.exit(-1)
        return None
    print(f'Detect {n} faces')

    param_lst, roi_box_lst = tddfa(img, boxes)
    #detection time
  #  detect_time = time.time()-begin
 #   print('detection time: '+str(detect_time), file=open('/mnt/lustre/jixinya/Home/3DDFA_V2/pose.txt', 'a'))
    # Visualization and serialization
    dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
  #  old_suffix = get_suffix(img_path)
    old_suffix = 'png'
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

    if args.opt == '2d_sparse':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '2d_dense':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '3d':
        render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'depth':

        # if `with_bf_flag` is False, the background is black
        depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'pncc':
        pncc(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'uv_tex':
        uv_tex(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'pose':
        all_pose = get_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=save_path, wnp = pose_path)
    elif args.opt == 'ply':
        ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    elif args.opt == 'obj':
        ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    else:
        raise ValueError(f'Unknown opt {args.opt}')

    return all_pose



def process_word(i):
    path = '/media/xinya/Backup Plus/sense_shixi_data/new_crop/MEAD_fomm_video_6/'
    save = '/media/xinya/Backup Plus/sense_shixi_data/new_crop/MEAD_fomm_pose_im/'
    pose = '/media/xinya/Backup Plus/sense_shixi_data/new_crop/MEAD_fomm_pose/'
    start = time.time()
    Dir = os.listdir(path)
    Dir.sort()
    word = Dir[i]
    wpath = os.path.join(path, word)
    print(wpath)
    pathDir = os.listdir(wpath)
    pose_file = os.path.join(pose,word)
    if not os.path.exists(pose_file):
        os.makedirs(pose_file)

    for j in range(len(pathDir)):
        name = pathDir[j]
     #   save_file = os.path.join(save,word,name)
     #   if not os.path.exists(save_file):
     #       os.makedirs(save_file)
        fpath = os.path.join(wpath,name)
        image_all = []
        videoCapture = cv2.VideoCapture(fpath)

        success, frame = videoCapture.read()

        n = 0
        while success :
            image_all.append(frame)
            n = n + 1
            success, frame = videoCapture.read()

     #   fDir = os.listdir(fpath)
        pose_all = np.zeros((len(image_all),7))
        for k in range(len(image_all)):
    #        index = fDir[k].split('.')[0]
    #        img_path = os.path.join(fpath,str(k)+'.png')

     #       pose_all[k] = main(args,image_all[k], os.path.join(save_file,str(k)+'.jpg'), None)
            pose_all[k] = main(args,image_all[k], None, None)
        np.save(os.path.join(pose,word,name.split('.')[0]+'.npy'),pose_all)
        st = time.time()-start
        print(str(i)+' '+word+' '+str(j)+' '+name+' '+str(k)+'time: '+str(st), file=open('/media/thea/Backup Plus/sense_shixi_data/new_crop/pose_mead6.txt', 'a'))
        print(i,word,j,name,k)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/0.png')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='pose',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='False', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()


    
    filepath = 'test/image/'
    pathDir = os.listdir(filepath)
    for i in range(len(pathDir)):
        image= cv2.imread(os.path.join(filepath,pathDir[i]))
        pose = main(args,image, None, None).reshape(1,7)

        np.save('test/pose/'+pathDir[i].split('.')[0]+'.npy',pose)
        print(i,pathDir[i])
        
        
'''





def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)
    print(f'Detect {n} faces')

    param_lst, roi_box_lst = tddfa(img, boxes)

    # Visualization and serialization
    dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
    old_suffix = get_suffix(args.img_fp)
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

    if args.opt == '2d_sparse':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '2d_dense':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '3d':
        render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'depth':
        # if `with_bf_flag` is False, the background is black
        depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'pncc':
        pncc(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'uv_tex':
        uv_tex(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'pose':
        viz_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'ply':
        ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    elif args.opt == 'obj':
        ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    else:
        raise ValueError(f'Unknown opt {args.opt}')
'''