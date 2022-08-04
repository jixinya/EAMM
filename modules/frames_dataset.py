import os
from skimage import io, img_as_float32, transform
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob
import pickle
import random
def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

def get_list(ipath,base_name):
#ipath = '/mnt/lustre/share/jixinya/LRW/pose/train_fo/'
    ipath = os.path.join(ipath,base_name)
    name_list = os.listdir(ipath)
    image_path = os.path.join('/mnt/lustre/share/jixinya/LRW/Image/',base_name)
    all = []
    for k in range(len(name_list)):
        name = name_list[k]
        path_ = os.path.join(ipath,name)
        Dir = os.listdir(path_)
        for i in range(len(Dir)):
            word = Dir[i]
            path = os.path.join(path_, word)
            if os.path.exists(os.path.join(image_path,name,word.split('.')[0])):
                all.append(name+'/'+word.split('.')[0])
            #print(k,name,i,word)
    print('get list '+os.path.basename(ipath))
    return all


class AudioDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir,'MFCC')
        self.image_dir = os.path.join(root_dir,'Image')
        self.landmark_dir =  os.path.join(root_dir,'Landmark') 
        self.pose_dir = os.path.join(root_dir,'pose')
      #  assert len(os.listdir(self.audio_dir)) == len(os.listdir(self.image_dir)), 'audio and image length not equal'
        
      
        df=open('../LRW/list/test_fo.txt','rb')
        self.videos=pickle.load(df)
        df.close()
      #  self.videos=np.load('../LRW/list/train_fo.npy')
      #  self.videos = os.listdir(self.landmark_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.pca = np.load('../LRW/list/U_106.npy')[:, :16]
        self.mean = np.load('../LRW/list/mean_106.npy')

        if os.path.exists(os.path.join(self.pose_dir, 'train_fo')):
            assert os.path.exists(os.path.join(self.pose_dir, 'test_fo'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(self.image_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = np.load('../LRW/list/train_fo.npy')# get_list(self.pose_dir, 'train_fo')
            df=open('../LRW/list/test_fo.txt','rb')
            test_videos=pickle.load(df)
            df.close()
         #   test_videos = np.load('../LRW/list/train_fo.npy')
            #get_list(self.pose_dir, 'test_fo')
        #    self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            self.landmark_dir = os.path.join(self.landmark_dir, 'train_fo' if is_train else 'test_fo')
            self.image_dir = os.path.join(self.image_dir, 'train_fo' if is_train else 'test_fo')
            self.audio_dir = os.path.join(self.audio_dir, 'train' if is_train else 'test')
            self.pose_dir = os.path.join(self.pose_dir, 'train_fo' if is_train else 'test_fo')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx].split('.')[0]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx].split('.')[0]
            landmark_path = os.path.join(self.landmark_dir, name+'.npy')
            
            audio_path = os.path.join(self.audio_dir, name)
            pose_path = os.path.join(self.pose_dir,name)
            path = os.path.join(self.image_dir, name)

        video_name = os.path.basename(path)

        if  os.path.isdir(path):
     #   if self.is_train and os.path.isdir(path):
            
            lmark = np.load(landmark_path).reshape(-1,212)/255
            if np.isnan(lmark).sum() or np.isinf(lmark).sum():
                print('Wrong lmark '+ video_name, file=open('log/wrong.txt', 'a'))
                lmark = np.zeros((29,212))  
            lmark = lmark - self.mean
            lmark = np.dot(lmark, self.pca)

            # mfcc loading
            
            r = random.choice([x for x in range(3, 8)])
            example_landmark = lmark[r, :]
            example_image = img_as_float32(io.imread(os.path.join(path, str(r)+'.png')))
          #  example_mfcc = mfcc[(r - 3) * 4: (r + 4) * 4, 1:]
           
            mfccs = []
            for ind in range(1, 17):
              #  t_mfcc = mfcc[(r + ind - 3) * 4: (r + ind + 4) * 4, 1:]
                try:
                    t_mfcc = np.load(os.path.join(audio_path,str(r + ind)+'.npy'),allow_pickle=True)[:, 1:]
                    if np.isnan(t_mfcc).sum() or np.isinf(t_mfcc).sum():
                        print('Wrong mfcc '+ video_name+str(r+ind), file=open('log/wrong.txt', 'a'))
                        t_mfcc = np.zeros((28,13))[:,1:]
                except:
                    t_mfcc = np.zeros((28,13))[:,1:]
                mfccs.append(t_mfcc)
            mfccs = np.array(mfccs)
            if not self.is_train:
                poses = []
                video_array = []
                for ind in range(1, 17):
              #  t_mfcc = mfcc[(r + ind - 3) * 4: (r + ind + 4) * 4, 1:]
                    t_pose = np.load(os.path.join(pose_path,str(r + ind)+'.npy'))[:-1]
                    poses.append(t_pose)
                    image = img_as_float32(io.imread(os.path.join(path, str(r + ind)+'.png')))
                    video_array.append(image)
                poses = np.array(poses)
                video_array = np.array(video_array)
            else:
                poses = []
                video_array = []
                for ind in range(1, 17):
              #  t_mfcc = mfcc[(r + ind - 3) * 4: (r + ind + 4) * 4, 1:]
                    t_pose = np.load(os.path.join(self.pose_dir,name+'.npy'))[r+ind,:-1]
                    if np.isnan(t_pose).sum() or np.isinf(t_pose).sum():
                        print('Wrong pose '+ video_name, file=open('log/wrong.txt', 'a'))
                        t_pose = np.zeros((6,))
                    poses.append(t_pose)
                    image = img_as_float32(io.imread(os.path.join(path, str(r + ind)+'.png')))
                    video_array.append(image)
                poses = np.array(poses)
                video_array = np.array(video_array)
            
            #mfccs = torch.FloatTensor(mfccs)
            landmark = lmark[r + 1: r + 17, :]
            index_32 = [0,4,8,12,16,20,24,28,32,33,35,67,68,40,42,52,55,72,73,58,61,75,76,46,47,51,84,87,90,93,98,102]
            driving_landmark = np.load(landmark_path)[r + 1: r + 17, :][:,index_32]
            source_landmark = np.load(landmark_path)[r, :][index_32]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if True:#self.is_train:
          #  a = img_as_float32(io.imread('/media/thea/Data/first-order-model/images_512/102.jpg'))
          #  source = np.array(a, dtype='float32')
         
            driving = np.array(video_array, dtype='float32')
            
            spatial_size = np.array(driving.shape[1:3][::-1])[np.newaxis]
      #      example_landmark = np.array(2*example_landmark / spatial_size -1, dtype='float32')
            driving_landmark = np.array(2*driving_landmark / spatial_size -1, dtype='float32')
            source_landmark = np.array(2*source_landmark / spatial_size -1, dtype='float32')
            driving_pose = np.array(poses, dtype='float32')
            example_landmark = np.array(example_landmark, dtype='float32')
            example_image = np.array(example_image, dtype='float32')
      #      source_cube = np.array(transform.resize(cube_array[0], (64,64)), dtype='float32')
      #      driving_cube = np.array(transform.resize(cube_array[1], (64,64)), dtype='float32')
       #     source_heatmap = np.array(heatmap_array[0] , dtype='float32')
       #     driving_heatmap = np.array(heatmap_array[1] , dtype='float32')
       #     out['source_cube'] = source_cube
       #     out['driving_cube'] = driving_cube
            out['example_landmark'] = example_landmark
            out['example_image'] = example_image.transpose((2, 0, 1))
            out['driving_landmark'] = driving_landmark
            out['source_landmark'] = source_landmark
            out['driving_pose'] = driving_pose
      #      out['source_heatmap'] = source_heatmap
     #       out['driving_heatmap'] = driving_heatmap
            out['driving'] = driving.transpose((0, 3, 1, 2))
        #    out['source'] = source.transpose((2, 0, 1))
            
        #    out['source_audio'] = np.array(audio_array[0], dtype='float32')
            out['driving_audio'] = np.array(mfccs, dtype='float32')
            out['gt_landmark'] = np.array(landmark, dtype='float32')
            out['pca'] = np.array(self.pca, dtype='float32')
            out['mean'] = np.array(self.mean, dtype='float32')
     

        out['name'] = video_name

        return out

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir,'audio/')
        self.image_dir = os.path.join(root_dir,'image/')
        self.landmark_dir =  os.path.join(root_dir,'cube/') 
      #  assert len(os.listdir(self.audio_dir)) == len(os.listdir(self.image_dir)), 'audio and image length not equal'
        
      
        df=open('/media/thea/新加卷/MEAD/neutral/train.txt','rb')
        self.videos=pickle.load(df)
        df.close()
      #  self.videos = os.listdir(self.landmark_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(self.image_dir, 'train')):
            assert os.path.exists(os.path.join(self.image_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(self.image_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(self.image_dir, 'train'))
            test_videos = os.listdir(os.path.join(self.image_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            self.landmark_dir = os.path.join(self.landmark_dir, 'train' if is_train else 'test')
            self.image_dir = os.path.join(self.image_dir, 'train' if is_train else 'test')
            self.audio_dir = os.path.join(self.audio_dir, 'train' if is_train else 'test')
            
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx].split('.')[0]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx].split('.')[0]
            landmark_path = os.path.join(self.landmark_dir, name)
            
            audio_path = os.path.join(self.audio_dir, name)
            path = os.path.join(self.image_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(audio_path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames-1, replace=True, size=2))
          #  landmark = np.load(landmark_path)#+'.npy'
          #  assert len(os.listdir(path)) == len(landmark), video_name+' length not equal'
            video_array = [img_as_float32(io.imread(os.path.join(path, str(idx)+'.png'))) for idx in frame_idx]
            cube_array = [img_as_float32(io.imread(os.path.join(landmark_path, str(idx)+'.jpg'))) for idx in frame_idx]
            audio_array = [np.load(os.path.join(audio_path, str(idx)+'.npy'))[:,1:] for idx in frame_idx]
            index_20 = [0,16,32,35,40,52,55,58,61,46,72,73,75,76,84,87,90,93,98,102]
            index_32 = [0,4,8,12,16,20,24,28,32,33,35,67,68,40,42,52,55,72,73,58,61,75,76,46,47,51,84,87,90,93,98,102]
         #   landmark_array = [landmark[idx] for idx in frame_idx]
          #  landmark_array = [landmark[idx][index_32] for idx in frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
          #  a = img_as_float32(io.imread('/media/thea/Data/first-order-model/images_512/102.jpg'))
          #  source = np.array(a, dtype='float32')
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            
            spatial_size = np.array(source.shape[:2][::-1])[np.newaxis]
           # source_landmark = np.array(2*landmark_array[0] / spatial_size -1, dtype='float32')
           # driving_landmark = np.array(2*landmark_array[1] / spatial_size -1, dtype='float32')
            source_cube = np.array(transform.resize(cube_array[0], (64,64)), dtype='float32')
            driving_cube = np.array(transform.resize(cube_array[1], (64,64)), dtype='float32')
       #     source_heatmap = np.array(heatmap_array[0] , dtype='float32')
       #     driving_heatmap = np.array(heatmap_array[1] , dtype='float32')
            out['source_cube'] = source_cube
            out['driving_cube'] = driving_cube
          #  out['source_landmark'] = source_landmark
          #  out['driving_landmark'] = driving_landmark
      #      out['source_heatmap'] = source_heatmap
     #       out['driving_heatmap'] = driving_heatmap
            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            
            out['source_audio'] = np.array(audio_array[0], dtype='float32')
            out['driving_audio'] = np.array(audio_array[1], dtype='float32')
            
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]#% self.dataset.__len__()


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
