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
from filter1 import OneEuroFilter
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

    def __init__(self, name, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, augmentation_params=None):
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir,'MFCC')
        self.image_dir = os.path.join(root_dir,'Image')
        self.pose_dir = os.path.join(root_dir,'pose')
      #  assert len(os.listdir(self.audio_dir)) == len(os.listdir(self.image_dir)), 'audio and image length not equal'

      #  self.videos=np.load('../LRW/list/train_fo.npy')
      #  self.videos = os.listdir(self.landmark_dir)
        self.frame_shape = tuple(frame_shape)
       
        self.id_sampling = id_sampling

        if os.path.exists(os.path.join(self.pose_dir, 'train_fo')):
            assert os.path.exists(os.path.join(self.pose_dir, 'test_fo'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(self.image_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos =  np.load('../LRW/list/train_fo.npy')# get_list(self.pose_dir, 'train_fo')
         #   df=open('../LRW/list/test_fo.txt','rb')
            test_videos=np.load('../LRW/list/test_fo.npy')
         #   df.close()
         #   test_videos = np.load('../LRW/list/train_fo.npy')
            #get_list(self.pose_dir, 'test_fo')
        #    self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
           
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
           
            audio_path = os.path.join(self.audio_dir, name)
            pose_path = os.path.join(self.pose_dir,name)
            path = os.path.join(self.image_dir, name)

        video_name = os.path.basename(path)

        if  os.path.isdir(path):
     #   if self.is_train and os.path.isdir(path):
         
            # mfcc loading
            r = random.choice([x for x in range(3, 8)])

            example_image = img_as_float32(io.imread(os.path.join(path, str(r)+'.png')))

            mfccs = []
            for ind in range(1, 17):
              #  t_mfcc = mfcc[(r + ind - 3) * 4: (r + ind + 4) * 4, 1:]
                t_mfcc = np.load(os.path.join(audio_path,str(r + ind)+'.npy'),allow_pickle=True)[:, 1:]
                mfccs.append(t_mfcc)
            mfccs = np.array(mfccs)
            
            poses = []
            video_array = []
            for ind in range(1, 17):
              
                t_pose = np.load(os.path.join(self.pose_dir,name+'.npy'))[r+ind,:-1]
                
                poses.append(t_pose)
                image = img_as_float32(io.imread(os.path.join(path, str(r + ind)+'.png')))
                video_array.append(image)
            poses = np.array(poses)
            video_array = np.array(video_array)

        else:
            print('Wrong, data path not an existing file.')

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
     
        driving = np.array(video_array, dtype='float32')
        spatial_size = np.array(driving.shape[1:3][::-1])[np.newaxis]
        driving_pose = np.array(poses, dtype='float32')
        example_image = np.array(example_image, dtype='float32')

        out['example_image'] = example_image.transpose((2, 0, 1))
        out['driving_pose'] = driving_pose
        out['driving'] = driving.transpose((0, 3, 1, 2))
        out['driving_audio'] = np.array(mfccs, dtype='float32')
    #    out['name'] = video_name

        return out

class VoxDataset(Dataset):
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
        self.image_dir = os.path.join(root_dir,'align_img')

        self.pose_dir = os.path.join(root_dir,'align_pose')
      #  assert len(os.listdir(self.audio_dir)) == len(os.listdir(self.image_dir)), 'audio and image length not equal'


     #   df=open('../LRW/list/test_fo.txt','rb')
     #  self.videos=pickle.load(df)
     #   df.close()
        self.videos=np.load('/mnt/lustre/share_data/jixinya/VoxCeleb1_Cut/right.npy')
      #  self.videos = os.listdir(self.landmark_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling

        if os.path.exists(os.path.join(self.pose_dir, 'train_fo')):
            assert os.path.exists(os.path.join(self.pose_dir, 'test_fo'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(self.image_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = np.load('/mnt/lustre/share_data/jixinya/VoxCeleb1_Cut/right.npy')# get_list(self.pose_dir, 'train_fo')
      
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

            audio_path = os.path.join(self.audio_dir, name+'.npy')
            pose_path = os.path.join(self.pose_dir,name+'.npy')
            path = os.path.join(self.image_dir, name)

        video_name = os.path.basename(path)

        if  os.path.isdir(path):
     #   if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
            mfcc = np.load(audio_path)
            pose = np.load(pose_path)

          #  print(audio_path,pose_path,len(mfcc))

            try:
                len(mfcc) > 16
            except:
                print('wrongmfcc len:',audio_path)
            if 16 < len(mfcc) < 24 :
                r = 0
            else:

                r = random.choice([x for x in range(3, len(mfcc)-20)])

            mfccs = []
            poses = []
            video_array = []
            for ind in range(1, 17):
                t_mfcc = mfcc[r+ind][:, 1:]
                mfccs.append(t_mfcc)
                t_pose = pose[r+ind,:-1]
                poses.append(t_pose)
                image = img_as_float32(io.imread(os.path.join(path, str(r + ind)+'.png')))
                video_array.append(image)
            mfccs = np.array(mfccs)
            poses = np.array(poses)
            video_array = np.array(video_array)

            example_image = img_as_float32(io.imread(os.path.join(path, str(r)+'.png')))


        else:
            print('Wrong, data path not an existing file.')

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}

        driving = np.array(video_array, dtype='float32')

        spatial_size = np.array(driving.shape[1:3][::-1])[np.newaxis]
        driving_pose = np.array(poses, dtype='float32')
        example_image = np.array(example_image, dtype='float32')
        out['example_image'] = example_image.transpose((2, 0, 1))
        out['driving_pose'] = driving_pose
        out['driving'] = driving.transpose((0, 3, 1, 2))

        out['driving_audio'] = np.array(mfccs, dtype='float32')
    #    out['name'] = video_name

        return out

class MeadDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, augmentation_params=None):
        self.root_dir = root_dir

        self.audio_dir = os.path.join(root_dir,'MEAD_MFCC')
        self.image_dir = os.path.join(root_dir,'MEAD_fomm_crop')

        self.pose_dir = os.path.join(root_dir,'MEAD_fomm_pose_crop')

        self.videos = np.load('/mnt/lustre/share_data/jixinya/MEAD/MEAD_fomm_audio_less_crop.npy')
        self.dict = np.load('/mnt/lustre/share_data/jixinya/MEAD/MEAD_fomm_neu_dic_crop.npy',allow_pickle=True).item()
       # self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)

        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
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
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.image_dir, name)

            video_name = os.path.basename(path)
            id_name = path.split('/')[-2]
            neu_list = self.dict[id_name]
            neu_path = os.path.join(self.image_dir, np.random.choice(neu_list))

            audio_path = os.path.join(self.audio_dir, name+'.npy')
            pose_path = os.path.join(self.pose_dir,name+'.npy')


        if self.is_train and os.path.isdir(path):

            mfcc = np.load(audio_path)
            pose_raw = np.load(pose_path)
            one_euro_filter = OneEuroFilter(mincutoff=0.01, beta=0.7, dcutoff=1.0, freq=100)
            pose = np.zeros((len(pose_raw),7))

            for j in range(len(pose_raw)):
                pose[j]=one_euro_filter.process(pose_raw[j])
          #  print(audio_path,pose_path,len(mfcc))

            neu_frames = os.listdir(neu_path)
            num_neu_frames = len(neu_frames)
            frame_idx = np.random.choice(num_neu_frames)
            example_image = img_as_float32(io.imread(os.path.join(neu_path, neu_frames[frame_idx])))
            try:
                len(mfcc) > 16
            except:
                print('wrongmfcc len:',audio_path)
            if 16 < len(mfcc) < 24 :
                r = 0
            else:

                r = random.choice([x for x in range(3, len(mfcc)-20)])

            mfccs = []
            poses = []
            video_array = []
            for ind in range(1, 17):
                t_mfcc = mfcc[r+ind][:, 1:]
                mfccs.append(t_mfcc)
                t_pose = pose[r+ind,:-1]
                poses.append(t_pose)
                image = img_as_float32(io.imread(os.path.join(path, str(r + ind)+'.png')))
                video_array.append(image)
            mfccs = np.array(mfccs)
            poses = np.array(poses)
            video_array = np.array(video_array)

        else:
            print('Wrong, data path not an existing file.')

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
      
            driving = np.array(video_array, dtype='float32')
            driving_pose = np.array(poses, dtype='float32')
            example_image = np.array(example_image, dtype='float32')


            out['example_image'] = example_image.transpose((2, 0, 1))
            out['driving_pose'] = driving_pose
            out['driving'] = driving.transpose((0, 3, 1, 2))
            out['driving_audio'] = np.array(mfccs, dtype='float32')

      #  out['name'] = id_name+'/'+video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
    #    self.dataset2 = dataset2
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
     #   if idx % 5 == 0:
     #       return self.dataset2[idx % self.dataset2.__len__()]#% self.dataset.__len__()
     #   else:
     #       return self.dataset[idx % self.dataset.__len__()]
        return self.dataset[idx % self.dataset.__len__()]

class TestsetRepeater(Dataset):
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
