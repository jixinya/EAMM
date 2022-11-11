import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import MeadDataset, AudioDataset, VoxDataset

from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector, Audio_Feature, KPDetector_a
from modules.util import AT_net,Emotion_k,get_logger
import torch

from train import train_part1, train_part1_fine_tune, train_part2
from reconstruction import reconstruction
from animate import animate

if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/train_part1.yaml", help="path to config")# required=True
    parser.add_argument("--mode", default="train_part1", choices=["train_part1", "train_part1_fine_tune", "train_part2"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default='124_52000.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--audio_checkpoint", default=None, help="path to audio_checkpoint to restore")
    parser.add_argument("--emo_checkpoint", default=None, help="path to audio_checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)  
    
    name = os.path.basename(opt.config).split('.')[0]
    if opt.checkpoint is not None:
   
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)
        
 #   logger = get_logger(os.path.join(log_dir, "log.txt"))  
    
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])

    if opt.verbose:
        print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
        
    
    
    if opt.verbose:
        print(discriminator)
   
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    kp_detector_a = KPDetector_a(**config['model_params']['kp_detector_params'],
                             **config['model_params']['audio_params'])

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
        kp_detector_a.to(opt.device_ids[0])

    audio_feature = AT_net()
    emo_feature = Emotion_k(block_expansion=32, num_channels=3, max_features=1024,
                 num_blocks=5, scale_factor=0.25, num_classes=8)
    
    if torch.cuda.is_available():
        audio_feature.to(opt.device_ids[0])
        emo_feature.to(opt.device_ids[0])

    if opt.verbose:
        print(kp_detector)
        print(kp_detector_a)
        print(audio_feature)
        print(emo_feature)
    
#    logger.info("Successfully load models.")
    
    if config['dataset_params']['name'] == 'Vox':
        dataset = VoxDataset(is_train=True, **config['dataset_params'])
        test_dataset = VoxDataset(is_train=False, **config['dataset_params'])
    elif config['dataset_params']['name'] == 'Lrw':
        dataset = AudioDataset(is_train=True, **config['dataset_params'])
        test_dataset = AudioDataset(is_train=False, **config['dataset_params'])
    elif config['dataset_params']['name'] == 'MEAD':
        dataset = MeadDataset(is_train=True, **config['dataset_params'])
        test_dataset = MeadDataset(is_train=False, **config['dataset_params'])


    

    if opt.mode == 'train_part1':
        print("Training part1...")
        train_part1(config, generator, discriminator, kp_detector, kp_detector_a,audio_feature, opt.checkpoint, opt.audio_checkpoint, log_dir, dataset, test_dataset,opt.device_ids, name)
    elif opt.mode == 'train_part1_fine_tune':
        print("Finetune part1...")
        train_part1_fine_tune(config, generator, discriminator, kp_detector, kp_detector_a,audio_feature, opt.checkpoint, opt.audio_checkpoint, log_dir, dataset, test_dataset,opt.device_ids, name)
    elif opt.mode == 'train_part2':
        print("Training part2...")
         train_part2(config, generator, discriminator, kp_detector, emo_feature,kp_detector_a,audio_feature, opt.checkpoint, opt.audio_checkpoint, opt.emo_checkpoint, log_dir, dataset,test_dataset,opt.device_ids, name)
