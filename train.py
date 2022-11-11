from tqdm import trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from logger import Logger
from modules.model import DiscriminatorFullModel, TrainPart1Model, TrainPart2Model
import itertools

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater,TestsetRepeater
import time
from tensorboardX import SummaryWriter

def train_part1(config, generator, discriminator, kp_detector, kp_detector_a,audio_feature, checkpoint, audio_checkpoint, log_dir, dataset, test_dataset, device_ids, name):
    train_params = config['train_params']

    optimizer_audio_feature = torch.optim.Adam(itertools.chain(audio_feature.parameters(),kp_detector_a.parameters()), lr=train_params['lr_audio_feature'], betas=(0.5, 0.999))


    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector, audio_feature,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector,
                                      None if train_params['lr_audio_feature'] == 0 else optimizer_audio_feature)
    if audio_checkpoint is not None:
        pretrain = torch.load(audio_checkpoint)
        kp_detector_a.load_state_dict(pretrain['kp_detector_a'])
        audio_feature.load_state_dict(pretrain['audio_feature'])
        optimizer_audio_feature.load_state_dict(pretrain['optimizer_audio_feature'])
        start_epoch = pretrain['epoch']

    else:
        start_epoch = 0

 
    scheduler_audio_feature = MultiStepLR(optimizer_audio_feature, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_audio_feature'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
        test_dataset = TestsetRepeater(test_dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)#6
    test_dataloader = DataLoader(test_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)#6
    num_steps_per_epoch = len(dataloader)
    num_steps_test_epoch = len(test_dataloader)
    generator_full = TrainPart1Model(kp_detector, kp_detector_a, audio_feature, generator, discriminator, train_params,device_ids)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
   
    if len(device_ids)>1:
        generator_full=torch.nn.DataParallel(generator_full)
        discriminator_full=torch.nn.DataParallel(discriminator_full)

    if torch.cuda.is_available():
        if len(device_ids) == 1:
            generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
            discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
        elif len(device_ids)>1:
            generator_full = generator_full.to(device_ids[0])
            discriminator_full = discriminator_full.to(device_ids[0])

    step = 0
    t0 = time.time()

    writer=SummaryWriter(comment=name)
    train_itr=0
    test_itr=0
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):

            for x in dataloader:

                losses_generator, generated = generator_full(x)
        
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                writer.add_scalar('Train',loss,train_itr)

                writer.add_scalar('Train_value',loss_values[0],train_itr)
                writer.add_scalar('Train_heatmap',loss_values[1],train_itr)
                writer.add_scalar('Train_jacobian',loss_values[2],train_itr)

                train_itr+=1
                loss.backward()

              
                optimizer_audio_feature.step()
                optimizer_audio_feature.zero_grad()
                d = time.time()
         
                # if train_params['loss_weights']['generator_gan'] != 0:
                #     optimizer_discriminator.zero_grad()
                # else:
                #     losses_discriminator = {}

                # losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)
                e = time.time()
          
                step += 1
             
                if(step % 500 == 0):
                    
                    logger.log_epoch(epoch,step, {'audio_feature': audio_feature,
                                     'kp_detector_a':kp_detector_a,
                                     'optimizer_audio_feature': optimizer_audio_feature}, inp=x, out=generated)

            scheduler_audio_feature.step()


            for x in test_dataloader:
                with torch.no_grad():
                    losses_generator, generated = generator_full(x)

                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)

                    writer.add_scalar('Test',loss,test_itr)

                    writer.add_scalar('Test_value',loss_values[0],test_itr)
                    writer.add_scalar('Test_heatmap',loss_values[1],test_itr)
                    writer.add_scalar('Test_jacobian',loss_values[2],test_itr)

                    test_itr+=1


              
def train_part1_fine_tune(config, generator, discriminator, kp_detector, kp_detector_a,audio_feature, checkpoint, audio_checkpoint, log_dir, dataset, dataset2, test_dataset, device_ids, name):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_audio_feature = torch.optim.Adam(itertools.chain(audio_feature.parameters(),kp_detector_a.parameters()), lr=train_params['lr_audio_feature'], betas=(0.5, 0.999))
  #  optimizer_kp_detector_a = torch.optim.Adam(kp_detector_a.parameters(), lr=train_params['lr_audio_feature'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector, audio_feature,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector,
                                      None if train_params['lr_audio_feature'] == 0 else optimizer_audio_feature)
    if audio_checkpoint is not None:
        pretrain = torch.load(audio_checkpoint)
        kp_detector_a.load_state_dict(pretrain['kp_detector_a'])
        audio_feature.load_state_dict(pretrain['audio_feature'])
   #     optimizer_kp_detector_a.load_state_dict(pretrain['optimizer_kp_detector_a'])
        optimizer_audio_feature.load_state_dict(pretrain['optimizer_audio_feature'])
        start_epoch = pretrain['epoch']


    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_audio_feature = MultiStepLR(optimizer_audio_feature, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_audio_feature'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
        test_dataset = TestsetRepeater(test_dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)#6
    test_dataloader = DataLoader(test_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)#6
    num_steps_per_epoch = len(dataloader)
    num_steps_test_epoch = len(test_dataloader)
    generator_full = TrainFullModel(kp_detector, kp_detector_a, audio_feature, generator, discriminator, train_params,device_ids)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    print('End dataload ', file=open('log/MEAD_LRW_test_a.txt', 'a'))
    if len(device_ids)>1:
        generator_full=torch.nn.DataParallel(generator_full)
        discriminator_full=torch.nn.DataParallel(discriminator_full)

    if torch.cuda.is_available():
        if len(device_ids) == 1:
            generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
            discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
        elif len(device_ids)>1:
            generator_full = generator_full.to(device_ids[0])
            discriminator_full = discriminator_full.to(device_ids[0])

    step = 0
    t0 = time.time()

    writer=SummaryWriter(comment=name)
    train_itr=0
    test_itr=0
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
          
            for x in dataloader:
      
         
                losses_generator, generated = generator_full(x)
             
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                writer.add_scalar('Train',loss,train_itr)

                writer.add_scalar('Train_value',loss_values[0],train_itr)
                writer.add_scalar('Train_heatmap',loss_values[1],train_itr)
                writer.add_scalar('Train_jacobian',loss_values[2],train_itr)
                writer.add_scalar('Train_perceptual',loss_values[3],train_itr)


                train_itr+=1
                loss.backward()

          

                optimizer_audio_feature.step()
                optimizer_audio_feature.zero_grad()
        
                optimizer_generator.step()
                optimizer_generator.zero_grad()
            #    optimizer_kp_detector_a.step()
            #    optimizer_kp_detector_a.zero_grad()
            
                if train_params['loss_weights']['discriminator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
           #         losses_discriminator = discriminator_full(x, generated)
           #         loss_values = [val.mean() for val in losses_discriminator.values()]
           #         loss = sum(loss_values)

           #         loss.backward()
           #         optimizer_discriminator.step()
           #         optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)
         
                step += 1
             
                if(step % 500 == 0):
       
                    logger.log_epoch(epoch,step, {'audio_feature': audio_feature,
                                     'kp_detector_a':kp_detector_a,
                                     'generator': generator,
                                     'optimizer_generator':optimizer_generator,
                                     'optimizer_audio_feature': optimizer_audio_feature}, inp=x, out=generated)
               
            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_audio_feature.step()

           
            for x in test_dataloader:
                with torch.no_grad():
                    losses_generator, generated = generator_full(x)

                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)

                    writer.add_scalar('Test',loss,test_itr)

                    writer.add_scalar('Test_value',loss_values[0],test_itr)
                    writer.add_scalar('Test_heatmap',loss_values[1],test_itr)
                    writer.add_scalar('Test_jacobian',loss_values[2],test_itr)
                    writer.add_scalar('Test_perceptual',loss_values[3],test_itr)

                    test_itr+=1


def train_part2(config, generator, discriminator, kp_detector, emo_detector, kp_detector_a,audio_feature, checkpoint, audio_checkpoint, emo_checkpoint, log_dir, dataset, test_dataset, device_ids, exp_name):
    train_params = config['train_params']
  
    optimizer_emo_detector = torch.optim.Adam(emo_detector.parameters(), lr=train_params['lr_audio_feature'], betas=(0.5, 0.999))
 
    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector, audio_feature,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector,
                                      None if train_params['lr_audio_feature'] == 0 else optimizer_audio_feature)
    if emo_checkpoint is not None:
        pretrain = torch.load(emo_checkpoint)
        tgt_state = emo_detector.state_dict()
        strip = 'module.'
        if 'emo_detector' in pretrain:
            emo_detector.load_state_dict(pretrain['emo_detector'])
            optimizer_emo_detector.load_state_dict(pretrain['optimizer_emo_detector'])
            print('emo_detector in pretrain + load', file=open('log/'+exp_name+'.txt', 'a'))
        for name, param in pretrain.items():
            if isinstance(param, nn.Parameter):
                param = param.data
            if strip is not None and name.startswith(strip):
                name = name[len(strip):]
            if name not in tgt_state:
                continue
            tgt_state[name].copy_(param)
            print(name)
    if audio_checkpoint is not None:
        pretrain = torch.load(audio_checkpoint)
        kp_detector_a.load_state_dict(pretrain['kp_detector_a'])
        audio_feature.load_state_dict(pretrain['audio_feature'])
        optimizer_audio_feature.load_state_dict(pretrain['optimizer_audio_feature'])
        if 'emo_detector' in pretrain:
            emo_detector.load_state_dict(pretrain['emo_detector'])
            optimizer_emo_detector.load_state_dict(pretrain['optimizer_emo_detector'])
        start_epoch = pretrain['epoch']
   
    else:
        start_epoch = 0


    scheduler_emo_detector = MultiStepLR(optimizer_emo_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_audio_feature'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
        test_dataset = TestsetRepeater(test_dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)#6
    test_dataloader = DataLoader(test_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)#6
    num_steps_per_epoch = len(dataloader)
    num_steps_test_epoch = len(test_dataloader)
    generator_full = TrainPart2Model(kp_detector, emo_detector,kp_detector_a, audio_feature,generator, discriminator, train_params,device_ids)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if len(device_ids)>1:
        generator_full=torch.nn.DataParallel(generator_full)
        discriminator_full=torch.nn.DataParallel(discriminator_full)
        
    if torch.cuda.is_available():
        if len(device_ids) == 1:
            generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
            discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
        elif len(device_ids)>1:
            generator_full = generator_full.to(device_ids[0])
            discriminator_full = discriminator_full.to(device_ids[0])
    
    step = 0
    t0 = time.time()
    
    writer=SummaryWriter(comment=exp_name)
    train_itr=0
    test_itr=0
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
         
            for x in dataloader:
      
                losses_generator, generated = generator_full(x)
               
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                
                writer.add_scalar('Train',loss,train_itr)
                
                writer.add_scalar('Train_value',loss_values[0],train_itr)
            #    writer.add_scalar('Train_heatmap',loss_values[1],train_itr)
                writer.add_scalar('Train_jacobian',loss_values[1],train_itr)
                writer.add_scalar('Train_classify',loss_values[2],train_itr)
               
                
                
                train_itr+=1
                loss.backward()
                
  
                optimizer_emo_detector.step()
                optimizer_emo_detector.zero_grad()
           
           
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)
             
                step += 1
               
                if(step % 1000 == 0):
                    
                    logger.log_epoch(epoch,step, {'audio_feature': audio_feature,
                                     'kp_detector_a':kp_detector_a,
                                     'emo_detector':emo_detector,
                                     'optimizer_emo_detector': optimizer_emo_detector,
             #                        'optimizer_kp_detector_a':optimizer_kp_detector_a,
                                     'optimizer_audio_feature': optimizer_audio_feature}, inp=x, out=generated)
                
            scheduler_emo_detector.step()
            
        
            for x in test_dataloader:
                with torch.no_grad():
                    losses_generator, generated = generator_full(x)

                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)
                
                    writer.add_scalar('Test',loss,test_itr)
                   
                    writer.add_scalar('Test_value',loss_values[0],test_itr)
                #    writer.add_scalar('Test_heatmap',loss_values[1],test_itr)
                    writer.add_scalar('Test_jacobian',loss_values[1],test_itr)
                    writer.add_scalar('Test_classify',loss_values[2],test_itr)
                    
                
                    test_itr+=1
                
                   
            


