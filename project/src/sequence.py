# -*- coding: utf-8 -*-
'''
@author     wzh
@file       sequence.py
@date       2023.12.13
@brief      training process for sequential networks
'''
# --------------------------------- preamble --------------------------------- #
import os
import sys
import yaml
import torch
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from data.dataset import NuScenesDataset
from network.network import SingleFrameSemanticSegamentation
from network.network import SequentialSemanticSegamentation
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from criterion.metric import WeightedCrossEntroy
from criterion.metric import EvaluationMetrics
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    assert os.path.exists(sys.argv[1]), f'couldn\'t find argument file {sys.argv[1]}'
    with open(sys.argv[1]) as stream:
        args = yaml.load(stream, Loader=yaml.Loader)
    # ---------------------------------------------------------------------------- #
    #                                   argument                                   #
    # ---------------------------------------------------------------------------- #
    from warnings import filterwarnings
    filterwarnings('ignore')
    def setup_seed(seed):
        import random
        import numpy as np
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(args['random_seed'])
    # ---------------------------------------------------------------------------- #
    #                         random seed & warning filter                         #
    # ---------------------------------------------------------------------------- #
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]  # ! ImageNet Param
    transform = Compose([ToTensor(), Normalize(mean, std)])
    # ---------------------------------------------------------------------------- #
    dataset = dict()
    loader  = dict()
    for key in ['train', 'val']:
        dataset[key] = NuScenesDataset(version    = args[key]['version'], 
                                       dataroot   = args[key]['dataroot'],
                                       labelroot  = args['labelroot'],
                                       resolution = args['resolution'],
                                       voxelsize  = args['voxelsize'],
                                       transform  = transform,
                                       batch_nbr  = args[key]['batch_nbr'],
                                       sequence   = True,
                                       base_len   = args['base_len'],
                                       rand_mask  = args['rand_mask'])
        loader[key]  = DataLoader(dataset[key], batch_size=1, pin_memory=True)
        print(f'\033[32m[INFO] {key} dataset has been loaded over\033[0m')
    # ---------------------------------------------------------------------------- #
    #                             dataset & dataloader                             #
    # ---------------------------------------------------------------------------- #
    model = SingleFrameSemanticSegamentation(resolution       = args['resolution'], 
                                             voxelsize        = args['voxelsize'],
                                             img_channels     = args['img_channels'],
                                             pcd_channels     = args['pcd_channels'],
                                             builtin_channels = args['builtin_channels'],
                                             output_channels  = args['output_channels'],
                                             pretrained       = args['pretrained'],
                                             classifier_head  = False)
    if '--load-single' in sys.argv:
        model.load_state_dict(torch.load(os.path.join(args['modelroot'], 'single.pth')))
    seq_model = SequentialSemanticSegamentation(model)
    if '--load-sequence' in sys.argv:
        seq_model.load_state_dict(torch.load(os.path.join(args['modelroot'], 'sequence.pth')))
    # ---------------------------------------------------------------------------- #
    #                                    network                                   #
    # ---------------------------------------------------------------------------- #
    loss_func  = WeightedCrossEntroy(args['weights'] if 'weights' in args else None)
    evaluation = EvaluationMetrics(args['resolution'][0] + 1)
    optimizer  = SGD(model.parameters(), args['lr'], args['momentum'])
    scheduler  = ExponentialLR(optimizer, args['decay'])
    # ---------------------------------------------------------------------------- #
    #                          loss & eval & optim & sched                         #
    # ---------------------------------------------------------------------------- #
    logname = datetime.datetime.now().strftime('sequence--%y-%m-%d--%H-%M-%S')
    logpath = os.path.join(args['logdir'], logname)
    os.mkdir(logpath)
    writer  = SummaryWriter(log_dir=logpath)
    print(f'\033[32m[INFO] tensorboard environment {logpath} has been created\033[0m')
    # ---------------------------------------------------------------------------- #
    #                          tensorboard summary writer                          #
    # ---------------------------------------------------------------------------- #
    def train_on_epoch(epoch):
        mIoU_sum = 0.
        mAcc_sum = 0.
        loss_sum = 0.
        sample_nbr = 0
        iterator = tqdm(loader['train'], desc='train', leave=False)
        model.train()
        for sensor_data, dense_label, pose_compen in iterator:
            optimizer.zero_grad()
            rotat, trans = pose_compen[0]
            past = (rotat.squeeze(0), trans.squeeze(0))
            loss = 0.
            hidden_state = torch.zeros(seq_model.hidden_shape, dtype=torch.float32)
            for data, label, current in zip(sensor_data, dense_label, pose_compen):
                image, point, intri, rotat, trans = data
                image = image.squeeze(0)
                point = point.squeeze(0)
                intri = intri.squeeze(0)
                rotat = rotat.squeeze(0)
                trans = trans.squeeze(0)
                label = label.squeeze(0)
                data = (image, point, intri, rotat, trans)
                rotat, trans = current
                current = (rotat.squeeze(0), trans.squeeze(0))
                grids, hidden_state = seq_model(data, current, past, hidden_state)
                past = current
                loss = loss + loss_func(grids, label)
                evaluation.update(grids, label)
                mIoU_sum = mIoU_sum + evaluation.MeanIntersectionOverUnion()
                mAcc_sum = mAcc_sum + evaluation.MeanAccuracy()
                sample_nbr = sample_nbr + 1
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.item()
        writer.add_scalar('train/loss', loss_sum / sample_nbr, epoch)
        writer.add_scalar('train/mIoU', mIoU_sum / sample_nbr, epoch)
        writer.add_scalar('train/mAcc', mAcc_sum / sample_nbr, epoch)
    # ---------------------------------------------------------------------------- #
    def val_on_epoch(epoch):
        mIoU_sum = 0.
        mAcc_sum = 0.
        loss_sum = 0.
        sample_nbr = 0
        iterator = tqdm(loader['val'], desc='val', leave=False)
        model.eval()
        with torch.no_grad():
            for sensor_data, dense_label, pose_compen in iterator:
                optimizer.zero_grad()
                rotat, trans = pose_compen[0]
                past = (rotat.squeeze(0), trans.squeeze(0))
                loss = 0.
                hidden_state = torch.zeros(seq_model.hidden_shape, dtype=torch.float32)
                for data, label, current in zip(sensor_data, dense_label, pose_compen):
                    image, point, intri, rotat, trans = data
                    image = image.squeeze(0)
                    point = point.squeeze(0)
                    intri = intri.squeeze(0)
                    rotat = rotat.squeeze(0)
                    trans = trans.squeeze(0)
                    label = label.squeeze(0)
                    data = (image, point, intri, rotat, trans)
                    rotat, trans = current
                    current = (rotat.squeeze(0), trans.squeeze(0))
                    grids, hidden_state = seq_model(data, current, past, hidden_state)
                    past = current
                    loss = loss + loss_func(grids, label)
                    evaluation.update(grids, label)
                    mIoU_sum = mIoU_sum + evaluation.MeanIntersectionOverUnion()
                    mAcc_sum = mAcc_sum + evaluation.MeanAccuracy()
                    sample_nbr = sample_nbr + 1
                loss.backward()
                optimizer.step()
                loss_sum = loss_sum + loss.item()
        writer.add_scalar('val/loss', loss_sum / sample_nbr, epoch)
        writer.add_scalar('val/mIoU', mIoU_sum / sample_nbr, epoch)
        writer.add_scalar('val/mAcc', mAcc_sum / sample_nbr, epoch)
        return mIoU_sum / sample_nbr, mAcc_sum / sample_nbr
    # ---------------------------------------------------------------------------- #
    best_mIoU = -float('inf')
    tolerance = 0
    for epoch in tqdm(range(args['epoch'])):
        train_on_epoch(epoch)
        mIoU, mAcc = val_on_epoch(epoch)
        if mIoU > best_mIoU:
            tolerance = 0
            best_mIoU = mIoU
        else:
            tolerance = tolerance + 1
        if tolerance > args['patience']:
            break
        scheduler.step()
    print('\033[32m[INFO] model has been trained over\033[0m')
    writer.close()
    torch.save(seq_model.state_dict(), os.path.join(args['modelroot'], 'sequence.pth'))
    # ---------------------------------------------------------------------------- #
    #                              train & validation                              #
    # ---------------------------------------------------------------------------- #