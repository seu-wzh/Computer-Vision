# -*- coding: utf-8 -*-
'''
@author     wzh
@file       single.py
@date       2023.12.13
@brief      training process for single frame networks
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
                                       sequence   = False,
                                       base_len   = args['base_len'],
                                       rand_mask  = args['rand_mask'])
        loader[key]  = DataLoader(dataset[key], 
                                  batch_size  = 1,
                                  num_workers = 2,
                                  pin_memory  = False)
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
                                             down_sample      = args['down_sample'],
                                             classifier_head  = True,
                                             use_cuda         = args['use_cuda'],
                                             cuda_devices     = args['cuda_devices'])
    if '--load' in sys.argv:
        model.load_state_dict(torch.load(os.path.join(args['modelroot'], 'single.pth')))
    # ---------------------------------------------------------------------------- #
    #                                    network                                   #
    # ---------------------------------------------------------------------------- #
    weights    = [args['weights'][category] for category in args['weights']]
    loss_func  = WeightedCrossEntroy(weights)
    evaluation = EvaluationMetrics(args['resolution'][0] + 1)
    optimizer  = SGD(model.parameters(), args['lr'], args['momentum'])
    scheduler  = ExponentialLR(optimizer, args['decay'])
    # ---------------------------------------------------------------------------- #
    #                          loss & eval & optim & sched                         #
    # ---------------------------------------------------------------------------- #
    logname = datetime.datetime.now().strftime('single--%y-%m-%d--%H-%M-%S')
    logpath = os.path.join(args['logdir'], logname)
    os.mkdir(logpath)
    writer  = SummaryWriter(log_dir=logpath)
    step = {'train': {'loss': 0, 'mIoU': 0, 'mAcc': 0}, 
            'val':   {'loss': 0, 'mIoU': 0, 'mAcc': 0}, 
            'y_pred': 0,                  'y_true': 0}
    print(f'\033[32m[INFO] tensorboard environment {logpath} has been created\033[0m')
    # ---------------------------------------------------------------------------- #
    #                          tensorboard summary writer                          #
    # ---------------------------------------------------------------------------- #
    def train_on_epoch():
        sample_nbr = 0
        iterator = tqdm(loader['train'], desc='train', leave=False)
        model.train()
        for image, point, intri, rotat, trans, label in iterator:
            image = image.squeeze(0)
            point = point.squeeze(0)
            intri = intri.squeeze(0)
            rotat = rotat.squeeze(0)
            trans = trans.squeeze(0)
            label = label.squeeze(0)
            optimizer.zero_grad()
            grids = model(image, point, intri, rotat, trans)
            label = label.to(args['cuda_devices']['cls'], non_blocking=True)
            loss  = loss_func(grids, label)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            evaluation.update(grids, label)
            mIoU = evaluation.MeanIntersectionOverUnion()
            mAcc = evaluation.MeanAccuracy()
            if sample_nbr % args['log_freq']['train'] == 0:
                iterator.set_description('mIoU = {:.3} | mAcc = {:.3}'.format(mIoU, mAcc))
                writer.add_scalar('train/loss', loss.item(), step['train']['loss'])
                writer.add_scalar('train/mIoU', mIoU, step['train']['mIoU'])
                writer.add_scalar('train/mAcc', mAcc, step['train']['mAcc'])
                step['train']['loss'] = step['train']['loss'] + 1
                step['train']['mIoU'] = step['train']['mIoU'] + 1
                step['train']['mAcc'] = step['train']['mAcc'] + 1
            sample_nbr = sample_nbr + 1
    # ---------------------------------------------------------------------------- #
    def val_on_epoch():
        mIoU_sum = 0.
        mAcc_sum = 0.
        loss_sum = 0.
        sample_nbr = 0
        iterator = tqdm(loader['val'], desc='val', leave=False)
        model.eval()
        with torch.no_grad():
            for image, point, intri, rotat, trans, label in iterator:
                image = image.squeeze(0)
                point = point.squeeze(0)
                intri = intri.squeeze(0)
                rotat = rotat.squeeze(0)
                trans = trans.squeeze(0)
                label = label.squeeze(0)
                grids = model(image, point, intri, rotat, trans)
                label = label.to(args['cuda_devices']['cls'], non_blocking=True)
                loss  = loss_func(grids, label)
                evaluation.update(grids, label)
                mIoU_sum = mIoU_sum + evaluation.MeanIntersectionOverUnion()
                mAcc_sum = mAcc_sum + evaluation.MeanAccuracy()
                loss_sum = loss_sum + loss.item()
                if sample_nbr % args['log_freq']['val'] == 0:
                    grids = grids.argmax(dim=0)
                    writer.add_figure('y_pred', 
                                      dataset['val'].render_semantic_grid_withfig(grids), 
                                      step['y_pred'])
                    writer.add_figure('y_true', 
                                      dataset['val'].render_semantic_grid_withfig(label), 
                                      step['y_true'])
                    step['y_pred'] = step['y_pred'] + 1
                    step['y_true'] = step['y_true'] + 1
                sample_nbr = sample_nbr + 1
        writer.add_scalar('val/loss', loss_sum / sample_nbr, step['val']['loss'])
        writer.add_scalar('val/mIoU', mIoU_sum / sample_nbr, step['val']['mIoU'])
        writer.add_scalar('val/mAcc', mAcc_sum / sample_nbr, step['val']['mAcc'])
        step['val']['loss'] = step['val']['loss'] + 1
        step['val']['mIoU'] = step['val']['mIoU'] + 1
        step['val']['mAcc'] = step['val']['mAcc'] + 1
        return mIoU_sum / sample_nbr, mAcc_sum / sample_nbr
    # ---------------------------------------------------------------------------- #
    best_mIoU = -float('inf')
    tolerance = 0
    for epoch in tqdm(range(args['epoch'])):
        train_on_epoch()
        mIoU, mAcc = val_on_epoch()
        if mIoU > best_mIoU:
            tolerance = 0
            best_mIoU = mIoU
        else:
            tolerance = tolerance + 1
        if tolerance > args['patience']:
            break
        scheduler.step()
    writer.close()
    torch.save(model.state_dict(), os.path.join(args['modelroot'], 'single.pth'))
    print('\033[32m[INFO] model has been trained over\033[0m')
    # ---------------------------------------------------------------------------- #
    #                              train & validation                              #
    # ---------------------------------------------------------------------------- #