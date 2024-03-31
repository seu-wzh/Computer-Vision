# -*- coding: utf-8 -*-
'''
@author     wzh
@file       network/module.py
@date       2023.12.13
@brief      complete network for 3D segamentation with multi-frame & modal data
'''
# --------------------------------- preamble --------------------------------- #
import sys
sys.path.append('src/network/')

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Conv3d
from torch.nn import ReLU
from module import ImageFeatureExtractor
from module import InversePerspectiveProjection
from module import PointCloudFeatureExtractor
from module import AdaptiveMultiModalFusion
from module import InterFramePoseCalibrater

class SingleFrameSemanticSegamentation(Module):

    def __init__(self, 
                 resolution      : list,
                 voxelsize       : float,
                 img_channels    : int,
                 pcd_channels    : int,
                 builtin_channels: list,
                 output_channels : int,
                 pretrained      : bool,
                 down_sample     : float, 
                 classifier_head : bool,
                 use_cuda        : bool,
                 cuda_devices    : dict[str:str]):
        '''
        @brief      constructor
        @param      resolution          特征栅格分辨率（含通道） -> [ C x D x H x W ]
                                        （此处的通道数 C 即为分类的类别总数）
        @param      voxelsize           语义栅格体素尺寸
        @param      img_channels        图像特征栅格通道数
        @param      pcd_channels        点云特征栅格通道数
        @param      builtin_channels    点云特征栅格提取器中3D特征金字塔的中间特征通道数
        @param      output_channels     融合输出的特征栅格通道数
        @param      pretrained          是否加载ResNet50预训练参数
        @param      down_sample         在训练时对原图进行降采样的倍数
        @param      classifier_head     模型推理时是否需要分类器头部
        @param      use_cuda            是否使用GPU加速
        @param      cuda_devices        模型 -> 设备映射
        '''
        super().__init__()
        self.resolution      = resolution
        self.voxelsize       = voxelsize
        self.img_extractor   = ImageFeatureExtractor(4, img_channels, down_sample, pretrained)
        self.inv_projection  = InversePerspectiveProjection()
        self.pcd_extractor   = PointCloudFeatureExtractor(self.resolution[1:], 
                                                          builtin_channels, 
                                                          pcd_channels)
        self.adaptive_fusion = AdaptiveMultiModalFusion(img_channels, 
                                                        pcd_channels, 
                                                        output_channels)
        self.classifer = Conv3d(output_channels, self.resolution[0] + 1, 1)
        if use_cuda and torch.cuda.is_available():
            self.cuda_devices    = cuda_devices
            self.img_extractor   = self.img_extractor.to(cuda_devices['img'], non_blocking=True)
            self.pcd_extractor   = self.pcd_extractor.to(cuda_devices['pcd'], non_blocking=True)
            self.adaptive_fusion = self.adaptive_fusion.to(cuda_devices['ada'], non_blocking=True)
            self.classifer       = self.classifer.to(cuda_devices['cls'], non_blocking=True)
        self.output_channels = output_channels
        self.classifer_head = classifier_head

    def forward(self,
                images     : Tensor,
                point_cloud: Tensor,
                intrinsic  : Tensor,
                rotation   : Tensor,
                translation: Tensor): 
        '''
        @brief      torch Module要求的forward方法
        @param      camera(RGB images)     [ N_{img} x 3 x H_{img} x W_{img} ]
        @param      lidar(3D point cloud)  [ N_{pcd} x 3 ]
        @param      intrinsic matrix       [ N_{img} x 3 x 3 ]
        @param      rotation matrix        [ N_{img} x 3 x 3 ]  (lidar  -> camera)
        @param      translation vector     [ N_{img} x 3 ]      (lidar  -> camera)
        @return     [ C_{out} x D x H x W ] or [ C x D x H x W ]
        '''
        img_grid = self.img_extractor(images.to(self.cuda_devices['img'], non_blocking=True))
        img_grid = self.inv_projection(img_grid.to(self.cuda_devices['inv'], non_blocking=True), 
                                       intrinsic.to(self.cuda_devices['inv'], non_blocking=True), 
                                       rotation.to(self.cuda_devices['inv'], non_blocking=True), 
                                       translation.to(self.cuda_devices['inv'], non_blocking=True), 
                                       self.resolution[1:], 
                                       self.voxelsize)            # ! C_{img} x D x H x W
        pcd_grid = self.pcd_extractor(point_cloud.to(self.cuda_devices['pcd']), 
                                      self.resolution[1:], 
                                      self.voxelsize)             # ! C_{pcd} x D x H x W
        fuse_grid = self.adaptive_fusion(img_grid.to(self.cuda_devices['ada'], non_blocking=True), 
                                         pcd_grid.to(self.cuda_devices['ada'], non_blocking=True))
                                                                  # ! C_{out} x D x H x W
        if not self.classifer_head:
            return fuse_grid
        fuse_grid = fuse_grid.to(self.cuda_devices['cls'], non_blocking=True)
        return self.classifer(fuse_grid.unsqueeze(0)).squeeze(0)  # ! C       x D x H x W

# ---------------------------------------------------------------------------- #
#                      Single Frame Semantic Segamentation                     #
# ---------------------------------------------------------------------------- #

class SequentialSemanticSegamentation(Module):

    def __init__(self, backbone: SingleFrameSemanticSegamentation):
        '''
        @brief      constructor
        @param      backbone    单帧预测模型
        '''
        super().__init__()
        self.backbone = backbone
        self.calibrater = InterFramePoseCalibrater()
        self.conv_hidden = Conv3d(backbone.output_channels, 
                                  backbone.output_channels, 
                                  3,            padding=1)
        self.conv_input = Conv3d(backbone.output_channels, 
                                 backbone.output_channels, 
                                 3,            padding=1)
        self.hidden_shape = backbone.resolution
        self.hidden_shape[0] = backbone.output_channels

    def forward(self, sensor_data, current_pose, past_pose, hidden_state):
        '''
        @brief      torch Module要求的forward方法
        @param      sensor_data     多帧的传感器数据
                    ├── + camera(RGB images):    [ N_{img} x 3 x H_{img} x W_{img} ]
                    ├── + lidar(3D point cloud): [ N_{pcd} x 3 ]
                    ├── + intrinsic matrix:      [ N_{img} x 3 x 3 ]
                    ├── + rotation matrix:       [ N_{img} x 3 x 3 ]
                    ├── + translation vector:    [ N_{img} x 3 ]
        @param      current_pose     当前坐标系相对于全局坐标系的位姿补偿
                    ├── + rotation matrix:       [ 3 x 3 ]
                    ├── + translation vectorx:   [ 3 ]
        @param      past_pose        过去坐标系相对于全局坐标系的位姿补偿
                    ├── + rotation matrix:       [ 3 x 3 ]
                    ├── + translation vectorx:   [ 3 ]
        @param      hidden_state    上一时刻的隐状态    [ C_{out} x D x H x W ]
        @return     [ C x D x H x W ]           当前时刻输出
                    [ C_{out} x D x H x W ]     当前时刻隐状态
        '''
        hidden_state = self.calibrater(hidden_state, 
                                       self.backbone.voxelsize, 
                                       past_pose, 
                                       current_pose).unsqueeze(0)
        features     = self.backbone(*sensor_data).unsqueeze(0)
        hidden_state = self.conv_hidden(hidden_state) + self.conv_input(features)
        hidden_state = ReLU(inplace=True)(hidden_state)
        return self.backbone.classifer(hidden_state).squeeze(0), hidden_state.squeeze(0)

# ---------------------------------------------------------------------------- #
#                       Sequential Semantic Segamentation                      #
# ---------------------------------------------------------------------------- #