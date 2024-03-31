# -*- coding: utf-8 -*-
'''
@author     wzh
@file       network/module.py
@date       2023.12.9
@brief      basic modules for 3D segamentation with multi-frame & modal data
'''
# --------------------------------- preamble --------------------------------- #
import torch
import numpy as np
import open3d as o3d
from torch import Tensor
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import ModuleList
from torch.nn import Conv3d
from torch.nn import MaxPool3d
from torch.nn import BatchNorm3d
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn.functional import interpolate
from torch.nn.functional import grid_sample
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

class ImageFeatureExtractor(Module):

    def __init__(self, scale: int, output_channels: int, down_sample: float, pretrained: bool):
        '''
        @brief      constructor
        @param      scale               输出特征图的缩放倍数
        @param      output_channels     输出特征图的通道数
        @param      down_sample         在训练时对原图进行降采样的倍数
        @param      pretrained          是否使用ResNet50的预训练参数
        '''
        super().__init__()
        self.SCALE = [4, 8, 16, 32]
        assert scale in self.SCALE, 'unexpected scale coefficient'
        self.LAYER_MAP  = dict([(f'layer{i}', f'scale{s}') 
                                for i, s in enumerate(self.SCALE, 1)])
        self.OUTPUT_KEY = f'scale{scale}'
        resnet = resnet50(pretrained=pretrained)
        self.backbone = IntermediateLayerGetter(resnet, self.LAYER_MAP)
        self.RESNET_CHANNELS = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(self.RESNET_CHANNELS, output_channels)
        self.down_sample = down_sample

    def forward(self, images: Tensor):
        '''
        @brief      torch Module要求的forward方法
        @param      images      RGB图像 [ N_{img} x 3 x H_{img} x W_{img} ]
        @return     经过提取、多尺度融合的特征图（相较于原图具有缩放系数）
                    [ N_{img} x output_channels x (H_{img} / scale) x (W_{img} / scale) ]
        '''
        if self.training and self.down_sample > 1:
            images = interpolate(images, scale_factor=(1. / self.down_sample))
        return self.fpn(self.backbone(images))[self.OUTPUT_KEY]

# ---------------------------------------------------------------------------- #
#                            Image Feature Extractor                           #
# ---------------------------------------------------------------------------- #

class InversePerspectiveProjection(Module):

    def __init__(self):
        '''
        @brief      constructor
        '''
        super().__init__()

    def forward(self, 
                images     : Tensor,
                intrinsic  : Tensor,
                rotation   : Tensor,
                translation: Tensor,
                resolution : list,
                voxelsize  : float):
        '''
        @brief      torch Module要求的forward方法
        @param      images          [ N_{img} x C x H_{img} x W_{img} ]
        @param      intrinsic       [ N_{img} x 3 x 3 ]
        @param      rotation        [ N_{img} x 3 x 3 ]     (lidar  -> camera)
        @param      translation     [ N_{img} x 3 ]         (lidar  -> camera)
        @param      resolution      语义栅格分辨率（不含通道维度） -> [ D x H x W ]
        @param      voxelsize       语义栅格体素尺寸
        @return     在像素空间采样后的语义栅格  [ C x D x H x W ]
        '''
        D, H, W = resolution[0], resolution[1], resolution[2]
        N_img, C, H_img, W_img = images.shape
        x = torch.arange(H, dtype=torch.float32) - (H - 1) / 2.
        y = torch.arange(W, dtype=torch.float32) - (W - 1) / 2.
        z = torch.arange(D, dtype=torch.float32) - (D - 1) / 2.
        x, y, z = torch.meshgrid(x, y, z)
        grid = torch.stack([x, y, z], dim=-1).flatten(0, 2) * voxelsize  # ! (H x W x D) x 3
        grid = grid.to(images.device, non_blocking=True)
        grid  = grid @ rotation.transpose(1, 2) + translation.unsqueeze(1)
        # ! lidar -> camera ========>>>>> N_{img} x (H x W x D) x 3
        grid  = grid @ intrinsic.transpose(1, 2)
        mask  = (grid[..., 2] < 0.1).unsqueeze(1)  # ? 设置近平面 N_{img} x 1 x (H x W x D)
        grid  = (grid[..., :2] / grid[..., 2:]).flip(-1)  # (u, v) -> (v, u)
        scale = torch.tensor([H_img, W_img], dtype=torch.float32) - 1
        grid  = 2 * grid / scale.to(images.device, non_blocking=True) - 1  # normalize
        # ! camera -> pixel ========>>>>> N_{img} x (H x W x D) x 2
        grid  = grid_sample(images, grid.unsqueeze(2), padding_mode='zeros').squeeze(3)
        # ! N_{img} x C x H_{img} x W_{img}  <-- sample --  N_{img} x (H x W x D) x 1 x 2
        grid  = grid.masked_fill(mask, value=torch.tensor(0.))
        return grid.mean(0).view((C, D, H, W))  # ! C x D x H x W

# ---------------------------------------------------------------------------- #
#                        Inverse Perspective Projection                        #
# ---------------------------------------------------------------------------- #

class PointCloudFeatureExtractor(Module):

    def __init__(self, resolution: list, builtin_channels: list, output_channels: int):
        '''
        @brief      constructor
        @param      resolution          语义栅格分辨率（不含通道维度） -> [ D x H x W ]
        @param      builtin_channels    3D特征金字塔的中间特征通道数
        @param      output_channels     输出的几何特征通道数
        '''
        super().__init__()
        self.resolution = resolution
        self.layer_nbr  = len(builtin_channels)
        self.conv_block = ModuleList()
        self.conv1x1    = ModuleList()
        self.conv_alias = ModuleList()
        for layer in range(self.layer_nbr):
            input_channels = 1 if layer == 0 else builtin_channels[layer - 1]
            conv3x3 = Conv3d(input_channels, builtin_channels[layer], 3, padding=1)
            bn      = BatchNorm3d(builtin_channels[layer])
            relu    = ReLU(inplace=True)
            maxpool = MaxPool3d(2)
            self.conv_block.append(Sequential(conv3x3, bn, relu, maxpool))
            self.conv_alias.append(Conv3d(output_channels, output_channels, 3, padding=1))
        for layer in range(self.layer_nbr + 1):
            input_channels = 1 if layer == 0 else builtin_channels[layer - 1]
            self.conv1x1.append(Conv3d(input_channels, output_channels, 1))

    def forward(self, point_cloud: Tensor, resolution: list, voxelsize: float):
        '''
        @brief      torch Module要求的forward方法
        @param      point_cloud     [ N_{pcd} x 3 ]
        @param      resolution      语义栅格分辨率（不含通道维度） -> [ D x H x W ]
        @param      voxelsize       语义栅格体素尺寸
        @return     体素化之后的结构经过3D滤波得到的geometry-level的特征 [ C x D x H x W ]
        '''
        D, H, W  = resolution[0], resolution[1], resolution[2]
        bound = (torch.tensor([D, H, W], dtype=torch.float32) - 1) / 2 * voxelsize
        bound = bound.to(point_cloud.device, non_blocking=True)
        bound = (point_cloud < bound) & (point_cloud > -bound)
        point_cloud = point_cloud[bound.all(dim=-1)]
        geometry = o3d.geometry.PointCloud()
        geometry.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
        geometry = o3d.geometry.VoxelGrid.create_from_point_cloud(geometry, voxelsize)
        geometry = np.stack([v.grid_index for v in geometry.get_voxels()])
        geometry = torch.from_numpy(geometry).type(torch.long)  # ! [ N_{within_bounds} x 3 ]
        geometry = geometry.to(point_cloud.device, non_blocking=True)
        grid = torch.zeros(resolution, dtype=torch.float32)
        grid = grid.to(point_cloud.device, non_blocking=True)
        grid[geometry[:, 0], geometry[:, 1], geometry[:, 2]] = 1
        grid = grid.unsqueeze(0).unsqueeze(1)  # ! 1 x 1 x D x H x W
        # ------------------------------- voxelization ------------------------------- #
        features = []
        for layer in range(self.layer_nbr):
            features.append(grid)
            grid = self.conv_block[layer](grid)
        # ---------------------------- feature extraction ---------------------------- #
        grid = self.conv1x1[-1](grid)
        for layer in range(self.layer_nbr - 1, -1, -1):
            grid = interpolate(grid, scale_factor=2, mode='trilinear')
            grid = grid + self.conv1x1[layer](features[layer])
            grid = self.conv_alias[layer](grid)
        # ---------------------------- multi-scale fusion ---------------------------- #
        return grid.squeeze(0)

# ---------------------------------------------------------------------------- #
#                         Point Cloud Feature Extractor                        #
# ---------------------------------------------------------------------------- #

class AdaptiveMultiModalFusion(Module):

    def __init__(self, img_channels: int, pcd_channels: int, output_channels: int):
        '''
        @brief      constructor
        @param      img_channels        图像提取的特征栅格通道数
        @param      pcd_channels        点云提取的特征栅格通道数
        @param      output_channels     融合输出的特征栅格通道数
        '''
        super().__init__()
        self.conv_block_img = Sequential(
            Conv3d(img_channels, img_channels, 3, padding=1), 
            BatchNorm3d(img_channels),   ReLU(inplace=True))
        self.conv_block_pcd = Sequential(
            Conv3d(pcd_channels, pcd_channels, 3, padding=1), 
            BatchNorm3d(pcd_channels),   ReLU(inplace=True))
        fusion_channels  = img_channels + pcd_channels
        self.conv_fusion = Conv3d(fusion_channels, output_channels, 3, padding=1)
        self.conv1x1_img = Conv3d(img_channels, output_channels, 1)
        self.conv1x1_pcd = Conv3d(pcd_channels, output_channels, 1)

    def forward(self, img_grid: Tensor, pcd_grid: Tensor):
        '''
        @brief      torch Module要求的forward方法
        @param      img_grid        图像特征栅格  [ C_{img} x D x H x W ]
        @param      pcd_grid        点云特征栅格  [ C_{pcd} x D x H x W ]
        @return     融合后的特征栅格 [ C_{out} x D x H x W ]
        '''
        img_weights = self.conv_block_img(img_grid.unsqueeze(0))
        pcd_weights = self.conv_block_pcd(pcd_grid.unsqueeze(0))
        fuse_weights = torch.cat([img_weights, pcd_weights], dim=1)
        fuse_weights = Sigmoid()(self.conv_fusion(fuse_weights))  # ! -+
        img_weights = self.conv1x1_img(img_grid.unsqueeze(0))     # !  +-> 1 x C_{out} x D x H x W
        pcd_weights = self.conv1x1_pcd(pcd_grid.unsqueeze(0))     # ! -+
        fuse_weights = fuse_weights * img_weights + (1 - fuse_weights) * pcd_weights
        return fuse_weights.squeeze(0)

# ---------------------------------------------------------------------------- #
#                          Adaptive Multi Modal Fusion                         #
# ---------------------------------------------------------------------------- #

class InterFramePoseCalibrater(Module):

    def __init__(self):
        '''
        @brief      constructor
        '''
        super().__init__()

    def forward(self, past_grid: Tensor, voxelsize: float, past: tuple, current: tuple):
        '''
        @brief      torch Module要求的forward方法
        @param      past_grid       过去的密集特征栅格  [ C x D x H x W ]
        @param      voxelsize       体素间距
        @param      past, current   过去、当前的位姿补偿
                        ├── + rotation matrix:       [ 3 x 3 ]
                        ├── + translation vectorx:   [ 3 ]
        @return     当前坐标栅格经过坐标系变换后对过去特征栅格采样的特征栅格 [ C x D x H x W ]
        '''
        C, D, H, W = past_grid.shape
        x = torch.arange(H, dtype=torch.float32) - (H - 1) / 2.
        y = torch.arange(W, dtype=torch.float32) - (W - 1) / 2.
        z = torch.arange(D, dtype=torch.float32) - (D - 1) / 2.
        x, y, z = torch.meshgrid(x, y, z)
        current_grid = torch.stack([x, y, z], dim=-1)  # ! H x W x D x 3
        # --------------------------------- mesh grid -------------------------------- #
        current_grid = current_grid * voxelsize
        rotation, translation = current
        current_grid = current_grid @ rotation.T + translation
        rotation, translation = past
        current_grid = (current_grid - translation) @ rotation
        # --------------------------- coordinate transform --------------------------- #
        current_grid = current_grid[..., [2, 0, 1]] / voxelsize
        grid_scale   = torch.tensor([D, H, W], dtype=torch.float32)
        current_grid = (2 * current_grid) / (grid_scale - 1)  # scale into [-1, 1]
        current_grid = current_grid.permute((2, 0, 1, 3)).unsqueeze(0)
        current_grid = grid_sample(input = past_grid.unsqueeze(0), 
                                   grid  = current_grid, 
                                   padding_mode='zeros')
        # ! 1 x C x D x H x W  <-- sample --  1 x D x H x W x 3
        return current_grid.squeeze(0)

# ---------------------------------------------------------------------------- #
#                          Inter Frame Pose Calibrater                         #
# ---------------------------------------------------------------------------- #