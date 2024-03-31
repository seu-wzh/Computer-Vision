# -*- coding: utf-8 -*-
'''
@author     wzh
@file       data/dataset.py
@date       2023.11.22
@brief      torch dataset interface for nuscenes
'''
# --------------------------------- preamble --------------------------------- #
import os
import sys
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset
from torch.nn import Module
from torch.nn.functional import one_hot
from pyquaternion import Quaternion

class NuScenesDataset(Dataset):

    def __init__(self, 
                 version   : str, 
                 dataroot  : str, 
                 labelroot : str, 
                 resolution: list[int], 
                 voxelsize : float, 
                 transform : Module, 
                 batch_nbr : int, 
                 sequence  : bool, 
                 base_len  : int, 
                 rand_mask : list[int]) -> None:
        '''
        @brief      constructor
        @param      version     nuscenes版本
        @param      dataroot    nuscenes数据集根目录
        @param      labelroot   密集语义标签根目录
        @param      resolution  密集语义标签分辨率 [C x D x H x W]
        @param      voxelsize   密集语义标签体素间距
        @param      transform   图像变换
        @param      batch_nbr   单个epoch中包含的batch数量
        @param      sequence    是否为序列模型
        @param      base_len    每个训练样本序列的基础长度
        @param      rand_mask   每个训练样本序列被随机mask的数量的取值范围
        '''
        super().__init__()
        assert version in ['v1.0-mini', 'v1.0-trainval'], 'unexpected nuscenes version'
        self.nuscenes   = NuScenes(version, dataroot, verbose=False)
        self.labelroot  = labelroot
        self.resolution = resolution
        self.voxelsize  = voxelsize
        self.transform  = transform
        self.batch_nbr  = batch_nbr
        self.sequence   = sequence
        self.base_len   = base_len
        self.rand_mask  = rand_mask
        self.SENSOR_CHANNEL = [
            'CAM_FRONT_LEFT',  'CAM_FRONT',  'CAM_FRONT_RIGHT', 
            'CAM_BACK_LEFT',   'CAM_BACK',   'CAM_BACK_RIGHT',   'LIDAR_TOP'
        ]
        self.collect_sensor_tokens()

    def __len__(self):
        '''
        @brief      torch数据集要求的__len__方法
        '''
        return self.batch_nbr

    def getitem_sequence(self, index):
        '''
        @brief      序列模型的 __getitem__ 方法
        @return     传感器数据，密集语义标签，每帧的位姿补偿
            ├── * sensor_data (list<len=T>)
            ├   ├── + camera(RGB images):    [ N_{img} x 3 x H_{img} x W_{img} ]
            ├   ├── + lidar(3D point cloud): [ N_{pcd} x 3 ]
            ├   ├── + intrinsic matrix:      [ N_{img} x 3 x 3 ]
            ├   ├── + rotation matrix:       [ N_{img} x 3 x 3 ]
            ├   ├── + translation vector:    [ N_{img} x 3 ]
            ├
            ├── * dense label (list<len=T>): [ C x D x H x W ]
            ├
            ├── * pose compensation (list<len=T>)
            ├   ├── + rotation matrix:       [ 3 x 3 ]
            ├   ├── + translation vectorx:   [ 3 ]
        '''
        # ! p(global) = p(frame_{i}) @ rotation_{i}.T + translation_{i}
        # ! p(frame_{j}) = (p(global) - translation_{j}) @ rotation_{j}
        # ? 如果使用该方法在使用 DataLoader 进行包装时 batch_size 只能设置为 1
        scene_id    = index % len(self.samples)
        main_sample = random.randint(self.base_len - 1, len(self.samples[scene_id]) - 1)
        auxi_sample = list(range(main_sample - self.base_len + 1, main_sample))
        auxi_sample = random.sample(auxi_sample, random.choice(self.rand_mask))
        sample_seq  = sorted(auxi_sample)
        sample_seq.append(main_sample)
        # ------------------------------- sample select ------------------------------ #
        sensor_data  = []
        dense_label  = []
        pose_compen  = []
        for sample_id in sample_seq:
            sensor_data.append(self.get_sensor_data(scene_id, sample_id))
            dense_label.append(self.read_semantic_grid(scene_id, sample_id))
        # ------------------------------- data & label ------------------------------- #
            lidar       = self.samples[scene_id][sample_id][-1]
            lidar       = self.nuscenes.get('sample_data', lidar)
            calib       = self.nuscenes.get('calibrated_sensor', lidar['calibrated_sensor_token'])
            rotat_lidar = Quaternion(calib['rotation']).rotation_matrix
            rotat_lidar = torch.tensor(rotat_lidar, dtype=torch.float32)
            trans_lidar = torch.tensor(calib['translation'], dtype=torch.float32)
            ego_pose    = self.nuscenes.get('ego_pose', lidar['ego_pose_token'])
            rotat_ego   = Quaternion(ego_pose['rotation']).rotation_matrix
            rotat_ego   = torch.tensor(rotat_ego, dtype=torch.float32)
            trans_ego   = torch.tensor(ego_pose['translation'], dtype=torch.float32)
            rotation    = rotat_ego @ rotat_lidar
            translation = trans_lidar @ rotat_ego.T + trans_ego
            pose_compen.append((rotation, translation))
        # ----------------------------- pose compensation ---------------------------- #
        return sensor_data, dense_label, pose_compen

    def getitem_single(self, index):
        '''
        @brief      单帧模型的 __getitem__ 方法
        @return     传感器数据，密集语义标签，每帧的位姿补偿
            ├── * camera(RGB images):    [ N_{img} x 3 x H_{img} x W_{img} ]
            ├── * lidar(3D point cloud): [ N_{pcd} x 3 ]
            ├── * intrinsic matrix:      [ N_{img} x 3 x 3 ]
            ├── * rotation matrix:       [ N_{img} x 3 x 3 ]
            ├── * translation vector:    [ N_{img} x 3 ]
            ├----------------------------------------------------------------
            ├── * dense label:           [ C x D x H x W ]
        '''
        # ? 如果使用该方法在使用 DataLoader 进行包装时 batch_size 只能设置为 1
        scene_id    = index % len(self.samples)
        sample_id   = random.randint(0, len(self.samples[scene_id]) - 1)
        sensor_data = self.get_sensor_data(scene_id, sample_id)
        dense_label = self.read_semantic_grid(scene_id, sample_id)
        images, point_cloud, intrinsic, rotation, translation = sensor_data
        return images, point_cloud, intrinsic, rotation, translation, dense_label

    def __getitem__(self, index):
        '''
        @brief      torch数据集要求的__getitem__方法
        '''
        return self.getitem_sequence(index) if self.sequence else self.getitem_single(index)

    def collect_sensor_tokens(self) -> None:
        '''
        @brief      收集所有的标注帧传感器数据的tokens
        '''
        self.samples = []
        # ! [scene(0) -> [<camera, lidar>, ...], scene(1) -> [<camera, lidar>, ...], ...]
        for scene in self.nuscenes.scene:
            scene_samples = []
            token = scene['first_sample_token']
            while token != '':
                sample = self.nuscenes.get('sample', token)
                sensor_data = []
                for sensor in self.SENSOR_CHANNEL:
                    sensor_data.append(sample['data'][sensor])
                scene_samples.append(sensor_data)
                token = sample['next']
            self.samples.append(scene_samples)

    def read_point_cloud(self, token):
        '''
        @brief      读取激光雷达点云数据 (lidar coordinate)
        @param      token       雷达数据的token
        @return     Tensor(float32) [ N_{pcd} x 3 ]
        '''
        lidar       = self.nuscenes.get('sample_data', token)
        calib       = self.nuscenes.get('calibrated_sensor', lidar['calibrated_sensor_token'])
        filepath    = os.path.join(self.nuscenes.dataroot, lidar['filename'])
        point_cloud = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
        # ! nuscenes .pcd.bin 点云格式：（x, y, z, 强度，ring-index）
        point_cloud.setflags(write=True)
        point_cloud = torch.from_numpy(point_cloud)
        return point_cloud[:, :3]

    def get_sensor_data(self, scene_id: int, sample_id: int):
        '''
        @brief      获取特定scene下特定sample的传感器数据
        @param      scene_id     scene索引序号
        @param      sample_id    sample索引序号
        @return     Tuple<Tensor(float32)>
            ├── * camera(RGB images):    [ N_{img} x 3 x H_{img} x W_{img} ]
            ├── * lidar(3D point cloud): [ N_{pcd} x 3 ]       (lidar coordinate)
            ├── * intrinsic matrix:      [ N_{img} x 3 x 3 ]   (camera ->  pixel)
            ├── * rotation matrix:       [ N_{img} x 3 x 3 ]   (lidar  -> camera)
            ├── * translation vector:    [ N_{img} x 3 ]       (lidar  -> camera)
        '''
        # ! p(camera coordinate) = p(lidar coordinate) @ rotation.T + translation
        # ! p(pixel  coordinate) = normalize(p(camera coordinate) @ intrinsic.T)
        sensor_data = self.samples[scene_id][sample_id]
        # ------------------------- token list of each sensor ------------------------ #
        images      = []
        intrinsic   = []
        rotation    = []
        translation = []
        # ---------------------------------------------------------------------------- #
        lidar       = self.nuscenes.get('sample_data', sensor_data[-1])
        calib       = self.nuscenes.get('calibrated_sensor', lidar['calibrated_sensor_token'])
        rotat_lidar = torch.tensor(Quaternion(calib['rotation']).rotation_matrix, 
                                   dtype=torch.float32)
        trans_lidar = torch.tensor(calib['translation'], dtype=torch.float32)
        ego_pose    = self.nuscenes.get('ego_pose', lidar['ego_pose_token'])
        rotat_ego_l = torch.tensor(Quaternion(ego_pose['rotation']).rotation_matrix, 
                                   dtype=torch.float32)
        trans_ego_l = torch.tensor(ego_pose['translation'], dtype=torch.float32)
        # -------------------------- lidar calib & ego pose -------------------------- #
        for camera_token in sensor_data[:-1]:
            camera      = self.nuscenes.get('sample_data', camera_token)
            calib       = self.nuscenes.get('calibrated_sensor', camera['calibrated_sensor_token'])
            rotat_cam   = torch.tensor(Quaternion(calib['rotation']).rotation_matrix, 
                                       dtype=torch.float32)
            trans_cam   = torch.tensor(calib['translation'], dtype=torch.float32)
            ego_pose    = self.nuscenes.get('ego_pose', camera['ego_pose_token'])
            rotat_ego_c = torch.tensor(Quaternion(ego_pose['rotation']).rotation_matrix, 
                                       dtype=torch.float32)
            trans_ego_c = torch.tensor(ego_pose['translation'], dtype=torch.float32)
        # -------------------------- camera calib & ego pose ------------------------- #
            filepath = os.path.join(self.nuscenes.dataroot, camera['filename'])
            images.append(self.transform(Image.open(filepath, mode='r')))
            intrinsic.append(torch.tensor(calib['camera_intrinsic'], dtype=torch.float32))
            rotation.append(rotat_cam.T @ rotat_ego_c.T @ rotat_ego_l @ rotat_lidar)
            trans_total = (trans_lidar @ rotat_ego_l.T) + trans_ego_l
            trans_total = (trans_total - trans_ego_c  ) @ rotat_ego_c
            trans_total = (trans_total - trans_cam    ) @ rotat_cam
            translation.append(trans_total)
        # ---------------------------------------------------------------------------- #
        images      = torch.stack(images)
        point_cloud = self.read_point_cloud(sensor_data[-1])
        intrinsic   = torch.stack(intrinsic)
        rotation    = torch.stack(rotation)
        translation = torch.stack(translation)
        return images, point_cloud, intrinsic, rotation, translation

    def read_semantic_grid(self, scene_id: int, sample_id: int, one_hot_encode: bool = False):
        '''
        @brief      读取某个sample下的密集语义标签 (lidar coordinate)
        @param      scene_id        scene索引序号
        @param      sample_id       sample索引序号
        @param      one_hot_encode  是否使用one-hot编码（默认不使用）
        @return     Tensor(float32) [ D x H x W ] (non-one-hot encoding)
                    Tensor(float32) [ C x D x H x W ] (one-hot encoding)
        '''
        token     = self.samples[scene_id][sample_id][-1]
        lidar     = self.nuscenes.get('sample_data', token)
        filename  = os.path.basename(lidar['filename']) + '.npy'
        label     = torch.from_numpy(np.load(os.path.join(self.labelroot, filename)))
        occupancy = label[:, :3].type(torch.long)
        category  = label[:, -1].type(torch.uint8) + 1
        category  = category.type(torch.long)
        # ----------------------------- origin label data ---------------------------- #
        label = torch.zeros(self.resolution[1:], dtype=torch.long)
        label[occupancy[:, 2], occupancy[:, 0], occupancy[:, 1]] = category
        if one_hot_encode:
            label = one_hot(label, num_classes=(self.resolution[0] + 1))
            label = label.permute((3, 0, 1, 2))
        return label.type(torch.long)

    def render_point_cloud(self, scene_id: int, sample_id: int, path:str):
        '''
        @brief      在图像像素坐标系下可视化某个sample的点云数据
                    点云按照lidar coordinate下与原点的距离着色
        @param      scene_id        scene索引序号
        @param      sample_id       sample索引序号
        @param      path            可视化图像的保存路径
        '''
        sensor_data = self.get_sensor_data(scene_id, sample_id)
        images, point_cloud, intrinsic, rotation, translation = sensor_data
        # -------------------------------- sensor data ------------------------------- #
        fig, axs = plt.subplots(2, 3, figsize=(18, 9))
        h_img, w_img = images.shape[2], images.shape[3]
        for i in range(2):
            for j in range(3):
                index = i * 3 + j
                axs[i, j].set_title(self.SENSOR_CHANNEL[index])
                axs[i, j].imshow(np.transpose(images[index].numpy(), (1, 2, 0)))
                pcd  = torch.matmul(point_cloud, rotation[index].T) + translation[index]
                pcd  = torch.matmul(pcd, intrinsic[index].T)
        # ----------------------- vehicle coord -> pixel coord ----------------------- #
                # // mask = pcd[:, 2] > 0   弃用方法，会产生“脏东西”（异常点）
                mask = pcd[:, 2] > 0.1  # ? 设置近平面防止以上情况的出现
                pcd  = pcd[:, :2] / pcd[:, 2:]
                mask = (pcd[:, 0] >= 0) & (pcd[:, 0] <= w_img - 1) &\
                       (pcd[:, 1] >= 0) & (pcd[:, 1] <= h_img - 1) & mask
                pcd  = pcd[mask].numpy()
        # ------------------------------- visible mask ------------------------------- #
                color = torch.linalg.norm(point_cloud[mask], axis=-1)
                color = (color - color.min()) / (color.max() - color.min() + 1e-5)
                color = cm.get_cmap('viridis')(color.numpy())
                axs[i, j].scatter(pcd[:, 0], pcd[:, 1], s=1, c=color)
        plt.savefig(path)

    def render_semantic_grid(self, scene_id: int, sample_id: int, path: str):
        '''
        @brief      可视化3D语义网格（占据的体素点根据其类别着色）
        @param      scene_id        scene索引序号
        @param      sample_id       sample索引序号
        @param      path            可视化图像的保存路径
        '''
        semantic_grid = self.read_semantic_grid(scene_id, sample_id, False)
        semantic_grid = semantic_grid.numpy().transpose((1, 2, 0))
        facecolors = semantic_grid / self.resolution[0]
        edgecolors = np.clip(2 * facecolors - 0.5, 0, 1)
        facecolors = cm.get_cmap('tab20')(facecolors)
        edgecolors = cm.get_cmap('tab20')(edgecolors)
        fig = plt.figure(figsize=(12, 12))
        axs = plt.subplot(1, 1, 1, projection='3d')
        axs.tick_params(axis='x', labelsize=10)
        axs.tick_params(axis='y', labelsize=10)
        axs.tick_params(axis='z', labelsize=10)
        ticks = [np.linspace(0, self.resolution[i], n + 1) 
                 for i, n in zip([2, 3, 1], [8, 8, 2])]
        axs.set_xticks(ticks[0], (ticks[0] - self.resolution[2] / 2) * self.voxelsize)
        axs.set_yticks(ticks[1], (ticks[1] - self.resolution[3] / 2) * self.voxelsize)
        axs.set_zticks(ticks[2], ticks[2] * self.voxelsize)
        axs.set_xlabel('X(m)', fontsize=17, labelpad=17)
        axs.set_ylabel('Y(m)', fontsize=17, labelpad=17)
        axs.set_zlabel('Z(m)', fontsize=17, labelpad=15)
        axs.set_title('Semantic Grids')
        axs.set_box_aspect([200, 200, 16])
        axs.voxels(semantic_grid, facecolors=facecolors, edgecolors=edgecolors)
        plt.savefig(path)

    def render_semantic_grid_withfig(self, grids, **kwargs):
        '''
        @brief      可视化3D语义网格并返回 figure 对象（占据的体素点根据其类别着色）
        @param      grids        语义网格
        @return     figure 对象
        '''
        grids = grids.cpu().numpy().transpose((1, 2, 0))
        facecolors = grids / self.resolution[0]
        edgecolors = np.clip(2 * facecolors - 0.5, 0, 1)
        facecolors = cm.get_cmap('tab20')(facecolors)
        edgecolors = cm.get_cmap('tab20')(edgecolors)
        fig = plt.figure(figsize=(12, 12))
        axs = plt.subplot(1, 1, 1, projection='3d')
        axs.tick_params(axis='x', labelsize=10)
        axs.tick_params(axis='y', labelsize=10)
        axs.tick_params(axis='z', labelsize=10)
        ticks = [np.linspace(0, self.resolution[i], n + 1) 
                 for i, n in zip([2, 3, 1], [8, 8, 2])]
        axs.set_xticks(ticks[0], (ticks[0] - self.resolution[2] / 2) * self.voxelsize)
        axs.set_yticks(ticks[1], (ticks[1] - self.resolution[3] / 2) * self.voxelsize)
        axs.set_zticks(ticks[2], ticks[2] * self.voxelsize)
        axs.set_xlabel('X(m)', fontsize=17, labelpad=17)
        axs.set_ylabel('Y(m)', fontsize=17, labelpad=17)
        axs.set_zlabel('Z(m)', fontsize=17, labelpad=15)
        axs.set_title('Semantic Grids')
        axs.set_box_aspect([200, 200, 16])
        axs.voxels(grids, facecolors=facecolors, edgecolors=edgecolors)
        if 'path' in kwargs:
            plt.savefig(kwargs['path'])
        return fig

if __name__ == '__main__':
    assert os.path.exists(sys.argv[1]), f'couldn\'t find argument file {sys.argv[1]}'

    from warnings import filterwarnings
    filterwarnings('ignore')
    config = {
        "font.family":'serif',
        "mathtext.fontset":'stix'
    }
    plt.rcParams.update(config)
    plt.style.use('seaborn-paper')

    with open(sys.argv[1]) as stream:
        args = yaml.load(stream, Loader=yaml.Loader)

    from torchvision.transforms import ToTensor
    from torchvision.transforms import Normalize
    from torchvision.transforms import Compose
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]  # ! ImageNet Param
    transform = Compose([ToTensor(), Normalize(mean, std)])
    dataset = NuScenesDataset(version    = args['val']['version'], 
                              dataroot   = args['val']['dataroot'],
                              labelroot  = args['labelroot'],
                              resolution = args['resolution'],
                              voxelsize  = args['voxelsize'],
                              transform  = transform,
                              batch_nbr  = args['val']['batch_nbr'],
                              sequence   = False,
                              base_len   = args['base_len'],
                              rand_mask  = args['rand_mask'])
    import open3d as o3d
    point_cloud = dataset.read_point_cloud(dataset.samples[0][0][-1])
    D, H, W  = dataset.resolution[1], dataset.resolution[2], dataset.resolution[3]
    bound = (torch.tensor([D, H, W], dtype=torch.float32) - 1) / 2 * dataset.voxelsize
    bound = bound.to(point_cloud.device, non_blocking=True)
    bound = (point_cloud < bound) & (point_cloud > -bound)
    point_cloud = point_cloud[bound.all(dim=-1)]
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
    geometry = o3d.geometry.VoxelGrid.create_from_point_cloud(geometry, dataset.voxelsize)
    geometry = np.stack([v.grid_index for v in geometry.get_voxels()])
    grid = np.zeros(dataset.resolution[1:], dtype=np.float32)
    grid[geometry[:, 0], geometry[:, 1], geometry[:, 2]] = 1
    dataset.render_semantic_grid_withfig(torch.from_numpy(grid), path='voxelization.pdf')
    # from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    # for sensor_data in loader:
    #     for data in sensor_data:
    #         print(data.shape)
    #     break

    # dataset.render_point_cloud(0, 0, 'docs/imgs/demo.pdf')
    # dataset.read_semantic_grid(0, 0)
    # dataset.render_semantic_grid(0, 0, 'demo.pdf')
    # pcd = dataset.read_point_cloud(dataset.samples[0][0][-1])
    # print(pcd.min(axis=0), pcd.max(axis=0))