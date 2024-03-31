# -*- coding: utf-8 -*-
'''
@author     wzh
@file       criterion/metric.py
@date       2023.12.13
@brief      loss function and evaluation metrics for networks
'''
# --------------------------------- preamble --------------------------------- #
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from sklearn.metrics import confusion_matrix

class WeightedCrossEntroy(Module):

    def __init__(self, weights: list):
        '''
        @brief      constructor
        @param      weights     每个类别的权重
        '''
        super().__init__()
        if weights is not None:
            self.cross_entropy = CrossEntropyLoss(torch.tensor(weights, dtype=torch.float32))
        else:
            self.cross_entropy = CrossEntropyLoss()

    def forward(self, y_pred: Tensor, y_true: Tensor):
        '''
        @brief      torch Module要求的forward方法
        @param      y_pred    [ C x D x H x W ]  预测的语义栅格（未经过softmax）
        @param      y_true    [ D x H x W ]      标注的语义栅格（非one-hot编码）
        @return     在通道维度加权的 cross entropy(y_pred, y_true)
        '''
        self.cross_entropy = self.cross_entropy.to(y_pred.device, non_blocking=True)
        return self.cross_entropy(y_pred.unsqueeze(0), y_true.unsqueeze(0))

class EvaluationMetrics(object):

    def __init__(self, class_nbr: int, epsilon: float=1e-5):
        '''
        @brief      constructor
        @param      class_nbr       类别总数
        @param      epsilon         防止数值计算溢出的极小值
        '''
        self.labels  = list(range(class_nbr))
        self.epsilon = epsilon

    def update(self, y_pred: Tensor, y_true: Tensor):
        '''
        @brief      更新混淆矩阵
        @param      y_pred    [ C x D x H x W ]  预测的语义栅格（未经过softmax）
        @param      y_true    [ D x H x W ]      标注的语义栅格（非one-hot编码）
        '''
        y_pred = y_pred.argmax(dim=0)  # ! [ D x H x W ]
        self.confusion = confusion_matrix(y_true.flatten().cpu(), 
                                          y_pred.flatten().cpu(), 
                                          labels=self.labels)

    def MeanIntersectionOverUnion(self):
        '''
        @brief      通过混淆矩阵计算 mIoU
        @return     mIoU 值
        '''
        mIoU = 0
        for i in self.labels:
            p_ii = self.confusion[i, i]
            p_ij = self.confusion[i, :].sum()
            p_ji = self.confusion[:, i].sum()
            mIoU = mIoU + p_ii / (p_ij + p_ji - p_ii + self.epsilon)
        return mIoU / len(self.labels)

    def MeanAccuracy(self):
        '''
        @brief      通过混淆矩阵计算 mAcc
        @return     mAcc 值
        '''
        mAcc = 0
        for i in self.labels:
            p_ii = self.confusion[i, i]
            p_ij = self.confusion[i, :].sum()
            mAcc = mAcc + p_ii / (p_ij + self.epsilon)
        return mAcc / len(self.labels)
            