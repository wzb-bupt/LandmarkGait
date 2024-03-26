import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..modules import SetBlockWrapper, BasicConv2d

import os
import time


class FeatureExtractor(nn.Module):
    '''
    input: [n, c, s, h, w]
    outpot: [n, c, s, h/4, w/4]
    '''
    def __init__(self, model_cfg):
        super(FeatureExtractor, self).__init__()
    
        in_channels = model_cfg['channels']  # in_channels = [32, 64, 128]

        self.model = nn.Sequential(BasicConv2d(1, in_channels[0], 7, 1, 3),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[0], in_channels[0], 5, 1, 2),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[0], in_channels[1], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[1], in_channels[1], 3, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[1], in_channels[2], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True))
        self.model = SetBlockWrapper(self.model)

    def forward(self, x):
        return self.model(x)


class SpatialSoftmax(torch.nn.Module):
    '''
    input: [n, c, s, h/4, w/4]
    feature_landmarks: [n, n_landmark, s, 2]
    '''
    def __init__(self, height, width, channel, lim=[-1., 1., -1., 1.], temperature=None):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(lim[0], lim[1], self.width),
            np.linspace(lim[2], lim[3], self.height))

        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        n, c, s, h, w = feature.size()

        feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x) * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y) * softmax_attention, dim=1, keepdim=True)
        
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_landmarks = expected_xy.view(n, self.channel, s, 2)

        return feature_landmarks


class LandmarkPredictor_1(nn.Module):
    '''
    input: [n, 1, s, h, w]
    return: [n, c, s, h, w]
    '''
    def __init__(self, model_cfg):
        super(LandmarkPredictor_1, self).__init__()

        in_channels = model_cfg['channels']  # in_channels = [32, 64, 128]

        self.model = nn.Sequential(BasicConv2d(1, in_channels[0], 7, 1, 3),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[0], in_channels[0], 5, 1, 2),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[0], in_channels[1], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[1], in_channels[1], 3, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[1], in_channels[2], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True))

        self.model = SetBlockWrapper(self.model)

    def forward(self, x):
        return self.model(x)


class LandmarkPredictor_2(nn.Module):
    '''
    input: [n, c, s, h/4, w/4]
    return: [n, num_landmark, s, h/4, w/4]
    '''
    def __init__(self, model_cfg, num_landmark, lim=[-1., 1., -1., 1.]):
        super(LandmarkPredictor_2, self).__init__()

        in_channels = model_cfg['channels']  # in_channels = [32, 64, 128]
        img_height = model_cfg['height']
        img_width = model_cfg['width']

        self.model_landmark = nn.Sequential(BasicConv2d(in_channels[2], num_landmark, 1, 1, 0))
        self.model_landmark = SetBlockWrapper(self.model_landmark)

        self.integrater = SpatialSoftmax(
            height=img_height//4, width=img_width//4, channel=num_landmark, lim=lim)

    def integrate(self, heatmap):
        return self.integrater(heatmap)

    def forward(self, x):
        heatmap = self.model_landmark(x)
        return self.integrate(heatmap)

class Refiner(nn.Module):
    '''
    input: [n, num_landmark, s, h/4, w/4]
    return: [n, 1, s, h, w]
    '''
    def __init__(self, model_cfg):
        super(Refiner, self).__init__()
        
        in_channels = model_cfg['channels']

        self.model = nn.Sequential(# BasicConv2d(in_channels[3], in_channels[3], 4, 2, 1),
                                    nn.ConvTranspose2d(in_channels[2], in_channels[2], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[2], in_channels[1], 3, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    # BasicConv2d(in_channels[2], in_channels[2], 4, 2, 1),
                                    nn.ConvTranspose2d(in_channels[1], in_channels[1], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[1], in_channels[0], 5, 1, 2),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[0], 1, 7, 1, 3))
        self.model = SetBlockWrapper(self.model)

    def forward(self, feat):
        return self.model(feat)

class landmarknet(nn.Module):
    def __init__(self, landmark_cfg):
        super(landmarknet, self).__init__()

        model_cfg = landmark_cfg

        self.inv_std = model_cfg['inv_std']  # 10.0
        self.height = model_cfg['height']    # 64
        self.width = model_cfg['width']      # 44
        self.num_landmark = model_cfg['num_landmark']  # n_landmark
        self.freeze = model_cfg['freeze_half']  # fix or not

        # visual feature extractor
        self.extract_feature = FeatureExtractor(model_cfg)

        # landmark predictor
        self.extract_landmark_1 = LandmarkPredictor_1(model_cfg)
        self.extract_landmark_2 = LandmarkPredictor_2(model_cfg, num_landmark=model_cfg['num_landmark'], lim=[-1., 1., -1., 1.])

        if model_cfg['freeze_half']:
            self.extract_feature.requires_grad_(False)
            self.extract_landmark_1.requires_grad_(False)
            self.extract_landmark_2.requires_grad_(False)

        # map the feature back to the image
        self.refine = Refiner(model_cfg)

    def landmark2heatmap(self, landmark, inv_std=10.):
        # landmark: N x n_landmark x 2
        # heatpmap: N x n_landmark x (H / 4) x (W / 4)
        # return: N x n_landmark x (H / 4) x (W / 4)
        height = self.height // 4
        width = self.width // 4

        mu_x, mu_y = landmark[:, :, :, :1].unsqueeze(-1), landmark[:, :, :, 1:].unsqueeze(-1)

        y = (torch.linspace(-1.0, 1.0, height).view(1, 1, 1, height, 1).to(mu_y.device)).detach() # H
        x = (torch.linspace(-1.0, 1.0, width).view(1, 1, 1, 1, width).to(mu_x.device)).detach() # W

        g_y = (y - mu_y)**2
        g_x = (x - mu_x)**2
        dist = (g_y + g_x) * inv_std**2

        hmap = torch.exp(-dist)

        return hmap

    def transport(self, src_feat, des_feat, src_hmap, des_hmap, des_feat_hmap=None):
        # src_feat: N x C × S x (H / 4) x (W / 4)
        # des_feat: N x C × S x (H / 4) x (W / 4)
        # src_hmap: N x n_landmark × S x (H / 4) x (W / 4)
        # des_hmap: N x n_landmark × S x (H / 4) x (W / 4)
        # des_feat_hmap = des_hmap * des_feat: N x C x (H / 4) * (W / 4)
        # mixed_feat: N x C × S x (H / 4) x (W / 4)
        src_hmap = torch.sum(src_hmap, 1, keepdim=True)
        des_hmap = torch.sum(des_hmap, 1, keepdim=True)
        src_digged = src_feat * (1. - src_hmap) * (1. - des_hmap)

        if des_feat_hmap is None:
            mixed_feat = src_digged + des_hmap * des_feat
        else:
            mixed_feat = src_digged + des_feat_hmap

        return mixed_feat
    
    def forward(self, sils):
        des = sils  # [n, c=1, s, h, w]
        src = torch.roll(sils, shifts=8, dims=2)  # [n, c=1, s, h, w]

        if self.freeze:
            with torch.no_grad():
                src_feat = self.extract_feature(src)  # [n, c, s, h/4, w/4]
                des_feat = self.extract_feature(des)  # [n, c, s, h/4, w/4]
                src_landmark = self.extract_landmark_1(src)  
                des_landmark = self.extract_landmark_1(des)  
                src_landmark = self.extract_landmark_2(src_landmark)  # [n, n_landmark, s, 2]
                des_landmark = self.extract_landmark_2(des_landmark)  # [n, n_landmark, s, 2]
        else:
            src_feat = self.extract_feature(src)  # [n, c, s, h/4, w/4]
            des_feat = self.extract_feature(des)  # [n, c, s, h/4, w/4]
            src_landmark = self.extract_landmark_1(src)  
            des_landmark = self.extract_landmark_1(des)  
            src_landmark = self.extract_landmark_2(src_landmark)  # [n, n_landmark, s, 2]
            des_landmark = self.extract_landmark_2(des_landmark)  # [n, n_landmark, s, 2]

        src_hmap = self.landmark2heatmap(src_landmark, self.inv_std)  # [n, n_landmark, s, h/4, w/4]
        des_hmap = self.landmark2heatmap(des_landmark, self.inv_std)  # [n, n_landmark, s, h/4, w/4]
        mixed_feat = self.transport(src_feat, des_feat, src_hmap, des_hmap)  # [n, c, s, h/4, w/4]
        des_pred = self.refine(mixed_feat)  # [n, c=1, s, h, w]

        return des, des_pred, des_hmap, src_hmap, des_feat, src_feat, des_landmark
