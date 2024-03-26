import torch
import torch.nn as nn
import numpy as np

from ..base_model import BaseModel
from kornia import morphology as morph

import torch.nn.functional as F
import torch.optim as optim
from utils import get_valid_args, get_attr_from, is_list_or_tuple
import os.path as osp

from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SetBlockWrapper, BasicConv2d, SeparateBNNecks


class Refiner(nn.Module):
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
                                    BasicConv2d(in_channels[0], 1, 7, 1, 3),
                                    nn.Sigmoid())
        self.model = SetBlockWrapper(self.model)

    def forward(self, feat):
        return self.model(feat)

class PMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PMBConv, self).__init__()
        self.PMBConv_3x3x3 = nn.Sequential(
                BasicConv3d(in_channels, out_channels, kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )
        self.PMBConv_1x3x3 = nn.Sequential(
            BasicConv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                        stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        self.PMBConv_1x1x1_A = nn.Sequential(
            BasicConv3d(in_channels, out_channels, kernel_size=(1, 1, 1),
                        stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )
        self.PMBConv_1x1x1_B = nn.Sequential(
            BasicConv3d(in_channels, out_channels, kernel_size=(1, 1, 1),
                        stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )
        self.PMBConv_3x3x3_B = nn.Sequential(
                BasicConv3d(out_channels, out_channels, kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )
        self.AvgPooling = nn.AvgPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        PMBConv_3x3x3 = self.PMBConv_3x3x3(x)
        # PMBConv_3x1x1 = self.PMBConv_3x1x1(x)
        PMBConv_1x3x3 = self.PMBConv_1x3x3(x)

        PMBConv_1x1x1_A = self.PMBConv_1x1x1_A(x)
        AvgPooling = self.AvgPooling(PMBConv_1x1x1_A)

        PMBConv_1x1x1_B = self.PMBConv_1x1x1_B(x)
        PMBConv_3x3x3_B = self.PMBConv_3x3x3_B(PMBConv_1x1x1_B)

        feat = PMBConv_3x3x3 + PMBConv_1x3x3 + AvgPooling + PMBConv_3x3x3_B
        return feat

class GeMHPP(nn.Module):
    def __init__(self, bin_num=[96], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)

class LandmarkGait_Recognition(BaseModel):
    def __init__(self, *args, **kargs):
        super(LandmarkGait_Recognition, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        # init
        self.num_landmark = model_cfg['landmarknet_cfg']['landmark_cfg']['num_landmark']
        self.landmark_lr = model_cfg['landmark_lr']
        self.freeze_half_bodypart = model_cfg['freeze_half_bodypart']
        self.shift_up = model_cfg['roll_up']
        self.shift_low = model_cfg['roll_low']

        # # ******************************************************** #
        # # **** LandmarkNet: LandmarkGait_Silh_to_Landmark ******** # 
        # # ******************************************************** #
        self.Backbone = self.get_backbone(model_cfg['landmarknet_cfg'])

        # # ******************************************************** #
        # # **** ParsingNet: LandmarkGait_Landmark_to_Parsing ****** # 
        # # ******************************************************** #
        self.refine = Refiner(model_cfg)
        self.kernel = torch.ones(
            (model_cfg['kernel_size'], model_cfg['kernel_size']))
        if model_cfg['freeze_half_bodypart']:
            self.refine.requires_grad_(False)

        # # ******************************************************** #
        # # **************** Appearance Feature ******************** # 
        # # ******************************************************** #
        in_c = model_cfg['channels_PMBConv']
        class_num = model_cfg['class_num']

        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        self.LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(
                3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )

        self.PMBConvA = PMBConv(in_c[0], in_c[1])
        self.MaxPool0 = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.PMBConvB = PMBConv(in_c[1], in_c[2])
        self.PMBConvC = PMBConv(in_c[2], in_c[2])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(96, in_c[2], in_c[2])
        
        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(in_c[2])
            self.Head1 = SeparateFCs(96, in_c[2], class_num)
            self.Bn_head = True

    def transport(self, src_feat, des_feat, src_hmap, des_hmap, des_hmap_part, des_feat_hmap=None):
        # src_feat: B × S x (nf * 4) x (H / 4) x (W / 4)
        # des_feat: B × S x (nf * 4) x (H / 4) x (W / 4)
        # src_hmap: B × S x n_kp x (H / 4) x (W / 4)
        # des_hmap: B × S x n_kp x (H / 4) x (W / 4)
        # des_feat_hmap = des_hmap * des_feat: B x (nf * 4) x (H / 4) * (W / 4)
        # mixed_feat: B × S x (nf * 4) x (H / 4) x (W / 4)
        src_hmap = torch.sum(src_hmap, 1, keepdim=True)
        des_hmap = torch.sum(des_hmap, 1, keepdim=True)
        src_digged = src_feat * (1. - src_hmap) * (1. - des_hmap)

        if des_feat_hmap is None:
            if des_hmap_part == None:
                mixed_feat = src_digged + des_hmap * des_feat
            else:
                des_hmap_part = torch.sum(des_hmap_part, 1, keepdim=True)
                mixed_feat = src_digged + des_hmap_part * des_feat
        else:
            mixed_feat = src_digged + des_feat_hmap

        return mixed_feat

    def finetune_parameters(self):
        fine_tune_params = list()
        others_params = list()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'Backbone' in name:
                fine_tune_params.append(p)
            else:
                others_params.append(p)
        return [{'params': fine_tune_params, 'lr': self.landmark_lr}, {'params': others_params}]

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(self.finetune_parameters(), **valid_arg)
        return optimizer

    def resume_ckpt(self, restore_hint):
        if is_list_or_tuple(restore_hint):
            for restore_hint_i in restore_hint:
                self.resume_ckpt(restore_hint_i)
            return
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def landmark_selection(self, landmarks, num_landmark):
        if num_landmark == 20:
            head_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[:1]
            up_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[1:9]
            low_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[9:]
        elif num_landmark == 25:
            head_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[:2]
            up_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[2:12]
            low_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[12:]
        elif num_landmark == 30:
            head_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[:2]
            up_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[2:16]
            low_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[16:]
        elif num_landmark == 35:
            head_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[:4]
            up_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[4:18]
            low_landmark = landmarks.sort(dim=0)[1][:,1].cpu().detach().tolist()[18:]
        else:
            raise ValueError(
                "The value of 'num_landmark' is not supported. You can define it yourself in 'landmark_selection' function.")
        return head_landmark, up_landmark, low_landmark

    def preprocess(self, sils):
        dilated_mask = (morph.dilation(sils, self.kernel.to(sils.device)).detach()) > 0.5  # Dilation
        return dilated_mask

    def PMBConv_Feature(self, sils):
        # img: B x 1 × S x H x W
        # ret: B x C × S x (H / 2) x (W / 2)
        outs = self.conv3d(sils)
        outs = self.LTA(outs)
        outs = self.PMBConvA(outs)
        outs = self.MaxPool0(outs)
        outs = self.PMBConvB(outs)
        outs = self.PMBConvC(outs)
        return outs

    def sils_roll(self, x, shifts):
        return torch.roll(x, shifts, 0)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts

        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        des_sils, des_pred_sils, des_hmap_sils, src_hmap_sils, des_feat_sils, src_feat_sils, des_landmark = self.Backbone(sils)
        
        # Parsing Part Reconstruction (Coarse)
        ## Landmark Grouping
        idx_batch = des_landmark.size(0)
        idx_frame = des_landmark.size(2)
        landmarks = des_landmark[np.random.randint(idx_batch), :, np.random.randint(idx_frame), ...]
        head_landmark, up_landmark, low_landmark = self.landmark_selection(landmarks, self.num_landmark)
        
        des_hmap_head = des_hmap_sils[:, head_landmark, :, :, :]
        des_hmap_up = des_hmap_sils[:, up_landmark, :, :, :]
        des_hmap_low = des_hmap_sils[:, low_landmark, :, :, :]

        ## Feature Swap
        mixed_feat_head = self.transport(src_feat_sils, des_feat_sils, src_hmap_sils, des_hmap_sils, des_hmap_head)  # [n, c=128, s, h, w]
        mixed_feat_up = self.transport(src_feat_sils, des_feat_sils, src_hmap_sils, des_hmap_sils, des_hmap_up)      # [n, c=128, s, h, w]
        mixed_feat_low = self.transport(src_feat_sils, des_feat_sils, src_hmap_sils, des_hmap_sils, des_hmap_low)    # [n, c=128, s, h, w]
        
        ## Reconstruct body parts
        if self.freeze_half_bodypart:
            with torch.no_grad():
                des_pred_head = self.refine(mixed_feat_head)  # [n, c=1, s, h, w]
                des_pred_up = self.refine(mixed_feat_up)      # [n, c=1, s, h, w]
                des_pred_low = self.refine(mixed_feat_low)    # [n, c=1, s, h, w]
        else:
            des_pred_head = self.refine(mixed_feat_head)  # [n, c=1, s, h, w]
            des_pred_up = self.refine(mixed_feat_up)      # [n, c=1, s, h, w]
            des_pred_low = self.refine(mixed_feat_low)    # [n, c=1, s, h, w]

        # Parsing Part Refinement (Fine)
        ## Mask threshold
        threshold = 0.9
        des_pred_head_th = des_pred_head > threshold*torch.max(des_pred_head)
        des_pred_up_th = des_pred_up > threshold*torch.max(des_pred_up)
        des_pred_low_th = des_pred_low > threshold*torch.max(des_pred_low)
        ## Mask shear & Mask Dilate
        up_mask = (des_pred_up_th==False)
        dilated_mask_head = self.preprocess((des_pred_head_th*des_pred_head*up_mask).view(n*s, 1, h, w))  # Dilation
        dilated_mask_up = self.preprocess((des_pred_up_th*des_pred_up).view(n*s, 1, h, w))                # Dilation
        dilated_mask_low = self.preprocess((des_pred_low_th*des_pred_low*up_mask).view(n*s, 1, h, w))     # Dilation
        ## Mask Union
        sils = sils.permute(0, 2, 1, 3, 4).contiguous()  # high-quality segmentation (original segmentation)
        head = dilated_mask_up.view(n, s, 1, h, w).logical_not() * sils * dilated_mask_low.view(n, s, 1, h, w).logical_not()
        up = dilated_mask_head.view(n, s, 1, h, w).logical_not() * sils * dilated_mask_low.view(n, s, 1, h, w).logical_not()
        low = dilated_mask_head.view(n, s, 1, h, w).logical_not() * sils * dilated_mask_up.view(n, s, 1, h, w).logical_not()

        # Parsing Parts Mixup
        sils_roll = torch.max(head, torch.max(self.sils_roll(up, self.shift_up), self.sils_roll(low, self.shift_low)))

        # # ***************************************************** # #
        # # ************** LandmarkGait: Recognition ************ # #
        # # ***************************************************** # #
        outs = self.PMBConv_Feature(des_sils)
        outs_head = self.PMBConv_Feature(head.permute(0, 2, 1, 3, 4).contiguous())
        outs_up = self.PMBConv_Feature(up.permute(0, 2, 1, 3, 4).contiguous())
        outs_low = self.PMBConv_Feature(low.permute(0, 2, 1, 3, 4).contiguous())
        outs_roll = self.PMBConv_Feature(sils_roll.permute(0, 2, 1, 3, 4).contiguous())

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_head = self.TP(outs_head, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_up = self.TP(outs_up, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_low = self.TP(outs_low, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_roll = self.TP(outs_roll, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]

        outs_pars = torch.max(torch.max(outs_head, outs_up), outs_low)
        outs = torch.cat([outs, outs_pars, outs_roll], dim=2)

        outs = self.HPP(outs)  # [n, c, p=1]
        gait = self.Head0(outs)  # [n, c, p]

        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]
            embed = gait

        # visual
        des_hmap_sils_max_c = torch.max(des_hmap_sils, dim=1)[0].unsqueeze(1)
        des_pred_part = torch.max(torch.cat([des_pred_up, des_pred_low, des_pred_head], dim=1), dim=1)[0].unsqueeze(1)

        _, _, _, h_landmark, w_landmark = des_hmap_sils_max_c.size()
        retval = {
            'training_feat': {
                'reconstruction1': {'des_pred': des_pred_sils, 'des': des_sils},
                'reconstruction2': {'des_pred': des_pred_part, 'des': des_sils},
                'triplet': {'embeddings': bnft, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs},
            },
            'visual_summary': {
                'image/sils': des_sils.view(n*s, 1, h, w),
                'image/head': head.view(n*s, 1, h, w),
                'image/up': up.view(n*s, 1, h, w),
                'image/low': low.view(n*s, 1, h, w),
                'image/roll': sils_roll.view(n*s, 1, h, w),
                # 'image/des_pred_sils': des_pred_sils.view(n*s, 1, h, w),
                # 'image/des_pred_head': des_pred_head.view(n*s, 1, h, w),
                # 'image/des_pred_up': des_pred_up.view(n*s, 1, h, w),
                # 'image/des_pred_low': des_pred_low.view(n*s, 1, h, w),
                'image/landmarks': des_hmap_sils_max_c.view(n*s, 1, h_landmark, w_landmark),
            },
            'inference_feat': {
                'embeddings': bnft
            }
        }

        return retval