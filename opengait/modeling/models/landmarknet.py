import torch
from ..base_model import BaseModel
from ..modules import BasicConv2d, SetBlockWrapper


class LandmarkGait_Silh_to_Landmark(BaseModel):
    ''' 
        LandmarkNet: Silhouette → Landmarks 
    '''
    def __init__(self, *args, **kargs):
        super(LandmarkGait_Silh_to_Landmark, self).__init__(*args, **kargs)
    
    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['landmarknet_cfg'])

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
        
        # LandmarkNet
        des_sils, des_pred_sils, des_hmap_sils, _, _, _, des_landmark = self.Backbone(sils)

        # landmarks_visual
        des_hmap_sils_max_c = torch.max(des_hmap_sils, dim=1)[0].unsqueeze(2)

        n, _, s, h, w = sils.size()
        _, _, _, h_kp, w_kp = des_hmap_sils_max_c.size()
        retval = {
            'training_feat': {
                'reconstruction': {'des_pred': des_pred_sils, 'des': des_sils},
            },
            'visual_summary': {
                'image/sils': des_sils.view(n*s, 1, h, w),
                'image/sils_pred': des_pred_sils.view(n*s, 1, h, w),
                'image/landmarks': des_hmap_sils_max_c.view(n*s, 1, h_kp, w_kp),
            },
            'inference_feat': {
                'sils': des_sils,
                'sils_pred': des_pred_sils,
                'sils_landmark': des_landmark
            }
        }
        return retval


