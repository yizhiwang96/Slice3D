import torch.nn as nn
import torch
import torch.nn.functional as F
from .unet_custom import UNet
import numpy as np
import torchvision
torch.set_printoptions(precision=8)
from torchvision import models
from .vgg_perceptual_loss import VGGPerceptualLoss
from .vgg16bn_feats import VGG16BNFeats

class Slices3DGTModel(nn.Module):
    def __init__(self, img_size=128, n_slices=12, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_encoder = VGG16BNFeats()
        self.img_size = img_size
        self.n_slices = n_slices
        self.att_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.att_decoder = nn.TransformerEncoder(self.att_layer, num_layers=3)
        self.fc_out = nn.Sequential(
            nn.Linear(128, 1),
        )
        self.pts_feat_extractor = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.fc_local = nn.Sequential(
            nn.Linear(1472, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.fc_global = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )


    def project_coord(self, coordinates, trans_mat_wo_rot_tp):
        n_bs, n_qry = coordinates.shape[0], coordinates.shape[1]

        size_lst = coordinates.shape
        homo_pc = torch.cat((coordinates, torch.ones((size_lst[0], size_lst[1], 1)).cuda()), axis=-1)
        pc_xyz = torch.bmm(homo_pc, trans_mat_wo_rot_tp)
        pc_xy = torch.divide(pc_xyz[:, :, :2], pc_xyz[:, :, 2:])
        ret = 2 * (pc_xy - 0.5) # from [0, 1] to [-1, 1]
        ret = torch.clamp(ret, min=-1, max=1)
        return ret
    
    def sample_from_planes(self, plane_features, projected_coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
        # assert padding_mode == 'zeros'
        n_planes = 1
        N, C, H, W = plane_features.shape
        _, M, _ = projected_coordinates.shape
        plane_features = plane_features.view(N, C, H, W)
        projected_coordinates = projected_coordinates.unsqueeze(1)
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=True).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        return output_features

    def forward(self, feed_dict):

        img_input = feed_dict['img_input']
        n_bs, _, n_w, n_h = img_input.shape

        if self.mode == 'test':
            qry_no_rot = feed_dict['qry_norot']
            qry_no_rot[:, :, 1:] *= -1
            qry_rot = qry_no_rot
        else:
            qry_no_rot = feed_dict['qry_norot']
            obj_rot_mat = feed_dict['obj_rot_mat']
            qry_rot = torch.bmm(qry_no_rot, obj_rot_mat)

        qry = qry_rot
        _, n_qry, _ = qry_no_rot.shape

        img_slices = feed_dict['img_slices']
        img_slices = img_slices.view(n_bs, self.n_slices, 3, n_w, n_h).view(n_bs * self.n_slices, 3, n_w, n_h)
        img_inpt_and_slices = torch.cat([img_input, img_slices], 0) # n_bs * n_slices
        feat_list, feats_global = self.img_encoder(img_slices) # n * n_slices, 3, w, h

        feat_interp = []
        img_pts = self.project_coord(qry_rot, feed_dict['trans_mat_wo_rot_tp'])
        img_pts = img_pts.view(n_bs, 1, n_qry, 2).expand(-1, self.n_slices, -1, -1).reshape(n_bs * (self.n_slices), n_qry, 2)
        for idx in range(len(feat_list)):
            feat_planes = feat_list[idx]
            feats_out = self.sample_from_planes(feat_planes, img_pts)
            feat_interp.append(feats_out.squeeze(1))
        feat_local_aggregated = torch.cat(feat_interp, dim=2) # n_bs * n_slices, n_qry, 1472
        feat_local_aggregated = feat_local_aggregated.view(n_bs, self.n_slices, n_qry, 1472).permute(0, 2, 1, 3).reshape(n_bs, n_qry, self.n_slices, 1472)

        feat_qry = self.pts_feat_extractor(qry_rot)
        feat_local_aggregated_ = self.fc_local(feat_local_aggregated)
        feat_slice = (feat_local_aggregated_).view(n_bs * n_qry, self.n_slices, 128)
        feat_input = torch.cat([feat_qry.view(n_bs * n_qry, 1, 128), feat_slice], 1)
        feat_attened = self.att_decoder(feat_input).view(n_bs, n_qry, self.n_slices + 1, 128)[:, :, 0, :]
        sdf_pred = self.fc_out(feat_attened).squeeze(-1)

        ret_dict = {}
        ret_dict['sdf_pred'] = sdf_pred

        return ret_dict

