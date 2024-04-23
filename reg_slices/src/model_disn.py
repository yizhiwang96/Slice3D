import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
torch.set_printoptions(precision=8)
from torchvision import models
from .vgg16bn_feats import VGG16BNFeats

class DISNModel(nn.Module):
    def __init__(self, img_size=224, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_encoder = VGG16BNFeats()
        self.img_size = img_size

        self.pts_feat_extractor = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.fc_local = nn.Sequential(
            nn.Linear(1472 + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )


        self.fc_global = nn.Sequential(
            nn.Linear(1000 + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )


    def project_coord(self, coordinates, trans_mat_right):
        n_bs, n_qry = coordinates.shape[0], coordinates.shape[1]
        # trans_mat_right = trans_mat_right.unsqueeze(1).expand(-1, n_qry, -1, -1)
        # coordinates = coordinates * 2
        size_lst = coordinates.shape
        homo_pc = torch.cat((coordinates, torch.ones((size_lst[0], size_lst[1], 1)).cuda()), axis=-1)
        pc_xyz = torch.bmm(homo_pc, trans_mat_right)
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
        trans_mat_right = feed_dict['trans_mat_right']
        n_bs, _, n_w, n_h = img_input.shape
        qry_no_rot = feed_dict['qry_norot']

        obj_rot_mat = feed_dict['obj_rot_mat']
        qry_rot = torch.bmm(qry_no_rot, obj_rot_mat)
        qry = qry_rot
        _, n_qry, _ = qry_no_rot.shape

        feat_list, feats_global = self.img_encoder(img_input) # n * 12, 3, w, h

        feat_interp = []

        img_pts = self.project_coord(qry_no_rot, trans_mat_right)
        # img_pts = img_pts.view(n_bs, 1, n_qry, 2).expand(-1, 12, -1, -1).reshape(n_bs * 12, n_qry, 2)

        for idx in range(len(feat_list)):

            n_bs_, n_c, n_h, n_w = feat_list[idx].shape
            feat_planes = feat_list[idx].view(n_bs_, n_c, n_h, n_w)

            feats_out = self.sample_from_planes(feat_planes, img_pts)

            feat_interp.append(feats_out.squeeze(1))

        feat_local_aggregated = torch.cat(feat_interp, dim=2) # n_bs, n_qry, 1472

        feats_global = feats_global.unsqueeze(1).expand(-1, n_qry, -1) # n_bs, n_qry, 1000

        # print(feat_local_aggregated.shape)
        # print(feats_global.shape)
        # input()
        # feat_local_aggregated_ = feat_local_aggregated.view(n_bs, n_qry, 1472).reshape(n_bs * n_qry, 1472)

        feat_qry = self.pts_feat_extractor(qry_rot)

        feat_local_cat_q = torch.cat([feat_local_aggregated, feat_qry], 2)
        feat_global_cat_q = torch.cat([feats_global, feat_qry], 2)



        sdf_pred = self.fc_local(feat_local_cat_q) + self.fc_global(feat_global_cat_q)


        ret_dict = {}
        ret_dict['sdf_pred'] = sdf_pred.squeeze(-1)
        # ret_dict['slices_rec'] = slices_rec_

        # img_slices_ = feed_dict['img_slices'].view(n_bs, 12, 3, n_w, n_h).view(n_bs * 12, 3, n_w, n_h)
        # print(slices_rec.shape)
        #print(img_slices_.shape)
        #input()
        # ret_dict['vgg_loss'] = self.vggptlossfunc(slices_rec, img_slices_)['pt_c_loss'] * 0.001

        return ret_dict

