import torch
import torch.nn as nn
import pdb

import numpy as np
import matplotlib.pyplot as plt

import torchvision.models as models
#from torchsummary import summary
import torch.nn.functional as F


class CameraNet(nn.Module):
    def __init__(self, backbone='vgg16_bn'): #,configs
        super(CameraNet, self).__init__()

        if backbone == 'vgg16_bn':
            encoder = models.vgg16_bn(pretrained=True)
            self.dim_last_conv_feat = 512
        else:
            encoder = models.resnet50(pretrained=True)
            self.dim_last_conv_feat = 2048
        
        self.global_features = nn.Sequential(*list(encoder.children())[:-2])

        self.fc = nn.Linear(4*4*self.dim_last_conv_feat, 1024)

        # ortho6d
        self.ortho6d_1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.ortho6d_2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.ortho6d_3 = nn.Sequential(nn.Linear(256, 6))

        self.branch_ortho6d = nn.Sequential(self.ortho6d_1, self.ortho6d_2, self.ortho6d_3)

        # distratio
        self.dist_1 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU())
        self.dist_2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.dist_3 = nn.Sequential(nn.Linear(64, 1))

        self.branch_dist = nn.Sequential(self.dist_1, self.dist_2, self.dist_3)


    def get_const(self, n_bs):

        CAM_MAX_DIST = torch.tensor(1.75)
        CAM_ROT = torch.tensor(np.asarray([[0.0, 0.0, 1.0],
                                            [1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0]], dtype=np.float32))
        R_camfix = torch.tensor(np.matrix(((1., 0., 0.), (0., -1., 0.), (0., 0., -1.)), dtype=np.float32))


        CAM_MAX_DIST = torch.tile(torch.reshape(CAM_MAX_DIST, [1,1,1]), [n_bs,1,1]).cuda()
        R_obj2cam_inv = torch.tile(CAM_ROT.unsqueeze(0), [n_bs, 1, 1]).cuda()
        R_camfix = torch.tile(R_camfix.unsqueeze(0), [n_bs, 1, 1]).cuda()

        return CAM_MAX_DIST, R_obj2cam_inv, R_camfix



    def normalize_vector(self, v):
        batch = v.shape[0]
        v_mag = torch.sqrt(torch.sum(torch.square(v), axis=1, keepdims=True))
        v_mag = torch.maximum(v_mag, torch.tensor(1e-8).cuda())
        v = v / v_mag
        return v

    def compute_rotation_matrix_from_ortho6d(self, poses):
        x_raw = poses[:, 0:3]
        y_raw = poses[:, 3:6]#batch*3
        x = self.normalize_vector(x_raw) #batch*3
        z = torch.linalg.cross(x, y_raw) #batch*3
        z = self.normalize_vector(z)#batch*3
        y = torch.linalg.cross(z,x) #batch*3

        x = torch.reshape(x, [-1, 3, 1])
        y = torch.reshape(y, [-1, 3, 1])
        z = torch.reshape(z, [-1, 3, 1])
        matrix = torch.cat((x,y,z), 2) #batch*3*3

        return matrix

    def forward(self, feed_dict):
        img = feed_dict['img_input']
        n_bs, _, _, _ = feed_dict['img_input'].shape
        global_feat = self.global_features(img) # n_bs, 512, 4, 4

        global_feat = global_feat.view(-1, 4*4*self.dim_last_conv_feat)
        global_feat = F.relu(self.fc(global_feat))

        rotation_pred = self.branch_ortho6d(global_feat)
        dist_pred = self.branch_dist(global_feat)

        # for branch_ortho6d
        pred_rotation_mat_inv = self.compute_rotation_matrix_from_ortho6d(rotation_pred)

        # for branch_dist
        distance_ratio = torch.sigmoid(dist_pred) * 0.35 + 0.7
        distance_ratio = distance_ratio.view(n_bs, 1, 1)

        CAM_MAX_DIST, R_obj2cam_inv, R_camfix = self.get_const(n_bs)

        cam_location_inv = torch.cat([distance_ratio * CAM_MAX_DIST, torch.zeros([n_bs, 1, 2]).cuda()], 2)
        R_camfix_inv = R_camfix.permute(0, 2, 1)

        pred_translation_inv = cam_location_inv @ R_obj2cam_inv @ R_camfix_inv * -1.0

        pred_RT_inv = torch.cat([pred_rotation_mat_inv, pred_translation_inv], 1) # n_bs, 4, 3

        # print(pred_translation_inv)
        # input()

        loss_pred, pred_trans_mat = self.get_loss(feed_dict, pred_RT_inv, n_bs)

        ret = {}
        ret['loss_pred'] = loss_pred
        ret['pred_trans_mat'] = pred_trans_mat
        ret['pred_rotation_mat_inv'] = pred_rotation_mat_inv
        
        return ret


    def get_inverse_norm_matrix(self, norm_params, batch_size):
        m = norm_params[:, 3]
        m_inv_padding = torch.multiply(torch.eye(3, m=4).unsqueeze(0).expand(batch_size, -1, -1), m[:, None, None])
        M_inv = torch.cat([m_inv_padding, torch.cat([torch.zeros([batch_size, 1, 3]), torch.ones([batch_size, 1, 1])], 2)], 1)

        T_inv_padding = torch.eye(3, m=4).unsqueeze(0).expand(batch_size, -1, -1)
        xyz1 = torch.unsqueeze(torch.concat((norm_params[:,:3], torch.ones([batch_size, 1])), axis=1), dim=1)
        T_inv = torch.cat((T_inv_padding, xyz1), 1)
        ret = torch.matmul(M_inv, T_inv)
        return ret.cuda()

    def get_loss(self, feed_dict, pred_RT, n_bs):

        # norm_mat_inv = self.get_inverse_norm_matrix(norm_params, n_bs)
        norm_mat_inv = feed_dict['norm_mat']
        K = feed_dict['K']

        # rot_mat_inv
        rot_mat_inv = np.array([[1., 0.,  0., 0.],
                               [0., 0.,  1., 0.],
                               [0., -1., 0., 0.],
                               [0., 0.,  0., 1.]], dtype=np.float32)
        rot_mat_inv_pl = torch.tensor(rot_mat_inv).cuda()
        rot_mat_inv_pl = rot_mat_inv_pl.unsqueeze(0)    # Convert to a len(yp) x 1 matrix.
        rot_mat_inv_pl = torch.tile(rot_mat_inv_pl, [n_bs, 1, 1])  # Create multiple columns.

        sample_pc = feed_dict['pcd']
        size_lst = sample_pc.shape
        homo_sample_pc = torch.cat((sample_pc, torch.ones((size_lst[0], size_lst[1], 1)).cuda()), axis= -1)

        regress_mat = feed_dict['regress_mat'] # n_bs, 4, 3

        # the chain mul to produce pred regress_mat: norm_mat_inv rot_mat_inv pred_RT
        pred_regress_mat = norm_mat_inv @ rot_mat_inv_pl @ pred_RT

        # camera loss point cloud rotation error
        # homo_sample_pc is [n_bs, 2048, 4], regress_mat is [n_bs, 4, 3]
        # the result will be [n_bs, 2048, 3]
        pc_trans_pred = torch.matmul(homo_sample_pc, pred_regress_mat)
        pc_trans_gt = torch.matmul(homo_sample_pc, regress_mat)

        l2_loss = nn.MSELoss()

        rotpc_loss = l2_loss(pc_trans_pred, pc_trans_gt)

        loss = rotpc_loss

        pred_trans_mat = (K @ pred_regress_mat.permute(0, 2, 1)).permute(0, 2, 1)
        # gt_trans_mat = (K @ regress_mat.permute(0, 2, 1)).permute(0, 2, 1)


        return loss, pred_trans_mat