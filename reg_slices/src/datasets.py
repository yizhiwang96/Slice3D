import numpy as np
import torch
from torch.utils.data import Dataset
import os
import trimesh
from scipy.spatial.transform import Rotation
from .utils import load_mesh, get_img_cam, getBlenderProj, get_rotate_matrix, get_norm_matrix, read_params, get_W2O_mat
from PIL import Image
import torchvision.transforms as T
import random
import h5py
import pickle

class Slice3DDataset(Dataset):
    def __init__(self, split, args) -> None:
        self.split = split
        self.n_qry = args.n_qry
        self.dir_dataset = os.path.join(args.dir_data, args.name_dataset)
        self.name_dataset = args.name_dataset
        self.img_size = args.img_size
        self.files = []
        if self.name_dataset == 'shapenet':
            if self.split in {'train', 'val'}:
                categories = args.categories_train.split(',')[:-1]
            else:
                categories = args.categories_test.split(',')[:-1]
            self.fext_mesh = 'obj'
        else:
            categories = ['']
            self.fext_mesh = 'ply'
        for category in categories:
            id_shapes = open(f'{self.dir_dataset}/03_splits/{category}/{split}.lst').read().split()
            for shape_id in id_shapes:
                self.files.append((category, shape_id))

        self.preprocess = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
        self.dir_sfd = f'{self.dir_dataset}/02_sdfs/'
        self.from_which_slices = args.from_which_slices
        if self.from_which_slices == 'gen':
            self.dir_img_slice = f'{self.dir_dataset}/04_img_slices_gen'
            self.preprocess_gen_slice = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        elif self.from_which_slices == 'gt':
            self.dir_img_slice = f'{self.dir_dataset}/01_img_slices'
        elif self.from_which_slices == 'gt_rec':
            self.dir_img_slice = f'{self.dir_dataset}/05_img_slices_rec'
            self.preprocess_gen_slice = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.dir_img_ipt = f'{self.dir_dataset}/00_img_input'
        self.camera_metainfo = f'{self.dir_dataset}/00_img_input'
        self.use_white_bg = args.use_white_bg
        self.n_views = args.n_views

    def __len__(self): return len(self.files)
    
    def get_sdf_h5(self, sdf_h5_file, cat_id, obj):
        h5_f = h5py.File(sdf_h5_file, 'r')
        try:
            if ('pc_sdf_original' in h5_f.keys()
                    and 'pc_sdf_sample' in h5_f.keys()
                    and 'norm_params' in h5_f.keys()):
                ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
                sample_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
                ori_pt = ori_sdf[:,:3]#, ori_sdf[:,3]
                ori_sdf_val = None
                sample_pt, sample_sdf_val = sample_sdf[:,:3], sample_sdf[:,3]
                norm_params = h5_f['norm_params'][:]
                sdf_params = h5_f['sdf_params'][:]
            else:
                raise Exception(cat_id, obj, "no sdf and sample")
        finally:
            h5_f.close()
        return ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params

    def png_2_whitebg(self, img):
        img_rgb = np.array(img)[:, :, 0:3]
        img_alpha = np.array(img)[:, :, 3:4]
        img_alpha_ = (img_alpha == 0).astype(np.float32)
        img_ret = np.ones(img_rgb.shape) * 255 * img_alpha_ + img_rgb * (1 - img_alpha_)
        ret = Image.fromarray(img_ret.astype(np.uint8))
        return ret

    def png_2_rgb(self, img):
        img_rgb = np.array(img)[:, :, 0:3]
        img_alpha = np.array(img)[:, :, 3:4]
        mask = (img_alpha / 255.)
        img_rgb_ = (img_rgb * mask).astype(np.uint8)
        return Image.fromarray(img_rgb_)

    def __getitem__(self, index):
        category, shape_id = self.files[index]

        if self.split == 'train':
            cmr_angle_idx = random.randint(0, self.n_views - 1)
        else:
            cmr_angle_idx = 4
        
        cmr_angle = "%03d"%cmr_angle_idx

        img_ipt = Image.open(f'{self.dir_img_ipt}/{shape_id}/{cmr_angle}.png')
        if self.use_white_bg: img_ipt = self.png_2_whitebg(img_ipt)
        else: img_ipt = self.png_2_rgb(img_ipt)
        img_ipt = self.preprocess(img_ipt)

        img_slices_list = []
        
        for axis in ['X', 'Z', 'Y']:
            if axis == 'Z':
                slice_list = ['4', '3', '2', '1']
            else:
                slice_list = ['1', '2', '3', '4']
            for part in slice_list:
                img_slice = Image.open(f'{self.dir_img_slice}/{shape_id}/{cmr_angle}/{axis}_{part}.png')
                if self.from_which_slices in ['gen', 'gt_rec']:
                    img_slice = self.preprocess_gen_slice(img_slice)
                else:
                    if self.use_white_bg: img_slice = self.png_2_whitebg(img_slice)
                    else: img_slice = self.png_2_rgb(img_slice)
                    img_slice = self.preprocess(img_slice)
                img_slices_list.append(img_slice)
        img_slices = torch.cat(img_slices_list, 0)

        # load camera information
        rot_mat = get_rotate_matrix(-np.pi / 2)
        path_cmr_info = f'{self.camera_metainfo}/{shape_id}/meta.pkl'
        with open(path_cmr_info, 'rb') as f:
            data = pickle.load(f)
        az = -data[1][cmr_angle_idx]
        el = data[2][cmr_angle_idx]
        distance = data[3][cmr_angle_idx]
        scale = data[5]
        offset = data[6]
        K, RT = getBlenderProj(az, el, distance, img_w=1, img_h=1)
        W2O_mat = get_W2O_mat((0, 0, 0))

        rot_full = np.linalg.multi_dot([RT, rot_mat])
        obj_rot_mat = np.transpose(rot_full)[:3, :]

        tmp = np.concatenate((np.eye(3), rot_full[:, 3:4]), axis=1) # Note: rot_full[:, 3:4] are constant values, not related to az, el
        trans_mat_wo_rot = np.linalg.multi_dot([K, tmp, W2O_mat])
        trans_mat_wo_rot_tp = np.transpose(trans_mat_wo_rot)

        sdf_npy = np.load(f'{self.dir_sfd}/{shape_id}.npy')
        sample_pt = sdf_npy[:, :3]
        sample_sdf_val = sdf_npy[:, 3]

        offset_ = np.array([offset[0], offset[2], -offset[1]])
        sample_pt = sample_pt * scale + offset_
        sample_sdf_val = (sample_sdf_val - 0.003) * scale # the sdfs were extracted at the level of 0.003

        qry = sample_pt
        sdf = sample_sdf_val
        occ = (sdf <= 0).astype(np.float32)

        if self.split == 'train':
            np.random.seed()
            perm = np.random.permutation(len(qry))[:self.n_qry]
            qry = qry[perm]
            occ = occ[perm]
            sdf = sdf[perm]
        else:
            np.random.seed(1234)
            perm = np.random.permutation(len(qry))[:self.n_qry]
            qry = qry[perm]
            occ = occ[perm]
            sdf = sdf[perm]  

        # sdf -= 0.003

        feed_dict = {
            'img_input': img_ipt,
            'qry_norot': torch.tensor(qry).float(), # qry_norot must be in the range of [-0.5, 0.5]
            'obj_rot_mat': torch.tensor(obj_rot_mat).float(),
            'trans_mat_wo_rot_tp': torch.tensor(trans_mat_wo_rot_tp).float(),
            'occ': torch.tensor(occ).float(),
            'sdf': torch.tensor(sdf).float(),
            'img_slices': img_slices,
        }

        return feed_dict
