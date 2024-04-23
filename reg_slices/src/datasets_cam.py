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

class CamEstDataset(Dataset):
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

        self.preprocess = T.Compose([T.Resize((128, 128)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.dir_img = f'/localhome/ywa439/Documents/03_blender_bisect/processed_{category}_input_addtrans_xrot/dataset_input'
        self.camera_metainfo = f'/localhome/ywa439/Documents/03_blender_bisect/processed_{category}_input_addtrans_xrot/metainfo_input'
        self.rot90y = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
        self.dir_pcd = f'/localhome/ywa439/Documents/datasets/ShapeNet/v1/{category}_pc'

        # self.K_ = cal_K((1., 1.))

    def __len__(self): return len(self.files)

    def rotate():
        batch_sdf_pt_rot[cnt, ...] = np.dot(sample_pt[choice, :], obj_rot_mat)
        return 
    
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
    
    def __getitem__(self, index):
        category, shape_id = self.files[index]

        # cmr_angle = 0
        if self.split in {'train', 'val'}:
            cmr_angle = str(random.randint(2, 7)) # "%02d"%cmr_angle
        else:
            cmr_angle = str(4) # "%02d"%cmr_angle
        
        img = Image.open(f'{self.dir_img}/train/{shape_id}/{cmr_angle}.png').convert('RGB')
        img = self.preprocess(img)

        pcd = np.load(f'{self.dir_pcd}/{shape_id}/2048.npy')

        rot_mat = get_rotate_matrix(-np.pi / 2)

        norm_params = [0, 0, 0, 1]

        norm_mat = get_norm_matrix(norm_params)

        with open(f"{self.camera_metainfo}/{shape_id}/{cmr_angle}.txt", 'r') as f:
            lines = f.read().splitlines()
            param_lst = read_params(lines) # (f"{self.dir_img}/{shape_id}/rendering/rendering_metadata.txt")
            camR, _ = get_img_cam(param_lst[0])
            obj_rot_mat = np.dot(self.rot90y, camR)
            az, el, distance_ratio = param_lst[0][0], param_lst[0][1], param_lst[0][3]
            K, RT = getBlenderProj(az, el, distance_ratio, img_w=1, img_h=1)
            W2O_mat = get_W2O_mat((param_lst[0][-3], param_lst[0][-1], -param_lst[0][-2]))
            trans_mat = np.linalg.multi_dot([K, RT, rot_mat, W2O_mat, norm_mat])
            trans_mat_right = np.transpose(trans_mat)
            regress_mat = np.transpose(np.linalg.multi_dot([RT, rot_mat, W2O_mat, norm_mat]))


        feed_dict = {
            'img_input': img,
            'obj_rot_mat': torch.tensor(obj_rot_mat).float(),
            'trans_mat_right': torch.tensor(trans_mat_right).float(),
            'pcd': torch.tensor(pcd).float(),
            'regress_mat': torch.tensor(regress_mat).float(),
            'norm_mat': torch.tensor(norm_mat).float(),
            'K': torch.tensor(K).float(),
        }

        return feed_dict
