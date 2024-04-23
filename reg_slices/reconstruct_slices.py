import torch
import torch.optim as optim
from torch import autograd
import numpy as np
import cv2
from tqdm import trange, tqdm
import trimesh
from src_convonet.utils import libmcubes
from src_convonet.common import make_3d_grid, normalize_coord, add_key, coord2index
from src_convonet.utils.libsimplify import simplify_mesh
from src_convonet.utils.libmise import MISE
import time
import math
from src import datasets
from src.models import Slices3DRegModel
from src.model_cam_est import CameraNet
from src.model_disn import DISNModel
from src.model_gt import Slices3DGTModel
from src.datasets import Slice3DDataset
from src.datasets_cam import CamEstDataset
import os
from options import get_parser
import copy
from src.utils import RGB2BGR, tensor2numpy, denorm

def show_imgs(img_input, img_slices_rec, dir_output, batch_id, shape_id):
    n_bs = 1
    res_img = np.zeros((128 * 5, 0, 3))
    dir_tgt = f'{dir_output}/{shape_id}'
    os.makedirs(dir_tgt, exist_ok=True)

    # X
    for idx in range(n_bs):
        for slices_idx in range(4):
            slice_img = RGB2BGR(tensor2numpy(denorm(img_slices_rec[idx, 3*slices_idx:3*(slices_idx+1), ...])))
            slice_img = cv2.resize(slice_img, (256, 256))
            cv2.imwrite(os.path.join(dir_tgt, f'X_{str(slices_idx + 1)}.png'), slice_img * 255.0)
        

    # Z
    for idx in range(n_bs):
        for slices_idx in range(4, 8):
            slice_img = RGB2BGR(tensor2numpy(denorm(img_slices_rec[idx, 3*slices_idx:3*(slices_idx+1), ...])))
            slice_img = cv2.resize(slice_img, (256, 256))
            cv2.imwrite(os.path.join(dir_tgt, f'Z_{str(8 - slices_idx)}.png'), slice_img * 255.0)

    # Y
    for idx in range(n_bs):
        for slices_idx in range(8, 12):
            slice_img = RGB2BGR(tensor2numpy(denorm(img_slices_rec[idx, 3*slices_idx:3*(slices_idx+1), ...])))
            slice_img = cv2.resize(slice_img, (256, 256))
            cv2.imwrite(os.path.join(dir_tgt, f'Y_{str(slices_idx - 7)}.png'), slice_img * 255.0)


if __name__ == '__main__':
    args = get_parser().parse_args()
    if args.name_model == 'slicenet':
        model = Slices3DRegModel(img_size=args.img_size, n_slices=args.n_slices, mode=args.mode)
    elif args.name_model == 'disn':
        model = DISNModel(img_size=args.img_size, mode=args.mode)
    else:
        model = Slices3DGTModel(img_size=args.img_size, n_slices=args.n_slices, mode=args.mode)
    path_ckpt = os.path.join('experiments', args.name_exp, 'ckpt', args.name_ckpt)
    model.load_state_dict(torch.load(path_ckpt)['model'])
    model = model.cuda()
    model = model.eval()

    if args.est_campose:
        
        model_cam = CameraNet()
        path_ckpt_cam = os.path.join('experiments', args.name_exp_cam, 'ckpt', args.name_ckpt_cam)
        model_cam.load_state_dict(torch.load(path_ckpt_cam)['model'])
        model_cam = model_cam.cuda()
        model_cam = model_cam.eval()

    path_res = os.path.join('experiments', args.name_exp, 'results', args.name_dataset)
    if not os.path.exists(path_res):
        os.makedirs(path_res)

    dataset = Slice3DDataset(split='test', args=args)

    dataset_cam = CamEstDataset(split='test', args=args)

    dir_dataset = os.path.join(args.dir_data, args.name_dataset)
    if args.name_dataset == 'shapenet':
        categories = args.categories_test.split(',')[:-1]
        id_shapes = []
        for category in categories:
            id_shapes_ = open(f'{dir_dataset}/04_splits/{category}/test.lst').read().split('\n')
            id_shapes += id_shapes_

    else:
        id_shapes = open(f'{dir_dataset}/04_splits/test.lst').read().split('\n')

    dir_output = os.path.join('experiments', args.name_exp, 'img_slices')
    os.makedirs(dir_output, exist_ok=True)

    with torch.no_grad():
        
        for idx in tqdm(range(len(dataset))):
            
            shape_id = id_shapes[idx]
            data = dataset[idx]
            path_mesh = os.path.join(path_res, '%s.obj'%shape_id)

            # if os.path.exists(path_mesh): continue
            for key in data:
                data[key] = data[key].unsqueeze(0).cuda()
            if args.est_campose:
                data_cam = dataset_cam[idx]
                for key in data_cam:
                    data_cam[key] = data_cam[key].unsqueeze(0).cuda()
            # if use estimated came
            if args.est_campose:
                print('using predicted pose')
                dict_ret_cam = model_cam(data_cam)

                cam_rot_pred = dict_ret_cam['pred_rotation_mat_inv']
                cam_rot_pred[0][0][1] *= -1.
                cam_rot_pred[0][0][2] *= -1.

                cam_rot_pred[0][2][1] *= -1.
                cam_rot_pred[0][2][2] *= -1.
                
                cam_rot_pred[0][1][0] *= -1.

                tmp = copy.deepcopy(cam_rot_pred[0][2][:])
                cam_rot_pred[0][2][:] = cam_rot_pred[0][1][:]
                cam_rot_pred[0][1][:] = tmp

                data['obj_rot_mat'] = cam_rot_pred

                data['trans_mat_right'] = dict_ret_cam['pred_trans_mat']

            ret_dict = model(data)
            img_slices_rec = ret_dict['slices_rec']
            img_input = data['img_input']
            
            show_imgs(img_input, img_slices_rec, dir_output, idx, shape_id)
