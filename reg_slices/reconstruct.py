import torch
import torch.optim as optim
from torch import autograd
import numpy as np
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

class Generator3D(object):
    '''  Generator class for Occupancy Networks.
    It provides functions to generate the final mesh as well refining options.
    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=64, upsampling_steps=2, chunk_size=3000,
                 with_normals=False, padding=0.0, sample=False,
                 input_type = None,
                 vol_info = None,
                 vol_bound = None,
                 simplify_nfaces=None, pred_type='occ'):
        #self.model = model.to(device)
        self.model = model
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.chunk_size = chunk_size
        self.pred_type = pred_type
        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info



    def eval_points(self, data):
        
        n_qry = data['qry_norot'].shape[1]
        chunk_size = self.chunk_size
        n_chunk = math.ceil(n_qry / chunk_size)

        ret = []

        for idx in range(n_chunk):
            data_chunk = {}
            for key in data:
                if key == 'qry_norot':
                    if idx < n_chunk - 1:
                        data_chunk[key] = data[key][:, chunk_size*idx:chunk_size*(idx+1), ...]
                    else:
                        data_chunk[key] = data[key][:, chunk_size*idx:n_qry, ...]
                else:
                    data_chunk[key] =  data[key]

            ret_dict = self.model(data_chunk)
            if self.pred_type == 'occ':
                ret.append(ret_dict['occ_pred'])
            else:
                ret.append(-ret_dict['sdf_pred'])

        
        ret = torch.cat(ret, -1)
        ret = ret.squeeze(0)
        return ret

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        # self.model.eval()
        device = self.device
        stats_dict = {}

        mesh = self.generate_from_latent(data, stats_dict=stats_dict)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh
    
    def generate_from_latent(self, c=None, stats_dict={}):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube
        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding
        
        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
                # (-1.0,)*3, (1.0,)*3, (nx,)*3
            )
            # print(np.min(pointsf), np.max(pointsf))
            # input()
            data['qry_norot'] = pointsf.unsqueeze(0).cuda()
            
            values = self.eval_points(data).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points
                pointsf = points / mesh_extractor.resolution
                # Normalize to bounding box
                pointsf = box_size * (pointsf - 0.5)
                # print(np.min(pointsf), np.max(pointsf))
                # input()
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                data['qry_norot'] = pointsf.unsqueeze(0).cuda()
                # Evaluate model and update
                values = self.eval_points(data).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.
        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
 
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1
        
        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (self.vol_bound['axis_n_crop'].max() * self.resolution0*2**self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else: 
            # Normalize to bounding box
            vertices /= np.array([n_x-1, n_y-1, n_z-1])
            vertices = box_size * (vertices - 0.5)
        
        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None


        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)
        


        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.
        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.
        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh

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

    generator = Generator3D(model, threshold=args.mc_threshold, resolution0=args.mc_res0, upsampling_steps=args.mc_up_steps, 
                            chunk_size=args.mc_chunk_size, pred_type=args.pred_type)
    dataset = Slice3DDataset(split='test', args=args)

    if args.est_campose:
        dataset_cam = CamEstDataset(split='test', args=args)

    dir_dataset = os.path.join(args.dir_data, args.name_dataset)
    if args.name_dataset == 'shapenet':
        categories = args.categories_test.split(',')[:-1]
        id_shapes = []
        for category in categories:
            id_shapes_ = open(f'{dir_dataset}/03_splits/{category}/test.lst').read().split('\n')
            id_shapes += id_shapes_

    else:
        id_shapes = open(f'{dir_dataset}/03_splits/test.lst').read().split('\n')
    with torch.no_grad():
        
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            path_mesh = os.path.join(path_res, '%s.obj'%id_shapes[idx])
            if not args.overwrite_res:
                if os.path.exists(path_mesh): continue
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

            out = generator.generate_mesh(data)
            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}
                

            mesh.export(path_mesh)

