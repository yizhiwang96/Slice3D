from src.utils import load_mesh, get_img_cam, getBlenderProj, get_rotate_matrix, get_norm_matrix, write_pointcloud, read_params, get_W2O_mat
import cv2
import numpy as np
import h5py
import trimesh
import pickle

def get_img_points(sample_pc, trans_mat_right, obj_rot_mat, K):
    # sample_pc N * 3
    size_lst = sample_pc.shape

    homo_pc = np.concatenate((sample_pc, np.ones((size_lst[0], 1))), axis=-1)
    pc_xyz = np.matmul(homo_pc, trans_mat_right)

    print(pc_xyz[0:5])
    
    pc_xyz_v2_tmp = np.matmul(sample_pc, obj_rot_mat)
    pc_xyz_v2_tmp = np.concatenate((pc_xyz_v2_tmp, np.ones((size_lst[0], 1))), axis=-1)
    pc_xyz_v2 = np.matmul(pc_xyz_v2_tmp, K)
    print(pc_xyz_v2[0:5])

    pc_xy = np.divide(pc_xyz[:,:2], pc_xyz[:,2:])

    pc_xy = pc_xy * 256
    mintensor = np.array([0.0,0.0])
    maxtensor = np.array([256,256.0])
    return np.minimum(maxtensor, np.maximum(mintensor, pc_xy))

def read_sdf_file(path_sdf, scale, offset):


    sdf_npy = np.load(path_sdf)

    sample_pt = sdf_npy[:, :3]
    sample_sdf_val = sdf_npy[:, 3]

    # (camX, camY, camZ) -> (camX, -camZ, camY)
    print(offset[0], offset[1], offset[2])
    offset_ = np.array([offset[0], offset[2], -offset[1]])

    sample_pt = sample_pt * scale + offset_
    sample_sdf_val = (sample_sdf_val) * scale

    return sample_pt, sample_sdf_val

def test_img_h5(dir_img, shape_id, angle_int):
    angle_str = "%02d"%angle_int
    img_arr, trans_mat, obj_rot_mat, K, sample_pt = get_img(dir_img, shape_id, angle_int)
    # march_obj_fl = f'/localhome/ywa439/Documents/datasets/ShapeNet/DISN_SDF_MC_Meshes/03001627/{shape_id}/isosurf.obj'
    sample_pt_ = sample_pt[0:100]

    pc_xy = get_img_points(sample_pt[0:100], trans_mat, obj_rot_mat, K)

    for j in range(pc_xy.shape[0]):
        y = int(pc_xy[j, 1])
        x = int(pc_xy[j, 0])

        cv2.circle(img_arr, (x, y), 3, (255, 0, 0), -1)
    
    # rot_pc = np.dot(new_pts, obj_rot_mat)

    cv2.imwrite(f"./{shape_id}_{angle_str}_proj_samplept_{str(angle_int)}.png", img_arr)


def get_points(obj_fl):
    sample_pc = np.zeros((0,3), dtype=np.float32)
    mesh_lst = trimesh.load_mesh(obj_fl, process=False)
    if not isinstance(mesh_lst, list):
        mesh_lst = [mesh_lst]
    for mesh in mesh_lst:
        choice = np.random.randint(mesh.vertices.shape[0], size=1000)
        sample_pc = np.concatenate((sample_pc, mesh.vertices[choice,...]), axis=0) #[choice,...]
    # color = [[255,0,0], [0,255,0], [0,0,255], [255, 0, 255]]
    color = 255*np.ones_like(sample_pc, dtype=np.uint8)
    color[:,0] = 0
    color[:,1] = 0
    return sample_pc, np.asarray(color, dtype=np.uint8)

def get_img(dir_img, shape_id, angle_int):
    rot90y = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    angle_str = "%03d"%(angle_int)
    dir_sfd = '/data/wangyz/11_slice3d/data/objaverse/02_sdfs'
    img_arr = cv2.imread(f'{dir_img}/{shape_id}/{angle_str}.png', cv2.IMREAD_UNCHANGED).astype(np.uint8)

    path_cmr_info = f'{dir_img}/{shape_id}/meta.pkl'
    # Open the file in binary mode
    with open(path_cmr_info, 'rb') as f:
        # Load the data
        data = pickle.load(f)
    # trans_mat = np.transpose(data[4][angle_int])
    az = -data[1][angle_int]
    el = data[2][angle_int]
    print(az, el)
    distance = data[3][angle_int]
    scale = data[5]
    offset = data[6]
    print('offset', offset)
    K, RT = getBlenderProj(az, el, distance, img_w=1, img_h=1)

    sample_pt, sample_sdf_val = read_sdf_file(f'{dir_sfd}/{shape_id}.npy', scale, offset)

    rot_mat = get_rotate_matrix(-np.pi / 2)
    W2O_mat = get_W2O_mat((0, 0, 0))
    trans_mat = np.linalg.multi_dot([K, RT, rot_mat, W2O_mat])
    trans_mat = np.transpose(trans_mat)

    rot_full = np.linalg.multi_dot([RT, rot_mat])
    tmp = np.concatenate((np.eye(3), rot_full[:, 3:4]), axis=1)
    obj_rot_mat = np.transpose(rot_full)[:3, :]

    K = np.transpose(np.linalg.multi_dot([K, tmp, W2O_mat]))
    return img_arr[:, :, :3].copy(), trans_mat, obj_rot_mat, K, sample_pt[(sample_sdf_val) > 0]


angle_int = 0
dir_img = '/data/wangyz/11_slice3d/data/objaverse/00_img_input'
shape_id = 'f598dfee0d22404983dc9d9c3f307202' # 00a1a602456f4eb188b522d7ef19e81b 0a5652c16e1a4575903dfc1696382502_00_proj_samplept_0
test_img_h5(dir_img, shape_id, angle_int)
