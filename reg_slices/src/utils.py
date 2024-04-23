import numpy as np 
import math
import trimesh
import cv2
import open3d as o3d
import struct

rot90y = np.array([[0, 0, -1],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=np.float32)

def read_params(lines):
    params = []
    for line in lines:
        line = line.strip()[1:-2]
        param = np.fromstring(line, dtype=float, sep=',')
        params.append(param)
    return params

def get_W2O_mat(shift):
    T_inv = np.asarray(
        [[1.0, 0., 0., shift[0]],
         [0., 1.0, 0., shift[1]],
         [0., 0., 1.0, shift[2]],
         [0., 0., 0., 1.]]
    )
    return T_inv
    
def getBlenderProj(az, el, distance, img_w=256, img_h=256):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    F_MM = 35.  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.
    CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                      [1.0, -4.371138828673793e-08, -0.0],
                      [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    # Calculate intrinsic matrix.
    scale = RESOLUTION_PCT / 100
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(-az)
    ca = np.cos(-az)
    se = np.sin(-el)
    ce = np.cos(-el)
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance,
                                           0,
                                           0)))
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))

    return K, RT

def get_img_cam(param):
    cam_mat, cam_pos = camera_info(degree2rad(param))

    return cam_mat, cam_pos

def degree2rad(params):
    params_new = np.zeros_like(params)
    params_new[0] = np.deg2rad(params[0] + 180.0)
    params_new[1] = np.deg2rad(params[1])
    params_new[2] = np.deg2rad(params[2])
    return params_new

def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def camera_info(param):
    az_mat = get_az(param[0])
    el_mat = get_el(param[1])
    inl_mat = get_inl(param[2])
    cam_mat = np.transpose(np.matmul(np.matmul(inl_mat, el_mat), az_mat))
    cam_pos = get_cam_pos(param)
    return cam_mat, cam_pos

def get_cam_pos(param):
    camX = 0
    camY = 0
    camZ = param[3]
    cam_pos = np.array([camX, camY, camZ])
    return -1 * cam_pos

def get_az(az):
    cos = np.cos(az)
    sin = np.sin(az)
    mat = np.asarray([cos, 0.0, sin, 0.0, 1.0, 0.0, -1.0 * sin, 0.0, cos], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat

def get_el(el):
    cos = np.cos(el)
    sin = np.sin(el)
    mat = np.asarray([1.0, 0.0, 0.0, 0.0, cos, -1.0 * sin, 0.0, sin, cos], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat

def get_inl(inl):
    cos = np.cos(inl)
    sin = np.sin(inl)
    # zeros = np.zeros_like(inl)
    # ones = np.ones_like(inl)
    mat = np.asarray([cos, -1.0 * sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat


def get_rotate_matrix(rotation_angle1):
    cosval = np.cos(rotation_angle1)
    sinval = np.sin(rotation_angle1)

    rotation_matrix_x = np.array([[1, 0,        0,      0],
                                  [0, cosval, -sinval, 0],
                                  [0, sinval, cosval, 0],
                                  [0, 0,        0,      1]])
    rotation_matrix_y = np.array([[cosval, 0, sinval, 0],
                                  [0,       1,  0,      0],
                                  [-sinval, 0, cosval, 0],
                                  [0,       0,  0,      1]])
    rotation_matrix_z = np.array([[cosval, -sinval, 0, 0],
                                  [sinval, cosval, 0, 0],
                                  [0,           0,  1, 0],
                                  [0,           0,  0, 1]])
    scale_y_neg = np.array([
        [1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0,  1, 0],
        [0, 0,  0, 1]
    ])

    neg = np.array([
        [-1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0,  -1, 0],
        [0, 0,  0, 1]
    ])
    # y,z swap = x rotate -90, scale y -1
    # new_pts0[:, 1] = new_pts[:, 2]
    # new_pts0[:, 2] = new_pts[:, 1]
    #
    # x y swap + negative = z rotate -90, scale y -1
    # new_pts0[:, 0] = - new_pts0[:, 1] = - new_pts[:, 2]
    # new_pts0[:, 1] = - new_pts[:, 0]

    # return np.linalg.multi_dot([rotation_matrix_z, rotation_matrix_y, rotation_matrix_y, scale_y_neg, rotation_matrix_z, scale_y_neg, rotation_matrix_x])
    return np.linalg.multi_dot([neg, rotation_matrix_z, rotation_matrix_z, scale_y_neg, rotation_matrix_x])


def get_norm_matrix(norm_params):
    #with h5py.File(sdf_h5_file, 'r') as h5_f:
        #norm_params = h5_f['norm_params'][:]
    center, m, = norm_params[:3], norm_params[3]
    x,y,z = center[0], center[1], center[2]
    M_inv = np.asarray(
        [[m, 0., 0., 0.],
            [0., m, 0., 0.],
            [0., 0., m, 0.],
            [0., 0., 0., 1.]]
    )
    T_inv = np.asarray(
        [[1.0 , 0., 0., x],
            [0., 1.0 , 0., y],
            [0., 0., 1.0 , z],
            [0., 0., 0., 1.]]
    )
    return np.matmul(T_inv, M_inv)

def load_mesh(fn):
    mesh = trimesh.load(fn, force='mesh', skip_materials=True, maintain_order=True, process=False)
    return mesh

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def concat_imgs(self, A2B, img_tensors_list):
    img_list = []
    for img_tensor in img_tensors_list:
        img_list.append(RGB2BGR(tensor2numpy(denorm(img_tensor[0]))))
    if A2B is None:
        A2B = np.concatenate(img_list, 1)
    else:
        A2B = np.concatenate((A2B, np.concatenate(img_list, 0)), 1)
    return A2B

def create_mesh_o3d(v,f):
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v),
        o3d.utility.Vector3iVector(f))
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def create_raycast_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene

def cast_rays(scene,rays):
    rays_o3dt = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    hits = scene.cast_rays(rays_o3dt)
    # dist
    hit_dists = hits['t_hit'].numpy() # real/inf
    # mask
    hit_geos = hits['geometry_ids'].numpy()
    hit_mask = hit_geos!=o3d.t.geometry.RaycastingScene.INVALID_ID
    # hit_ids = np.where(hit_mask)[0]
    hit_dists[~hit_mask] = 1.0
    rdf = np.full_like(hit_dists, 1.0, dtype='float32')
    mask = np.full_like(hit_dists, 0.0, dtype='float32')
    rdf[hit_mask] = hit_dists[hit_mask] 
    mask[hit_mask] = hit_mask[hit_mask]
    return rdf, mask

def fibonacci_sphere(n=48,offset=False):
    """Sample points on sphere using fibonacci spiral.

    # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

    :param int n: number of sample points, defaults to 48
    :param bool offset: set True to get more uniform samplings when n is large , defaults to False
    :return array: points samples
    """

    golden_ratio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / golden_ratio

    if offset:
        if n >= 600000:
            epsilon = 214
        elif n>= 400000:
            epsilon = 75
        elif n>= 11000:
            epsilon = 27
        elif n>= 890:
            epsilon = 10
        elif n>= 177:
            epsilon = 3.33
        elif n>= 24:
            epsilon = 1.33
        else:
            epsilon = 0.33
        phi = np.arccos(1 - 2*(i+epsilon)/(n-1+2*epsilon))
    else:
        phi = np.arccos(1 - 2*(i+0.5)/n)

    x = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],axis=-1)
    return x


def write_pointcloud(filename, xyz_points, rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()



def cal_K(img_size):
    F_MM = 35.  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.

    # Calculate intrinsic matrix.
    # 2 atan(35 / 2*32)
    scale = RESOLUTION_PCT / 100
    f_u = F_MM * img_size[1] * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_size[0] * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    u_0 = img_size[1] * scale / 2
    v_0 = img_size[0] * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)), dtype=np.float32)
    return K