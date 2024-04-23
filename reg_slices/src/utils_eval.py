import torch
import numpy as np
import math
from scipy.spatial import cKDTree, distance
from src_convonet.utils.libmesh import check_mesh_contains

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def eval_iou(mesh, qry, occ_tgt):

    if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        occ = check_mesh_contains(mesh, qry)
        iou = compute_iou(occ, occ_tgt)
    else:
        iou = 0.0
    return iou


def points_dist(p1, p2, k=1, return_ind=False):
    '''distance from p1 to p2'''
    tree = cKDTree(p2)
    dist, ind = tree.query(p1, k=k)
    if return_ind == True:
        return dist, ind
    else:
        return dist

def chamfer_dist(p1, p2):
    d1 = points_dist(p1, p2) ** 2
    d2 = points_dist(p2, p1) ** 2
    return d1, d2

def np2th(array, device='cuda'):
    tensor = array
    if type(array) is not torch.Tensor:
        tensor = torch.tensor(array).float()
    if type(tensor) is torch.Tensor:
        if device=='cuda':
            return tensor.cuda()
        return tensor.cpu()
    else:
        return array

def eval_chamfer(p1, p2, f_thresh=0.01):
    """ p1: reconstructed points
        p2: reference ponits
        shapes: (N, 3)
    """
    d1, d2  = chamfer_dist(p1, p2)

    d1sqrt, d2sqrt = (d1**.5), (d2**.5)
    chamfer_L1 = 0.5 * (d1sqrt.mean(axis=-1) + d2sqrt.mean(axis=-1))
    chamfer_L2 = 0.5 * (d1.mean(axis=-1) + d2.mean(axis=-1))

    precision  = (d1sqrt < f_thresh).sum(axis=-1) / p1.shape[0]
    recall = (d2sqrt < f_thresh).sum(axis=-1) / p2.shape[0]
    fscore = 2 * (recall * precision /  recall + precision )

    return [chamfer_L1, chamfer_L2, fscore, precision, recall]

def eval_hausdoff(p1, p2):
    """ p1: reconstructed points
        p2: reference ponits
        shapes: (N, 3)
    """
    dist_rec2ref, _, _ = distance.directed_hausdorff(p1, p2)
    dist_ref2rec, _, _ = distance.directed_hausdorff(p2, p1)
    dist = max(dist_rec2ref, dist_ref2rec)
    return dist_rec2ref, dist_ref2rec, dist

