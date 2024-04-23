import os
import sys
from PIL import Image
import numpy as np
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_slices", type=str, default='logs/2024-04-23T02-11-33_objaverse-ldm-kl-8/images_testing_sampled')
    parser.add_argument("--type_slices", type=str, default='gen', choices=['gen', 'rec'])
    parser.add_argument("--name_dataset", type=str, default='objaverse')
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--n_bs", type=int, default=8)
    parser.add_argument("--n_views", type=int, default=12)
    args = parser.parse_args()
    return args

def crop_slices(args):
    dir_slices = args.dir_slices

    if args.type_slices == 'gen':
        dir_tgt = f'../data/{args.name_dataset}/04_img_slices_gen'
        shape_uids = open('../data/objaverse/03_splits/test.lst', 'r').read().split('\n')
    else:
        dir_tgt = f'../data/{args.name_dataset}/05_img_slices_rec'
        shape_uids_ = open('../data/objaverse/03_splits/trainval.lst', 'r').read().split('\n')
        shape_uids_num = len(shape_uids_)
        shape_uids = shape_uids_ * args.n_views

    axis_list = ['X', 'Z', 'Y']
    part_list = ['1', '2', '3', '4']
    part_list_ = ['4', '3', '2', '1']

    img_size = args.img_size
    n_bs = args.n_bs

    for idx, shape_uid in enumerate(shape_uids):
        batch_id = idx // n_bs
        case_id = idx % n_bs
        if args.type_slices == 'gen':
            view_id = '004'
        else:
            view_id = "%03d"%(idx // shape_uids_num)
        
        if not os.path.exists(f'{dir_slices}/{batch_id}_{case_id}.png'): continue
        img = Image.open(f'{dir_slices}/{batch_id}_{case_id}.png')
        os.makedirs(f'{dir_tgt}/{shape_uid}/{view_id}', exist_ok=True)
        for idx_i in range(3):
            for idx_j in range(4):
                axis = axis_list[idx_i]
                if idx_i != 1:
                    part_name = part_list[idx_j]
                else:
                    part_name = part_list_[idx_j]
                dir_save = f'{dir_tgt}/{shape_uid}/{view_id}/{axis}_{part_name}.png'
                if os.path.exists(dir_save): continue
                crop_area = (idx_j * img_size, idx_i * img_size, (idx_j + 1) * img_size, (idx_i + 1) * img_size)
                img_slice = img.crop(crop_area)
                img_slice.save(dir_save)

if __name__ == "__main__":
    args = get_parser()
    crop_slices(args)