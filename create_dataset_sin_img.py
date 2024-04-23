import os
import shutil
import numpy as np
import argparse
import pickle
from PIL import Image

def save_pickle(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default='./imgs/demo/input.png')
    parser.add_argument("--name_dataset", type=str, default='custom_sin_img')
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--resize_img", type=bool, default=False)
    parser.add_argument("--center_obj", type=bool, default=True)
    args = parser.parse_args()
    return args

def create_dataset(args):
    dir_tgt = f'./data/{args.name_dataset}'
    os.makedirs(dir_tgt, exist_ok=True)
    object_uid = '00000'
    dir_names = ['00_img_input', '01_img_slices', '02_sdfs', '03_splits']
    for dir_name in dir_names:
        os.makedirs(f'{dir_tgt}/{dir_name}', exist_ok=True)
    
    # save image
    img = Image.open(args.img_path)
    os.makedirs(f'{dir_tgt}/00_img_input/{object_uid}', exist_ok=True)
    img_path_new = f'{dir_tgt}/00_img_input/{object_uid}/004.png'
    assert (img.mode == 'RGBA')
    if args.center_obj:
        # we assumed the camera points to the centre of the object
        # centre the 2D bbox could be helpful, but does not guarantee that object 3D centre is aligned
        alpha = img.split()[3]
        bbox = alpha.getbbox()
        width, height = img.size
        object_width = bbox[2] - bbox[0]
        object_height = bbox[3] - bbox[1]
        offset_x = (width - object_width) // 2 - bbox[0]
        offset_y = (height - object_height) // 2 - bbox[1]
        new_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        new_image.paste(img, (offset_x, offset_y), mask=alpha)
        img = new_image
    if args.resize_img:
        img_rs = img.resize((args.img_size, args.img_size), Image.ANTIALIAS)
        img_rs.save(img_path_new)
    else:
        # shutil.copy(args.img_path, img_path_new)
        img.save(img_path_new, 'PNG')
    # write meta pkl
    K = np.zeros((3, 3))
    azimuths = np.zeros(12)
    elevations = np.zeros(12)
    distances = np.ones(12) * 1.2
    cam_poses = np.zeros((12, 3, 4))
    scale_rand = 1.0
    offset_rand = np.zeros(3)
    save_pickle([K, azimuths, elevations, distances, cam_poses, scale_rand, offset_rand], f'{dir_tgt}/00_img_input/{object_uid}/meta.pkl')

    # 01_img_slices
    os.makedirs(f'{dir_tgt}/01_img_slices/{object_uid}/004', exist_ok=True)
    for axis in ['X', 'Y', 'Z']:
        for part in ['1', '2', '3', '4']:
            img_slice = Image.new("RGBA", (args.img_size, args.img_size))
            img_slice.save(f'{dir_tgt}/01_img_slices/{object_uid}/004/{axis}_{part}.png')
    
    # 02_sdfs
    arr_sdf = np.zeros((16384, 4))
    np.save(f'{dir_tgt}/02_sdfs/{object_uid}.npy', arr_sdf)

    # 03_splits
    os.makedirs(f'{dir_tgt}/03_splits', exist_ok=True)
    for split in ['train', 'val', 'test']:
        fout = open(f'{dir_tgt}/03_splits/{split}.lst', 'w')
        # create 00_img_input
        fout.write(object_uid)
        fout.close()


if __name__ == "__main__":
    args = get_parser()
    create_dataset(args)