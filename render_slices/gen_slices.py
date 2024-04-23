import os
from joblib import Parallel, delayed
from csv import reader
import json

path_root_objaverse = '/data/wangyz/03_datasets/'
path_output = '../data/objaverse/01_img_slices'
path_view = '../data/objaverse/00_img_input'
path_blender = '/data/wangyz/04_blender_renderer/blender-3.6.0-linux-x64'

def is_file_bigger_than_100MB(file_path):
    # Get the size of the file in bytes
    file_size = os.path.getsize(file_path)
    # Convert bytes to megabytes
    file_size_MB = file_size / (1024 * 1024)
    return file_size_MB > 100

def gen_slices(sid, spath):
    # check if the mesh file is bigger than 100MB, large files tend to make Blender crash when slicing
    object_path = f'{path_root_objaverse}/{spath}'
    object_uid = os.path.basename(object_path).split(".")[0]
    if is_file_bigger_than_100MB(object_path): return
    if os.path.exists(f'{path_output}/{object_uid}/011/Z_4.png'): return
    
    try:
        cmd = f'{path_blender}/blender -b -P blender_script_slices.py -- \
                --object_path {object_path} \
                --output_dir ./{path_output} \
                --engine CYCLES \
                --view_path {path_view} \
                --slice_direction camera \
                --num_images 12 '
        
        os.system(cmd)
    except:
        f = open(f'./logs/failed/{sid}.txt', 'w')
        f.close()
        return

def get_shape_ids_and_paths():

    shape_ids = []
    sid2spath = {}
    shape_paths = []
    # Open the JSON file

    with open('../data/objaverse/input_models_path-lvis.json', 'r') as file:
        data = json.load(file)
        for index, item in enumerate(data):
            if os.path.exists(f'{path_root_objaverse}/{item}'):
                shape_id = item.split('/')[-1].split('.')[0]
                shape_ids.append(shape_id)
                shape_paths.append(item)
                sid2spath[shape_id] = item
    
    return shape_ids, sid2spath

def main():
    shape_ids, sid2spath = get_shape_ids_and_paths()
    shape_ids = ['0a0c7e40a66d4fd090f549599f2f2c9d']  # this is an example, delete this line when creating a dataset
    with Parallel(n_jobs=8) as p:
        p(delayed(gen_slices)(sid=sid, spath=sid2spath[sid]) for idx, sid in enumerate(shape_ids))

main()
