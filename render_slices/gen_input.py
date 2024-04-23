import os
from joblib import Parallel, delayed
from csv import reader
import json

path_root_objaverse = '/data/wangyz/03_datasets/'
path_output = '../data/objaverse/00_img_input'
path_blender = '/data/wangyz/04_blender_renderer/blender-3.6.0-linux-x64'

def gen_input(sid, spath):

    try:
        cmd = f'{path_blender}/blender -b -P blender_script_input.py -- \
                --object_path {path_root_objaverse}/{spath} \
                --output_dir ./{path_output} \
                --engine CYCLES \
                --num_images 12 '

        os.system(cmd)
    except:
        f = open(f'./processed_objaverse_input_lvis/dataset_input/failed/{sid}.txt', 'w')
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
                shape_ids.append(item.split('/')[-1].split('.')[0])
                shape_paths.append(item)
                sid2spath[item.split('/')[-1].split('.')[0]] = item
    return shape_ids, sid2spath

def main():

    shape_ids, sid2spath = get_shape_ids_and_paths()
    shape_ids = ['0a0c7e40a66d4fd090f549599f2f2c9d'] # this is an example, delete this line when creating a dataset
    with Parallel(n_jobs=8) as p:
        p(delayed(gen_input)(sid=sid, spath=sid2spath[sid]) for idx, sid in enumerate(shape_ids))

main()
